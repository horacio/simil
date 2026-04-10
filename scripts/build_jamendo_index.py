#!/usr/bin/env python3
"""Build a distributable simil index from the Jamendo catalog.

Jamendo hosts 600k+ Creative Commons-licensed tracks with a public API.
This script fetches track metadata, downloads audio, indexes it with
effnet-discogs, and packages the result for upload to GitHub Releases.

Usage
-----
    # Free API key: https://devportal.jamendo.com  (takes ~1 minute)
    export JAMENDO_CLIENT_ID=your_client_id

    uv run scripts/build_jamendo_index.py                  # electronic preset (~30k tracks)
    uv run scripts/build_jamendo_index.py --preset ambient # ambient/drone subset
    uv run scripts/build_jamendo_index.py --limit 5000     # smaller test run

Options
-------
    --preset        Track selection preset: electronic (default), ambient, all
    --limit         Max tracks to download (0 = no limit)
    --client-id     Jamendo API client ID (overrides JAMENDO_CLIENT_ID env var)
    --cache-dir     Cache directory for downloads (default: ~/.simil/build_cache/jamendo)
    --output        Output directory for the archive (default: ./dist)
    --workers       Embedding worker threads (default: 4)
    --download-workers  Parallel audio download threads (default: 8)
    --skip-download Already have audio? Pass the directory and skip to indexing.

Presets
-------
    electronic  tags: electronic, techno, house, synth, downtempo, chillout (≈30k)
    ambient     tags: ambient, drone, experimental, atmospheric, space (≈15k)
    all         no tag filter — full catalog (≈600k, takes many hours)

After the script finishes it prints the exact values to paste into registry.json,
then you upload the archive and push.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------

PRESETS: dict[str, dict[str, Any]] = {
    "electronic": {
        "tags": ["electronic", "techno", "house", "synth", "downtempo", "chillout"],
        "description": "Jamendo — ~30,000 electronic/techno/house/downtempo tracks (CC-licensed)",
        "archive_slug": "jamendo-electronic",
    },
    "ambient": {
        "tags": ["ambient", "drone", "experimental", "atmospheric", "space"],
        "description": "Jamendo — ~15,000 ambient/drone/experimental tracks (CC-licensed)",
        "archive_slug": "jamendo-ambient",
    },
    "all": {
        "tags": [],  # no filter
        "description": "Jamendo — full catalog, ~600,000 CC-licensed tracks",
        "archive_slug": "jamendo-all",
    },
}

JAMENDO_API = "https://api.jamendo.com/v3.0/tracks/"
PAGE_SIZE = 200  # Jamendo max per request
REQUEST_DELAY = 0.5  # seconds between API pages (be a good citizen)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            sha.update(chunk)
    return sha.hexdigest()


def _make_archive(index_dir: Path, output_path: Path) -> None:
    print(f"  Packing archive {output_path.name}…")
    with tarfile.open(output_path, "w:gz", compresslevel=6) as tf:
        for f in sorted(index_dir.iterdir()):
            if f.is_file():
                tf.add(f, arcname=f.name)
                print(f"    + {f.name}  ({f.stat().st_size / 1e6:.1f} MB)")


# ---------------------------------------------------------------------------
# Step 1: Fetch track metadata from Jamendo API
# ---------------------------------------------------------------------------


def fetch_metadata(
    client_id: str,
    tags: list[str],
    cache_path: Path,
    limit: int = 0,
) -> list[dict]:
    """Page through Jamendo API and collect track metadata.

    Results are cached to *cache_path* so re-runs are free.

    Args:
        client_id: Jamendo API client ID.
        tags:      Genre/tag filter list. Empty = no filter.
        cache_path: JSON file to cache results.
        limit:     Max tracks (0 = all).

    Returns:
        List of track dicts with id, name, artist_name, audiodownload, duration.
    """
    import httpx

    if cache_path.exists():
        tracks = json.loads(cache_path.read_text())
        if limit and len(tracks) > limit:
            tracks = tracks[:limit]
        print(f"  [cache] Loaded {len(tracks):,} tracks from {cache_path}")
        return tracks

    tracks: list[dict] = []
    offset = 0
    tags_str = " ".join(tags)  # Jamendo accepts space-separated tags (OR logic)

    print(f"  Fetching metadata from Jamendo API…")
    if tags:
        print(f"  Tags filter : {', '.join(tags)}")
    print(f"  This uses ~{(limit or 50_000) // PAGE_SIZE + 1} API requests.")

    with httpx.Client(timeout=30) as client:
        while True:
            params: dict[str, Any] = {
                "client_id": client_id,
                "format": "json",
                "limit": PAGE_SIZE,
                "offset": offset,
                "audiodownload_allowed": 1,
                "audioformat": "mp32",
                "fields": "id,name,artist_name,duration,audiodownload",
            }
            if tags_str:
                params["tags"] = tags_str

            try:
                resp = client.get(JAMENDO_API, params=params)
                resp.raise_for_status()
                data = resp.json()
            except Exception as exc:
                print(f"\n  API error at offset {offset}: {exc}", file=sys.stderr)
                print(  "  Retrying in 5s…", file=sys.stderr)
                time.sleep(5)
                continue

            results = data.get("results", [])
            if not results:
                break

            # Filter out tracks without a download URL or zero duration
            valid = [
                t for t in results
                if t.get("audiodownload") and t.get("duration", 0) > 10
            ]
            tracks.extend(valid)

            fetched = offset + len(results)
            print(f"  \r  Fetched {fetched:,} | kept {len(tracks):,}", end="", flush=True)

            if limit and len(tracks) >= limit:
                tracks = tracks[:limit]
                break

            if len(results) < PAGE_SIZE:
                break  # last page

            offset += PAGE_SIZE
            time.sleep(REQUEST_DELAY)

    print()  # newline after \r progress

    # Cache to disk
    cache_path.write_text(json.dumps(tracks, indent=2))
    print(f"  Cached {len(tracks):,} tracks to {cache_path}")
    return tracks


# ---------------------------------------------------------------------------
# Step 2: Download audio files
# ---------------------------------------------------------------------------


def download_audio(
    tracks: list[dict],
    audio_dir: Path,
    download_workers: int = 8,
) -> list[Path]:
    """Download MP3 files for *tracks* into *audio_dir*.

    Already-downloaded files are skipped. Returns the list of local paths
    that succeeded (failed downloads are skipped with a warning).
    """
    import httpx

    audio_dir.mkdir(parents=True, exist_ok=True)

    # Build work list — skip already-downloaded files
    work = []
    for t in tracks:
        dest = audio_dir / f"{t['id']}.mp3"
        if not dest.exists():
            work.append((t, dest))

    already = len(tracks) - len(work)
    if already:
        print(f"  Skipping {already:,} already-downloaded files.")
    print(f"  Downloading {len(work):,} tracks with {download_workers} workers…")

    done = already
    failed = 0

    def _download_one(task: tuple[dict, Path]) -> Path | None:
        t, dest = task
        tmp = dest.with_suffix(".tmp")
        try:
            with httpx.stream("GET", t["audiodownload"], timeout=60,
                              follow_redirects=True) as resp:
                resp.raise_for_status()
                with tmp.open("wb") as fh:
                    for chunk in resp.iter_bytes(65536):
                        fh.write(chunk)
            tmp.rename(dest)
            return dest
        except Exception as exc:
            tmp.unlink(missing_ok=True)
            return None

    with ThreadPoolExecutor(max_workers=download_workers) as pool:
        futures = {pool.submit(_download_one, task): task for task in work}
        for future in as_completed(futures):
            result = future.result()
            done += 1
            if result is None:
                failed += 1
            if done % 100 == 0 or done == len(tracks):
                print(
                    f"  \r  {done:,}/{len(tracks):,} done  "
                    f"({failed} failed)",
                    end="", flush=True,
                )

    print()  # newline
    print(f"  Download complete: {done - failed:,} tracks, {failed} failures.")

    return [audio_dir / f"{t['id']}.mp3" for t in tracks
            if (audio_dir / f"{t['id']}.mp3").exists()]


# ---------------------------------------------------------------------------
# Step 3: Build metadata sidecar for catalog enrichment
# ---------------------------------------------------------------------------


def write_metadata_sidecar(tracks: list[dict], audio_dir: Path) -> None:
    """Write a JSON sidecar next to each MP3 so the indexer picks up artist/title.

    simil's metadata extractor (mutagen) can't read artist/title from bare MP3
    URLs — they have no ID3 tags. This sidecar is read by a custom metadata
    hook added below.
    """
    lookup = {str(t["id"]): t for t in tracks}
    sidecar_path = audio_dir / "_jamendo_meta.json"
    sidecar_path.write_text(json.dumps(lookup, indent=2))
    print(f"  Wrote metadata sidecar → {sidecar_path.name}")


# ---------------------------------------------------------------------------
# Step 4: Index with simil (with metadata injection)
# ---------------------------------------------------------------------------


def build_index(
    audio_dir: Path,
    tracks_meta: list[dict],
    library_name: str,
    embedder_name: str,
    workers: int,
) -> Path:
    """Run simil's Indexer over *audio_dir* and return the index directory."""
    from simil.catalog import TrackCatalog
    from simil.cli.main import _get_index_dir
    from simil.config import Settings
    from simil.core.models import Track
    from simil.embedders import get_embedder
    from simil.index.numpy_index import NumpyIndex
    from simil.library.indexer import Indexer
    from simil.library.scanner import content_id

    # Build a lookup so we can inject artist/title that Jamendo provides
    # but the MP3 files themselves don't have in ID3 tags.
    meta_lookup: dict[str, dict] = {}
    for t in tracks_meta:
        fname = f"{t['id']}.mp3"
        meta_lookup[fname] = t

    index_dir = _get_index_dir(library_name)
    index_dir.mkdir(parents=True, exist_ok=True)

    emb = get_embedder(embedder_name)
    idx = NumpyIndex(
        embedding_dim=emb.embedding_dim,
        embedder_name=embedder_name,
        library_id=library_name,
        audio_config=emb.audio_config,
    )
    cat = TrackCatalog(library_id=library_name)
    settings = Settings(library_name=library_name, workers=workers)

    # Monkey-patch metadata extraction to inject Jamendo artist/title
    import simil.library.metadata as _meta_mod
    _orig_extract = _meta_mod.extract_metadata

    def _patched_extract(path: Path) -> dict:
        result = _orig_extract(path)
        fname = path.name
        if fname in meta_lookup:
            jt = meta_lookup[fname]
            result.setdefault("title", jt.get("name"))
            result.setdefault("artist", jt.get("artist_name"))
            result["duration"] = jt.get("duration")
        return result

    _meta_mod.extract_metadata = _patched_extract  # type: ignore[assignment]

    try:
        indexer = Indexer(embedder=emb, index=idx, catalog=cat, settings=settings)
        result = indexer.build(audio_dir, full=True)
    finally:
        _meta_mod.extract_metadata = _orig_extract  # restore

    print(
        f"  Indexed {result.indexed:,} | "
        f"skipped {result.skipped} | "
        f"failed {len(result.failed)} | "
        f"{result.duration_seconds:.0f}s"
    )
    if result.failed:
        n = min(5, len(result.failed))
        print(f"  First {n} failures:")
        for p in result.failed[:n]:
            print(f"    {p}")

    if not (index_dir / "meta.json").exists():
        sys.exit("Error: indexing produced no meta.json — check errors above.")

    return index_dir


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--preset", choices=list(PRESETS), default="electronic",
        help="Track selection preset (default: electronic)",
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Max tracks to index (0 = no limit, default: 0)",
    )
    parser.add_argument(
        "--client-id", default=os.environ.get("JAMENDO_CLIENT_ID", ""),
        help="Jamendo API client ID (or set JAMENDO_CLIENT_ID env var). "
             "Free at https://devportal.jamendo.com",
    )
    parser.add_argument(
        "--embedder", default="effnet-discogs",
        help="Embedder name (default: effnet-discogs)",
    )
    parser.add_argument(
        "--cache-dir", type=Path,
        default=Path.home() / ".simil" / "build_cache" / "jamendo",
        help="Cache directory for downloads (default: ~/.simil/build_cache/jamendo)",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("dist"),
        help="Output directory for archive (default: ./dist)",
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Embedding worker threads (default: 4)",
    )
    parser.add_argument(
        "--download-workers", type=int, default=8,
        help="Parallel audio download threads (default: 8)",
    )
    parser.add_argument(
        "--skip-download", type=Path, default=None,
        help="Skip download — use this existing audio directory directly.",
    )
    args = parser.parse_args()

    if not args.client_id and not args.skip_download:
        print(
            "Error: Jamendo client ID required.\n"
            "  Get a free one at: https://devportal.jamendo.com\n"
            "  Then: export JAMENDO_CLIENT_ID=your_id\n"
            "  Or:   --client-id your_id",
            file=sys.stderr,
        )
        sys.exit(1)

    preset = PRESETS[args.preset]
    archive_name = f"{preset['archive_slug']}-{args.embedder}.tar.gz"
    library_name = f"build_{preset['archive_slug'].replace('-', '_')}"

    args.output.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    archive_path = args.output / archive_name

    limit_str = f"{args.limit:,}" if args.limit else "no limit"
    print(f"\nsimil Jamendo index builder")
    print(f"  Preset     : {args.preset}")
    print(f"  Tags       : {', '.join(preset['tags']) or '(all)'}")
    print(f"  Limit      : {limit_str}")
    print(f"  Embedder   : {args.embedder}")
    print(f"  Cache dir  : {args.cache_dir}")
    print(f"  Output     : {archive_path}")
    print()

    # ── Step 1: Metadata ───────────────────────────────────────────────────────
    if args.skip_download:
        audio_dir = args.skip_download.expanduser().resolve()
        if not audio_dir.is_dir():
            sys.exit(f"Error: --skip-download path is not a directory: {audio_dir}")
        print(f"[1/4] Skipping download, using: {audio_dir}")
        tracks_meta: list[dict] = []
    else:
        print("[1/4] Fetching track metadata from Jamendo…")
        metadata_cache = args.cache_dir / f"{args.preset}_tracks.json"
        tracks_meta = fetch_metadata(
            client_id=args.client_id,
            tags=preset["tags"],
            cache_path=metadata_cache,
            limit=args.limit,
        )
        print(f"  Total tracks to process: {len(tracks_meta):,}")

        # ── Step 2: Download audio ─────────────────────────────────────────────
        print(f"\n[2/4] Downloading audio…")
        audio_dir = args.cache_dir / args.preset
        download_audio(tracks_meta, audio_dir, download_workers=args.download_workers)
        write_metadata_sidecar(tracks_meta, audio_dir)

    # ── Step 3: Build index ────────────────────────────────────────────────────
    print(f"\n[3/4] Building index with {args.embedder}…")
    index_dir = build_index(
        audio_dir=audio_dir,
        tracks_meta=tracks_meta,
        library_name=library_name,
        embedder_name=args.embedder,
        workers=args.workers,
    )

    # ── Step 4: Package ────────────────────────────────────────────────────────
    print(f"\n[4/4] Packaging archive…")
    _make_archive(index_dir, archive_path)

    sha = _sha256(archive_path)
    size_bytes = archive_path.stat().st_size
    size_mb = size_bytes / (1024 * 1024)

    # Read track count from the built catalog
    catalog_data = json.loads((index_dir / "catalog.json").read_text())
    track_count = len(catalog_data.get("tracks", []))

    release_url = (
        f"https://github.com/horacio/simil/releases/download/indexes/{archive_name}"
    )

    print(f"\n{'─' * 62}")
    print(f"  Archive    : {archive_path}")
    print(f"  Tracks     : {track_count:,}")
    print(f"  Size       : {size_mb:.1f} MB  ({size_bytes:,} bytes)")
    print(f"  SHA-256    : {sha}")
    print(f"{'─' * 62}")
    print()
    print("Next steps:")
    print(f"  1. Upload to GitHub Releases:")
    print(f"       gh release upload indexes {archive_path}")
    print()
    print(f"  2. Add to registry.json:")
    print(f'     {{')
    print(f'       "name":        "{preset["archive_slug"]}",')
    print(f'       "description": "{preset["description"]}",')
    print(f'       "embedder":    "{args.embedder}",')
    print(f'       "tracks":      {track_count},')
    print(f'       "url":         "{release_url}",')
    print(f'       "sha256":      "{sha}",')
    print(f'       "size_bytes":  {size_bytes}')
    print(f'     }}')
    print()
    print(f"  3. git add registry.json && git commit -m 'Add {preset['archive_slug']} index' && git push")
    print(f"     → users can now: simil fetch {preset['archive_slug']}")


if __name__ == "__main__":
    main()
