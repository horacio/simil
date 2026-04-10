#!/usr/bin/env python3
"""Build a distributable simil index from the MTG-Jamendo dataset.

MTG-Jamendo is a 55,000-track CC-licensed dataset curated by the Music
Technology Group at Universitat Pompeu Fabra — the same researchers who
created the EffNet-discogs embedding model used by simil and cosine.club.
This makes it the highest-quality dataset for effnet-discogs embeddings.

Dataset: https://github.com/MTG/mtg-jamendo-dataset
Paper:   https://arxiv.org/abs/1906.02687

Usage
-----
    uv run scripts/build_mtgjamendo_index.py

Options
-------
    --subset     all (default), electronic, ambient
    --limit      Max tracks (0 = all)
    --output     Output directory for archive (default: ./dist)
    --cache-dir  Cache directory (default: ~/.simil/build_cache/mtg-jamendo)
    --workers    Embedding worker threads (default: 4)
    --download-workers  Parallel download threads (default: 8)

Subsets
-------
    all         All 55,701 tracks across all genres
    electronic  Tracks tagged with electronic/techno/house/ambient/experimental
    ambient     Tracks tagged with ambient/drone/experimental/space

No API key required — downloads directly from Jamendo CDN via track IDs
in the MTG-Jamendo metadata files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# MTG-Jamendo metadata files (from their GitHub repo)
# ---------------------------------------------------------------------------

# Raw metadata TSV hosted on GitHub — contains track_id, artist, title, tags, duration
MTG_METADATA_URL = (
    "https://raw.githubusercontent.com/MTG/mtg-jamendo-dataset/"
    "master/data/tracks.tsv"
)

# Jamendo audio download URL template
AUDIO_URL_TEMPLATE = "https://mp3l.jamendo.com/?trackid={track_id}&format=mp32"

# Tags that map to our subsets
SUBSET_TAGS: dict[str, set[str]] = {
    "electronic": {
        "electronic", "techno", "house", "synth", "electronica",
        "downtempo", "chillout", "trance", "edm", "electro",
        "idm", "breakbeat", "dubstep", "drum and bass", "dnb",
    },
    "ambient": {
        "ambient", "drone", "experimental", "atmospheric", "space",
        "newage", "meditation", "chillout", "soundtrack",
    },
    "all": set(),  # empty = no filter
}


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
    print(f"  Packing {output_path.name}…")
    with tarfile.open(output_path, "w:gz", compresslevel=6) as tf:
        for f in sorted(index_dir.iterdir()):
            if f.is_file():
                tf.add(f, arcname=f.name)
                print(f"    + {f.name}  ({f.stat().st_size / 1e6:.1f} MB)")


# ---------------------------------------------------------------------------
# Step 1: Download and parse MTG-Jamendo metadata TSV
# ---------------------------------------------------------------------------


def fetch_metadata(
    cache_dir: Path,
    subset: str,
    limit: int = 0,
) -> list[dict]:
    """Download MTG-Jamendo metadata TSV and filter to *subset*.

    The TSV is cached locally — re-runs skip the download.

    Args:
        cache_dir: Directory to cache the TSV.
        subset:    One of "all", "electronic", "ambient".
        limit:     Max tracks (0 = all).

    Returns:
        List of track dicts: id, name, artist_name, tags, duration.
    """
    import httpx

    tsv_path = cache_dir / "tracks.tsv"

    if not tsv_path.exists():
        print(f"  Downloading MTG-Jamendo metadata TSV…")
        resp = httpx.get(MTG_METADATA_URL, timeout=60, follow_redirects=True)
        resp.raise_for_status()
        tsv_path.write_bytes(resp.content)
        print(f"  Saved to {tsv_path}")
    else:
        print(f"  Using cached metadata: {tsv_path}")

    # Parse TSV
    # Format: TRACK_ID\tARTIST_ID\tALBUM_ID\tPATH\tDURATION\tTAGS...
    # Header line starts with #
    lines = tsv_path.read_text(encoding="utf-8").splitlines()
    # Find header
    header_line = next((l for l in lines if l.startswith("TRACK")), None)
    if header_line is None:
        # Try first non-comment line
        header_line = next((l for l in lines if not l.startswith("#")), lines[0])

    headers = header_line.strip().split("\t")
    tracks = []

    for line in lines:
        if line.startswith("#") or line == header_line or not line.strip():
            continue
        parts = line.strip().split("\t")
        if len(parts) < 5:
            continue

        # TSV columns vary by version — try to map robustly
        row: dict[str, Any] = {}
        for i, h in enumerate(headers):
            if i < len(parts):
                row[h.lower().strip()] = parts[i]

        # Normalise key names
        track_id = (
            row.get("track_id") or row.get("id") or
            (parts[0] if parts else None)
        )
        if not track_id:
            continue

        # Tags are space-separated in the last column(s)
        tag_str = row.get("tags", "") or " ".join(parts[5:]) if len(parts) > 5 else ""
        tags = set(t.lower().strip() for t in tag_str.replace(",", " ").split() if t.strip())

        # Artist name: Jamendo doesn't include it in the TSV, we add from track path
        path_field = row.get("path", "") or (parts[3] if len(parts) > 3 else "")
        # path looks like: 01/0123456.mp3
        artist_id = row.get("artist_id") or (parts[1] if len(parts) > 1 else "")
        title = row.get("title") or Path(path_field).stem if path_field else track_id

        try:
            duration = float(row.get("duration") or parts[4])
        except (ValueError, IndexError):
            duration = 0.0

        if duration < 10:
            continue

        # Apply subset filter
        filter_tags = SUBSET_TAGS.get(subset, set())
        if filter_tags and not (tags & filter_tags):
            continue

        tracks.append({
            "id": track_id.strip(),
            "name": title,
            "artist_id": artist_id.strip(),
            "artist_name": f"Artist-{artist_id.strip()[:8]}",  # enriched later
            "duration": duration,
            "tags": sorted(tags),
        })

    print(f"  {len(tracks):,} tracks match subset '{subset}'")

    if limit and len(tracks) > limit:
        tracks = tracks[:limit]
        print(f"  Limited to {len(tracks):,} tracks.")

    return tracks


# ---------------------------------------------------------------------------
# Step 2: Enrich with artist names via Jamendo API (optional, best-effort)
# ---------------------------------------------------------------------------


def enrich_artist_names(
    tracks: list[dict],
    cache_dir: Path,
    client_id: str | None = None,
) -> None:
    """Best-effort: fill in artist names from Jamendo API if client_id given.

    Without a client_id, artist names stay as "Artist-XXXXXX".
    Enriched data is cached to avoid repeat API calls.
    """
    if not client_id:
        return

    enriched_path = cache_dir / "enriched_tracks.json"
    if enriched_path.exists():
        enriched = json.loads(enriched_path.read_text())
        id_map = {str(e["id"]): e for e in enriched}
        for t in tracks:
            if t["id"] in id_map:
                t["artist_name"] = id_map[t["id"]].get("artist_name", t["artist_name"])
                t["name"] = id_map[t["id"]].get("name", t["name"])
        print(f"  Loaded enriched metadata for {len(id_map):,} tracks from cache.")
        return

    import httpx

    print(f"  Enriching artist names via Jamendo API (batches of 200)…")
    ids = [t["id"] for t in tracks]
    id_to_track = {t["id"]: t for t in tracks}
    total = len(ids)
    done = 0

    with httpx.Client(timeout=30) as client:
        for i in range(0, len(ids), 200):
            batch = ids[i:i + 200]
            try:
                resp = client.get(
                    "https://api.jamendo.com/v3.0/tracks/",
                    params={
                        "client_id": client_id,
                        "format": "json",
                        "id[]": batch,
                        "fields": "id,name,artist_name",
                        "limit": 200,
                    },
                )
                resp.raise_for_status()
                for r in resp.json().get("results", []):
                    tid = str(r["id"])
                    if tid in id_to_track:
                        id_to_track[tid]["artist_name"] = r.get("artist_name", "")
                        id_to_track[tid]["name"] = r.get("name", id_to_track[tid]["name"])
                done += len(batch)
                print(f"  \r  {done:,}/{total:,}", end="", flush=True)
                time.sleep(0.3)
            except Exception as exc:
                print(f"\n  Warning: enrichment batch failed: {exc}", file=sys.stderr)

    print()
    enriched_path.write_text(json.dumps(tracks, indent=2))
    print(f"  Saved enriched metadata.")


# ---------------------------------------------------------------------------
# Step 3: Download audio
# ---------------------------------------------------------------------------


def download_audio(
    tracks: list[dict],
    audio_dir: Path,
    download_workers: int = 8,
) -> None:
    """Download MP3s for all tracks, skipping existing files."""
    import httpx

    audio_dir.mkdir(parents=True, exist_ok=True)

    work = [
        (t, audio_dir / f"{t['id']}.mp3")
        for t in tracks
        if not (audio_dir / f"{t['id']}.mp3").exists()
    ]
    already = len(tracks) - len(work)
    if already:
        print(f"  Skipping {already:,} already-downloaded tracks.")
    print(f"  Downloading {len(work):,} tracks ({download_workers} parallel)…")

    total = len(tracks)
    done = already
    failed = 0

    def _get(task: tuple[dict, Path]) -> bool:
        t, dest = task
        url = AUDIO_URL_TEMPLATE.format(track_id=t["id"])
        tmp = dest.with_suffix(".tmp")
        try:
            with httpx.stream("GET", url, timeout=60, follow_redirects=True) as resp:
                resp.raise_for_status()
                with tmp.open("wb") as fh:
                    for chunk in resp.iter_bytes(65536):
                        fh.write(chunk)
            tmp.rename(dest)
            return True
        except Exception:
            tmp.unlink(missing_ok=True)
            return False

    with ThreadPoolExecutor(max_workers=download_workers) as pool:
        futures = {pool.submit(_get, task): task for task in work}
        for future in as_completed(futures):
            ok = future.result()
            done += 1
            if not ok:
                failed += 1
            if done % 200 == 0 or done == total:
                print(f"  \r  {done:,}/{total:,}  ({failed} failed)", end="", flush=True)

    print()
    print(f"  Done: {done - failed:,} tracks, {failed} failures.")


# ---------------------------------------------------------------------------
# Step 4: Build index
# ---------------------------------------------------------------------------


def build_index(
    audio_dir: Path,
    tracks_meta: list[dict],
    library_name: str,
    embedder_name: str,
    workers: int,
) -> Path:
    from simil.catalog import TrackCatalog
    from simil.cli.main import _get_index_dir
    from simil.config import Settings
    from simil.embedders import get_embedder
    from simil.index.numpy_index import NumpyIndex
    from simil.library.indexer import Indexer
    import simil.library.metadata as _meta_mod

    meta_lookup = {f"{t['id']}.mp3": t for t in tracks_meta}
    _orig_extract = _meta_mod.extract_metadata

    def _patched_extract(path: Path) -> dict:
        result = _orig_extract(path)
        if path.name in meta_lookup:
            jt = meta_lookup[path.name]
            result.setdefault("title", jt.get("name"))
            result.setdefault("artist", jt.get("artist_name"))
            if jt.get("duration"):
                result["duration"] = jt["duration"]
        return result

    _meta_mod.extract_metadata = _patched_extract  # type: ignore[assignment]

    index_dir = _get_index_dir(library_name)
    index_dir.mkdir(parents=True, exist_ok=True)

    emb = get_embedder(embedder_name)

    # Load existing index/catalog if present so incremental builds resume cleanly.
    if (index_dir / "meta.json").exists():
        try:
            idx = NumpyIndex.load(index_dir)
            cat = TrackCatalog.load(index_dir / "catalog.json", library_id=library_name)
            already = cat.size
            print(f"  Resuming — {already:,} tracks already indexed.")
        except Exception as exc:
            print(f"  Warning: could not load existing index ({exc}). Starting fresh.")
            idx = NumpyIndex(
                embedding_dim=emb.embedding_dim, embedder_name=embedder_name,
                library_id=library_name, audio_config=emb.audio_config,
            )
            cat = TrackCatalog(library_id=library_name)
    else:
        idx = NumpyIndex(
            embedding_dim=emb.embedding_dim, embedder_name=embedder_name,
            library_id=library_name, audio_config=emb.audio_config,
        )
        cat = TrackCatalog(library_id=library_name)

    settings = Settings(library_name=library_name, workers=workers)
    indexer = Indexer(embedder=emb, index=idx, catalog=cat, settings=settings)

    try:
        # full=False → incremental: skips tracks already in the index.
        # If the build was interrupted, re-running resumes from where it left off.
        result = indexer.build(audio_dir, full=False)
    finally:
        _meta_mod.extract_metadata = _orig_extract

    print(
        f"  Indexed {result.indexed:,} | skipped {result.skipped} | "
        f"failed {len(result.failed)} | {result.duration_seconds:.0f}s"
    )
    if not (index_dir / "meta.json").exists():
        sys.exit("Error: indexing produced no meta.json.")

    return index_dir


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--subset", choices=["all", "electronic", "ambient"], default="all",
        help="Track subset to index (default: all)",
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Max tracks (0 = all)",
    )
    parser.add_argument(
        "--embedder", default="effnet-discogs",
    )
    parser.add_argument(
        "--cache-dir", type=Path,
        default=Path.home() / ".simil" / "build_cache" / "mtg-jamendo",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("dist"),
    )
    parser.add_argument(
        "--workers", type=int, default=4,
    )
    parser.add_argument(
        "--download-workers", type=int, default=8,
    )
    parser.add_argument(
        "--client-id", default="",
        help="Optional Jamendo client ID for artist name enrichment.",
    )
    args = parser.parse_args()

    subset_slug = f"mtg-jamendo-{args.subset}" if args.subset != "all" else "mtg-jamendo"
    archive_name = f"{subset_slug}-{args.embedder}.tar.gz"
    library_name = f"build_{subset_slug.replace('-', '_')}"

    args.output.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    archive_path = args.output / archive_name

    print(f"\nsimil MTG-Jamendo index builder")
    print(f"  Subset     : {args.subset}")
    print(f"  Embedder   : {args.embedder}")
    print(f"  Cache      : {args.cache_dir}")
    print(f"  Output     : {archive_path}")
    print()

    # ── Step 1: Metadata ───────────────────────────────────────────────────
    print("[1/5] Fetching MTG-Jamendo metadata…")
    tracks = fetch_metadata(args.cache_dir, args.subset, args.limit)
    print(f"  Processing {len(tracks):,} tracks.")

    # ── Step 2: Enrich artist names (optional) ─────────────────────────────
    if args.client_id:
        print("\n[2/5] Enriching artist/title metadata…")
        enrich_artist_names(tracks, args.cache_dir, args.client_id)
    else:
        print("\n[2/5] Skipping enrichment (no --client-id / JAMENDO_CLIENT_ID).")
        print("  Artist names will be generic. Pass --client-id for real names.")

    # ── Step 3: Download audio ─────────────────────────────────────────────
    print(f"\n[3/5] Downloading audio…")
    audio_dir = args.cache_dir / "audio" / args.subset
    download_audio(tracks, audio_dir, download_workers=args.download_workers)

    # ── Step 4: Build index ────────────────────────────────────────────────
    print(f"\n[4/5] Building index with {args.embedder}…")
    index_dir = build_index(audio_dir, tracks, library_name, args.embedder, args.workers)

    # ── Step 5: Package ────────────────────────────────────────────────────
    print(f"\n[5/5] Packaging archive…")
    _make_archive(index_dir, archive_path)

    sha = _sha256(archive_path)
    size_bytes = archive_path.stat().st_size
    size_mb = size_bytes / (1024 * 1024)
    catalog_data = json.loads((index_dir / "catalog.json").read_text())
    track_count = len(catalog_data.get("tracks", []))

    release_url = (
        f"https://github.com/horacio/simil/releases/download/indexes/{archive_name}"
    )

    description = (
        f"MTG-Jamendo {args.subset} — {track_count:,} CC-licensed tracks "
        f"(curated by MTG Barcelona, the EffNet-discogs model creators)"
    )

    print(f"\n{'─' * 62}")
    print(f"  Archive    : {archive_path}")
    print(f"  Tracks     : {track_count:,}")
    print(f"  Size       : {size_mb:.1f} MB  ({size_bytes:,} bytes)")
    print(f"  SHA-256    : {sha}")
    print(f"{'─' * 62}")
    print()
    print("Next steps:")
    print(f"  1. gh release upload indexes {archive_path}")
    print()
    print(f"  2. Add to registry.json:")
    print(f'     {{')
    print(f'       "name":        "{subset_slug}",')
    print(f'       "description": "{description}",')
    print(f'       "embedder":    "{args.embedder}",')
    print(f'       "tracks":      {track_count},')
    print(f'       "url":         "{release_url}",')
    print(f'       "sha256":      "{sha}",')
    print(f'       "size_bytes":  {size_bytes}')
    print(f'     }}')
    print()
    print(f"  3. git add registry.json && git push")
    print(f"     → simil fetch {subset_slug}")


if __name__ == "__main__":
    main()
