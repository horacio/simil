#!/usr/bin/env python3
"""Build a distributable simil index from the Free Music Archive (FMA).

This is a one-time script run by maintainers to produce the pre-built index
that users download via ``simil fetch``.  It downloads the FMA audio dataset,
runs the simil indexer over it, and packages the result as a tar.gz archive
ready for upload to GitHub Releases.

Usage
-----
    uv run scripts/build_fma_index.py [OPTIONS]

Options
-------
    --subset    fma_small (8k tracks, ~7 GB) or fma_medium (25k, ~22 GB)
                Default: fma_small
    --embedder  Embedder name.  Default: effnet-discogs
    --output    Directory for the output archive.  Default: ./dist
    --audio-dir Use a pre-downloaded audio directory instead of downloading.
    --workers   Number of embedding worker threads.  Default: 4

After this script finishes, upload the .tar.gz to GitHub Releases under the
tag ``indexes``, then update ``registry.json`` with the printed URL, sha256,
and size_bytes.

Example
-------
    uv run scripts/build_fma_index.py
    # → dist/fma-small-effnet-discogs.tar.gz
    # → sha256: abc123…
    # → size:   42.3 MB

    gh release create indexes --title "Pre-built indexes" \\
        dist/fma-small-effnet-discogs.tar.gz

    # Then edit registry.json with the printed values.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

# ---------------------------------------------------------------------------
# FMA dataset metadata
# ---------------------------------------------------------------------------

FMA_SUBSETS = {
    "fma_small": {
        "audio_url": "https://os.unil.cloud.switch.ch/fma/fma_small.zip",
        "audio_size_hint": "7.2 GB",
        "tracks": 8000,
        "description": (
            "Free Music Archive — 8,000 tracks across 8 genres "
            "(30-second clips, CC-licensed)"
        ),
    },
    "fma_medium": {
        "audio_url": "https://os.unil.cloud.switch.ch/fma/fma_medium.zip",
        "audio_size_hint": "22 GB",
        "tracks": 25000,
        "description": (
            "Free Music Archive — 25,000 tracks across 16 genres "
            "(30-second clips, CC-licensed)"
        ),
    },
}

# Short name mapping used for output filenames
_SUBSET_SHORTNAME = {
    "fma_small": "fma-small",
    "fma_medium": "fma-medium",
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


def _download(url: str, dest: Path) -> None:
    """Download *url* to *dest* using yt-dlp's httpx or plain urllib as fallback."""
    print(f"  Downloading {url}")
    print(f"  → {dest}")
    try:
        import httpx
        from rich.progress import (
            BarColumn, DownloadColumn, Progress, TextColumn,
            TimeRemainingColumn, TransferSpeedColumn,
        )

        with Progress(
            TextColumn("  [cyan]{task.fields[name]}"),
            BarColumn(), DownloadColumn(),
            TransferSpeedColumn(), TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task("dl", name=dest.name, total=None)
            with httpx.stream("GET", url, follow_redirects=True, timeout=300) as resp:
                resp.raise_for_status()
                total = int(resp.headers.get("content-length", 0)) or None
                progress.update(task, total=total)
                with dest.open("wb") as fh:
                    for chunk in resp.iter_bytes(65536):
                        fh.write(chunk)
                        progress.update(task, advance=len(chunk))
    except ImportError:
        # Fallback: use curl if httpx not available
        subprocess.run(["curl", "-L", "-o", str(dest), url], check=True)


def _unzip(archive: Path, dest_dir: Path) -> None:
    print(f"  Extracting {archive.name}…")
    with zipfile.ZipFile(archive, "r") as zf:
        members = zf.namelist()
        for i, member in enumerate(members, 1):
            zf.extract(member, dest_dir)
            if i % 1000 == 0:
                print(f"    {i}/{len(members)}")
    print(f"    Done ({len(members)} files)")


def _make_archive(index_dir: Path, output_path: Path) -> None:
    """Pack index_dir contents (no wrapping directory) into output_path."""
    print(f"  Creating archive {output_path.name}…")
    with tarfile.open(output_path, "w:gz", compresslevel=6) as tf:
        for f in sorted(index_dir.iterdir()):
            if f.is_file():
                tf.add(f, arcname=f.name)
                print(f"    + {f.name}  ({f.stat().st_size / 1e6:.1f} MB)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--subset", choices=list(FMA_SUBSETS), default="fma_small",
        help="FMA subset to index (default: fma_small)",
    )
    parser.add_argument(
        "--embedder", default="effnet-discogs",
        help="Embedder name (default: effnet-discogs)",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("dist"),
        help="Output directory for the archive (default: ./dist)",
    )
    parser.add_argument(
        "--audio-dir", type=Path, default=None,
        help="Path to a pre-downloaded FMA audio directory. Skips the download step.",
    )
    parser.add_argument(
        "--cache-dir", type=Path,
        default=Path.home() / ".simil" / "build_cache",
        help=(
            "Directory to cache the downloaded zip so reruns skip the download "
            "(default: ~/.simil/build_cache)"
        ),
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Number of parallel embedding workers (default: 4)",
    )
    args = parser.parse_args()

    subset_info = FMA_SUBSETS[args.subset]
    short_name = _SUBSET_SHORTNAME[args.subset]
    archive_name = f"{short_name}-{args.embedder}.tar.gz"

    args.output.mkdir(parents=True, exist_ok=True)
    archive_path = args.output / archive_name

    print(f"\nsimil index builder")
    print(f"  Subset   : {args.subset}  ({subset_info['tracks']:,} tracks)")
    print(f"  Embedder : {args.embedder}")
    print(f"  Output   : {archive_path}")
    print()

    # ── Step 1: Obtain audio ────────────────────────────────────────────────
    if args.audio_dir:
        audio_dir = args.audio_dir.expanduser().resolve()
        if not audio_dir.is_dir():
            sys.exit(f"Error: --audio-dir {audio_dir} is not a directory")
        print(f"[1/4] Using existing audio directory: {audio_dir}")
    else:
        cache_dir = args.cache_dir.expanduser()
        cache_dir.mkdir(parents=True, exist_ok=True)
        zip_path = cache_dir / f"{args.subset}.zip"
        audio_dir = cache_dir / args.subset

        if audio_dir.is_dir():
            # Already extracted from a previous run — skip download + unzip
            print(f"[1/4] Using cached audio: {audio_dir}")
        else:
            if zip_path.exists():
                print(f"[1/4] Using cached zip (skipping download): {zip_path}")
            else:
                print(
                    f"[1/4] Downloading FMA audio ({subset_info['audio_size_hint']})…\n"
                    f"  Cached to: {zip_path}\n"
                    f"  Re-runs will skip this step automatically."
                )
                _download(subset_info["audio_url"], zip_path)

            audio_dir.mkdir(parents=True, exist_ok=True)
            _unzip(zip_path, audio_dir)
            # FMA zips extract into a subdirectory — unwrap if needed
            subdirs = [d for d in audio_dir.iterdir() if d.is_dir()]
            if len(subdirs) == 1:
                audio_dir = subdirs[0]
            print(f"  Audio ready at: {audio_dir}")

    try:
        # ── Step 2: Build index ─────────────────────────────────────────────
        print(f"\n[2/4] Building index with {args.embedder}…")
        library_name = f"build_{short_name}"

        from simil.catalog import TrackCatalog
        from simil.cli.main import _get_index_dir
        from simil.config import Settings
        from simil.embedders import get_embedder
        from simil.index.numpy_index import NumpyIndex
        from simil.library.indexer import Indexer

        index_dir = _get_index_dir(library_name)
        index_dir.mkdir(parents=True, exist_ok=True)

        emb = get_embedder(args.embedder)
        idx = NumpyIndex(
            embedding_dim=emb.embedding_dim,
            embedder_name=args.embedder,
            library_id=library_name,
            audio_config=emb.audio_config,
        )
        cat = TrackCatalog(library_id=library_name)
        settings = Settings(library_name=library_name, workers=args.workers)
        indexer = Indexer(embedder=emb, index=idx, catalog=cat, settings=settings)
        build_result = indexer.build(audio_dir, full=True)
        print(
            f"  Indexed {build_result.indexed:,} tracks, "
            f"skipped {build_result.skipped}, "
            f"failed {len(build_result.failed)} "
            f"in {build_result.duration_seconds:.1f}s"
        )

        if not (index_dir / "meta.json").exists():
            sys.exit("Error: indexing produced no meta.json")

        # ── Step 3: Verify output ───────────────────────────────────────────
        print(f"\n[3/4] Verifying index…")
        meta = json.loads((index_dir / "meta.json").read_text())
        catalog_data = json.loads((index_dir / "catalog.json").read_text())
        track_count = len(catalog_data.get("tracks", []))
        print(f"  Tracks   : {track_count:,}")
        print(f"  Embedder : {meta['embedder']}")
        print(f"  Dim      : {meta['embedding_dim']}")

        # ── Step 4: Package ─────────────────────────────────────────────────
        print(f"\n[4/4] Packaging archive…")
        _make_archive(index_dir, archive_path)

        sha = _sha256(archive_path)
        size_bytes = archive_path.stat().st_size
        size_mb = size_bytes / (1024 * 1024)

        release_url = (
            f"https://github.com/horacio/simil/releases/download/indexes/{archive_name}"
        )

        print(f"\n{'─' * 60}")
        print(f"  Archive  : {archive_path}")
        print(f"  Size     : {size_mb:.1f} MB  ({size_bytes:,} bytes)")
        print(f"  SHA-256  : {sha}")
        print(f"{'─' * 60}")
        print()
        print("Next steps:")
        print(f"  1. Upload to GitHub Releases (tag: indexes):")
        print(f"       gh release create indexes --title 'Pre-built indexes' \\")
        print(f"           {archive_path}")
        print()
        print(f"  2. Update registry.json with:")
        print(f'       "url":        "{release_url}",')
        print(f'       "sha256":     "{sha}",')
        print(f'       "size_bytes": {size_bytes}')
        print()
        print(f"  3. Commit and push registry.json — users can now `simil fetch {short_name}`")

    finally:
        pass  # audio cached at ~/.simil/build_cache — intentionally kept


if __name__ == "__main__":
    main()
