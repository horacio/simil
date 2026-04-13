#!/usr/bin/env python3
"""Build a simil index from Deezer 30-second preview clips.

Enumerates tracks via the Deezer public API (no API key required):
  - Phase A: genre → top artists → top tracks
  - Phase B: per-genre charts

Then downloads 30-second preview MP3s, embeds with EffNet-discogs,
and builds a searchable index.

Usage
-----
    uv run scripts/build_deezer_index.py

Options
-------
    --enumerate-only      Only build tracks.tsv — skip download and embed
    --skip-enumerate      Use existing tracks.tsv — skip API enumeration
    --limit N             Max tracks to process (0 = all)
    --cache-dir PATH      Cache dir  (default: ~/.simil/build_cache/deezer)
    --embedder NAME       Embedder   (default: effnet-discogs)
    --workers N           Embed workers (default: 4)
    --download-workers N  Parallel download threads (default: 32)

No API key required — Deezer's public API is open.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import httpx

DEEZER_BASE = "https://api.deezer.com"
LIBRARY_NAME = "deezer"

TSV_FIELDS = [
    "track_id", "title", "artist", "album",
    "album_art_url", "preview_url", "duration_sec",
]


# ---------------------------------------------------------------------------
# Deezer API helpers
# ---------------------------------------------------------------------------


def _dz_get(path: str, params: dict[str, Any] | None = None) -> dict:
    """GET from Deezer API with retry on transient errors and 429 backoff."""
    url = f"{DEEZER_BASE}{path}" if path.startswith("/") else path
    for attempt in range(4):
        try:
            resp = httpx.get(url, params=params, timeout=15.0, follow_redirects=True)
        except httpx.RequestError:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)
            continue
        if resp.status_code == 429:
            wait = int(resp.headers.get("Retry-After", "5")) + 1
            print(f"\n  Rate limited — waiting {wait}s…", flush=True)
            time.sleep(wait)
            continue
        resp.raise_for_status()
        data = resp.json()
        # Deezer returns {"error": {...}} on logical errors
        if "error" in data:
            raise RuntimeError(f"Deezer API error: {data['error']}")
        return data
    raise RuntimeError(f"Exhausted retries: {url}")


# ---------------------------------------------------------------------------
# TSV helpers
# ---------------------------------------------------------------------------


def _save_tsv(tracks: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=TSV_FIELDS, delimiter="\t", extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(tracks)
    print(f"  Saved {len(tracks):,} tracks → {path}")


def _load_tsv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


# ---------------------------------------------------------------------------
# Track extraction helpers
# ---------------------------------------------------------------------------


def _extract(t: dict) -> dict | None:
    """Pull the fields we need from a Deezer track object."""
    tid = str(t.get("id", ""))
    if not tid:
        return None
    preview_url = t.get("preview", "")
    if not preview_url:
        return None
    artist_obj = t.get("artist") or {}
    artist = artist_obj.get("name", "") if isinstance(artist_obj, dict) else ""
    album_obj = t.get("album") or {}
    album = album_obj.get("title", "") if isinstance(album_obj, dict) else ""
    # album art: prefer medium (cover_medium), fall back to cover
    album_art = (
        album_obj.get("cover_medium")
        or album_obj.get("cover_big")
        or album_obj.get("cover")
        or ""
    ) if isinstance(album_obj, dict) else ""
    return {
        "track_id": tid,
        "title": t.get("title", ""),
        "artist": artist,
        "album": album,
        "album_art_url": album_art,
        "preview_url": preview_url,
        "duration_sec": str(t.get("duration", "")),
    }


# ---------------------------------------------------------------------------
# Step 1: Enumerate tracks → tracks.tsv
# ---------------------------------------------------------------------------


def enumerate_tracks(cache_dir: Path, limit: int = 0) -> list[dict]:
    """Build tracks.tsv from Deezer genres, artists, and charts.

    Resumes from an existing TSV. Skips tracks without a preview URL.
    """
    tsv_path = cache_dir / "tracks.tsv"
    seen: set[str] = set()
    tracks: list[dict] = []

    if tsv_path.exists():
        for row in _load_tsv(tsv_path):
            if row["track_id"] not in seen:
                seen.add(row["track_id"])
                tracks.append(row)
        print(f"  Resumed: {len(tracks):,} tracks from existing TSV.")

    def _add(t: dict | None) -> None:
        if not t:
            return
        row = _extract(t)
        if row and row["track_id"] not in seen:
            seen.add(row["track_id"])
            tracks.append(row)

    # ── Phase A: Search by diverse queries (paginated) ───────────────────
    SEARCH_QUERIES = [
        # genres
        "rock", "pop", "jazz", "blues", "classical", "hip hop", "soul", "funk",
        "reggae", "metal", "punk", "folk", "country", "latin", "r&b", "electronic",
        "dance", "ambient", "world music", "indie", "alternative", "singer songwriter",
        "techno", "house", "drum and bass", "dubstep", "gospel", "bossa nova",
        "flamenco", "tango", "salsa", "afrobeat", "k-pop", "j-pop", "bollywood",
        # moods / styles
        "chill", "workout", "party", "sad", "happy", "romantic", "focus",
        "acoustic", "instrumental", "live", "lo-fi", "psychedelic",
        # decades
        "60s hits", "70s hits", "80s hits", "90s hits", "2000s hits", "2010s hits",
        # regions
        "french music", "italian music", "spanish music", "brazilian music",
        "african music", "caribbean music",
    ]
    print(f"  [A] Search enumeration ({len(SEARCH_QUERIES)} queries × 10 pages)…")
    for qi, query in enumerate(SEARCH_QUERIES):
        for offset in range(0, 1000, 100):  # up to 1000 results per query
            try:
                data = _dz_get("/search", params={"q": query, "limit": 100, "index": offset})
                items = data.get("data", [])
                if not items:
                    break
                for t in items:
                    _add(t)
                time.sleep(0.06)
            except Exception:
                break
        if (qi + 1) % 10 == 0 or qi + 1 == len(SEARCH_QUERIES):
            print(
                f"      {qi + 1}/{len(SEARCH_QUERIES)} queries  →  {len(tracks):,} unique tracks",
                end="\r", flush=True,
            )
    print(f"\n      After search: {len(tracks):,} unique tracks.")

    # Collect genre list for phases B and C
    try:
        genres_data = _dz_get("/genre")
        genres = [g for g in genres_data.get("data", []) if g.get("id", 0) != 0]
    except Exception:
        genres = []

    # ── Phase B: Per-genre charts ─────────────────────────────────────────
    print("  [B] Genre charts…")
    for genre in genres:
        genre_id = genre["id"]
        try:
            chart_data = _dz_get(f"/chart/{genre_id}/tracks", params={"limit": 100})
            for t in chart_data.get("data", []):
                _add(t)
            time.sleep(0.05)
        except Exception:
            pass

    # Also pull global chart
    try:
        global_chart = _dz_get("/chart/0/tracks", params={"limit": 100})
        for t in global_chart.get("data", []):
            _add(t)
    except Exception:
        pass

    print(f"      After genre charts: {len(tracks):,} unique tracks.")

    # ── Phase C: Editorial playlists ──────────────────────────────────────
    print("  [C] Editorial playlists…")
    try:
        editorial_data = _dz_get("/editorial")
        editorials = editorial_data.get("data", [])
        print(f"      {len(editorials)} editorial sections.")
        for ed in editorials:
            ed_id = ed.get("id")
            if not ed_id:
                continue
            try:
                sel_data = _dz_get(f"/editorial/{ed_id}/selection")
                for album in sel_data.get("data", []):
                    album_id = album.get("id")
                    if not album_id:
                        continue
                    try:
                        tracks_data = _dz_get(f"/album/{album_id}/tracks")
                        for t in tracks_data.get("data", []):
                            # album tracks don't embed album art — patch it in
                            if isinstance(t, dict) and "album" not in t:
                                t["album"] = album
                            _add(t)
                        time.sleep(0.05)
                    except Exception:
                        pass
                time.sleep(0.1)
            except Exception:
                pass
    except Exception as exc:
        print(f"      Editorial fetch failed: {exc}")

    print(f"      After editorial playlists: {len(tracks):,} unique tracks.")

    _save_tsv(tracks, tsv_path)

    if limit and len(tracks) > limit:
        tracks = tracks[:limit]
        print(f"  Limited to {limit:,} tracks.")

    return tracks


# ---------------------------------------------------------------------------
# Step 2: Download preview MP3s
# ---------------------------------------------------------------------------


def download_previews(
    tracks: list[dict],
    audio_dir: Path,
    workers: int = 32,
) -> None:
    """Download Deezer 30-second previews to audio_dir/{track_id}.mp3."""
    audio_dir.mkdir(parents=True, exist_ok=True)

    work = [
        (t, audio_dir / f"{t['track_id']}.mp3")
        for t in tracks
        if not (audio_dir / f"{t['track_id']}.mp3").exists()
    ]
    already = len(tracks) - len(work)
    if already:
        print(f"  Skipping {already:,} already-downloaded previews.")
    print(f"  Downloading {len(work):,} previews ({workers} parallel)…")

    total = len(tracks)
    done = already
    failed = 0

    def _fetch(task: tuple[dict, Path]) -> bool:
        t, dest = task
        tmp = dest.with_suffix(".tmp")
        try:
            resp = httpx.get(t["preview_url"], timeout=30.0, follow_redirects=True)
            resp.raise_for_status()
            tmp.write_bytes(resp.content)
            tmp.rename(dest)
            return True
        except Exception:
            tmp.unlink(missing_ok=True)
            return False

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_fetch, task): task for task in work}
        for future in as_completed(futures):
            done += 1
            if not future.result():
                failed += 1
            if done % 500 == 0 or done == total:
                print(f"  {done:,}/{total:,}  ({failed} failed)", end="\r", flush=True)

    print(f"\n  Done: {done - failed:,} downloaded, {failed} failed.")


# ---------------------------------------------------------------------------
# Step 3: Build NumpyIndex + TrackCatalog
# ---------------------------------------------------------------------------


def build_index(
    audio_dir: Path,
    tracks_meta: list[dict],
    embedder_name: str,
    workers: int,
) -> Path:
    """Embed audio and build the index, resuming any previous run."""
    from simil.catalog import TrackCatalog
    from simil.cli.main import _get_index_dir
    from simil.config import Settings
    from simil.embedders import get_embedder
    from simil.index.numpy_index import NumpyIndex
    from simil.library.indexer import Indexer
    import simil.library.metadata as _meta_mod

    meta_lookup = {f"{t['track_id']}.mp3": t for t in tracks_meta}
    _orig_extract = _meta_mod.extract_metadata

    def _patched_extract(path: Path) -> dict:
        result = _orig_extract(path)
        dz = meta_lookup.get(path.name)
        if dz:
            result.setdefault("title", dz.get("title") or None)
            result.setdefault("artist", dz.get("artist") or None)
            result.setdefault("album", dz.get("album") or None)
            dur = dz.get("duration_sec")
            if dur:
                try:
                    result["duration"] = float(dur)
                except (ValueError, TypeError):
                    pass
        return result

    _meta_mod.extract_metadata = _patched_extract  # type: ignore[assignment]

    index_dir = _get_index_dir(LIBRARY_NAME)
    index_dir.mkdir(parents=True, exist_ok=True)
    emb = get_embedder(embedder_name)

    if (index_dir / "meta.json").exists():
        try:
            idx = NumpyIndex.load(index_dir)
            cat = TrackCatalog.load(index_dir / "catalog.json", library_id=LIBRARY_NAME)
            print(f"  Resuming — {cat.size:,} tracks already indexed.")
        except Exception as exc:
            print(f"  Warning: could not load existing index ({exc}). Starting fresh.")
            idx = NumpyIndex(
                embedding_dim=emb.embedding_dim,
                embedder_name=embedder_name,
                library_id=LIBRARY_NAME,
                audio_config=emb.audio_config,
            )
            cat = TrackCatalog(library_id=LIBRARY_NAME)
    else:
        idx = NumpyIndex(
            embedding_dim=emb.embedding_dim,
            embedder_name=embedder_name,
            library_id=LIBRARY_NAME,
            audio_config=emb.audio_config,
        )
        cat = TrackCatalog(library_id=LIBRARY_NAME)

    settings = Settings(library_name=LIBRARY_NAME, workers=workers)
    indexer = Indexer(embedder=emb, index=idx, catalog=cat, settings=settings)

    try:
        result = indexer.build(audio_dir, full=False)
    finally:
        _meta_mod.extract_metadata = _orig_extract

    print(
        f"  Indexed {result.indexed:,} | skipped {result.skipped:,} | "
        f"failed {len(result.failed)} | {result.duration_seconds:.0f}s"
    )
    return index_dir


# ---------------------------------------------------------------------------
# Step 4: Enrich catalog with Deezer metadata
# ---------------------------------------------------------------------------


def enrich_catalog(index_dir: Path, tracks_by_id: dict[str, dict]) -> None:
    """Inject deezer_id, deezer_url, preview_url, album_art_url into track.extra."""
    from simil.catalog import TrackCatalog

    catalog_path = index_dir / "catalog.json"
    cat = TrackCatalog.load(catalog_path, library_id=LIBRARY_NAME)

    enriched = 0
    for track in cat.all_tracks():
        deezer_id = Path(track.path).stem  # filename stem = deezer track id
        dz = tracks_by_id.get(deezer_id)
        if dz:
            track.extra["deezer_id"] = deezer_id
            track.extra["deezer_url"] = f"https://www.deezer.com/track/{deezer_id}"
            track.extra["preview_url"] = dz.get("preview_url", "")
            track.extra["album_art_url"] = dz.get("album_art_url", "")
            enriched += 1

    cat.save(catalog_path)
    print(f"  Enriched {enriched:,} tracks with Deezer metadata.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--enumerate-only", action="store_true",
        help="Only build tracks.tsv — skip download and embed",
    )
    parser.add_argument(
        "--skip-enumerate", action="store_true",
        help="Use existing tracks.tsv — skip API enumeration",
    )
    parser.add_argument("--limit", type=int, default=0, help="Max tracks (0 = all)")
    parser.add_argument(
        "--cache-dir", type=Path,
        default=Path.home() / ".simil" / "build_cache" / "deezer",
    )
    parser.add_argument("--embedder", default="effnet-discogs")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--download-workers", type=int, default=32)
    args = parser.parse_args()

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    tsv_path = args.cache_dir / "tracks.tsv"

    print(f"\nsimil Deezer index builder")
    print(f"  Embedder     : {args.embedder}")
    print(f"  Cache        : {args.cache_dir}")
    print(f"  Library      : {LIBRARY_NAME}")
    print()

    # ── Step 1: Enumerate ─────────────────────────────────────────────────
    if args.skip_enumerate:
        if not tsv_path.exists():
            sys.exit(f"Error: --skip-enumerate requires {tsv_path} to exist first.")
        print("[1/4] Loading existing tracks.tsv…")
        tracks = _load_tsv(tsv_path)
        if args.limit:
            tracks = tracks[: args.limit]
        print(f"  {len(tracks):,} tracks loaded.")
    else:
        print("[1/4] Enumerating tracks from Deezer…")
        tracks = enumerate_tracks(args.cache_dir, limit=args.limit)
        print(f"  Total: {len(tracks):,} tracks with previews.")

    if args.enumerate_only:
        print(f"\nEnumeration complete. Tracks saved to: {tsv_path}")
        return

    tracks_by_id = {t["track_id"]: t for t in tracks}
    audio_dir = args.cache_dir / "audio"

    # ── Step 2: Download ──────────────────────────────────────────────────
    print(f"\n[2/4] Downloading previews…")
    download_previews(tracks, audio_dir, workers=args.download_workers)

    # ── Step 3: Build index ───────────────────────────────────────────────
    print(f"\n[3/4] Building index ({args.embedder})…")
    index_dir = build_index(audio_dir, tracks, args.embedder, args.workers)

    # ── Step 4: Enrich catalog ────────────────────────────────────────────
    print(f"\n[4/4] Enriching catalog with Deezer metadata…")
    enrich_catalog(index_dir, tracks_by_id)

    print(f"\nDone! Index at: {index_dir}")
    print(f"Load with:  simil search 'query' --library {LIBRARY_NAME}")


if __name__ == "__main__":
    main()
