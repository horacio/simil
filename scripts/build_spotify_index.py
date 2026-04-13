#!/usr/bin/env python3
"""Build a simil index from Spotify 30-second preview clips.

Enumerates tracks via the Spotify Web API (genre search + editorial
category playlists), downloads 30-second preview MP3s, embeds with
EffNet-discogs, and builds a searchable index.

Usage
-----
    uv run scripts/build_spotify_index.py

Options
-------
    --enumerate-only      Only build tracks.tsv — skip download and embed
    --skip-enumerate      Use existing tracks.tsv — skip API enumeration
    --limit N             Max tracks to process (0 = all)
    --cache-dir PATH      Cache dir  (default: ~/.simil/build_cache/spotify)
    --embedder NAME       Embedder   (default: effnet-discogs)
    --workers N           Embed workers (default: 4)
    --download-workers N  Parallel download threads (default: 32)

Requirements
------------
    SIMIL_SPOTIFY_CLIENT_ID and SIMIL_SPOTIFY_CLIENT_SECRET must be set.
    Register an application at https://developer.spotify.com/dashboard.
"""

from __future__ import annotations

import argparse
import base64
import csv
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import httpx

TOKEN_URL = "https://accounts.spotify.com/api/token"
API_BASE = "https://api.spotify.com/v1"
LIBRARY_NAME = "spotify"

# Hardcoded genre list (mirrors the former recommendations/available-genre-seeds
# endpoint, which was removed by Spotify in late 2024).
GENRES = [
    "acoustic", "afrobeat", "alt-rock", "alternative", "ambient", "anime",
    "black-metal", "bluegrass", "blues", "bossanova", "brazil", "breakbeat",
    "british", "cantopop", "chicago-house", "children", "chill", "classical",
    "club", "country", "dance", "dancehall", "death-metal", "deep-house",
    "detroit-techno", "disco", "drum-and-bass", "dub", "dubstep", "edm",
    "electro", "electronic", "emo", "folk", "french", "funk", "garage",
    "gospel", "goth", "grindcore", "groove", "grunge", "guitar", "happy",
    "hard-rock", "hardcore", "hardstyle", "heavy-metal", "hip-hop",
    "honky-tonk", "house", "idm", "indie", "indie-pop", "industrial",
    "j-pop", "j-rock", "jazz", "k-pop", "latin", "metal", "metalcore",
    "minimal-techno", "new-age", "opera", "piano", "pop", "post-dubstep",
    "progressive-house", "psych-rock", "punk", "punk-rock", "r-n-b",
    "reggae", "reggaeton", "rock", "rock-n-roll", "romance", "sad", "salsa",
    "samba", "singer-songwriter", "ska", "soul", "soundtracks", "swedish",
    "synth-pop", "tango", "techno", "trance", "trip-hop", "world-music",
]

TSV_FIELDS = [
    "track_id", "title", "artist", "album",
    "album_art_url", "preview_url", "duration_ms",
]


# ---------------------------------------------------------------------------
# Spotify API client
# ---------------------------------------------------------------------------


class SpotifyClient:
    """Thin Spotify API wrapper with auto-refreshing token and 429 backoff."""

    def __init__(self, client_id: str, client_secret: str) -> None:
        self._client_id = client_id
        self._client_secret = client_secret
        self._token = ""
        self._token_expires_at = 0.0
        self._http = httpx.Client(timeout=15.0)

    def _get_token(self) -> str:
        now = time.monotonic()
        if self._token and now < self._token_expires_at:
            return self._token
        creds = base64.b64encode(
            f"{self._client_id}:{self._client_secret}".encode()
        ).decode()
        resp = self._http.post(
            TOKEN_URL,
            headers={"Authorization": f"Basic {creds}"},
            data={"grant_type": "client_credentials"},
        )
        resp.raise_for_status()
        data = resp.json()
        self._token = data["access_token"]
        self._token_expires_at = now + data["expires_in"] - 60
        return self._token

    def get(self, path: str, params: dict[str, Any] | None = None) -> dict:
        """Authenticated GET with exponential backoff on 429."""
        url = f"{API_BASE}{path}" if path.startswith("/") else path
        for attempt in range(4):
            try:
                resp = self._http.get(
                    url,
                    headers={"Authorization": f"Bearer {self._get_token()}"},
                    params=params,
                )
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
            return resp.json()
        raise RuntimeError(f"Exhausted retries: {url}")

    def close(self) -> None:
        self._http.close()


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
# Step 1: Enumerate tracks → tracks.tsv
# ---------------------------------------------------------------------------


def enumerate_tracks(
    client: SpotifyClient,
    cache_dir: Path,
    limit: int = 0,
) -> list[dict]:
    """Build tracks.tsv via genre recommendations + category playlists.

    Resumes from an existing TSV — only adds new tracks.
    Skips any track that lacks a ``preview_url``.
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
        tid = t.get("id")
        if not tid or tid in seen:
            return
        preview_url = t.get("preview_url")
        if not preview_url:
            return
        artists = t.get("artists") or []
        artist = ", ".join(a["name"] for a in artists if a.get("name"))
        album_obj = t.get("album") or {}
        album_name = album_obj.get("name", "") if isinstance(album_obj, dict) else ""
        images = album_obj.get("images") or [] if isinstance(album_obj, dict) else []
        album_art = images[0]["url"] if images else ""
        seen.add(tid)
        tracks.append({
            "track_id": tid,
            "title": t.get("name", ""),
            "artist": artist,
            "album": album_name,
            "album_art_url": album_art,
            "preview_url": preview_url,
            "duration_ms": str(t.get("duration_ms", "")),
        })

    # ── Phase A: Genre search (3 pages × 50 tracks per genre) ────────────
    print(f"  [A] Genre search ({len(GENRES)} genres × 3 pages)…")
    for i, genre in enumerate(GENRES):
        for offset in (0, 50, 100):
            try:
                data = client.get(
                    "/search",
                    params={
                        "q": f"genre:{genre}",
                        "type": "track",
                        "limit": 50,
                        "offset": offset,
                        "market": "US",
                    },
                )
                for t in (data.get("tracks") or {}).get("items") or []:
                    _add(t)
                time.sleep(0.1)
            except Exception as exc:
                print(f"\n      Skipping genre {genre!r} offset {offset}: {exc}")
                break
        if (i + 1) % 20 == 0 or i + 1 == len(GENRES):
            print(
                f"      {i + 1}/{len(GENRES)} genres  →  {len(tracks):,} unique tracks",
                end="\r", flush=True,
            )
    print(f"\n      After genre search: {len(tracks):,} unique tracks.")

    # ── Phase B: Browse categories → playlists → tracks ──────────────────
    print("  [B] Category playlists…")
    try:
        cats_resp = client.get("/browse/categories", params={"limit": 50})
        cat_items = (cats_resp.get("categories") or {}).get("items") or []
        print(f"      {len(cat_items)} categories available.")
        for cat in cat_items:
            cat_id = cat.get("id")
            if not cat_id:
                continue
            try:
                pl_resp = client.get(
                    f"/browse/categories/{cat_id}/playlists",
                    params={"limit": 10},
                )
                playlists = (pl_resp.get("playlists") or {}).get("items") or []
                for pl in playlists[:5]:
                    if not pl:
                        continue
                    pl_id = pl.get("id")
                    if not pl_id:
                        continue
                    try:
                        items_resp = client.get(
                            f"/playlists/{pl_id}/tracks",
                            params={
                                "limit": 100,
                                "market": "US",
                                "fields": "items(track(id,name,artists,album,preview_url,duration_ms))",
                            },
                        )
                        for item in items_resp.get("items") or []:
                            _add((item or {}).get("track"))
                        time.sleep(0.1)
                    except Exception:
                        pass
                time.sleep(0.2)
            except Exception as exc:
                print(f"\n      Category {cat_id!r} skipped: {exc}")
    except Exception as exc:
        print(f"\n      Category browse failed: {exc}")
    print(f"      After category playlists: {len(tracks):,} unique tracks.")

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
    """Download Spotify 30-second previews to audio_dir/{track_id}.mp3."""
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
        sp = meta_lookup.get(path.name)
        if sp:
            result.setdefault("title", sp.get("title") or None)
            result.setdefault("artist", sp.get("artist") or None)
            result.setdefault("album", sp.get("album") or None)
            dur_ms = sp.get("duration_ms")
            if dur_ms:
                try:
                    result["duration"] = int(dur_ms) / 1000.0
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
# Step 4: Enrich catalog with Spotify metadata
# ---------------------------------------------------------------------------


def enrich_catalog(index_dir: Path, tracks_by_id: dict[str, dict]) -> None:
    """Inject spotify_id, spotify_url, preview_url, album_art_url into each track's extra."""
    from simil.catalog import TrackCatalog

    catalog_path = index_dir / "catalog.json"
    cat = TrackCatalog.load(catalog_path, library_id=LIBRARY_NAME)

    enriched = 0
    for track in cat.all_tracks():
        spotify_id = Path(track.path).stem  # filename is {spotify_track_id}.mp3
        sp = tracks_by_id.get(spotify_id)
        if sp:
            track.extra["spotify_id"] = spotify_id
            track.extra["spotify_url"] = f"https://open.spotify.com/track/{spotify_id}"
            track.extra["preview_url"] = sp.get("preview_url", "")
            track.extra["album_art_url"] = sp.get("album_art_url", "")
            enriched += 1

    cat.save(catalog_path)
    print(f"  Enriched {enriched:,} tracks with Spotify metadata.")


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
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Max tracks (0 = all)",
    )
    parser.add_argument(
        "--cache-dir", type=Path,
        default=Path.home() / ".simil" / "build_cache" / "spotify",
    )
    parser.add_argument("--embedder", default="effnet-discogs")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--download-workers", type=int, default=32)
    args = parser.parse_args()

    client_id = os.environ.get("SIMIL_SPOTIFY_CLIENT_ID", "")
    client_secret = os.environ.get("SIMIL_SPOTIFY_CLIENT_SECRET", "")
    if not client_id or not client_secret:
        sys.exit(
            "Error: SIMIL_SPOTIFY_CLIENT_ID and SIMIL_SPOTIFY_CLIENT_SECRET must be set.\n"
            "Register an app at https://developer.spotify.com/dashboard"
        )

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    tsv_path = args.cache_dir / "tracks.tsv"

    print(f"\nsimil Spotify index builder")
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
        print("[1/4] Enumerating tracks from Spotify…")
        client = SpotifyClient(client_id, client_secret)
        try:
            tracks = enumerate_tracks(client, args.cache_dir, limit=args.limit)
        finally:
            client.close()
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
    print(f"\n[4/4] Enriching catalog with Spotify metadata…")
    enrich_catalog(index_dir, tracks_by_id)

    print(f"\nDone! Index at: {index_dir}")
    print(f"Load with:  simil search 'query' --library {LIBRARY_NAME}")
    print(f"Or point the web UI at library: {LIBRARY_NAME}")


if __name__ == "__main__":
    main()
