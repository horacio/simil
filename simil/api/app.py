"""FastAPI application factory for the simil web UI.

Routes
------
GET  /                       Serve the web UI (index.html)
GET  /static/*               Static assets
POST /api/search             Search: JSON body {source: str} OR multipart file upload
GET  /api/audio/{track_id}   Stream the audio file for a track
GET  /api/status             Library stats (track count, embedder, built_at, etc.)
"""

from __future__ import annotations

import logging
import mimetypes
import os
import tempfile
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from simil.config import Settings
from simil.core.exceptions import (
    EmbeddingError,
    ResolverError,
    SimILError,
    SpotifyPreviewUnavailableError,
    UnsupportedURLError,
)

logger = logging.getLogger(__name__)

_STATIC_DIR = Path(__file__).parent.parent / "static"


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        settings: Application settings.  A default :class:`~simil.config.Settings`
            instance is created if ``None``.

    Returns:
        Configured :class:`~fastapi.FastAPI` application.
    """
    if settings is None:
        settings = Settings()

    app = FastAPI(
        title="simil",
        description="Find similar-sounding tracks in your music library.",
        version="0.1.0",
    )

    # ── Shared state (loaded lazily on first request) ─────────────────────────

    _state: dict = {}

    def _get_engine():  # type: ignore[return]
        """Lazily load and cache the SearchEngine."""
        if "engine" not in _state:
            from simil.catalog import TrackCatalog
            from simil.embedders import get_embedder
            from simil.index.numpy_index import NumpyIndex
            from simil.search.engine import SearchEngine

            index_dir = settings.index_dir
            meta_path = index_dir / "meta.json"

            if not meta_path.exists():
                raise HTTPException(
                    status_code=503,
                    detail=(
                        f"Index not built yet. "
                        f"Run: simil index PATH --library {settings.library_name}"
                    ),
                )

            try:
                idx = NumpyIndex.load(index_dir)
                cat = TrackCatalog.load(index_dir / "catalog.json")
            except SimILError as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc

            embedder_name: str = idx._embedder_name or settings.embedder  # type: ignore[attr-defined]
            try:
                emb = get_embedder(embedder_name)
            except SimILError as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc

            _state["engine"] = SearchEngine(
                embedder=emb,
                index=idx,
                catalog=cat,
                settings=settings,
            )

        return _state["engine"]

    # ── Routes ────────────────────────────────────────────────────────────────

    @app.get("/", include_in_schema=False)
    async def ui_root() -> FileResponse:
        """Serve the web UI."""
        html_path = _STATIC_DIR / "index.html"
        if not html_path.exists():
            raise HTTPException(status_code=404, detail="Web UI not found.")
        return FileResponse(html_path)

    @app.post("/api/search")
    async def api_search(
        request: Request,
        top_k: Annotated[int, Query(ge=1, le=100)] = 10,
        min_score: Annotated[float, Query(ge=-1.0, le=1.0)] = 0.0,
    ) -> JSONResponse:
        """Search for similar tracks.

        Accepts either:
        - ``application/json`` body: ``{"source": "<path|url|text>"}``
        - ``multipart/form-data`` with a ``file`` field (uploaded audio)

        Returns:
            JSON array of result objects with fields:
            ``rank``, ``score``, ``raw_score``, ``title``, ``artist``,
            ``album``, ``duration``, ``track_id``, ``has_audio``.
        """
        engine = _get_engine()
        content_type = request.headers.get("content-type", "")

        # ── File upload ───────────────────────────────────────────────────────
        if "multipart/form-data" in content_type:
            form = await request.form()
            uploaded: UploadFile | None = form.get("file")  # type: ignore[assignment]
            if uploaded is None or not uploaded.filename:
                raise HTTPException(status_code=422, detail="No file uploaded.")

            suffix = Path(uploaded.filename).suffix or ".audio"
            tmp_file = tempfile.NamedTemporaryFile(
                delete=False, suffix=suffix, prefix="simil_upload_"
            )
            try:
                contents = await uploaded.read()
                tmp_file.write(contents)
                tmp_file.flush()
                tmp_path = Path(tmp_file.name)
            finally:
                tmp_file.close()

            try:
                results = engine.search(tmp_path, top_k=top_k, min_score=min_score)
            except (EmbeddingError, SimILError) as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            finally:
                tmp_path.unlink(missing_ok=True)

        # ── JSON / text body ──────────────────────────────────────────────────
        else:
            try:
                body = await request.json()
                source: str = body.get("source", "")
            except Exception:
                raise HTTPException(status_code=422, detail="Expected JSON body with 'source' key.")

            if not source:
                raise HTTPException(status_code=422, detail="'source' must not be empty.")

            # Only allow http/https URLs (no file:// from browser context)
            if source.startswith("file://"):
                raise HTTPException(status_code=422, detail="file:// URLs are not allowed via the API.")

            try:
                results = engine.search(source, top_k=top_k, min_score=min_score)
            except UnsupportedURLError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            except SpotifyPreviewUnavailableError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            except ResolverError as exc:
                raise HTTPException(status_code=502, detail=str(exc)) from exc
            except (EmbeddingError, SimILError) as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc

        return JSONResponse(
            content=[_format_result(r) for r in results]  # type: ignore[union-attr]
        )

    @app.get("/api/audio/{track_id}")
    async def api_audio(track_id: str) -> StreamingResponse:
        """Stream the audio file for a given track.

        Args:
            track_id: The ``content_id`` of the track (24-char hex string).

        Returns:
            Audio file streamed with appropriate MIME type.

        Raises:
            404: If the track is not in the catalog or the file no longer exists.
        """
        engine = _get_engine()
        track = engine.catalog.get(track_id)

        if track is None:
            raise HTTPException(status_code=404, detail=f"Track {track_id!r} not found.")

        audio_path = track.path
        if not audio_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Audio file no longer exists: {audio_path}",
            )

        mime, _ = mimetypes.guess_type(str(audio_path))
        mime = mime or "audio/mpeg"

        def _iter_file():
            with open(audio_path, "rb") as fh:
                while chunk := fh.read(65536):
                    yield chunk

        return StreamingResponse(
            _iter_file(),
            media_type=mime,
            headers={
                "Accept-Ranges": "bytes",
                "Content-Length": str(audio_path.stat().st_size),
            },
        )

    @app.get("/api/status")
    async def api_status() -> JSONResponse:
        """Return current library stats."""
        index_dir = settings.index_dir
        meta_path = index_dir / "meta.json"

        if not meta_path.exists():
            return JSONResponse(
                content={
                    "ready": False,
                    "library": settings.library_name,
                    "message": (
                        "Index not built yet. "
                        "Run \u2018simil index PATH\u2019 to build from local files, "
                        "or \u2018simil fetch\u2019 to download a pre-built index."
                    ),
                    "hint_fetch": "simil fetch",
                    "hint_index": "simil index PATH",
                }
            )

        import json as _json

        meta = _json.loads(meta_path.read_text())
        catalog_path = index_dir / "catalog.json"
        track_count = 0
        if catalog_path.exists():
            cat_raw = _json.loads(catalog_path.read_text())
            track_count = len(cat_raw.get("tracks", []))

        disk_bytes = sum(
            p.stat().st_size for p in index_dir.iterdir() if p.is_file()
        )

        return JSONResponse(
            content={
                "ready": True,
                "library": settings.library_name,
                "tracks": track_count,
                "embedder": meta.get("embedder"),
                "embedding_dim": meta.get("embedding_dim"),
                "built_at": meta.get("built_at"),
                "index_dir": str(index_dir),
                "disk_bytes": disk_bytes,
            }
        )

    # ── Static files (must be last to avoid shadowing API routes) ────────────
    if _STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

    return app


# ── Helpers ───────────────────────────────────────────────────────────────────


def _format_result(r: object) -> dict:
    """Serialize a SearchResult to a JSON-safe dict."""
    from simil.core.models import SearchResult

    result: SearchResult = r  # type: ignore[assignment]
    return {
        "rank": result.rank,
        "score": round(result.score, 6),
        "raw_score": round(result.raw_score, 6),
        "track_id": result.track.id,
        "title": result.track.title,
        "artist": result.track.artist,
        "album": result.track.album,
        "duration": result.track.duration_seconds,
        "has_audio": result.track.path.exists(),
        "extra": result.track.extra,
    }
