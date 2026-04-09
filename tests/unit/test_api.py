"""Unit tests for the FastAPI application.

Uses TestClient with mocked SearchEngine — no index on disk required.

Important: all patches AND TestClient requests must live inside the same
`with` blocks, because the lazy `_get_engine()` closure reads settings at
request time (not at app-creation time).
"""

from __future__ import annotations

import json
import wave
from contextlib import contextmanager, ExitStack
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest
from fastapi.testclient import TestClient

from simil.api.app import create_app
from simil.config import Settings
from simil.core.models import SearchResult, Track


# ── Helpers ───────────────────────────────────────────────────────────────────


def _write_wav(path: Path, sr: int = 22050) -> None:
    n = sr * 2
    t = np.linspace(0, 2.0, n, endpoint=False)
    samples = (np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(samples.tobytes())


def _make_result(path: Path, rank: int = 1) -> SearchResult:
    return SearchResult(
        track=Track(
            id=f"abc{rank:03d}",
            path=path,
            title=f"Track {rank}",
            artist="Test Artist",
        ),
        raw_score=0.9 - 0.05 * rank,
        score=1.0 - 0.1 * rank,
        rank=rank,
    )


def _make_engine(results: list[SearchResult], catalog_get=None) -> MagicMock:
    engine = MagicMock()
    engine.search.return_value = results
    engine.catalog.get.return_value = catalog_get or (results[0].track if results else None)
    return engine


@contextmanager
def _mocked_client(tmp_path: Path, engine: MagicMock):
    """Context manager that yields a TestClient with a fully mocked backend.

    All patches are held open for the lifetime of the context so that
    lazy imports inside _get_engine() resolve to mocks.
    """
    settings = Settings(library_name="test")
    with ExitStack() as stack:
        # Redirect index_dir to tmp_path for this settings instance
        stack.enter_context(
            patch.object(
                type(settings), "index_dir",
                new_callable=lambda: property(lambda s: tmp_path),
            )
        )
        # Patch the source modules that _get_engine() lazy-imports
        mock_idx = MagicMock()
        mock_idx._embedder_name = "mock"
        stack.enter_context(
            patch("simil.index.numpy_index.NumpyIndex.load", return_value=mock_idx)
        )
        stack.enter_context(
            patch("simil.catalog.TrackCatalog.load", return_value=MagicMock())
        )
        stack.enter_context(patch("simil.embedders.get_embedder"))
        stack.enter_context(
            patch("simil.search.engine.SearchEngine", return_value=engine)
        )
        api = create_app(settings=settings)
        yield TestClient(api, raise_server_exceptions=False)


def _write_index(tmp_path: Path, embedder: str = "mock", tracks: int = 0) -> None:
    meta = {
        "schema_version": 1,
        "embedder": embedder,
        "embedding_dim": 64,
        "built_at": "2026-01-01T00:00:00+00:00",
    }
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    catalog = {
        "schema_version": 1,
        "library_id": "test",
        "built_at": "",
        "tracks": [
            {"id": f"id{i}", "path": f"/t/{i}.mp3", "title": f"T{i}", "artist": "A",
             "album": None, "duration_seconds": None, "mtime": None, "extra": {}}
            for i in range(tracks)
        ],
    }
    (tmp_path / "catalog.json").write_text(json.dumps(catalog))


# ── App factory ───────────────────────────────────────────────────────────────


def test_create_app_returns_fastapi_app() -> None:
    from fastapi import FastAPI

    api = create_app(settings=Settings())
    assert isinstance(api, FastAPI)


# ── GET / ─────────────────────────────────────────────────────────────────────


def test_ui_root_responds(tmp_path: Path) -> None:
    """GET / returns some response (200 if index.html exists, 404 if not)."""
    engine = _make_engine([])
    _write_index(tmp_path)
    with _mocked_client(tmp_path, engine) as client:
        resp = client.get("/")
    assert resp.status_code in (200, 404)


# ── GET /api/status ───────────────────────────────────────────────────────────


def test_api_status_not_ready(tmp_path: Path) -> None:
    """Returns ready=False when the index directory is empty."""
    settings = Settings(library_name="test-no-index")
    with patch.object(
        type(settings), "index_dir",
        new_callable=lambda: property(lambda s: tmp_path),
    ):
        api = create_app(settings=settings)
        client = TestClient(api)
        resp = client.get("/api/status")

    assert resp.status_code == 200
    assert resp.json()["ready"] is False


def test_api_status_ready(tmp_path: Path) -> None:
    """Returns ready=True with correct stats when index files exist."""
    _write_index(tmp_path, tracks=3)

    settings = Settings(library_name="test-ready")
    with patch.object(
        type(settings), "index_dir",
        new_callable=lambda: property(lambda s: tmp_path),
    ):
        api = create_app(settings=settings)
        client = TestClient(api)
        resp = client.get("/api/status")

    assert resp.status_code == 200
    data = resp.json()
    assert data["ready"] is True
    assert data["tracks"] == 3
    assert data["embedder"] == "mock"


# ── POST /api/search ──────────────────────────────────────────────────────────


def test_search_json_returns_results(tmp_path: Path, wav_440: Path) -> None:
    """POST /api/search with JSON body returns results."""
    _write_index(tmp_path)
    results = [_make_result(wav_440, rank=i) for i in range(1, 4)]
    engine = _make_engine(results)

    with _mocked_client(tmp_path, engine) as client:
        resp = client.post("/api/search", json={"source": "dark ambient drone"})

    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 3
    assert data[0]["rank"] == 1
    assert data[0]["title"] == "Track 1"
    assert "score" in data[0]
    assert "track_id" in data[0]


def test_search_results_have_required_fields(tmp_path: Path, wav_440: Path) -> None:
    """Each result object contains all expected fields."""
    _write_index(tmp_path)
    results = [_make_result(wav_440, rank=1)]
    engine = _make_engine(results)

    with _mocked_client(tmp_path, engine) as client:
        resp = client.post("/api/search", json={"source": "test"})

    data = resp.json()
    assert data
    r = data[0]
    for field in ("rank", "score", "raw_score", "track_id", "title", "artist", "has_audio"):
        assert field in r, f"Missing field: {field}"


def test_search_empty_source_returns_422(tmp_path: Path) -> None:
    _write_index(tmp_path)
    engine = _make_engine([])
    with _mocked_client(tmp_path, engine) as client:
        resp = client.post("/api/search", json={"source": ""})
    assert resp.status_code == 422


def test_search_file_url_returns_422(tmp_path: Path) -> None:
    """file:// URLs are rejected (security: prevents server-side path traversal)."""
    _write_index(tmp_path)
    engine = _make_engine([])
    with _mocked_client(tmp_path, engine) as client:
        resp = client.post("/api/search", json={"source": "file:///etc/passwd"})
    assert resp.status_code == 422


def test_search_no_index_returns_503(tmp_path: Path) -> None:
    """Returns 503 when the index has not been built yet."""
    # tmp_path has no meta.json
    settings = Settings(library_name="test-503")
    with patch.object(
        type(settings), "index_dir",
        new_callable=lambda: property(lambda s: tmp_path),
    ):
        api = create_app(settings=settings)
        client = TestClient(api, raise_server_exceptions=False)
        resp = client.post("/api/search", json={"source": "test"})
    assert resp.status_code == 503


def test_search_returns_empty_list_on_no_results(tmp_path: Path, wav_440: Path) -> None:
    _write_index(tmp_path)
    engine = _make_engine([])
    with _mocked_client(tmp_path, engine) as client:
        resp = client.post("/api/search", json={"source": "test query"})
    assert resp.status_code == 200
    assert resp.json() == []


def test_search_file_upload(tmp_path: Path, wav_440: Path) -> None:
    """POST /api/search with multipart file upload returns results."""
    _write_index(tmp_path)
    results = [_make_result(wav_440, rank=1)]
    engine = _make_engine(results)

    with _mocked_client(tmp_path, engine) as client:
        with open(wav_440, "rb") as fh:
            resp = client.post(
                "/api/search",
                files={"file": ("test.wav", fh, "audio/wav")},
            )

    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    assert data[0]["title"] == "Track 1"


def test_search_upload_no_file_returns_422(tmp_path: Path) -> None:
    """Multipart upload without a file field returns 422."""
    _write_index(tmp_path)
    engine = _make_engine([])
    with _mocked_client(tmp_path, engine) as client:
        resp = client.post("/api/search", data={})
    # 422 for missing file
    assert resp.status_code == 422


# ── GET /api/audio/{track_id} ─────────────────────────────────────────────────


def test_audio_streams_existing_file(tmp_path: Path, wav_440: Path) -> None:
    """GET /api/audio/{id} streams the audio file content."""
    _write_index(tmp_path)
    track = Track(id="testtrack", path=wav_440, title="Test")
    engine = MagicMock()
    engine.catalog.get.return_value = track

    with _mocked_client(tmp_path, engine) as client:
        resp = client.get("/api/audio/testtrack")

    assert resp.status_code == 200
    assert len(resp.content) > 0


def test_audio_missing_track_returns_404(tmp_path: Path) -> None:
    """GET /api/audio/{id} returns 404 for an unknown track_id."""
    _write_index(tmp_path)
    engine = MagicMock()
    engine.catalog.get.return_value = None

    with _mocked_client(tmp_path, engine) as client:
        resp = client.get("/api/audio/nonexistent")

    assert resp.status_code == 404


def test_audio_missing_file_on_disk_returns_404(tmp_path: Path) -> None:
    """GET /api/audio/{id} returns 404 when the audio file was deleted."""
    _write_index(tmp_path)
    ghost_path = tmp_path / "deleted.mp3"  # does not exist
    track = Track(id="ghost", path=ghost_path, title="Ghost")
    engine = MagicMock()
    engine.catalog.get.return_value = track

    with _mocked_client(tmp_path, engine) as client:
        resp = client.get("/api/audio/ghost")

    assert resp.status_code == 404
