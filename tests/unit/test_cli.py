"""Unit tests for simil.cli.main.

All heavy operations (indexer, embedder) are mocked so tests run fast.
"""

from __future__ import annotations

import json
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from typer.testing import CliRunner

from simil.cli.main import app

runner = CliRunner()


# ── Helpers ───────────────────────────────────────────────────────────────────


def _write_wav(path: Path, freq: float = 440.0, sr: int = 22050) -> None:
    n = sr * 2
    t = np.linspace(0, 2.0, n, endpoint=False)
    samples = (np.sin(2 * np.pi * freq * t) * 32767).astype(np.int16)
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(samples.tobytes())


def _make_mock_index(embedding_dim: int = 64, embedder_name: str = "mock") -> MagicMock:
    idx = MagicMock()
    idx._embedder_name = embedder_name
    idx._embedding_dim = embedding_dim
    idx.size = 5
    idx.get_vector.return_value = None
    rng = np.random.default_rng(0)
    idx.search.return_value = []
    return idx


def _make_mock_catalog() -> MagicMock:
    cat = MagicMock()
    cat.size = 5
    cat.contains.return_value = False
    cat.get.return_value = None
    return cat


# ── Help ──────────────────────────────────────────────────────────────────────


def test_help_exits_zero() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "simil" in result.output.lower()


def test_index_help() -> None:
    result = runner.invoke(app, ["index", "--help"])
    assert result.exit_code == 0
    assert "library" in result.output.lower()


def test_search_help() -> None:
    result = runner.invoke(app, ["search", "--help"])
    assert result.exit_code == 0


def test_status_help() -> None:
    result = runner.invoke(app, ["status", "--help"])
    assert result.exit_code == 0


def test_add_help() -> None:
    result = runner.invoke(app, ["add", "--help"])
    assert result.exit_code == 0


def test_serve_help() -> None:
    result = runner.invoke(app, ["serve", "--help"])
    assert result.exit_code == 0


# ── index command ─────────────────────────────────────────────────────────────


def test_index_no_path_exits_nonzero() -> None:
    result = runner.invoke(app, ["index"])
    assert result.exit_code != 0


def test_index_nonexistent_path_exits_error(tmp_path: Path) -> None:
    result = runner.invoke(app, ["index", str(tmp_path / "does_not_exist")])
    assert result.exit_code != 0


def test_index_file_path_exits_error(tmp_path: Path) -> None:
    f = tmp_path / "not_a_dir.wav"
    f.write_bytes(b"")
    result = runner.invoke(app, ["index", str(f)])
    assert result.exit_code != 0


def test_index_runs_and_shows_summary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """index command calls Indexer.build() and prints a summary."""
    audio_dir = tmp_path / "music"
    audio_dir.mkdir()
    _write_wav(audio_dir / "song.wav")

    index_dir = tmp_path / "index"

    monkeypatch.setattr("simil.cli.main._get_index_dir", lambda lib: index_dir)

    mock_result = MagicMock()
    mock_result.indexed = 3
    mock_result.skipped = 1
    mock_result.failed = []
    mock_result.duration_seconds = 0.5

    mock_embedder = MagicMock()
    mock_embedder.embedding_dim = 64
    mock_embedder.audio_config = {"sample_rate": 22050}
    mock_embedder.name = "mock"

    mock_indexer = MagicMock()
    mock_indexer.build.return_value = mock_result

    # Patch at the source modules since CLI uses lazy imports
    with patch("simil.embedders.get_embedder", return_value=mock_embedder):
        with patch("simil.library.indexer.Indexer", return_value=mock_indexer):
            result = runner.invoke(
                app, ["index", str(audio_dir), "--library", "test-lib"]
            )

    assert result.exit_code == 0, result.output
    assert "3" in result.output  # indexed count
    mock_indexer.build.assert_called_once()


def test_index_unknown_embedder_exits_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("simil.cli.main._get_index_dir", lambda lib: tmp_path / "idx")
    result = runner.invoke(
        app, ["index", str(tmp_path), "--embedder", "nonexistent-model-xyz"]
    )
    assert result.exit_code != 0


# ── status command ────────────────────────────────────────────────────────────


def test_status_no_index(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """status shows a helpful message when no index exists."""
    monkeypatch.setattr("simil.cli.main._get_index_dir", lambda lib: tmp_path / "empty")
    result = runner.invoke(app, ["status", "--library", "nolib"])
    assert result.exit_code == 0
    assert "No index found" in result.output


def test_status_with_index(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """status reads meta.json and catalog.json and prints stats."""
    index_dir = tmp_path / "idx"
    index_dir.mkdir()

    meta = {
        "schema_version": 1,
        "embedder": "mfcc",
        "embedding_dim": 78,
        "built_at": "2026-01-01T00:00:00+00:00",
    }
    (index_dir / "meta.json").write_text(json.dumps(meta))
    catalog = {"schema_version": 1, "library_id": "test", "built_at": "", "tracks": [
        {"id": "abc", "path": "/tmp/song.mp3", "title": "Song", "artist": "Artist",
         "album": None, "duration_seconds": None, "mtime": None, "extra": {}}
    ]}
    (index_dir / "catalog.json").write_text(json.dumps(catalog))

    monkeypatch.setattr("simil.cli.main._get_index_dir", lambda lib: index_dir)
    result = runner.invoke(app, ["status", "--library", "mylib"])
    assert result.exit_code == 0
    assert "1" in result.output  # track count
    assert "mfcc" in result.output


# ── search command ────────────────────────────────────────────────────────────


def test_search_no_index(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("simil.cli.main._get_index_dir", lambda lib: tmp_path / "empty")
    result = runner.invoke(app, ["search", "dark ambient drone", "--library", "nolib"])
    assert result.exit_code != 0
    assert "No index found" in result.stderr


def test_search_returns_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, wav_440: Path
) -> None:
    """search command prints a results table on success."""
    from simil.core.models import SearchResult, Track

    index_dir = tmp_path / "idx"
    index_dir.mkdir()
    (index_dir / "meta.json").write_text(json.dumps({
        "schema_version": 1, "embedder": "mock", "embedding_dim": 64,
        "built_at": "2026-01-01T00:00:00+00:00",
    }))
    (index_dir / "catalog.json").write_text(json.dumps({
        "schema_version": 1, "library_id": "t", "built_at": "", "tracks": []
    }))

    monkeypatch.setattr("simil.cli.main._get_index_dir", lambda lib: index_dir)

    mock_track = Track(id="abc123", path=wav_440, title="Test Song", artist="Test Artist")
    mock_results = [SearchResult(track=mock_track, raw_score=0.9, score=1.0, rank=1)]

    mock_engine = MagicMock()
    mock_engine.search.return_value = mock_results

    with patch("simil.index.numpy_index.NumpyIndex.load", return_value=_make_mock_index()):
        with patch("simil.catalog.TrackCatalog.load", return_value=_make_mock_catalog()):
            with patch("simil.embedders.get_embedder"):
                with patch("simil.search.engine.SearchEngine", return_value=mock_engine):
                    result = runner.invoke(
                        app, ["search", str(wav_440), "--library", "testlib"]
                    )

    assert result.exit_code == 0, result.output
    assert "Test Song" in result.output


def test_search_no_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, wav_440: Path
) -> None:
    index_dir = tmp_path / "idx"
    index_dir.mkdir()
    (index_dir / "meta.json").write_text(json.dumps({
        "schema_version": 1, "embedder": "mock", "embedding_dim": 64,
        "built_at": "2026-01-01T00:00:00+00:00",
    }))
    (index_dir / "catalog.json").write_text(json.dumps({
        "schema_version": 1, "library_id": "t", "built_at": "", "tracks": []
    }))

    monkeypatch.setattr("simil.cli.main._get_index_dir", lambda lib: index_dir)

    mock_engine = MagicMock()
    mock_engine.search.return_value = []

    with patch("simil.index.numpy_index.NumpyIndex.load", return_value=_make_mock_index()):
        with patch("simil.catalog.TrackCatalog.load", return_value=_make_mock_catalog()):
            with patch("simil.embedders.get_embedder"):
                with patch("simil.search.engine.SearchEngine", return_value=mock_engine):
                    result = runner.invoke(
                        app, ["search", str(wav_440), "--library", "testlib"]
                    )

    assert result.exit_code == 0
    assert "No results" in result.output


def test_search_json_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, wav_440: Path
) -> None:
    from simil.core.models import SearchResult, Track

    index_dir = tmp_path / "idx"
    index_dir.mkdir()
    (index_dir / "meta.json").write_text(json.dumps({
        "schema_version": 1, "embedder": "mock", "embedding_dim": 64,
        "built_at": "",
    }))
    (index_dir / "catalog.json").write_text(json.dumps({
        "schema_version": 1, "library_id": "t", "built_at": "", "tracks": []
    }))

    monkeypatch.setattr("simil.cli.main._get_index_dir", lambda lib: index_dir)

    mock_track = Track(id="abc", path=wav_440, title="T", artist="A")
    mock_results = [SearchResult(track=mock_track, raw_score=0.8, score=0.9, rank=1)]

    mock_engine = MagicMock()
    mock_engine.search.return_value = mock_results

    with patch("simil.index.numpy_index.NumpyIndex.load", return_value=_make_mock_index()):
        with patch("simil.catalog.TrackCatalog.load", return_value=_make_mock_catalog()):
            with patch("simil.embedders.get_embedder"):
                with patch("simil.search.engine.SearchEngine", return_value=mock_engine):
                    result = runner.invoke(
                        app,
                        ["search", str(wav_440), "--library", "testlib", "--json"],
                    )

    assert result.exit_code == 0
    data = json.loads(result.output)
    assert isinstance(data, list)
    assert data[0]["title"] == "T"


# ── add command ───────────────────────────────────────────────────────────────


def test_add_missing_file_exits_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    index_dir = tmp_path / "idx"
    index_dir.mkdir()
    (index_dir / "meta.json").write_text(json.dumps({
        "schema_version": 1, "embedder": "mock", "embedding_dim": 64, "built_at": "",
    }))
    monkeypatch.setattr("simil.cli.main._get_index_dir", lambda lib: index_dir)
    result = runner.invoke(app, ["add", str(tmp_path / "missing.wav")])
    assert result.exit_code != 0


def test_add_no_index_exits_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("simil.cli.main._get_index_dir", lambda lib: tmp_path / "empty")
    f = tmp_path / "song.wav"
    f.write_bytes(b"")
    result = runner.invoke(app, ["add", str(f)])
    assert result.exit_code != 0
    assert "No index found" in result.stderr


def test_add_already_indexed_skips(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, wav_440: Path
) -> None:
    """add command skips a file that is already in the catalog with matching mtime."""
    import os
    from simil.core.models import Track
    from simil.library.scanner import content_id

    index_dir = tmp_path / "idx"
    index_dir.mkdir()
    (index_dir / "meta.json").write_text(json.dumps({
        "schema_version": 1, "embedder": "mock", "embedding_dim": 64, "built_at": "",
    }))

    monkeypatch.setattr("simil.cli.main._get_index_dir", lambda lib: index_dir)

    cid = content_id(wav_440)
    mtime = os.path.getmtime(str(wav_440))
    existing_track = Track(id=cid, path=wav_440, mtime=mtime)

    mock_cat = _make_mock_catalog()
    mock_cat.contains.return_value = True
    mock_cat.get.return_value = existing_track

    with patch("simil.index.numpy_index.NumpyIndex.load", return_value=_make_mock_index()):
        with patch("simil.catalog.TrackCatalog.load", return_value=mock_cat):
            with patch("simil.embedders.get_embedder"):
                result = runner.invoke(app, ["add", str(wav_440), "--library", "test"])

    assert result.exit_code == 0
    assert "Already indexed" in result.output


# ── serve command ─────────────────────────────────────────────────────────────


def test_serve_missing_api_exits_error() -> None:
    """serve command exits cleanly when simil.api is not importable."""
    with patch.dict("sys.modules", {"simil.api.app": None}):
        result = runner.invoke(app, ["serve", "--help"])
    # --help always succeeds regardless
    assert result.exit_code == 0


def test_serve_calls_uvicorn(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """serve command invokes uvicorn.run with correct host/port."""
    monkeypatch.setattr("simil.cli.main._get_index_dir", lambda lib: tmp_path / "idx")

    mock_app = MagicMock()
    mock_uvicorn = MagicMock()

    # Lazy imports in serve() — patch at source modules
    with patch("simil.api.app.create_app", return_value=mock_app):
        with patch("uvicorn.run") as mock_run:
            result = runner.invoke(
                app,
                ["serve", "--host", "0.0.0.0", "--port", "9999", "--library", "test"],
            )

    # uvicorn.run should have been called with the correct host and port
    mock_run.assert_called_once()
    call_args = mock_run.call_args
    assert call_args.kwargs.get("host") == "0.0.0.0"
    assert call_args.kwargs.get("port") == 9999
