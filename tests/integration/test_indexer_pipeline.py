"""Integration tests for the full indexer pipeline."""

from __future__ import annotations

import os
import time
import wave
from pathlib import Path

import numpy as np
import pytest

from simil.catalog import TrackCatalog
from simil.config import Settings
from simil.index.numpy_index import NumpyIndex
from simil.library.indexer import Indexer

pytestmark = pytest.mark.integration


def _write_wav(path: Path, freq: float = 440.0, duration: float = 1.0, sr: int = 22050) -> None:
    """Write a synthetic sine-wave WAV file."""
    n = int(sr * duration)
    t = np.linspace(0, duration, n, endpoint=False)
    samples = (np.sin(2 * np.pi * freq * t) * 32767).astype(np.int16)
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(samples.tobytes())


@pytest.fixture
def library_dir(tmp_path: Path) -> Path:
    """Create a temp library with 5 WAV files."""
    lib = tmp_path / "library"
    lib.mkdir()
    freqs = [220.0, 330.0, 440.0, 550.0, 660.0]
    for i, freq in enumerate(freqs):
        _write_wav(lib / f"tone_{i}.wav", freq=freq)
    return lib


@pytest.fixture
def index_dir(tmp_path: Path) -> Path:
    """Temp directory for index storage."""
    d = tmp_path / "index"
    d.mkdir()
    return d


@pytest.fixture
def settings(index_dir: Path) -> Settings:
    """Settings with a custom index_dir, patched to use the temp directory."""
    s = Settings(library_name="_test_pipeline_")

    # Monkey-patch the computed property so the indexer saves to our tmp dir.
    # We use a simple wrapper that overrides __class__ to bypass pydantic's
    # computed_field descriptor.
    class _PatchedSettings(type(s)):
        @property
        def index_dir(self) -> Path:  # type: ignore[override]
            return index_dir

    s.__class__ = _PatchedSettings
    return s


def _make_indexer(mock_embedder: object, settings: Settings) -> Indexer:
    """Construct a fresh Indexer with a MockEmbedder."""
    dim = mock_embedder.embedding_dim
    idx = NumpyIndex(
        embedding_dim=dim,
        embedder_name=mock_embedder.name,
        library_id="test_lib",
    )
    cat = TrackCatalog(library_id="test_lib")
    return Indexer(embedder=mock_embedder, index=idx, catalog=cat, settings=settings)


def _reload_indexer(mock_embedder: object, settings: Settings) -> Indexer:
    """Reload index + catalog from disk and return a new Indexer."""
    idx = NumpyIndex.load(settings.index_dir)
    cat = TrackCatalog.load(settings.index_dir / "catalog.json")
    return Indexer(embedder=mock_embedder, index=idx, catalog=cat, settings=settings)


class TestIndexerPipeline:
    """End-to-end indexer pipeline tests."""

    def test_initial_index_counts_all_files(
        self, library_dir: Path, mock_embedder: object, settings: Settings
    ) -> None:
        """First run indexes all 5 WAV files."""
        indexer = _make_indexer(mock_embedder, settings)
        result = indexer.build(library_dir)
        assert result.indexed == 5
        assert result.skipped == 0
        assert result.failed == []

    def test_incremental_skips_unchanged(
        self, library_dir: Path, mock_embedder: object, settings: Settings
    ) -> None:
        """Second run (incremental) skips all 5 unchanged files."""
        # Initial index
        indexer = _make_indexer(mock_embedder, settings)
        indexer.build(library_dir)

        # Reload and run again
        indexer2 = _reload_indexer(mock_embedder, settings)
        result2 = indexer2.build(library_dir)
        assert result2.skipped == 5
        assert result2.indexed == 0

    def test_incremental_reindexes_changed_file(
        self, library_dir: Path, mock_embedder: object, settings: Settings
    ) -> None:
        """After touching one file, incremental run re-indexes only that file."""
        # Initial index
        indexer = _make_indexer(mock_embedder, settings)
        indexer.build(library_dir)

        # Modify the mtime of one file (write new content)
        target = library_dir / "tone_0.wav"
        time.sleep(0.05)  # ensure mtime changes
        _write_wav(target, freq=777.0)

        # Incremental run
        indexer2 = _reload_indexer(mock_embedder, settings)
        result2 = indexer2.build(library_dir)
        assert result2.indexed == 1
        assert result2.skipped == 4

    def test_full_rebuild_reindexes_all(
        self, library_dir: Path, mock_embedder: object, settings: Settings
    ) -> None:
        """full=True forces re-indexing of all files."""
        # Initial index
        indexer = _make_indexer(mock_embedder, settings)
        indexer.build(library_dir)

        # Full rebuild
        indexer2 = _reload_indexer(mock_embedder, settings)
        result2 = indexer2.build(library_dir, full=True)
        assert result2.indexed == 5
        assert result2.skipped == 0

    def test_indexed_tracks_searchable(
        self, library_dir: Path, mock_embedder: object, settings: Settings
    ) -> None:
        """After indexing, catalog contains all 5 tracks."""
        indexer = _make_indexer(mock_embedder, settings)
        indexer.build(library_dir)

        cat = indexer.catalog
        assert cat.size == 5

        # All indexed files should be in catalog
        for wav in library_dir.glob("*.wav"):
            from simil.library.scanner import content_id as cid_fn
            cid = cid_fn(wav)
            assert cat.contains(cid), f"Missing from catalog: {wav.name}"
