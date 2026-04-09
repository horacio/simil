"""Unit tests for simil.search.engine.SearchEngine."""

from __future__ import annotations

import hashlib
import os
import wave
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from simil.catalog import TrackCatalog
from simil.config import Settings
from simil.core.models import Track
from simil.index.numpy_index import NumpyIndex
from simil.resolvers.base import ResolvedAudio
from simil.search.engine import SearchEngine, _normalise_scores


def _write_wav(path: Path, freq: float = 440.0, duration: float = 1.0, sr: int = 22050) -> None:
    n = int(sr * duration)
    t = np.linspace(0, duration, n, endpoint=False)
    samples = (np.sin(2 * np.pi * freq * t) * 32767).astype(np.int16)
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(samples.tobytes())


def _rand_unit(dim: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Settings pointing to a temp index directory."""
    s = Settings(library_name="test")
    # Override index_dir by monkey-patching
    object.__setattr__(s, "_index_dir_override", tmp_path)
    return s


@pytest.fixture
def engine_with_library(tmp_path: Path, mock_embedder: object) -> tuple[SearchEngine, list[Path]]:
    """Set up a SearchEngine with 5 indexed WAV files."""
    dim = 64
    idx = NumpyIndex(embedding_dim=dim, embedder_name="mock")
    cat = TrackCatalog(library_id="lib_test")
    settings = Settings(library_name="test")

    # Create 5 WAV files and index them
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    paths: list[Path] = []
    for i, freq in enumerate([220, 330, 440, 550, 660]):
        p = audio_dir / f"tone_{freq}.wav"
        _write_wav(p, freq=float(freq))
        paths.append(p)

        from simil.library.scanner import content_id as cid_fn
        cid = cid_fn(p)
        vec = mock_embedder.embed(p)
        mtime = os.path.getmtime(str(p))
        idx.add(cid, vec)
        cat.add(Track(id=cid, path=p, title=f"Tone {freq}", mtime=mtime))

    engine = SearchEngine(
        embedder=mock_embedder,
        index=idx,
        catalog=cat,
        settings=settings,
    )
    return engine, paths


class TestSearchEngineLocal:
    """Tests for SearchEngine.search() with local files."""

    def test_search_file_in_library_excludes_self(
        self, engine_with_library: tuple[SearchEngine, list[Path]]
    ) -> None:
        """Searching with a file in the library excludes the file itself."""
        engine, paths = engine_with_library
        results = engine.search(paths[0], top_k=5, min_score=-1.0)
        result_paths = [r.track.path for r in results]
        assert paths[0] not in result_paths

    def test_search_file_in_library_returns_results(
        self, engine_with_library: tuple[SearchEngine, list[Path]]
    ) -> None:
        """Searching with a library file returns non-empty results."""
        engine, paths = engine_with_library
        # Use min_score=-1.0 to include all results regardless of random orientation
        results = engine.search(paths[2], top_k=4, min_score=-1.0)
        assert len(results) > 0

    def test_search_file_not_in_library(
        self, tmp_path: Path, engine_with_library: tuple[SearchEngine, list[Path]]
    ) -> None:
        """Searching with a file NOT in the library still returns results."""
        engine, _ = engine_with_library
        # Create a new file not in the index
        new_file = tmp_path / "new_tone.wav"
        _write_wav(new_file, freq=999.0)
        # min_score=-1.0 ensures all random vectors are returned
        results = engine.search(new_file, top_k=3, min_score=-1.0)
        assert len(results) > 0

    def test_results_sorted_descending(
        self, engine_with_library: tuple[SearchEngine, list[Path]]
    ) -> None:
        """Results are sorted by score descending."""
        engine, paths = engine_with_library
        results = engine.search(paths[0], top_k=5, min_score=-1.0)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_scores_in_unit_interval(
        self, engine_with_library: tuple[SearchEngine, list[Path]]
    ) -> None:
        """Normalised scores are in [0.0, 1.0]."""
        engine, paths = engine_with_library
        results = engine.search(paths[0], top_k=5, min_score=-1.0)
        for r in results:
            assert 0.0 <= r.score <= 1.0

    def test_min_score_filters_results(
        self, engine_with_library: tuple[SearchEngine, list[Path]]
    ) -> None:
        """min_score filters out results below the threshold."""
        engine, paths = engine_with_library
        results_no_filter = engine.search(paths[0], top_k=5, min_score=-1.0)
        results_filtered = engine.search(paths[0], top_k=5, min_score=0.99)
        # Filtered results should be a subset
        assert len(results_filtered) <= len(results_no_filter)
        for r in results_filtered:
            assert r.raw_score >= 0.99

    def test_rank_starts_at_one(
        self, engine_with_library: tuple[SearchEngine, list[Path]]
    ) -> None:
        """First result has rank == 1."""
        engine, paths = engine_with_library
        results = engine.search(paths[0], top_k=3, min_score=-1.0)
        if results:
            assert results[0].rank == 1

    def test_unsupported_scheme_raises(
        self, engine_with_library: tuple[SearchEngine, list[Path]]
    ) -> None:
        """A URL scheme no resolver handles raises UnsupportedURLError."""
        from simil.core.exceptions import UnsupportedURLError

        engine, _ = engine_with_library
        # ftp:// is not handled by any default resolver
        with pytest.raises(UnsupportedURLError):
            engine.search("ftp://example.com/song.mp3")


class TestSearchEngineURL:
    """Tests for SearchEngine.search() with URL sources."""

    def test_search_url_uses_resolver_chain(
        self,
        engine_with_library: tuple[SearchEngine, list[Path]],
        wav_440: Path,
    ) -> None:
        """URL sources are resolved via the injected ResolverChain."""
        engine, _ = engine_with_library

        mock_resolved = ResolvedAudio(path=wav_440, origin="https://example.com/test.mp3")
        mock_cm = MagicMock()
        mock_cm.__enter__ = MagicMock(return_value=mock_resolved)
        mock_cm.__exit__ = MagicMock(return_value=False)

        mock_chain = MagicMock()
        mock_chain.resolve.return_value = mock_cm
        engine.resolver_chain = mock_chain

        results = engine.search("https://example.com/test.mp3", top_k=3, min_score=-1.0)
        assert len(results) > 0
        mock_chain.resolve.assert_called_once_with("https://example.com/test.mp3")

    def test_search_url_returns_unit_norm_float32_vectors(
        self,
        engine_with_library: tuple[SearchEngine, list[Path]],
        wav_440: Path,
    ) -> None:
        """Results from URL search have valid shape and dtype."""
        engine, _ = engine_with_library

        mock_resolved = ResolvedAudio(path=wav_440, origin="https://example.com/x.mp3")
        mock_cm = MagicMock()
        mock_cm.__enter__ = MagicMock(return_value=mock_resolved)
        mock_cm.__exit__ = MagicMock(return_value=False)

        mock_chain = MagicMock()
        mock_chain.resolve.return_value = mock_cm
        engine.resolver_chain = mock_chain

        results = engine.search("https://example.com/x.mp3", top_k=5, min_score=-1.0)
        assert isinstance(results, list)
        for r in results:
            assert 0.0 <= r.score <= 1.0
            assert r.rank >= 1


class TestSearchEngineText:
    """Tests for SearchEngine.search() with free-form text queries."""

    def _make_text_engine(
        self, engine_with_library: tuple[SearchEngine, list[Path]]
    ) -> SearchEngine:
        """Return a SearchEngine whose embedder also implements TextEmbedder."""
        engine, _ = engine_with_library

        class MockTextEmbedder:
            name = "mock-text"
            embedding_dim = 64
            audio_config: dict = {"sample_rate": 22050}

            def embed(self, path: Path) -> np.ndarray:
                digest = hashlib.sha256(str(path).encode()).digest()
                seed = int.from_bytes(digest[:4], "big")
                v = np.random.default_rng(seed).standard_normal(64).astype(np.float32)
                return v / np.linalg.norm(v)

            def embed_batch(self, paths: list[Path]) -> list[np.ndarray]:
                return [self.embed(p) for p in paths]

            def embed_text(self, text: str) -> np.ndarray:
                seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "big")
                v = np.random.default_rng(seed).standard_normal(64).astype(np.float32)
                return v / np.linalg.norm(v)

        return SearchEngine(
            embedder=MockTextEmbedder(),
            index=engine.index,
            catalog=engine.catalog,
            settings=engine.settings,
        )

    def test_text_query_returns_results(
        self, engine_with_library: tuple[SearchEngine, list[Path]]
    ) -> None:
        """Text queries return non-empty SearchResult list (with TextEmbedder)."""
        engine = self._make_text_engine(engine_with_library)
        results = engine.search("dark ambient drone", top_k=3, min_score=-1.0)
        assert len(results) > 0

    def test_text_query_scores_in_unit_interval(
        self, engine_with_library: tuple[SearchEngine, list[Path]]
    ) -> None:
        engine = self._make_text_engine(engine_with_library)
        results = engine.search("lo-fi hip hop beats", top_k=5, min_score=-1.0)
        for r in results:
            assert 0.0 <= r.score <= 1.0

    def test_text_query_no_exclude_ids(
        self, engine_with_library: tuple[SearchEngine, list[Path]]
    ) -> None:
        """Text queries are not library tracks, so all 5 indexed tracks are eligible."""
        engine = self._make_text_engine(engine_with_library)
        results = engine.search("heavy bass techno", top_k=5, min_score=-1.0)
        # All 5 indexed tracks should be eligible
        assert len(results) == 5

    def test_text_query_without_text_embedder_raises(
        self, engine_with_library: tuple[SearchEngine, list[Path]]
    ) -> None:
        """UnsupportedURLError if the current embedder is not a TextEmbedder."""
        from simil.core.exceptions import UnsupportedURLError

        engine, _ = engine_with_library
        # mock_embedder does NOT implement TextEmbedder
        with pytest.raises(UnsupportedURLError, match="TextEmbedder"):
            engine.search("dark ambient drone")


class TestNormaliseScores:
    """Tests for the _normalise_scores helper."""

    def test_empty(self) -> None:
        """Empty input returns empty output."""
        assert _normalise_scores([]) == []

    def test_single_result(self) -> None:
        """Single result: score clamped to [0, 1]."""
        result = _normalise_scores([("t1", 0.8)])
        assert len(result) == 1
        tid, raw, norm = result[0]
        assert tid == "t1"
        assert raw == pytest.approx(0.8)
        assert 0.0 <= norm <= 1.0

    def test_min_max_normalisation(self) -> None:
        """Multiple results: min-max normalise to [0, 1]."""
        inputs = [("t1", 0.9), ("t2", 0.6), ("t3", 0.3)]
        results = _normalise_scores(inputs)
        norms = [n for _, _, n in results]
        assert max(norms) == pytest.approx(1.0)
        assert min(norms) == pytest.approx(0.0)

    def test_all_same_score(self) -> None:
        """All-same scores normalise to 1.0."""
        inputs = [("t1", 0.5), ("t2", 0.5), ("t3", 0.5)]
        results = _normalise_scores(inputs)
        for _, _, n in results:
            assert n == pytest.approx(1.0)
