"""Unit tests for simil.index.numpy_index.NumpyIndex."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from simil.core.exceptions import IndexDimensionError, IndexSchemaError, SimILIndexError
from simil.index.numpy_index import NumpyIndex, SCHEMA_VERSION


def _rand_unit(dim: int, seed: int = 0) -> np.ndarray:
    """Return a random L2-normalised float32 vector."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


class TestAddAndSearch:
    """Tests for basic add + search behaviour."""

    def test_self_similarity(self) -> None:
        """A query vector should have cosine similarity ≈ 1.0 with itself."""
        idx = NumpyIndex(embedding_dim=32)
        v = _rand_unit(32, seed=1)
        idx.add("t1", v)

        results = idx.search(v, top_k=1)
        assert len(results) == 1
        tid, score = results[0]
        assert tid == "t1"
        assert score == pytest.approx(1.0, abs=1e-5)

    def test_size_increases(self) -> None:
        """size property reflects added vectors."""
        idx = NumpyIndex(embedding_dim=16)
        assert idx.size == 0
        idx.add("a", _rand_unit(16, 0))
        assert idx.size == 1
        idx.add("b", _rand_unit(16, 1))
        assert idx.size == 2

    def test_search_empty_index(self) -> None:
        """Searching an empty index returns an empty list."""
        idx = NumpyIndex(embedding_dim=32)
        results = idx.search(_rand_unit(32), top_k=5)
        assert results == []

    def test_search_k_greater_than_size(self) -> None:
        """top_k > index size returns all available results."""
        idx = NumpyIndex(embedding_dim=16)
        for i in range(3):
            idx.add(f"t{i}", _rand_unit(16, i))
        results = idx.search(_rand_unit(16, 99), top_k=100)
        assert len(results) == 3

    def test_results_sorted_descending(self) -> None:
        """Results are sorted by score descending."""
        idx = NumpyIndex(embedding_dim=32)
        for i in range(10):
            idx.add(f"t{i}", _rand_unit(32, i))
        results = idx.search(_rand_unit(32, 42), top_k=5)
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)


class TestAddBatch:
    """Tests for add_batch."""

    def test_add_batch_100_vectors(self) -> None:
        """add_batch correctly handles 100 vectors, top-1 is self."""
        dim = 64
        idx = NumpyIndex(embedding_dim=dim)
        vectors = [_rand_unit(dim, i) for i in range(100)]
        ids = [f"t{i}" for i in range(100)]
        idx.add_batch(ids, vectors)
        assert idx.size == 100

        # Query with vector 42 — should return t42 as top-1
        results = idx.search(vectors[42], top_k=1)
        assert len(results) == 1
        assert results[0][0] == "t42"
        assert results[0][1] == pytest.approx(1.0, abs=1e-5)


class TestExcludeIds:
    """Tests for exclude_ids parameter."""

    def test_excluded_track_not_in_results(self) -> None:
        """Excluded track IDs do not appear in search results."""
        idx = NumpyIndex(embedding_dim=32)
        v = _rand_unit(32, 1)
        idx.add("self", v)
        idx.add("other", _rand_unit(32, 2))

        results = idx.search(v, top_k=10, exclude_ids=["self"])
        ids = [tid for tid, _ in results]
        assert "self" not in ids


class TestRemove:
    """Tests for the remove() method."""

    def test_remove_decreases_size(self) -> None:
        """Removing a track decreases the size."""
        idx = NumpyIndex(embedding_dim=16)
        idx.add("a", _rand_unit(16, 0))
        idx.add("b", _rand_unit(16, 1))
        assert idx.size == 2
        idx.remove("a")
        assert idx.size == 1

    def test_removed_track_not_found(self) -> None:
        """Removed track does not appear in search results."""
        idx = NumpyIndex(embedding_dim=32)
        v = _rand_unit(32, 1)
        idx.add("target", v)
        idx.add("other", _rand_unit(32, 2))
        idx.remove("target")
        results = idx.search(v, top_k=10)
        ids = [tid for tid, _ in results]
        assert "target" not in ids

    def test_remove_nonexistent_raises(self) -> None:
        """Removing a non-existent track raises SimILIndexError."""
        idx = NumpyIndex(embedding_dim=16)
        idx.add("a", _rand_unit(16, 0))
        with pytest.raises(SimILIndexError):
            idx.remove("does_not_exist")


class TestPersistence:
    """Tests for save() and load() round-trip."""

    def test_save_load_round_trip(self, tmp_path: Path) -> None:
        """Save and load preserves all vectors and metadata."""
        dim = 32
        idx = NumpyIndex(
            embedding_dim=dim,
            embedder_name="test_embedder",
            library_id="lib_abc",
        )
        vectors = {f"t{i}": _rand_unit(dim, i) for i in range(5)}
        for tid, v in vectors.items():
            idx.add(tid, v)

        idx.save(tmp_path)
        idx2 = NumpyIndex.load(tmp_path)

        assert idx2.size == 5
        assert idx2.embedding_dim == dim
        assert idx2._embedder_name == "test_embedder"
        assert idx2._library_id == "lib_abc"

        # Values preserved
        for tid, v in vectors.items():
            loaded_v = idx2.get_vector(tid)
            assert loaded_v is not None
            np.testing.assert_allclose(loaded_v, v, atol=1e-6)

    def test_load_wrong_schema_version(self, tmp_path: Path) -> None:
        """Loading an index with wrong schema_version raises IndexSchemaError."""
        import json

        idx = NumpyIndex(embedding_dim=8)
        idx.add("t1", _rand_unit(8))
        idx.save(tmp_path)

        # Tamper with schema_version
        meta_path = tmp_path / "meta.json"
        meta = json.loads(meta_path.read_text())
        meta["schema_version"] = 999
        meta_path.write_text(json.dumps(meta))

        with pytest.raises(IndexSchemaError):
            NumpyIndex.load(tmp_path)


class TestValidation:
    """Tests for dimension and NaN/Inf validation."""

    def test_dimension_mismatch_raises(self) -> None:
        """Adding a vector with wrong dimension raises IndexDimensionError."""
        idx = NumpyIndex(embedding_dim=32)
        with pytest.raises(IndexDimensionError):
            idx.add("t1", _rand_unit(16))

    def test_nan_vector_raises(self) -> None:
        """Adding a vector with NaN values raises SimILIndexError."""
        idx = NumpyIndex(embedding_dim=8)
        bad = np.full(8, float("nan"), dtype=np.float32)
        with pytest.raises(SimILIndexError):
            idx.add("t1", bad)

    def test_inf_vector_raises(self) -> None:
        """Adding a vector with Inf values raises SimILIndexError."""
        idx = NumpyIndex(embedding_dim=8)
        bad = np.full(8, float("inf"), dtype=np.float32)
        with pytest.raises(SimILIndexError):
            idx.add("t1", bad)
