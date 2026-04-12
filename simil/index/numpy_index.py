"""Pure-NumPy vector similarity index."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import filelock
import numpy as np

from simil.core.exceptions import IndexDimensionError, IndexSchemaError, SimILIndexError
from simil.core.models import EmbeddingVector, IndexStats

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1


class NumpyIndex:
    """In-memory vector index backed by a NumPy matrix.

    New vectors accumulate in ``_pending`` and are lazily materialised into
    ``_matrix`` on the first ``search()`` or ``save()`` call. This avoids
    ``np.vstack`` inside a loop.

    Args:
        embedding_dim: Dimensionality of embedding vectors.
        embedder_name: Name of the embedder that produced the vectors.
        library_id: Identifier for the owning library.
        audio_config: Audio loading parameters stored in metadata.
    """

    def __init__(
        self,
        embedding_dim: int,
        embedder_name: str = "",
        library_id: str = "",
        audio_config: dict[str, object] | None = None,
    ) -> None:
        """Initialise an empty NumpyIndex."""
        self._embedding_dim = embedding_dim
        self._embedder_name = embedder_name
        self._library_id = library_id
        self._audio_config: dict[str, object] = audio_config or {}

        # Materialised matrix (N, D) — rows aligned with self._ids
        self._matrix: np.ndarray = np.empty((0, embedding_dim), dtype=np.float32)
        # Pending vectors not yet stacked into _matrix
        self._pending: list[np.ndarray] = []
        # Parallel list of track IDs for _matrix rows
        self._ids: list[str] = []
        # Fast id → row index lookup (valid only after materialise)
        self._id_to_idx: dict[str, int] = {}

        self._built_at: str = datetime.now(timezone.utc).isoformat()
        # Corpus mean, subtracted from query vectors at search time.
        # None until center() is called.
        self._centroid: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        """Total number of vectors (materialised + pending)."""
        return len(self._ids)

    @property
    def embedding_dim(self) -> int:
        """Dimensionality of stored vectors."""
        return self._embedding_dim

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add(self, track_id: str, vector: EmbeddingVector) -> None:
        """Add a single vector to the index.

        Args:
            track_id: Unique track identifier.
            vector: Embedding vector. Must have shape ``(embedding_dim,)``
                and contain only finite values.

        Raises:
            IndexDimensionError: If vector dimension does not match embedding_dim.
            SimILIndexError: If vector contains NaN or Inf values.
        """
        self._validate_vector(vector, track_id)
        vec32 = np.asarray(vector, dtype=np.float32)
        self._ids.append(track_id)
        self._pending.append(vec32)
        # _id_to_idx will be rebuilt on next _materialise()

    def add_batch(self, track_ids: Sequence[str], vectors: Sequence[EmbeddingVector]) -> None:
        """Add multiple vectors to the index.

        Args:
            track_ids: Sequence of unique track identifiers.
            vectors: Corresponding sequence of embedding vectors.

        Raises:
            IndexDimensionError: If any vector has wrong dimension.
            SimILIndexError: If any vector contains non-finite values.
        """
        for tid, vec in zip(track_ids, vectors):
            self._validate_vector(vec, tid)

        new_vecs = [np.asarray(v, dtype=np.float32) for v in vectors]
        self._ids.extend(track_ids)
        self._pending.extend(new_vecs)

    def center(self) -> np.ndarray:
        """Remove the corpus mean from all stored vectors (in-place).

        EffNet-discogs embeddings have a very strong DC component — every track
        has ~96% of its embedding mass pointing in the same "music" direction,
        leaving only ~4% for actual musical differences.  Subtracting the corpus
        mean and renormalising collapses that bias and lets cosine distance
        reflect true similarity rather than a shared "sounds like music" signal.

        Call this once after all vectors have been added.  The centroid is saved
        with the index and applied to query vectors at search time so both sides
        of the dot product live in the same centred space.

        Returns:
            The centroid vector that was subtracted (shape ``(embedding_dim,)``).
        """
        self._materialise()
        if self._matrix.shape[0] == 0:
            return np.zeros(self._embedding_dim, dtype=np.float32)

        centroid = self._matrix.mean(axis=0)  # (D,)
        centered = self._matrix - centroid[None, :]  # (N, D)
        norms = np.linalg.norm(centered, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        self._matrix = (centered / norms).astype(np.float32)
        self._centroid = centroid.astype(np.float32)
        logger.info(
            "Centred NumpyIndex: corpus mean norm=%.4f  (size=%d)",
            float(np.linalg.norm(centroid)),
            self.size,
        )
        return self._centroid

    def remove(self, track_id: str) -> None:
        """Remove a track from the index.

        This is O(N) because it rebuilds the internal matrix. Use sparingly.

        Args:
            track_id: The content_id of the track to remove.

        Raises:
            SimILIndexError: If track_id is not in the index.
        """
        self._materialise()
        if track_id not in self._id_to_idx:
            raise SimILIndexError(f"track_id {track_id!r} not found in index")

        idx = self._id_to_idx[track_id]
        keep = [i for i in range(len(self._ids)) if i != idx]
        self._matrix = self._matrix[keep]
        self._ids = [self._ids[i] for i in keep]
        self._id_to_idx = {tid: i for i, tid in enumerate(self._ids)}
        logger.debug("Removed track %s from index (new size=%d)", track_id, self.size)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query: EmbeddingVector,
        top_k: int = 10,
        exclude_ids: Sequence[str] | None = None,
        min_score: float = -1.0,
    ) -> list[tuple[str, float]]:
        """Return the top-k most similar tracks.

        Args:
            query: Query embedding vector.
            top_k: Maximum number of results.
            exclude_ids: Track IDs to mask out (score set to -2.0).
            min_score: Minimum cosine similarity threshold.

        Returns:
            List of ``(track_id, cosine_score)`` tuples, sorted descending.
        """
        self._materialise()

        if self._matrix.shape[0] == 0:
            return []

        # Apply corpus centering (if available) then L2-normalise.
        # Both the stored vectors and the query must live in the same centred
        # space so the dot product measures true similarity, not just the shared
        # "DC" direction that dominates raw EffNet-discogs embeddings.
        query32 = np.asarray(query, dtype=np.float32)
        if self._centroid is not None:
            query32 = query32 - self._centroid
        norm = np.linalg.norm(query32)
        if norm > 0.0:
            query32 = query32 / norm

        # Compute all cosine similarities in one shot
        scores: np.ndarray = (self._matrix @ query32).astype(np.float64)

        # Mask excluded tracks
        if exclude_ids:
            for eid in exclude_ids:
                idx = self._id_to_idx.get(eid)
                if idx is not None:
                    scores[idx] = -2.0

        n = self._matrix.shape[0]
        k = min(top_k, n)

        if k == 0:
            return []

        # O(N) partial sort via argpartition, then sort the top-k
        top_idx = np.argpartition(scores, -k)[-k:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]

        results = [
            (self._ids[int(i)], float(scores[i]))
            for i in top_idx
            if float(scores[i]) >= min_score
        ]
        return results

    def get_vector(self, track_id: str) -> EmbeddingVector | None:
        """Retrieve the stored vector for a track.

        Args:
            track_id: The content_id of the track.

        Returns:
            Float32 embedding vector, or None if not found.
        """
        self._materialise()
        idx = self._id_to_idx.get(track_id)
        if idx is None:
            return None
        return self._matrix[idx].copy()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        """Atomically save the index to disk.

        Writes to ``.tmp`` sibling files then renames them into place.
        Acquires ``path / ".lock"`` before writing.

        Args:
            path: Directory to write ``vectors.npy`` and ``meta.json`` into.
        """
        self._materialise()
        path.mkdir(parents=True, exist_ok=True)
        self._built_at = datetime.now(timezone.utc).isoformat()

        lock_path = path / ".lock"
        with filelock.FileLock(str(lock_path)):
            # --- vectors.npy ---
            # Note: np.save appends ".npy" if the filename doesn't end with it,
            # so we use "vectors.tmp.npy" as the staging name.
            vectors_tmp = path / "vectors.tmp.npy"
            vectors_path = path / "vectors.npy"
            np.save(str(vectors_tmp), self._matrix)
            os.replace(str(vectors_tmp), str(vectors_path))

            # --- ids.json (parallel id list) ---
            ids_tmp = path / "ids.json.tmp"
            ids_path = path / "ids.json"
            ids_tmp.write_text(json.dumps(self._ids, ensure_ascii=False), encoding="utf-8")
            os.replace(str(ids_tmp), str(ids_path))

            # --- centroid.npy (optional) ---
            if self._centroid is not None:
                centroid_tmp = path / "centroid.tmp.npy"
                centroid_path = path / "centroid.npy"
                np.save(str(centroid_tmp), self._centroid)
                os.replace(str(centroid_tmp), str(centroid_path))

            # --- meta.json ---
            meta = {
                "schema_version": SCHEMA_VERSION,
                "embedder": self._embedder_name,
                "embedding_dim": self._embedding_dim,
                "built_at": self._built_at,
                "index_type": "numpy",
                "library_id": self._library_id,
                "audio_config": self._audio_config,
            }
            meta_tmp = path / "meta.json.tmp"
            meta_path = path / "meta.json"
            meta_tmp.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
            os.replace(str(meta_tmp), str(meta_path))

        logger.info("Saved NumpyIndex (%d vectors, dim=%d) to %s", self.size, self._embedding_dim, path)

    @classmethod
    def load(cls, path: Path) -> NumpyIndex:
        """Load a NumpyIndex from disk.

        Args:
            path: Directory containing ``vectors.npy``, ``ids.json``, and ``meta.json``.

        Returns:
            Populated NumpyIndex instance.

        Raises:
            IndexSchemaError: If schema_version in meta.json is not SCHEMA_VERSION.
            SimILIndexError: If the vectors and IDs have mismatched lengths.
        """
        meta_path = path / "meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

        if meta.get("schema_version") != SCHEMA_VERSION:
            raise IndexSchemaError(
                f"Index schema version {meta.get('schema_version')!r} != {SCHEMA_VERSION}. "
                "Run with --full to rebuild."
            )

        ids_path = path / "ids.json"
        ids: list[str] = json.loads(ids_path.read_text(encoding="utf-8"))

        vectors_path = path / "vectors.npy"
        matrix = np.load(str(vectors_path))

        if len(ids) != matrix.shape[0]:
            raise SimILIndexError(
                f"IDs list length ({len(ids)}) != matrix rows ({matrix.shape[0]})"
            )

        index = cls(
            embedding_dim=meta["embedding_dim"],
            embedder_name=meta.get("embedder", ""),
            library_id=meta.get("library_id", ""),
            audio_config=meta.get("audio_config", {}),
        )
        index._matrix = matrix.astype(np.float32)
        index._ids = ids
        index._id_to_idx = {tid: i for i, tid in enumerate(ids)}
        index._built_at = meta.get("built_at", "")

        centroid_path = path / "centroid.npy"
        if centroid_path.exists():
            index._centroid = np.load(str(centroid_path)).astype(np.float32)
            logger.info("Loaded centroid for NumpyIndex (norm=%.4f)", float(np.linalg.norm(index._centroid)))

        logger.info("Loaded NumpyIndex (%d vectors, dim=%d) from %s", len(ids), meta["embedding_dim"], path)
        return index

    def get_stats(self) -> IndexStats:
        """Return statistics about the current index state."""
        return IndexStats(
            total_tracks=self.size,
            embedder_name=self._embedder_name,
            embedding_dim=self._embedding_dim,
            index_type="numpy",
            built_at=self._built_at,
            library_id=self._library_id,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _materialise(self) -> None:
        """Merge ``_pending`` vectors into ``_matrix``.

        Called lazily before any read operation (search, save, get_vector).
        """
        if not self._pending:
            return

        if self._matrix.shape[0] == 0:
            self._matrix = np.stack(self._pending, axis=0).astype(np.float32)
        else:
            pending_array = np.stack(self._pending, axis=0).astype(np.float32)
            self._matrix = np.concatenate([self._matrix, pending_array], axis=0)

        self._pending.clear()
        # Rebuild id → index mapping
        self._id_to_idx = {tid: i for i, tid in enumerate(self._ids)}

    def _validate_vector(self, vector: EmbeddingVector, label: str) -> None:
        """Validate that a vector has the correct shape and finite values.

        Args:
            vector: The vector to validate.
            label: A label (track_id or path) used in error messages.

        Raises:
            IndexDimensionError: If vector.shape != (embedding_dim,).
            SimILIndexError: If vector contains NaN or Inf values.
        """
        arr = np.asarray(vector)
        if arr.shape != (self._embedding_dim,):
            raise IndexDimensionError(
                f"Vector for {label!r} has shape {arr.shape}, "
                f"expected ({self._embedding_dim},)"
            )
        if not np.isfinite(arr).all():
            raise SimILIndexError(
                f"Vector for {label!r} contains NaN or Inf values"
            )
