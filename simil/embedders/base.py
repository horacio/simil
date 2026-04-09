"""Abstract base class for audio embedders."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Sequence

import numpy as np

from simil.core.exceptions import EmbeddingError
from simil.core.models import EmbeddingVector

logger = logging.getLogger(__name__)


class BaseEmbedder(ABC):
    """Abstract base class for audio embedders.

    Subclasses must implement ``name``, ``embedding_dim``, ``audio_config``,
    and ``embed``. A default ``embed_batch`` implementation is provided that
    simply loops over ``embed``.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name for this embedder (e.g. "mfcc", "clap")."""

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Dimensionality of the output embedding vectors."""

    @property
    @abstractmethod
    def audio_config(self) -> dict[str, object]:
        """Audio loading parameters (sample_rate, n_mels, hop_length, mono)."""

    @abstractmethod
    def embed(self, audio_path: Path) -> EmbeddingVector:
        """Embed a single audio file.

        Args:
            audio_path: Path to the audio file.

        Returns:
            L2-normalised float32 embedding vector of shape ``(embedding_dim,)``.

        Raises:
            EmbeddingError: If embedding fails for any reason.
        """

    def embed_batch(self, audio_paths: Sequence[Path]) -> list[EmbeddingVector]:
        """Embed multiple audio files sequentially.

        This default implementation calls ``embed`` for each path. Subclasses
        may override for more efficient batched inference.

        Args:
            audio_paths: Sequence of paths to audio files.

        Returns:
            List of L2-normalised float32 embedding vectors.
        """
        results: list[EmbeddingVector] = []
        for path in audio_paths:
            results.append(self.embed(path))
        return results

    def _validate_vector(self, vec: np.ndarray, path: Path) -> None:
        """Validate that a vector has the correct shape and finite values.

        Args:
            vec: The embedding vector to validate.
            path: Source audio path, used in error messages.

        Raises:
            EmbeddingError: If vec has wrong shape or contains NaN/Inf.
        """
        if vec.shape != (self.embedding_dim,):
            raise EmbeddingError(
                f"Embedding for {path!r} has shape {vec.shape}, "
                f"expected ({self.embedding_dim},)"
            )
        if not np.isfinite(vec).all():
            raise EmbeddingError(
                f"Embedding for {path!r} contains NaN or Inf values"
            )
