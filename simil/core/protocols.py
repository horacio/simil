"""Runtime-checkable Protocols for simil abstractions."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, Sequence, runtime_checkable

from simil.core.models import EmbeddingVector, IndexStats


@runtime_checkable
class Embedder(Protocol):
    """Protocol for audio embedding models.

    An Embedder converts an audio file into a fixed-dimension float32 vector.
    """

    @property
    def name(self) -> str:
        """Unique name for this embedder (e.g. "mfcc", "clap")."""
        ...

    @property
    def embedding_dim(self) -> int:
        """Dimensionality of the output embedding vectors."""
        ...

    @property
    def audio_config(self) -> dict[str, object]:
        """Audio loading parameters (sr, n_mels, hop_length, etc.) for index metadata."""
        ...

    def embed(self, audio_path: Path) -> EmbeddingVector:
        """Embed a single audio file.

        Args:
            audio_path: Path to the audio file.

        Returns:
            L2-normalised float32 embedding vector of shape (embedding_dim,).
        """
        ...

    def embed_batch(self, audio_paths: Sequence[Path]) -> list[EmbeddingVector]:
        """Embed multiple audio files.

        Args:
            audio_paths: Sequence of paths to audio files.

        Returns:
            List of L2-normalised float32 embedding vectors.
        """
        ...


@runtime_checkable
class Index(Protocol):
    """Protocol for vector similarity indices."""

    @property
    def size(self) -> int:
        """Number of vectors currently stored."""
        ...

    @property
    def embedding_dim(self) -> int:
        """Dimensionality of stored vectors."""
        ...

    def add(self, track_id: str, vector: EmbeddingVector) -> None:
        """Add a single vector to the index.

        Args:
            track_id: Unique track identifier (content_id).
            vector: L2-normalised float32 embedding vector.
        """
        ...

    def add_batch(self, track_ids: Sequence[str], vectors: Sequence[EmbeddingVector]) -> None:
        """Add multiple vectors to the index at once.

        Args:
            track_ids: Sequence of unique track identifiers.
            vectors: Corresponding sequence of embedding vectors.
        """
        ...

    def search(
        self,
        query: EmbeddingVector,
        top_k: int = 10,
        exclude_ids: Sequence[str] | None = None,
        min_score: float = -1.0,
    ) -> list[tuple[str, float]]:
        """Search for the top-k most similar vectors.

        Args:
            query: Query embedding vector.
            top_k: Maximum number of results to return.
            exclude_ids: Track IDs to exclude from results (e.g. the query track itself).
            min_score: Minimum cosine similarity threshold.

        Returns:
            List of (track_id, cosine_score) tuples, sorted by score descending.
        """
        ...

    def remove(self, track_id: str) -> None:
        """Remove a track from the index.

        Args:
            track_id: The content_id of the track to remove.
        """
        ...

    def save(self, path: Path) -> None:
        """Atomically persist the index to disk.

        Args:
            path: Directory path where index files are stored.
        """
        ...

    @classmethod
    def load(cls, path: Path) -> Index:
        """Load an index from disk.

        Args:
            path: Directory path where index files are stored.

        Returns:
            Populated Index instance.
        """
        ...

    def get_stats(self) -> IndexStats:
        """Return statistics about the current index state."""
        ...


@runtime_checkable
class TextEmbedder(Protocol):
    """Protocol for embedders that can embed free-form text queries.

    Implemented by CLAP-based embedders, which share an embedding space
    between audio and text — enabling queries like "dark ambient drone" to
    return similar tracks from an audio-indexed library.

    Not all Embedders are TextEmbedders. Check ``isinstance(embedder, TextEmbedder)``
    before calling ``embed_text``.
    """

    def embed_text(self, text: str) -> EmbeddingVector:
        """Embed a free-form text description into the shared audio/text space.

        Args:
            text: Natural-language description (e.g. "melancholic lo-fi jazz").

        Returns:
            L2-normalised float32 embedding vector compatible with the audio index.

        Raises:
            EmbeddingError: If text embedding fails.
        """
        ...
