"""Data models for simil."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

# Type alias for embedding vectors
EmbeddingVector = NDArray[np.float32]


@dataclass
class Track:
    """Represents a single audio track in the library.

    Attributes:
        id: Content ID (sha256 of first 160KB of file). Stable across renames.
        path: Absolute path to the audio file.
        title: Track title from metadata, or None if unavailable.
        artist: Track artist from metadata, or None if unavailable.
        album: Album name from metadata, or None if unavailable.
        duration_seconds: Duration in seconds from metadata, or None.
        mtime: File modification time (os.path.getmtime) at index time, or None.
        extra: Additional metadata fields.
    """

    id: str
    path: Path
    title: str | None = None
    artist: str | None = None
    album: str | None = None
    duration_seconds: float | None = None
    mtime: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation of this Track."""
        return {
            "id": self.id,
            "path": self.path.as_posix(),
            "title": self.title,
            "artist": self.artist,
            "album": self.album,
            "duration_seconds": self.duration_seconds,
            "mtime": self.mtime,
            "extra": self.extra,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Track:
        """Construct a Track from a JSON-deserialisable dict."""
        return cls(
            id=data["id"],
            path=Path(data["path"]),
            title=data.get("title"),
            artist=data.get("artist"),
            album=data.get("album"),
            duration_seconds=data.get("duration_seconds"),
            mtime=data.get("mtime"),
            extra=data.get("extra", {}),
        )


@dataclass
class SearchResult:
    """A single search result.

    Attributes:
        track: The matched Track.
        raw_score: Cosine similarity in [-1.0, 1.0].
        score: Min-max normalised score in [0.0, 1.0] within the result set.
        rank: 1-indexed rank in the result list.
    """

    track: Track
    raw_score: float
    score: float
    rank: int

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation of this SearchResult."""
        return {
            "rank": self.rank,
            "score": self.score,
            "raw_score": self.raw_score,
            "track": self.track.as_dict(),
        }


@dataclass
class IndexStats:
    """Statistics about the current vector index.

    Attributes:
        total_tracks: Number of tracks indexed.
        embedder_name: Name of the embedder used to build the index.
        embedding_dim: Dimensionality of each embedding vector.
        index_type: Backend type (e.g. "numpy").
        built_at: ISO-8601 UTC timestamp of last build/save.
        library_id: SHA256-derived library identifier.
    """

    total_tracks: int
    embedder_name: str
    embedding_dim: int
    index_type: str
    built_at: str
    library_id: str


@dataclass
class IndexerResult:
    """Result of an Indexer.build() run.

    Attributes:
        indexed: Number of tracks successfully indexed.
        skipped: Number of tracks skipped (already up-to-date).
        failed: Paths of tracks that could not be indexed.
        duration_seconds: Wall-clock time for the entire indexing run.
    """

    indexed: int
    skipped: int
    failed: list[Path]
    duration_seconds: float


@dataclass
class AudioConfig:
    """Audio loading/feature-extraction configuration.

    Attributes:
        sample_rate: Target sample rate in Hz.
        n_mels: Number of mel filter banks (0 = unused for MFCC).
        hop_length: Hop length for spectrogram computation.
        mono: Whether to mix down to mono.
    """

    sample_rate: int = 22050
    n_mels: int = 0
    hop_length: int = 512
    mono: bool = True

    def as_dict(self) -> dict[str, int | bool]:
        """Return a JSON-serialisable representation."""
        return {
            "sample_rate": self.sample_rate,
            "n_mels": self.n_mels,
            "hop_length": self.hop_length,
            "mono": self.mono,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AudioConfig:
        """Construct an AudioConfig from a dict."""
        return cls(
            sample_rate=int(data.get("sample_rate", 22050)),
            n_mels=int(data.get("n_mels", 0)),
            hop_length=int(data.get("hop_length", 512)),
            mono=bool(data.get("mono", True)),
        )
