"""Resolver base class and ResolvedAudio context manager."""

from __future__ import annotations

import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ResolvedAudio:
    """A resolved audio file, optionally backed by a temporary directory.

    Use as a context manager to ensure temporary files are cleaned up::

        with resolver.resolve(url) as resolved:
            vec = embedder.embed(resolved.path)

    Attributes:
        path: Absolute path to the audio file (may be in a temp directory).
        origin: The original URL or path string that was resolved.
        title: Track title from source metadata, if available.
        artist: Artist name from source metadata, if available.
    """

    path: Path
    origin: str
    title: str | None = None
    artist: str | None = None
    _tmp_dir: Path | None = field(default=None, repr=False, compare=False)

    def __enter__(self) -> ResolvedAudio:
        return self

    def __exit__(self, *args: object) -> None:
        if self._tmp_dir is not None and self._tmp_dir.exists():
            shutil.rmtree(self._tmp_dir, ignore_errors=True)


class BaseResolver(ABC):
    """Abstract base class for audio source resolvers.

    Each resolver handles a specific category of source (local files,
    YouTube/Bandcamp via yt-dlp, Spotify previews, etc.).
    Resolvers are combined in a :class:`~simil.resolvers.ResolverChain`.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier for this resolver (e.g. ``"local"``, ``"ytdlp"``)."""
        ...

    @abstractmethod
    def can_handle(self, source: str) -> bool:
        """Return ``True`` if this resolver can handle *source*.

        Must be fast and free of network I/O — inspect the source string only.

        Args:
            source: URL string or file path.
        """
        ...

    @abstractmethod
    def resolve(self, source: str) -> ResolvedAudio:
        """Resolve *source* to a local audio file path.

        Args:
            source: URL string or file path.

        Returns:
            :class:`ResolvedAudio` pointing at a local audio file.
            Use as a context manager to ensure temp files are removed.

        Raises:
            ResolverError: If resolution or download fails.
            UnsupportedURLError: If the source is not handled by this resolver.
        """
        ...
