"""Local file resolver — handles plain paths and file:// URLs."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import unquote, urlparse

from simil.core.exceptions import ResolverError
from simil.resolvers.base import BaseResolver, ResolvedAudio


class LocalResolver(BaseResolver):
    """Resolves local file paths and ``file://`` URLs.

    Always registered first in the resolver chain to short-circuit network
    activity for files already on disk.
    """

    @property
    def name(self) -> str:
        return "local"

    def can_handle(self, source: str) -> bool:
        """Return ``True`` for ``file://`` URLs or existing local paths."""
        if source.startswith("file://"):
            return True
        return Path(source).exists()

    def resolve(self, source: str) -> ResolvedAudio:
        """Return a :class:`~simil.resolvers.base.ResolvedAudio` for a local file.

        Args:
            source: A ``file://`` URL or absolute/relative file path string.

        Raises:
            ResolverError: If the file does not exist or is not a regular file.
        """
        if source.startswith("file://"):
            parsed = urlparse(source)
            path = Path(unquote(parsed.path)).resolve()
        else:
            path = Path(source).resolve()

        if not path.exists():
            raise ResolverError(f"Local file not found: {path}")
        if not path.is_file():
            raise ResolverError(f"Path is not a regular file: {path}")

        return ResolvedAudio(path=path, origin=source)
