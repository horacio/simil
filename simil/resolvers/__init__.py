"""Audio source resolver chain.

Resolvers translate a source string (local path, YouTube URL, Spotify link)
into a local audio file that the embedder can process.

Usage::

    from simil.resolvers import ResolverChain

    chain = ResolverChain()
    with chain.resolve("https://youtu.be/dQw4w9WgXcQ") as resolved:
        vec = embedder.embed(resolved.path)
        # temp file cleaned up automatically on exit

Adding a custom resolver::

    from simil.resolvers.base import BaseResolver, ResolvedAudio

    class MyResolver(BaseResolver):
        ...

    chain = ResolverChain(resolvers=[MyResolver(), *ResolverChain.default_resolvers()])
"""

from __future__ import annotations

import logging
from typing import Sequence

from simil.core.exceptions import UnsupportedURLError
from simil.resolvers.base import BaseResolver, ResolvedAudio
from simil.resolvers.local import LocalResolver
from simil.resolvers.spotify import SpotifyResolver
from simil.resolvers.ytdlp import YtDlpResolver

__all__ = [
    "BaseResolver",
    "ResolvedAudio",
    "ResolverChain",
    "LocalResolver",
    "SpotifyResolver",
    "YtDlpResolver",
]

logger = logging.getLogger(__name__)


class ResolverChain:
    """Tries resolvers in order until one reports it can handle the source.

    The default ordering is intentional:

    1. :class:`~simil.resolvers.local.LocalResolver` — fast, no network.
    2. :class:`~simil.resolvers.spotify.SpotifyResolver` — Spotify track URLs
       (before yt-dlp so Spotify previews use the dedicated 30-s API clip).
    3. :class:`~simil.resolvers.ytdlp.YtDlpResolver` — catch-all for any
       remaining ``http://`` / ``https://`` URL.

    Args:
        resolvers: Custom resolver list.  If ``None``, the default order is used.
    """

    def __init__(self, resolvers: Sequence[BaseResolver] | None = None) -> None:
        if resolvers is None:
            self._resolvers: list[BaseResolver] = [
                LocalResolver(),
                SpotifyResolver(),
                YtDlpResolver(),
            ]
        else:
            self._resolvers = list(resolvers)

    @staticmethod
    def default_resolvers() -> list[BaseResolver]:
        """Return a fresh list of the default resolvers (useful for extension)."""
        return [LocalResolver(), SpotifyResolver(), YtDlpResolver()]

    def resolve(self, source: str) -> ResolvedAudio:
        """Resolve *source* to a local audio file.

        Iterates through resolvers in order; returns the result of the first
        resolver whose :meth:`~simil.resolvers.base.BaseResolver.can_handle`
        returns ``True``.

        Args:
            source: URL string or file path.

        Returns:
            :class:`~simil.resolvers.base.ResolvedAudio` pointing at a local
            audio file.  Use as a context manager for automatic cleanup.

        Raises:
            UnsupportedURLError: If no resolver can handle *source*.
            ResolverError: If a resolver claims to handle *source* but fails.
        """
        for resolver in self._resolvers:
            if resolver.can_handle(source):
                logger.debug("Resolver %r handling %r", resolver.name, source)
                return resolver.resolve(source)

        raise UnsupportedURLError(
            f"No resolver can handle {source!r}. "
            f"Available resolvers: {[r.name for r in self._resolvers]}"
        )
