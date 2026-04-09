"""Similarity search engine."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from simil.catalog import TrackCatalog
from simil.config import Settings
from simil.core.exceptions import UnsupportedURLError
from simil.core.models import EmbeddingVector, SearchResult
from simil.core.protocols import Embedder
from simil.index.numpy_index import NumpyIndex
from simil.library.scanner import content_id

if TYPE_CHECKING:
    from simil.resolvers import ResolverChain

logger = logging.getLogger(__name__)


def _is_url(source: str) -> bool:
    """Return True if *source* looks like a URL (has a scheme)."""
    return "://" in source


@dataclass
class SearchEngine:
    """Orchestrates similarity search over an indexed library.

    Args:
        embedder: Audio embedder to use for query tracks not in the index.
        index: The vector index to search.
        catalog: Track catalog for metadata lookups.
        settings: Application settings.
        resolver_chain: Optional custom resolver chain for URL resolution.
            If ``None``, a default :class:`~simil.resolvers.ResolverChain`
            is created on first use.
    """

    embedder: Embedder
    index: NumpyIndex
    catalog: TrackCatalog
    settings: Settings
    resolver_chain: ResolverChain | None = field(default=None, repr=False)

    def search(
        self,
        source: Path | str,
        top_k: int | None = None,
        min_score: float | None = None,
    ) -> list[SearchResult]:
        """Find the most similar tracks to *source*.

        *source* may be:

        - A :class:`~pathlib.Path` → local audio file.
        - A string pointing to an existing local file.
        - A URL (``http://``, ``https://``, ``file://``, Spotify, etc.) →
          resolved and embedded on-the-fly via the resolver chain.
        - A free-form text query (e.g. ``"dark ambient drone"``) →
          embedded via :meth:`~simil.core.protocols.TextEmbedder.embed_text`
          if the current embedder supports it (requires the CLAP embedder).

        If the source file is already in the catalog **and** its mtime matches,
        the stored vector is retrieved directly (no re-embedding).

        Args:
            source: Audio source — path, URL, or text query string.
            top_k: Number of results (falls back to ``settings.top_k``).
            min_score: Minimum raw cosine similarity threshold
                (falls back to ``settings.min_score``).

        Returns:
            List of :class:`~simil.core.models.SearchResult` objects sorted by
            normalised score descending.

        Raises:
            UnsupportedURLError: If *source* is a URL not handled by any resolver,
                or a text query with a non-text-capable embedder.
            EmbeddingError: If audio loading or inference fails.
            ResolverError: If a URL resolver fails.
        """
        k = top_k if top_k is not None else self.settings.top_k
        threshold = min_score if min_score is not None else self.settings.min_score

        if isinstance(source, Path):
            return self._search_local(source.resolve(), k, threshold)

        source_str = str(source)

        if _is_url(source_str):
            return self._search_url(source_str, k, threshold)

        source_path = Path(source_str).expanduser()
        if source_path.exists():
            return self._search_local(source_path.resolve(), k, threshold)

        # Not a URL and not an existing path — treat as a free-form text query.
        return self._search_text(source_str, k, threshold)

    # ── Private search routes ─────────────────────────────────────────────────

    def _search_local(
        self, source_path: Path, k: int, threshold: float
    ) -> list[SearchResult]:
        """Embed and search a local audio file."""
        query_vector: EmbeddingVector
        exclude_ids: list[str] = []

        cid = content_id(source_path)
        current_mtime = os.path.getmtime(str(source_path))

        if self.catalog.contains(cid):
            existing = self.catalog.get(cid)
            if existing is not None and existing.mtime == current_mtime:
                cached = self.index.get_vector(cid)
                if cached is not None:
                    query_vector = cached
                    exclude_ids = [cid]
                    logger.debug("Using cached vector for %s", source_path.name)
                else:
                    # Catalog/index desync — re-embed
                    query_vector = self._embed_and_update(source_path, cid, current_mtime)
                    exclude_ids = [cid]
            else:
                # mtime changed — re-embed and update
                query_vector = self._embed_and_update(source_path, cid, current_mtime)
                exclude_ids = [cid]
        else:
            query_vector = self.embedder.embed(source_path)
            logger.debug("Embedded fresh (not in library): %s", source_path.name)

        raw_results = self.index.search(
            query_vector,
            top_k=k,
            exclude_ids=exclude_ids if exclude_ids else None,
            min_score=threshold,
        )
        return self._build_results(_normalise_scores(raw_results))

    def _search_url(self, url: str, k: int, threshold: float) -> list[SearchResult]:
        """Resolve a URL to audio, embed it, and search."""
        from simil.resolvers import ResolverChain  # noqa: PLC0415

        chain = self.resolver_chain or ResolverChain()
        with chain.resolve(url) as resolved:
            query_vector = self.embedder.embed(resolved.path)

        raw_results = self.index.search(query_vector, top_k=k, min_score=threshold)
        return self._build_results(_normalise_scores(raw_results))

    def _search_text(self, text: str, k: int, threshold: float) -> list[SearchResult]:
        """Embed a free-form text query and search (requires a TextEmbedder)."""
        from simil.core.protocols import TextEmbedder  # noqa: PLC0415

        if not isinstance(self.embedder, TextEmbedder):
            raise UnsupportedURLError(
                f"Text queries require a TextEmbedder (e.g. CLAP). "
                f"Current embedder {self.embedder.name!r} does not support text. "
                f"Switch with: simil index --embedder clap"
            )
        query_vector = self.embedder.embed_text(text)  # type: ignore[attr-defined]
        raw_results = self.index.search(query_vector, top_k=k, min_score=threshold)
        return self._build_results(_normalise_scores(raw_results))

    def _build_results(
        self, normalised: list[tuple[str, float, float]]
    ) -> list[SearchResult]:
        """Convert normalised score triples to :class:`~simil.core.models.SearchResult` objects."""
        results: list[SearchResult] = []
        for rank, (track_id, raw_score, norm_score) in enumerate(normalised, start=1):
            track = self.catalog.get(track_id)
            if track is None:
                logger.warning("Track %s in index but missing from catalog", track_id)
                continue
            results.append(
                SearchResult(
                    track=track,
                    raw_score=raw_score,
                    score=norm_score,
                    rank=rank,
                )
            )
        return results

    def _embed_and_update(
        self, path: Path, cid: str, mtime: float
    ) -> EmbeddingVector:
        """Re-embed a track and update the index + catalog in place."""
        vec = self.embedder.embed(path)
        try:
            self.index.remove(cid)
        except Exception:
            pass
        self.index.add(cid, vec)

        existing = self.catalog.get(cid)
        if existing is not None:
            from dataclasses import replace  # noqa: PLC0415

            self.catalog.add(replace(existing, mtime=mtime))

        logger.debug("Re-embedded and updated: %s", path.name)
        return vec


def _normalise_scores(
    results: list[tuple[str, float]],
) -> list[tuple[str, float, float]]:
    """Min-max normalise raw cosine similarity scores to [0.0, 1.0].

    Args:
        results: List of ``(track_id, raw_score)`` pairs.

    Returns:
        List of ``(track_id, raw_score, normalised_score)`` triples.
    """
    if not results:
        return []

    raw_scores = [r[1] for r in results]

    if len(raw_scores) < 2:
        clamped = float(np.clip(raw_scores[0], 0.0, 1.0))
        return [(results[0][0], raw_scores[0], clamped)]

    mn = min(raw_scores)
    mx = max(raw_scores)

    if mx == mn:
        return [(tid, rs, 1.0) for tid, rs in results]

    return [(tid, rs, float((rs - mn) / (mx - mn))) for tid, rs in results]
