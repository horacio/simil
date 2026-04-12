"""Similarity search engine."""

from __future__ import annotations

import logging
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
        """Embed and search a local audio file.

        The query vector is always produced by a fresh embed call.  Using a
        cached (stored) vector is intentionally avoided: the index stores
        *centred* vectors but ``index.search`` centres the query itself, so
        reusing a cached vector would double-centre it and produce garbage
        scores.  Embedding is fast enough (~0.5 s) that the cache hit would
        not be worth the correctness risk.
        """
        # Embed fresh — always in the same (raw, un-centred) space that
        # index.search() expects to receive and will centre internally.
        query_vector: EmbeddingVector = self.embedder.embed(source_path)
        logger.debug("Embedded query: %s", source_path.name)

        # If the file is already in the library, exclude it from results so
        # the query track does not appear as its own best match.
        exclude_ids: list[str] = []
        cid = content_id(source_path)
        if self.catalog.contains(cid):
            exclude_ids = [cid]

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


def _normalise_scores(
    results: list[tuple[str, float]],
) -> list[tuple[str, float, float]]:
    """Attach a display score to each result.

    The ``score`` field is the raw cosine similarity clipped to [0, 1] — it
    represents actual similarity to the query, not relative rank among results.

    Min-max normalisation over the result set was intentionally removed: it
    stretched a tiny range (e.g. 0.81–0.83) to [0, 1], making the worst result
    look 0% similar and the best look 100%, even when neither was meaningfully
    similar to the query.

    Args:
        results: List of ``(track_id, raw_score)`` pairs, sorted descending.

    Returns:
        List of ``(track_id, raw_score, display_score)`` triples where
        ``display_score`` is the raw cosine clipped to [0.0, 1.0].
    """
    return [
        (tid, rs, float(np.clip(rs, 0.0, 1.0)))
        for tid, rs in results
    ]
