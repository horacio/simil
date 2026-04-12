"""Library indexer — scans, embeds, and persists audio tracks."""

from __future__ import annotations

import logging
import os
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from simil.catalog import TrackCatalog
from simil.config import Settings
from simil.core.exceptions import EmbeddingError, IndexEmbedderMismatch
from simil.core.models import IndexerResult, Track
from simil.core.protocols import Embedder
from simil.index.numpy_index import NumpyIndex
from simil.library.metadata import extract_metadata
from simil.library.scanner import content_id, scan_library

logger = logging.getLogger(__name__)


def _load_audio_task(path: Path) -> tuple[Path, str, float, dict[str, Any]]:
    """I/O-bound task executed in the thread pool.

    Computes the content_id, reads mtime, and extracts metadata for a path.
    Audio loading / embedding is intentionally left to the main thread.

    Args:
        path: Path to the audio file.

    Returns:
        ``(path, cid, mtime, metadata_dict)``
    """
    cid = content_id(path)
    mtime = os.path.getmtime(str(path))
    meta = extract_metadata(path)
    return path, cid, mtime, meta


@dataclass
class Indexer:
    """Orchestrates library scanning, embedding, and index persistence.

    Threading model:
    - A ``ThreadPoolExecutor`` runs :func:`_load_audio_task` concurrently
      (disk I/O: hashing + metadata).
    - The main thread consumes completed futures and calls
      ``embedder.embed()`` serially (inference is single-threaded).

    Args:
        embedder: Audio embedder to use.
        index: Target NumpyIndex.
        catalog: Target TrackCatalog.
        settings: Application settings (workers, checkpoint_every, etc.).
    """

    embedder: Embedder
    index: NumpyIndex
    catalog: TrackCatalog
    settings: Settings

    def build(self, library_path: Path, full: bool = False) -> IndexerResult:
        """Scan, embed, and index all audio files under ``library_path``.

        Args:
            library_path: Root directory to scan.
            full: If True, re-embed all tracks ignoring the existing catalog.

        Returns:
            :class:`~simil.core.models.IndexerResult` with indexed/skipped/failed counts.

        Raises:
            IndexEmbedderMismatch: If the index was built with a different embedder
                and ``full=False``.
        """
        start_time = time.monotonic()

        # -- Embedder mismatch guard --
        if not full and self.index.size > 0:
            stored_embedder = self.index._embedder_name
            if stored_embedder and stored_embedder != self.embedder.name:
                raise IndexEmbedderMismatch(
                    f"Index was built with embedder {stored_embedder!r} but current embedder "
                    f"is {self.embedder.name!r}. Run with --full to rebuild."
                )

        if full:
            logger.info("Full rebuild requested — clearing existing index and catalog.")
            # Reset index
            self.index.__init__(  # type: ignore[misc]
                embedding_dim=self.embedder.embedding_dim,
                embedder_name=self.embedder.name,
                library_id=self.index._library_id,
                audio_config=self.embedder.audio_config,
            )
            self.catalog._tracks.clear()

        audio_paths = scan_library(library_path)
        logger.info("Found %d audio files in %s", len(audio_paths), library_path)

        indexed = 0
        skipped = 0
        failed: list[Path] = []

        # Submit I/O tasks to thread pool
        futures: list[Future[tuple[Path, str, float, dict[str, Any]]]] = []
        with ThreadPoolExecutor(max_workers=self.settings.workers) as executor:
            for ap in audio_paths:
                futures.append(executor.submit(_load_audio_task, ap))

            for future in futures:
                try:
                    path, cid, mtime, meta = future.result()
                except Exception as exc:
                    logger.warning("I/O task failed: %s", exc)
                    failed.append(path if "path" in dir() else Path("<unknown>"))
                    continue

                # -- Incremental skip check --
                if not full and self.catalog.contains(cid):
                    existing = self.catalog.get(cid)
                    if existing is not None and existing.mtime == mtime:
                        logger.debug("Skipping unchanged track: %s", path.name)
                        skipped += 1
                        continue

                # -- Embed (main thread, serial) --
                try:
                    vec = self.embedder.embed(path)
                except EmbeddingError as exc:
                    logger.warning("Embedding failed for %s: %s", path, exc)
                    failed.append(path)
                    continue

                # -- Build Track --
                track = Track(
                    id=cid,
                    path=path,
                    title=meta.get("title"),  # type: ignore[arg-type]
                    artist=meta.get("artist"),  # type: ignore[arg-type]
                    album=meta.get("album"),  # type: ignore[arg-type]
                    duration_seconds=meta.get("duration"),  # type: ignore[arg-type]
                    mtime=mtime,
                )

                self.index.add(cid, vec)
                self.catalog.add(track)
                indexed += 1

                # -- Checkpoint --
                if indexed % self.settings.checkpoint_every == 0:
                    logger.info("Checkpoint at %d indexed tracks…", indexed)
                    self._checkpoint()

        # Final save
        self._checkpoint()

        # Centre the index: subtract corpus mean so cosine distance reflects
        # actual musical similarity rather than a shared embedding-space bias.
        if self.index.size > 0:
            self.index.center()
            self._checkpoint()

        duration = time.monotonic() - start_time
        result = IndexerResult(
            indexed=indexed,
            skipped=skipped,
            failed=failed,
            duration_seconds=duration,
        )
        logger.info(
            "Indexing complete: indexed=%d skipped=%d failed=%d in %.1fs",
            indexed,
            skipped,
            len(failed),
            duration,
        )
        return result

    def _checkpoint(self) -> None:
        """Save the index and catalog to disk (checkpoint)."""
        index_dir = self.settings.index_dir
        catalog_path = index_dir / "catalog.json"
        try:
            self.index.save(index_dir)
            self.catalog.save(catalog_path)
        except Exception as exc:
            logger.error("Checkpoint save failed: %s", exc)
