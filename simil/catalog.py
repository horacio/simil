"""Track catalog — maps content IDs to Track metadata."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import filelock

from simil.core.exceptions import CatalogSchemaError
from simil.core.models import Track

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1


class TrackCatalog:
    """Persistent mapping from content IDs to Track objects.

    The catalog is stored as ``catalog.json`` alongside the vector index.
    Both share the same ``.lock`` file for coordinated writes.

    Args:
        library_id: SHA256-derived identifier for the owning library.
    """

    def __init__(self, library_id: str = "") -> None:
        """Initialise an empty TrackCatalog."""
        self._tracks: dict[str, Track] = {}
        self._library_id = library_id

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def library_id(self) -> str:
        """The library identifier associated with this catalog."""
        return self._library_id

    @property
    def size(self) -> int:
        """Number of tracks in the catalog."""
        return len(self._tracks)

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add(self, track: Track) -> None:
        """Add or update a track in the catalog.

        Args:
            track: The Track to add. If a track with the same ID already
                exists it is replaced.
        """
        self._tracks[track.id] = track
        logger.debug("Catalog add: %s (%s)", track.id, track.path)

    def remove(self, track_id: str) -> None:
        """Remove a track from the catalog.

        Args:
            track_id: The content_id of the track to remove.

        Raises:
            KeyError: If track_id is not in the catalog.
        """
        del self._tracks[track_id]
        logger.debug("Catalog remove: %s", track_id)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get(self, track_id: str) -> Track | None:
        """Retrieve a track by its content_id.

        Args:
            track_id: The content_id to look up.

        Returns:
            The Track, or None if not found.
        """
        return self._tracks.get(track_id)

    def contains(self, track_id: str) -> bool:
        """Check whether a track is in the catalog.

        Args:
            track_id: The content_id to check.

        Returns:
            True if the track exists.
        """
        return track_id in self._tracks

    def all_ids(self) -> list[str]:
        """Return a list of all track IDs in the catalog."""
        return list(self._tracks.keys())

    def all_tracks(self) -> list[Track]:
        """Return a list of all Track objects in the catalog."""
        return list(self._tracks.values())

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        """Atomically save the catalog to ``catalog.json``.

        Acquires the shared ``path.parent / ".lock"`` before writing.

        Args:
            path: Full path to the target ``catalog.json`` file.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = path.parent / ".lock"

        payload = {
            "schema_version": SCHEMA_VERSION,
            "library_id": self._library_id,
            "built_at": datetime.now(timezone.utc).isoformat(),
            "tracks": [t.as_dict() for t in self._tracks.values()],
        }

        with filelock.FileLock(str(lock_path)):
            tmp_path = path.parent / (path.name + ".tmp")
            tmp_path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            os.replace(str(tmp_path), str(path))

        logger.info("Saved catalog (%d tracks) to %s", self.size, path)

    @classmethod
    def load(cls, path: Path, library_id: str = "") -> TrackCatalog:
        """Load a TrackCatalog from ``catalog.json``.

        Args:
            path: Full path to the ``catalog.json`` file.
            library_id: Expected library ID. A mismatch produces a warning
                but does not raise (paths may still be valid after migration).

        Returns:
            Populated TrackCatalog instance.

        Raises:
            CatalogSchemaError: If schema_version != SCHEMA_VERSION.
        """
        raw = json.loads(path.read_text(encoding="utf-8"))

        if raw.get("schema_version") != SCHEMA_VERSION:
            raise CatalogSchemaError(
                f"Catalog schema version {raw.get('schema_version')!r} != {SCHEMA_VERSION}."
            )

        stored_library_id: str = raw.get("library_id", "")
        if library_id and stored_library_id and stored_library_id != library_id:
            logger.warning(
                "Catalog library_id %r != expected %r — catalog may have been moved.",
                stored_library_id,
                library_id,
            )

        catalog = cls(library_id=stored_library_id or library_id)
        for track_data in raw.get("tracks", []):
            track = Track.from_dict(track_data)
            catalog._tracks[track.id] = track

        logger.info("Loaded catalog (%d tracks) from %s", catalog.size, path)
        return catalog
