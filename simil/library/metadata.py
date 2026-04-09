"""Audio file metadata extraction using mutagen."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def extract_metadata(path: Path) -> dict[str, str | float | None]:
    """Extract basic metadata from an audio file using mutagen.

    All exceptions are caught — corrupt or unsupported files return a dict
    with ``None`` for all fields rather than raising.

    Args:
        path: Path to the audio file.

    Returns:
        Dict with keys ``"title"``, ``"artist"``, ``"album"``,
        ``"duration"`` (float seconds or None).
    """
    result: dict[str, str | float | None] = {
        "title": None,
        "artist": None,
        "album": None,
        "duration": None,
    }

    try:
        import mutagen  # noqa: PLC0415

        audio_file: Any = mutagen.File(str(path), easy=True)
        if audio_file is None:
            logger.debug("mutagen returned None for %s", path)
            return result

        tags = audio_file.tags or {}

        def _first(key: str) -> str | None:
            """Return the first value for a tag key, or None."""
            val = tags.get(key)
            if val and isinstance(val, (list, tuple)) and len(val) > 0:
                return str(val[0])
            if isinstance(val, str):
                return val
            return None

        result["title"] = _first("title")
        result["artist"] = _first("artist")
        result["album"] = _first("album")

        # Duration from audio_file.info if available
        info = getattr(audio_file, "info", None)
        if info is not None:
            dur = getattr(info, "length", None)
            if dur is not None:
                result["duration"] = float(dur)

    except Exception as exc:  # noqa: BLE001
        logger.debug("Could not extract metadata from %s: %s", path, exc)

    return result
