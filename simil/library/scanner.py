"""Library file scanner and content-ID utilities."""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path

from simil.core.exceptions import LibraryError

logger = logging.getLogger(__name__)

AUDIO_EXTENSIONS: frozenset[str] = frozenset(
    {".mp3", ".flac", ".wav", ".ogg", ".m4a", ".opus", ".aac", ".aif", ".aiff", ".wma"}
)


def scan_library(library_path: Path, max_depth: int = 20) -> list[Path]:
    """Recursively scan a directory for audio files.

    Uses ``os.walk`` with ``followlinks=False``.  Only files whose
    lower-cased extension is in :data:`AUDIO_EXTENSIONS` are returned.

    Args:
        library_path: Root directory to scan.
        max_depth: Maximum recursion depth (default 20).  Directories
            deeper than this are not entered.

    Returns:
        Sorted list of absolute ``Path`` objects for audio files found.
    """
    found: list[Path] = []
    root_parts = len(library_path.parts)

    for dirpath, dirnames, filenames in os.walk(str(library_path), followlinks=False):
        current_depth = len(Path(dirpath).parts) - root_parts
        if current_depth >= max_depth:
            # Don't recurse deeper — clear dirnames in-place to prune walk
            dirnames.clear()
            continue

        for filename in filenames:
            ext = os.path.splitext(filename)[1].lower()
            if ext in AUDIO_EXTENSIONS:
                found.append(Path(dirpath) / filename)

    found.sort()
    logger.debug("scan_library(%s): found %d audio files", library_path, len(found))
    return found


def content_id(path: Path, sample_bytes: int = 163840) -> str:
    """Compute a stable content identifier for an audio file.

    Hashes the first ``sample_bytes`` (default 160 KB) of the file using
    SHA-256.  This is stable across renames and machine boundaries as long
    as the audio content does not change.

    Args:
        path: Path to the audio file.
        sample_bytes: Number of bytes to hash (default 163840 = 160 KB).

    Returns:
        First 24 hex characters of the SHA-256 digest (96 bits of uniqueness).

    Raises:
        LibraryError: If the file cannot be opened or read.
    """
    try:
        with open(path, "rb") as fh:
            data = fh.read(sample_bytes)
    except FileNotFoundError as exc:
        raise LibraryError(f"File not found: {path!r}") from exc
    except OSError as exc:
        raise LibraryError(f"Cannot read file {path!r}: {exc}") from exc

    digest = hashlib.sha256(data).hexdigest()
    return digest[:24]
