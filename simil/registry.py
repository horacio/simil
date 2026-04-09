"""Remote index registry — list and download pre-built indexes.

The registry manifest lives at REGISTRY_URL (a JSON file in the GitHub repo).
It lists pre-built indexes with their download URLs and SHA-256 checksums so
that users who don't have a local music library can still search against a
community-maintained catalog.

Typical usage::

    from simil.registry import fetch_registry, download_index

    entries = fetch_registry()
    for e in entries:
        print(e.name, e.tracks, e.size_mb)

    download_index(entries[0], dest_dir=Path.home() / ".simil/libraries/fma-small")
"""

from __future__ import annotations

import hashlib
import json
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path

import httpx

from simil.core.exceptions import SimILError

# ── Constants ─────────────────────────────────────────────────────────────────

# Points at the raw JSON file in the repo — updated in-place when new indexes
# are published, so clients always get the latest list without a code release.
REGISTRY_URL = (
    "https://raw.githubusercontent.com/horacio/simil/main/registry.json"
)


# ── Exceptions ─────────────────────────────────────────────────────────────────


class RegistryError(SimILError):
    """Raised when the registry cannot be fetched, parsed, or validated."""


class ChecksumError(SimILError):
    """Raised when a downloaded archive fails SHA-256 verification."""


# ── Data model ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class IndexEntry:
    """A single entry in the remote registry manifest."""

    name: str
    description: str
    embedder: str
    tracks: int
    url: str
    sha256: str
    size_bytes: int

    _FIELDS = frozenset(
        {"name", "description", "embedder", "tracks", "url", "sha256", "size_bytes"}
    )

    @classmethod
    def from_dict(cls, d: dict) -> "IndexEntry":
        try:
            return cls(**{k: v for k, v in d.items() if k in cls._FIELDS})
        except TypeError as exc:
            raise RegistryError(f"Malformed registry entry: {exc}") from exc

    @property
    def size_mb(self) -> float:
        return self.size_bytes / (1024 * 1024)


# ── Registry fetching ──────────────────────────────────────────────────────────


def fetch_registry(
    url: str = REGISTRY_URL,
    *,
    timeout: float = 15.0,
) -> list[IndexEntry]:
    """Fetch and parse the remote registry manifest.

    Args:
        url:     URL of the registry JSON file.
        timeout: HTTP request timeout in seconds.

    Returns:
        List of :class:`IndexEntry` objects available for download.

    Raises:
        RegistryError: On network failure or unexpected response format.
    """
    try:
        resp = httpx.get(url, timeout=timeout, follow_redirects=True)
        resp.raise_for_status()
        data = resp.json()
    except httpx.HTTPStatusError as exc:
        raise RegistryError(
            f"Registry returned HTTP {exc.response.status_code}: {url}"
        ) from exc
    except httpx.HTTPError as exc:
        raise RegistryError(f"Could not reach registry at {url}: {exc}") from exc
    except (json.JSONDecodeError, ValueError) as exc:
        raise RegistryError(f"Malformed registry response: {exc}") from exc

    try:
        raw_entries: list[dict] = data["indexes"]
    except (KeyError, TypeError) as exc:
        raise RegistryError(f"Unexpected registry schema (missing 'indexes'): {exc}") from exc

    return [IndexEntry.from_dict(entry) for entry in raw_entries]


# ── Download, verify, extract ─────────────────────────────────────────────────


def download_index(
    entry: IndexEntry,
    dest_dir: Path,
    *,
    timeout: float = 120.0,
) -> None:
    """Download, verify, and extract *entry* into *dest_dir*.

    The archive is staged inside a temporary directory so an interrupted or
    failed download never leaves a partial index in *dest_dir*.

    Args:
        entry:    Registry entry to download.
        dest_dir: Directory to extract the index into (will be created).
        timeout:  Per-chunk read timeout in seconds.

    Raises:
        RegistryError:  On download failure or unsafe archive contents.
        ChecksumError:  If the downloaded archive does not match ``entry.sha256``.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="simil_fetch_") as tmp:
        archive = Path(tmp) / "index.tar.gz"
        _stream_download(entry, archive, timeout=timeout)
        _verify_checksum(archive, entry.sha256)
        _safe_extract(archive, dest_dir)


def _stream_download(entry: IndexEntry, dest: Path, *, timeout: float) -> None:
    """Stream *entry.url* to *dest*, showing a Rich progress bar."""
    from rich.progress import (
        BarColumn,
        DownloadColumn,
        Progress,
        TextColumn,
        TimeRemainingColumn,
        TransferSpeedColumn,
    )

    try:
        with Progress(
            TextColumn("  [bold cyan]{task.fields[name]}"),
            BarColumn(),
            DownloadColumn(),
            TransferSpeedColumn(),
            TimeRemainingColumn(),
            transient=True,
        ) as progress:
            task = progress.add_task("dl", name=entry.name, total=entry.size_bytes or None)
            with httpx.stream(
                "GET", entry.url, timeout=timeout, follow_redirects=True
            ) as resp:
                resp.raise_for_status()
                # Update total from Content-Length if registry size was 0
                cl = resp.headers.get("content-length")
                if cl and not entry.size_bytes:
                    progress.update(task, total=int(cl))
                with dest.open("wb") as fh:
                    for chunk in resp.iter_bytes(65536):
                        fh.write(chunk)
                        progress.update(task, advance=len(chunk))
    except httpx.HTTPStatusError as exc:
        raise RegistryError(
            f"Download failed — HTTP {exc.response.status_code}: {entry.url}"
        ) from exc
    except httpx.HTTPError as exc:
        raise RegistryError(f"Download failed: {exc}") from exc


def _verify_checksum(path: Path, expected: str) -> None:
    """Raise :class:`ChecksumError` if *path* does not match *expected* SHA-256."""
    sha = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            sha.update(chunk)
    actual = sha.hexdigest()
    if actual != expected:
        raise ChecksumError(
            f"SHA-256 mismatch.\n"
            f"  Expected : {expected}\n"
            f"  Got      : {actual}\n"
            "The download may be corrupted. Try again."
        )


def _safe_extract(archive: Path, dest_dir: Path) -> None:
    """Extract *archive* into *dest_dir*, rejecting path-traversal members."""
    with tarfile.open(archive, "r:gz") as tf:
        for member in tf.getmembers():
            member_path = Path(member.name)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise RegistryError(
                    f"Refusing to extract unsafe path in archive: {member.name!r}"
                )
        tf.extractall(dest_dir)  # noqa: S202 — all paths verified above
