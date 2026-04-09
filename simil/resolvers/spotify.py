"""Spotify resolver — downloads 30-second preview clips via the Spotify Web API.

Authentication:
    Requires OAuth2 client credentials.  Set environment variables:

    .. code-block:: bash

        export SIMIL_SPOTIFY_CLIENT_ID="your-client-id"
        export SIMIL_SPOTIFY_CLIENT_SECRET="your-client-secret"

    Register an application at https://developer.spotify.com/dashboard.

Limitations:
    - Only Spotify *track* URLs are supported (not albums, playlists, or artists).
    - Preview clips are 30 seconds.  Some tracks have no preview — these raise
      :exc:`~simil.core.exceptions.SpotifyPreviewUnavailableError`.
"""

from __future__ import annotations

import base64
import logging
import os
import re
import shutil
import tempfile
import time
from pathlib import Path

import httpx

from simil.core.exceptions import (
    ResolverError,
    SpotifyPreviewUnavailableError,
    UnsupportedURLError,
)
from simil.resolvers.base import BaseResolver, ResolvedAudio

logger = logging.getLogger(__name__)

_TRACK_URL_RE = re.compile(r"https?://open\.spotify\.com/track/([A-Za-z0-9]+)")
_TOKEN_URL = "https://accounts.spotify.com/api/token"
_API_BASE = "https://api.spotify.com/v1"


class SpotifyResolver(BaseResolver):
    """Downloads Spotify track preview clips (30 s MP3) via the Web API.

    Args:
        client_id: Spotify application client ID.
            Falls back to the ``SIMIL_SPOTIFY_CLIENT_ID`` environment variable.
        client_secret: Spotify application client secret.
            Falls back to the ``SIMIL_SPOTIFY_CLIENT_SECRET`` environment variable.
    """

    def __init__(
        self,
        client_id: str | None = None,
        client_secret: str | None = None,
    ) -> None:
        self._client_id = client_id or os.environ.get("SIMIL_SPOTIFY_CLIENT_ID") or ""
        self._client_secret = (
            client_secret or os.environ.get("SIMIL_SPOTIFY_CLIENT_SECRET") or ""
        )
        self._token: str = ""
        self._token_expires_at: float = 0.0

    @property
    def name(self) -> str:
        return "spotify"

    def can_handle(self, source: str) -> bool:
        """Return ``True`` only for Spotify *track* URLs."""
        return bool(_TRACK_URL_RE.match(source))

    def resolve(self, source: str) -> ResolvedAudio:
        """Download the Spotify preview for *source*.

        Args:
            source: A Spotify track URL, e.g.
                ``"https://open.spotify.com/track/4iV5W9uYEdYUVa79Axb7Rh"``.

        Returns:
            :class:`~simil.resolvers.base.ResolvedAudio` pointing at the
            downloaded ``preview.mp3`` in a temporary directory.

        Raises:
            UnsupportedURLError: If *source* is not a Spotify track URL.
            ResolverError: If credentials are missing or the API call fails.
            SpotifyPreviewUnavailableError: If the track has no 30-s preview.
        """
        m = _TRACK_URL_RE.match(source)
        if not m:
            raise UnsupportedURLError(f"Not a Spotify track URL: {source!r}")
        track_id = m.group(1)

        if not self._client_id or not self._client_secret:
            raise ResolverError(
                "Spotify credentials are required. "
                "Set SIMIL_SPOTIFY_CLIENT_ID and SIMIL_SPOTIFY_CLIENT_SECRET."
            )

        token = self._get_token()
        track = self._get_track(track_id, token)

        preview_url: str | None = track.get("preview_url")
        if not preview_url:
            raise SpotifyPreviewUnavailableError(
                f"No 30-s preview available for Spotify track {track_id!r}. "
                "This is common for tracks in certain markets or by some labels."
            )

        title: str | None = track.get("name")
        artists: list[dict] = track.get("artists") or []
        artist: str | None = ", ".join(a["name"] for a in artists) or None

        tmp_dir = Path(tempfile.mkdtemp(prefix="simil_spotify_"))
        try:
            audio_path = _download_preview(preview_url, tmp_dir)
        except Exception as exc:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise ResolverError(
                f"Failed to download Spotify preview for {track_id!r}: {exc}"
            ) from exc

        logger.info("Spotify preview downloaded: %s – %s", artist, title)
        return ResolvedAudio(
            path=audio_path,
            origin=source,
            title=title,
            artist=artist,
            _tmp_dir=tmp_dir,
        )

    # ── Internal ──────────────────────────────────────────────────────────────

    def _get_token(self) -> str:
        """Return a valid access token, refreshing if expired."""
        now = time.monotonic()
        if self._token and now < self._token_expires_at:
            return self._token

        credentials = base64.b64encode(
            f"{self._client_id}:{self._client_secret}".encode()
        ).decode()

        resp = httpx.post(
            _TOKEN_URL,
            headers={"Authorization": f"Basic {credentials}"},
            data={"grant_type": "client_credentials"},
            timeout=10.0,
        )
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise ResolverError(
                f"Spotify token request failed: {exc.response.status_code}"
            ) from exc

        data = resp.json()
        self._token = data["access_token"]
        self._token_expires_at = now + data["expires_in"] - 30  # 30 s safety margin
        return self._token

    def _get_track(self, track_id: str, token: str) -> dict:
        """Fetch track metadata from the Spotify Web API."""
        resp = httpx.get(
            f"{_API_BASE}/tracks/{track_id}",
            headers={"Authorization": f"Bearer {token}"},
            timeout=10.0,
        )
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise ResolverError(
                f"Spotify API request failed for track {track_id!r}: "
                f"{exc.response.status_code}"
            ) from exc
        return resp.json()


def _download_preview(preview_url: str, dest_dir: Path) -> Path:
    """Download *preview_url* to ``dest_dir/preview.mp3``.

    Args:
        preview_url: Direct MP3 URL from the Spotify track object.
        dest_dir: Destination directory.

    Returns:
        Path to the downloaded file.

    Raises:
        ResolverError: If the HTTP request fails.
    """
    resp = httpx.get(preview_url, timeout=30.0, follow_redirects=True)
    try:
        resp.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise ResolverError(
            f"Preview download failed ({exc.response.status_code}): {preview_url}"
        ) from exc
    audio_path = dest_dir / "preview.mp3"
    audio_path.write_bytes(resp.content)
    return audio_path
