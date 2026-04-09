"""yt-dlp resolver — streams audio from YouTube, Bandcamp, SoundCloud, and more.

Requires:
    - yt-dlp (``pip install yt-dlp``, included in simil core deps)
    - ffmpeg in PATH (for WAV conversion; optional — see note below)

Notes:
    Without ffmpeg, yt-dlp downloads in the native container format (m4a,
    webm, opus, etc.) and librosa attempts decoding via ``audioread``.
    For reliable audio reading across all sources, installing ffmpeg is
    strongly recommended.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path

from simil.core.exceptions import ResolverError
from simil.resolvers.base import BaseResolver, ResolvedAudio

logger = logging.getLogger(__name__)

_HTTP_SCHEMES = ("http://", "https://")


class YtDlpResolver(BaseResolver):
    """Downloads audio from any URL supported by yt-dlp.

    The downloaded file lives in a temporary directory that is automatically
    removed when the returned :class:`~simil.resolvers.base.ResolvedAudio`
    context manager exits.

    For playlist URLs, only the first item is downloaded.
    """

    @property
    def name(self) -> str:
        return "ytdlp"

    def can_handle(self, source: str) -> bool:
        """Return ``True`` for any ``http://`` or ``https://`` URL."""
        return any(source.startswith(s) for s in _HTTP_SCHEMES)

    def resolve(self, source: str) -> ResolvedAudio:
        """Download audio from *source* and return a
        :class:`~simil.resolvers.base.ResolvedAudio` backed by a temp dir.

        Args:
            source: Any yt-dlp-compatible URL.

        Raises:
            ResolverError: If yt-dlp is not installed or the download fails.
        """
        try:
            import yt_dlp  # noqa: PLC0415
        except ImportError as exc:
            raise ResolverError(
                "yt-dlp is not installed. Run: pip install yt-dlp"
            ) from exc

        tmp_dir = Path(tempfile.mkdtemp(prefix="simil_ytdlp_"))
        try:
            info = _run_download(yt_dlp, source, tmp_dir)
        except ResolverError:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise
        except Exception as exc:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise ResolverError(
                f"yt-dlp download failed for {source!r}: {exc}"
            ) from exc

        audio_files = [f for f in tmp_dir.iterdir() if f.is_file()]
        if not audio_files:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise ResolverError(f"yt-dlp produced no output file for {source!r}")

        # Pick the largest file (avoids thumbnail images and .json sidecar files)
        audio_path = max(audio_files, key=lambda f: f.stat().st_size)
        logger.info("yt-dlp resolved %r → %s", source, audio_path.name)

        info = info or {}
        title: str | None = info.get("title") or info.get("fulltitle")
        artist: str | None = info.get("artist") or info.get("uploader")

        return ResolvedAudio(
            path=audio_path,
            origin=source,
            title=title,
            artist=artist,
            _tmp_dir=tmp_dir,
        )


def _run_download(yt_dlp: object, url: str, dest_dir: Path) -> dict | None:
    """Invoke yt-dlp and return the info dict.

    Attempts WAV conversion first (requires ffmpeg); falls back to raw audio
    container format if ffmpeg is unavailable.

    Args:
        yt_dlp: The imported ``yt_dlp`` module.
        url: URL to download.
        dest_dir: Directory to write the output file into.

    Returns:
        yt-dlp info dict, or ``None`` if metadata extraction failed.
    """
    base_opts: dict = {
        "format": "bestaudio/best",
        "outtmpl": str(dest_dir / "audio.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
        "playlistend": 1,  # first item only for playlists
    }

    # Prefer WAV for maximum librosa compatibility (requires ffmpeg)
    wav_opts = {
        **base_opts,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "0",
            }
        ],
    }

    with yt_dlp.YoutubeDL(wav_opts) as ydl:  # type: ignore[attr-defined]
        try:
            return ydl.extract_info(url, download=True)
        except Exception:
            pass  # fall through if ffmpeg is missing or conversion fails

    # Fallback: native container format, let librosa handle decoding
    with yt_dlp.YoutubeDL(base_opts) as ydl:  # type: ignore[attr-defined]
        return ydl.extract_info(url, download=True)
