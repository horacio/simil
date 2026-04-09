"""Unit tests for the resolver layer.

All tests are fully mocked — no network access required.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from simil.core.exceptions import (
    ResolverError,
    SpotifyPreviewUnavailableError,
    UnsupportedURLError,
)
from simil.resolvers import ResolverChain
from simil.resolvers.base import ResolvedAudio
from simil.resolvers.local import LocalResolver
from simil.resolvers.spotify import SpotifyResolver
from simil.resolvers.ytdlp import YtDlpResolver


# ── ResolvedAudio context manager ─────────────────────────────────────────────


def test_resolved_audio_context_manager_cleans_up(tmp_path: Path) -> None:
    """Context manager removes _tmp_dir on exit."""
    tmp_dir = tmp_path / "tmp"
    tmp_dir.mkdir()
    (tmp_dir / "audio.wav").write_bytes(b"")

    resolved = ResolvedAudio(path=tmp_dir / "audio.wav", origin="test", _tmp_dir=tmp_dir)
    with resolved:
        assert tmp_dir.exists()
    assert not tmp_dir.exists()


def test_resolved_audio_no_tmp_dir_leaves_file(tmp_path: Path) -> None:
    """ResolvedAudio without _tmp_dir does not delete the path on exit."""
    p = tmp_path / "audio.wav"
    p.write_bytes(b"")
    resolved = ResolvedAudio(path=p, origin="test")
    with resolved:
        pass
    assert p.exists()


def test_resolved_audio_ignores_missing_tmp_dir(tmp_path: Path) -> None:
    """No error if _tmp_dir was already removed before __exit__."""
    tmp_dir = tmp_path / "gone"  # never created
    p = tmp_path / "audio.wav"
    p.write_bytes(b"")
    resolved = ResolvedAudio(path=p, origin="test", _tmp_dir=tmp_dir)
    with resolved:  # should not raise
        pass


# ── LocalResolver ─────────────────────────────────────────────────────────────


def test_local_can_handle_existing_file(tmp_path: Path) -> None:
    p = tmp_path / "audio.wav"
    p.write_bytes(b"")
    assert LocalResolver().can_handle(str(p))


def test_local_can_handle_file_url(tmp_path: Path) -> None:
    p = tmp_path / "audio.wav"
    p.write_bytes(b"")
    assert LocalResolver().can_handle(f"file://{p}")


def test_local_cannot_handle_missing_path() -> None:
    assert not LocalResolver().can_handle("/absolutely/does/not/exist.wav")


def test_local_cannot_handle_http_url() -> None:
    assert not LocalResolver().can_handle("https://example.com/audio.mp3")


def test_local_resolve_existing_file(tmp_path: Path) -> None:
    p = tmp_path / "audio.wav"
    p.write_bytes(b"")
    resolved = LocalResolver().resolve(str(p))
    assert resolved.path == p.resolve()
    assert resolved._tmp_dir is None


def test_local_resolve_file_url(tmp_path: Path) -> None:
    p = tmp_path / "audio.wav"
    p.write_bytes(b"")
    resolved = LocalResolver().resolve(f"file://{p}")
    assert resolved.path.exists()


def test_local_resolve_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(ResolverError, match="not found"):
        LocalResolver().resolve(str(tmp_path / "missing.wav"))


def test_local_resolve_directory_raises(tmp_path: Path) -> None:
    with pytest.raises(ResolverError, match="not a regular file"):
        LocalResolver().resolve(str(tmp_path))


# ── YtDlpResolver ─────────────────────────────────────────────────────────────


def test_ytdlp_can_handle_http() -> None:
    assert YtDlpResolver().can_handle("http://example.com/audio.mp3")


def test_ytdlp_can_handle_https() -> None:
    assert YtDlpResolver().can_handle("https://www.youtube.com/watch?v=dQw4w9WgXcQ")


def test_ytdlp_cannot_handle_ftp() -> None:
    assert not YtDlpResolver().can_handle("ftp://example.com/file.wav")


def test_ytdlp_cannot_handle_local_path() -> None:
    assert not YtDlpResolver().can_handle("/some/local/file.wav")


def test_ytdlp_cannot_handle_spotify() -> None:
    assert not YtDlpResolver().can_handle("spotify:track:abc123")


def test_ytdlp_raises_if_yt_dlp_missing() -> None:
    """ResolverError with install hint when yt-dlp is not importable."""
    with patch.dict("sys.modules", {"yt_dlp": None}):
        with pytest.raises(ResolverError, match="yt-dlp"):
            YtDlpResolver().resolve("https://example.com/audio.mp3")


def test_ytdlp_resolve_mocked(tmp_path: Path) -> None:
    """YtDlpResolver.resolve() returns ResolvedAudio when _run_download succeeds."""
    audio_file = tmp_path / "audio.wav"
    audio_file.write_bytes(b"fake audio" * 200)  # non-empty file

    with patch("simil.resolvers.ytdlp._run_download") as mock_run:
        mock_run.return_value = {"title": "Test Track", "artist": "Test Artist"}
        with patch("simil.resolvers.ytdlp.tempfile.mkdtemp", return_value=str(tmp_path)):
            with patch.dict("sys.modules", {"yt_dlp": MagicMock()}):
                resolved = YtDlpResolver().resolve("https://youtube.com/watch?v=test")

    assert resolved.path == audio_file
    assert resolved.title == "Test Track"
    assert resolved.artist == "Test Artist"
    assert resolved._tmp_dir == tmp_path


def test_ytdlp_resolve_no_output_file_raises(tmp_path: Path) -> None:
    """ResolverError when yt-dlp produces no output file."""
    # tmp_path is empty — no audio file created

    with patch("simil.resolvers.ytdlp._run_download", return_value=None):
        with patch("simil.resolvers.ytdlp.tempfile.mkdtemp", return_value=str(tmp_path)):
            with patch.dict("sys.modules", {"yt_dlp": MagicMock()}):
                with pytest.raises(ResolverError, match="no output file"):
                    YtDlpResolver().resolve("https://youtube.com/watch?v=test")


def test_ytdlp_resolve_cleans_up_on_error(tmp_path: Path) -> None:
    """tmp_dir is removed when _run_download raises."""
    tmp_dir = tmp_path / "ytdlp_work"
    tmp_dir.mkdir()

    with patch(
        "simil.resolvers.ytdlp._run_download", side_effect=RuntimeError("boom")
    ):
        with patch("simil.resolvers.ytdlp.tempfile.mkdtemp", return_value=str(tmp_dir)):
            with patch.dict("sys.modules", {"yt_dlp": MagicMock()}):
                with pytest.raises(ResolverError):
                    YtDlpResolver().resolve("https://youtube.com/watch?v=test")

    assert not tmp_dir.exists()


# ── SpotifyResolver ───────────────────────────────────────────────────────────


def test_spotify_can_handle_track_url() -> None:
    assert SpotifyResolver().can_handle(
        "https://open.spotify.com/track/4iV5W9uYEdYUVa79Axb7Rh"
    )


def test_spotify_can_handle_track_url_with_query() -> None:
    """Query params (e.g. ?si=...) should still match."""
    assert SpotifyResolver().can_handle(
        "https://open.spotify.com/track/4iV5W9uYEdYUVa79Axb7Rh?si=abc"
    )


def test_spotify_cannot_handle_playlist() -> None:
    assert not SpotifyResolver().can_handle(
        "https://open.spotify.com/playlist/37i9dQZF1DX4sWSpwq3LiO"
    )


def test_spotify_cannot_handle_youtube() -> None:
    assert not SpotifyResolver().can_handle("https://www.youtube.com/watch?v=abc")


def test_spotify_cannot_handle_bare_spotify_scheme() -> None:
    assert not SpotifyResolver().can_handle("spotify:track:4iV5W9uYEdYUVa79Axb7Rh")


def test_spotify_resolve_raises_without_credentials() -> None:
    """ResolverError when no client credentials are configured."""
    resolver = SpotifyResolver(client_id="", client_secret="")
    with pytest.raises(ResolverError, match="credentials"):
        resolver.resolve("https://open.spotify.com/track/abc123")


def test_spotify_resolve_raises_no_preview() -> None:
    """SpotifyPreviewUnavailableError when preview_url is null."""
    resolver = SpotifyResolver(client_id="id", client_secret="secret")
    with patch.object(resolver, "_get_token", return_value="tok"):
        with patch.object(
            resolver,
            "_get_track",
            return_value={"name": "Test", "artists": [], "preview_url": None},
        ):
            with pytest.raises(SpotifyPreviewUnavailableError):
                resolver.resolve("https://open.spotify.com/track/abc123")


def test_spotify_resolve_success(tmp_path: Path) -> None:
    """Full happy-path: token + track + preview download → ResolvedAudio."""
    resolver = SpotifyResolver(client_id="id", client_secret="secret")

    mock_resp = MagicMock()
    mock_resp.content = b"ID3" + b"\x00" * 200
    mock_resp.raise_for_status = MagicMock()

    with patch.object(resolver, "_get_token", return_value="tok"):
        with patch.object(
            resolver,
            "_get_track",
            return_value={
                "name": "Test Track",
                "artists": [{"name": "Test Artist"}],
                "preview_url": "https://preview.spotify.com/track/abc.mp3",
            },
        ):
            with patch("simil.resolvers.spotify.tempfile.mkdtemp", return_value=str(tmp_path)):
                with patch("simil.resolvers.spotify.httpx.get", return_value=mock_resp):
                    resolved = resolver.resolve(
                        "https://open.spotify.com/track/abc123"
                    )

    assert resolved.path.exists()
    assert resolved.path.name == "preview.mp3"
    assert resolved.title == "Test Track"
    assert resolved.artist == "Test Artist"
    assert resolved._tmp_dir == tmp_path


def test_spotify_resolve_cleans_up_on_download_failure(tmp_path: Path) -> None:
    """tmp_dir is removed when the preview download fails."""
    tmp_dir = tmp_path / "spotify_work"
    tmp_dir.mkdir()

    resolver = SpotifyResolver(client_id="id", client_secret="secret")

    with patch.object(resolver, "_get_token", return_value="tok"):
        with patch.object(
            resolver,
            "_get_track",
            return_value={
                "name": "X",
                "artists": [],
                "preview_url": "https://preview.spotify.com/x.mp3",
            },
        ):
            with patch(
                "simil.resolvers.spotify.tempfile.mkdtemp", return_value=str(tmp_dir)
            ):
                with patch(
                    "simil.resolvers.spotify.httpx.get",
                    side_effect=RuntimeError("network error"),
                ):
                    with pytest.raises(ResolverError):
                        resolver.resolve("https://open.spotify.com/track/abc123")

    assert not tmp_dir.exists()


def test_spotify_token_caching() -> None:
    """Token is reused within its expiry window."""
    import httpx as _httpx

    resolver = SpotifyResolver(client_id="id", client_secret="secret")

    mock_resp = MagicMock()
    mock_resp.json.return_value = {"access_token": "tok123", "expires_in": 3600}
    mock_resp.raise_for_status = MagicMock()

    with patch("simil.resolvers.spotify.httpx.post", return_value=mock_resp) as mock_post:
        t1 = resolver._get_token()
        t2 = resolver._get_token()  # should use cached token

    assert t1 == t2 == "tok123"
    assert mock_post.call_count == 1  # only one network call


# ── ResolverChain ─────────────────────────────────────────────────────────────


def test_chain_routes_local_file(tmp_path: Path) -> None:
    p = tmp_path / "audio.wav"
    p.write_bytes(b"")
    chain = ResolverChain()
    resolved = chain.resolve(str(p))
    assert resolved.path == p.resolve()


def test_chain_raises_for_unsupported_scheme() -> None:
    """Scheme not handled by any resolver raises UnsupportedURLError."""
    chain = ResolverChain()
    with pytest.raises(UnsupportedURLError):
        chain.resolve("ftp://example.com/audio.mp3")


def test_chain_raises_for_unknown_scheme() -> None:
    """Custom or unknown schemes raise UnsupportedURLError."""
    chain = ResolverChain()
    with pytest.raises(UnsupportedURLError):
        chain.resolve("myapp://something/audio")


def test_chain_skips_non_matching_resolver(tmp_path: Path) -> None:
    """Resolvers that return can_handle=False are skipped."""
    p = tmp_path / "audio.wav"
    p.write_bytes(b"")

    mock_resolver = MagicMock()
    mock_resolver.can_handle.return_value = False

    chain = ResolverChain(resolvers=[mock_resolver, LocalResolver()])
    resolved = chain.resolve(str(p))
    assert resolved.path.exists()
    mock_resolver.resolve.assert_not_called()


def test_chain_spotify_before_ytdlp() -> None:
    """Spotify URLs are handled by SpotifyResolver, not YtDlpResolver."""
    spotify_url = "https://open.spotify.com/track/abc123"

    mock_spotify = MagicMock(spec=SpotifyResolver)
    mock_spotify.can_handle.return_value = True
    mock_spotify.resolve.return_value = ResolvedAudio(
        path=Path("/tmp/preview.mp3"), origin=spotify_url
    )

    mock_ytdlp = MagicMock(spec=YtDlpResolver)
    mock_ytdlp.can_handle.return_value = True  # would also match https

    chain = ResolverChain(resolvers=[LocalResolver(), mock_spotify, mock_ytdlp])
    result = chain.resolve(spotify_url)

    mock_spotify.resolve.assert_called_once()
    mock_ytdlp.resolve.assert_not_called()
    assert result.origin == spotify_url


def test_chain_custom_resolvers() -> None:
    """ResolverChain accepts a custom resolver list."""
    mock = MagicMock()
    mock.can_handle.return_value = True
    mock.resolve.return_value = ResolvedAudio(
        path=Path("/tmp/audio.wav"), origin="custom://source"
    )

    chain = ResolverChain(resolvers=[mock])
    result = chain.resolve("custom://source")
    assert result.origin == "custom://source"


def test_chain_default_resolvers_utility() -> None:
    """default_resolvers() returns LocalResolver, SpotifyResolver, YtDlpResolver."""
    resolvers = ResolverChain.default_resolvers()
    names = [r.name for r in resolvers]
    assert names == ["local", "spotify", "ytdlp"]
