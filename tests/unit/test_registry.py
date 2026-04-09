"""Unit tests for simil.registry.

All network I/O and filesystem operations are mocked — no downloads required.
"""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from simil.registry import (
    ChecksumError,
    IndexEntry,
    RegistryError,
    _safe_extract,
    _verify_checksum,
    download_index,
    fetch_registry,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


def _make_entry(**kwargs) -> IndexEntry:
    defaults = dict(
        name="fma-small",
        description="Test index",
        embedder="effnet-discogs",
        tracks=8000,
        url="https://example.com/fma-small.tar.gz",
        sha256="a" * 64,
        size_bytes=10 * 1024 * 1024,
    )
    defaults.update(kwargs)
    return IndexEntry(**defaults)


def _make_tar_gz(files: dict[str, bytes], dest: Path) -> None:
    """Write a tar.gz containing *files* (name → content) to *dest*."""
    with tarfile.open(dest, "w:gz") as tf:
        for name, content in files.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(content)
            tf.addfile(info, io.BytesIO(content))


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ── IndexEntry ────────────────────────────────────────────────────────────────


def test_index_entry_from_dict_all_fields() -> None:
    d = {
        "name": "fma-small",
        "description": "desc",
        "embedder": "effnet-discogs",
        "tracks": 8000,
        "url": "https://example.com/x.tar.gz",
        "sha256": "a" * 64,
        "size_bytes": 1024,
        "extra_ignored": "value",
    }
    entry = IndexEntry.from_dict(d)
    assert entry.name == "fma-small"
    assert entry.tracks == 8000
    assert entry.size_bytes == 1024


def test_index_entry_size_mb() -> None:
    entry = _make_entry(size_bytes=10 * 1024 * 1024)
    assert abs(entry.size_mb - 10.0) < 0.01


def test_index_entry_from_dict_missing_required_field() -> None:
    d = {"name": "x", "description": "d", "embedder": "e", "tracks": 1}
    # Missing url, sha256, size_bytes
    with pytest.raises(RegistryError):
        IndexEntry.from_dict(d)


def test_index_entry_is_frozen() -> None:
    entry = _make_entry()
    with pytest.raises(Exception):  # FrozenInstanceError
        entry.name = "modified"  # type: ignore[misc]


# ── fetch_registry ────────────────────────────────────────────────────────────


def _mock_httpx_get(payload: dict, status: int = 200):
    """Return a mock that simulates httpx.get returning *payload* JSON."""
    mock_resp = MagicMock()
    mock_resp.status_code = status
    mock_resp.json.return_value = payload
    if status >= 400:
        import httpx
        mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "error", request=MagicMock(), response=mock_resp
        )
    else:
        mock_resp.raise_for_status.return_value = None
    return mock_resp


def test_fetch_registry_returns_entries() -> None:
    payload = {
        "schema_version": 1,
        "indexes": [
            {
                "name": "fma-small",
                "description": "FMA small",
                "embedder": "effnet-discogs",
                "tracks": 8000,
                "url": "https://example.com/fma.tar.gz",
                "sha256": "a" * 64,
                "size_bytes": 1024,
            }
        ],
    }
    with patch("httpx.get", return_value=_mock_httpx_get(payload)):
        entries = fetch_registry("https://example.com/registry.json")

    assert len(entries) == 1
    assert entries[0].name == "fma-small"
    assert entries[0].tracks == 8000


def test_fetch_registry_returns_empty_list() -> None:
    payload = {"schema_version": 1, "indexes": []}
    with patch("httpx.get", return_value=_mock_httpx_get(payload)):
        entries = fetch_registry()
    assert entries == []


def test_fetch_registry_http_error_raises() -> None:
    import httpx

    with patch("httpx.get", side_effect=httpx.ConnectError("refused")):
        with pytest.raises(RegistryError, match="Could not reach registry"):
            fetch_registry()


def test_fetch_registry_404_raises() -> None:
    mock_resp = _mock_httpx_get({}, status=404)
    with patch("httpx.get", return_value=mock_resp):
        with pytest.raises(RegistryError, match="HTTP 404"):
            fetch_registry()


def test_fetch_registry_bad_json_raises() -> None:
    import httpx

    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None
    mock_resp.json.side_effect = json.JSONDecodeError("bad", "", 0)
    with patch("httpx.get", return_value=mock_resp):
        with pytest.raises(RegistryError, match="Malformed"):
            fetch_registry()


def test_fetch_registry_missing_indexes_key_raises() -> None:
    mock_resp = _mock_httpx_get({"schema_version": 1})  # no "indexes" key
    with patch("httpx.get", return_value=mock_resp):
        with pytest.raises(RegistryError, match="indexes"):
            fetch_registry()


# ── _verify_checksum ──────────────────────────────────────────────────────────


def test_verify_checksum_correct(tmp_path: Path) -> None:
    data = b"hello simil"
    f = tmp_path / "file.bin"
    f.write_bytes(data)
    _verify_checksum(f, _sha256_bytes(data))  # should not raise


def test_verify_checksum_mismatch_raises(tmp_path: Path) -> None:
    f = tmp_path / "file.bin"
    f.write_bytes(b"hello")
    with pytest.raises(ChecksumError, match="SHA-256 mismatch"):
        _verify_checksum(f, "wrong" * 12 + "cafe")


# ── _safe_extract ─────────────────────────────────────────────────────────────


def test_safe_extract_creates_files(tmp_path: Path) -> None:
    archive = tmp_path / "test.tar.gz"
    _make_tar_gz({"meta.json": b'{"v":1}', "catalog.json": b'{"tracks":[]}'}, archive)
    dest = tmp_path / "out"
    dest.mkdir()
    _safe_extract(archive, dest)
    assert (dest / "meta.json").exists()
    assert (dest / "catalog.json").exists()


def test_safe_extract_blocks_absolute_path(tmp_path: Path) -> None:
    archive = tmp_path / "bad.tar.gz"
    with tarfile.open(archive, "w:gz") as tf:
        info = tarfile.TarInfo(name="/etc/passwd")
        info.size = 5
        tf.addfile(info, io.BytesIO(b"pwned"))
    dest = tmp_path / "out"
    dest.mkdir()
    with pytest.raises(RegistryError, match="Refusing"):
        _safe_extract(archive, dest)


def test_safe_extract_blocks_path_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "traversal.tar.gz"
    with tarfile.open(archive, "w:gz") as tf:
        info = tarfile.TarInfo(name="../../evil.sh")
        info.size = 5
        tf.addfile(info, io.BytesIO(b"rm -rf"))
    dest = tmp_path / "out"
    dest.mkdir()
    with pytest.raises(RegistryError, match="Refusing"):
        _safe_extract(archive, dest)


# ── download_index ────────────────────────────────────────────────────────────


def _build_test_archive(tmp_path: Path) -> tuple[Path, str]:
    """Build a minimal valid index archive, return (archive_path, sha256)."""
    content = {
        "meta.json": b'{"schema_version":1,"embedder":"effnet-discogs","embedding_dim":1280,"built_at":""}',
        "catalog.json": b'{"schema_version":1,"library_id":"test","built_at":"","tracks":[]}',
        "vectors.npy": b"\x93NUMPY" + b"\x00" * 100,
    }
    archive = tmp_path / "index.tar.gz"
    _make_tar_gz(content, archive)
    return archive, _sha256_bytes(archive.read_bytes())


def test_download_index_full_flow(tmp_path: Path) -> None:
    """Happy path: download → verify → extract."""
    archive_path, sha = _build_test_archive(tmp_path)
    archive_bytes = archive_path.read_bytes()

    entry = _make_entry(sha256=sha, size_bytes=len(archive_bytes))
    dest = tmp_path / "dest"

    # Mock the streaming download to yield our archive bytes
    mock_resp = MagicMock()
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    mock_resp.raise_for_status = MagicMock()
    mock_resp.headers = {"content-length": str(len(archive_bytes))}
    mock_resp.iter_bytes = MagicMock(
        return_value=iter([archive_bytes[i:i+1024] for i in range(0, len(archive_bytes), 1024)])
    )

    mock_stream_cm = MagicMock()
    mock_stream_cm.__enter__ = MagicMock(return_value=mock_resp)
    mock_stream_cm.__exit__ = MagicMock(return_value=False)

    with patch("httpx.stream", return_value=mock_stream_cm):
        download_index(entry, dest)

    assert (dest / "meta.json").exists()
    assert (dest / "catalog.json").exists()


def test_download_index_checksum_mismatch_raises(tmp_path: Path) -> None:
    archive_path, _ = _build_test_archive(tmp_path)
    archive_bytes = archive_path.read_bytes()

    entry = _make_entry(sha256="wrong" * 12 + "dead", size_bytes=len(archive_bytes))
    dest = tmp_path / "dest"

    mock_resp = MagicMock()
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    mock_resp.raise_for_status = MagicMock()
    mock_resp.headers = {}
    mock_resp.iter_bytes = MagicMock(return_value=iter([archive_bytes]))

    mock_stream_cm = MagicMock()
    mock_stream_cm.__enter__ = MagicMock(return_value=mock_resp)
    mock_stream_cm.__exit__ = MagicMock(return_value=False)

    with patch("httpx.stream", return_value=mock_stream_cm):
        with pytest.raises(ChecksumError):
            download_index(entry, dest)


def test_download_index_creates_dest_dir(tmp_path: Path) -> None:
    """dest_dir is created if it does not exist."""
    archive_path, sha = _build_test_archive(tmp_path)
    archive_bytes = archive_path.read_bytes()
    entry = _make_entry(sha256=sha, size_bytes=len(archive_bytes))

    dest = tmp_path / "new" / "library" / "dir"
    assert not dest.exists()

    mock_resp = MagicMock()
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    mock_resp.raise_for_status = MagicMock()
    mock_resp.headers = {}
    mock_resp.iter_bytes = MagicMock(return_value=iter([archive_bytes]))

    mock_stream_cm = MagicMock()
    mock_stream_cm.__enter__ = MagicMock(return_value=mock_resp)
    mock_stream_cm.__exit__ = MagicMock(return_value=False)

    with patch("httpx.stream", return_value=mock_stream_cm):
        download_index(entry, dest)

    assert dest.exists()


# ── CLI fetch command ─────────────────────────────────────────────────────────


def test_cli_fetch_list_shows_table(monkeypatch: pytest.MonkeyPatch) -> None:
    from typer.testing import CliRunner
    from simil.cli.main import app

    monkeypatch.setattr(
        "simil.registry.fetch_registry",
        lambda *a, **kw: [_make_entry()],
    )

    result = CliRunner().invoke(app, ["fetch"])
    assert result.exit_code == 0
    assert "fma-small" in result.output


def test_cli_fetch_empty_registry_shows_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    from typer.testing import CliRunner
    from simil.cli.main import app

    monkeypatch.setattr(
        "simil.registry.fetch_registry",
        lambda *a, **kw: [],
    )

    result = CliRunner().invoke(app, ["fetch"])
    assert result.exit_code == 0
    assert "No pre-built indexes" in result.output


def test_cli_fetch_unknown_name_exits_error(monkeypatch: pytest.MonkeyPatch) -> None:
    from typer.testing import CliRunner
    from simil.cli.main import app

    monkeypatch.setattr(
        "simil.registry.fetch_registry",
        lambda *a, **kw: [_make_entry()],
    )

    result = CliRunner().invoke(app, ["fetch", "does-not-exist"])
    assert result.exit_code != 0


def test_cli_fetch_downloads_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from typer.testing import CliRunner
    from simil.cli.main import app

    entry = _make_entry()
    monkeypatch.setattr("simil.registry.fetch_registry", lambda *a, **kw: [entry])
    monkeypatch.setattr("simil.cli.main._get_index_dir", lambda lib: tmp_path / lib)

    called_with = {}

    def mock_download(e, dest):
        called_with["entry"] = e
        called_with["dest"] = dest
        # Simulate successful extraction
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "meta.json").write_text('{"embedder":"effnet-discogs","tracks":8000}')

    monkeypatch.setattr("simil.registry.download_index", mock_download)

    result = CliRunner().invoke(app, ["fetch", "fma-small", "--yes"])
    assert result.exit_code == 0, result.output
    assert called_with["entry"].name == "fma-small"


def test_cli_fetch_prompts_on_existing_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from typer.testing import CliRunner
    from simil.cli.main import app

    index_dir = tmp_path / "fma-small"
    index_dir.mkdir()
    (index_dir / "meta.json").write_text("{}")

    monkeypatch.setattr("simil.registry.fetch_registry", lambda *a, **kw: [_make_entry()])
    monkeypatch.setattr("simil.cli.main._get_index_dir", lambda lib: index_dir)
    monkeypatch.setattr("simil.registry.download_index", lambda *a, **kw: None)

    # Answer "n" to the confirmation prompt — should exit 0 without downloading
    result = CliRunner().invoke(app, ["fetch", "fma-small"], input="n\n")
    assert result.exit_code == 0


def test_cli_fetch_yes_flag_skips_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from typer.testing import CliRunner
    from simil.cli.main import app

    index_dir = tmp_path / "fma-small"
    index_dir.mkdir()
    (index_dir / "meta.json").write_text("{}")

    downloaded = []
    monkeypatch.setattr("simil.registry.fetch_registry", lambda *a, **kw: [_make_entry()])
    monkeypatch.setattr("simil.cli.main._get_index_dir", lambda lib: index_dir)
    monkeypatch.setattr(
        "simil.registry.download_index",
        lambda e, d: downloaded.append(e.name),
    )

    result = CliRunner().invoke(app, ["fetch", "fma-small", "--yes"])
    assert result.exit_code == 0
    assert "fma-small" in downloaded


def test_cli_fetch_registry_error_aborts(monkeypatch: pytest.MonkeyPatch) -> None:
    from typer.testing import CliRunner
    from simil.cli.main import app
    from simil.registry import RegistryError

    monkeypatch.setattr(
        "simil.registry.fetch_registry",
        lambda *a, **kw: (_ for _ in ()).throw(RegistryError("network down")),
    )

    result = CliRunner().invoke(app, ["fetch"])
    assert result.exit_code != 0
