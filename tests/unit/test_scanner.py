"""Unit tests for simil.library.scanner."""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest

from simil.library.scanner import AUDIO_EXTENSIONS, content_id, scan_library


def _write_wav(path: Path, freq: float = 440.0, duration: float = 0.5, sr: int = 22050) -> None:
    """Write a minimal WAV file."""
    n = int(sr * duration)
    t = np.linspace(0, duration, n, endpoint=False)
    samples = (np.sin(2 * np.pi * freq * t) * 32767).astype(np.int16)
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(samples.tobytes())


class TestScanLibrary:
    """Tests for scan_library()."""

    def test_finds_wav_files(self, tmp_path: Path) -> None:
        """scan_library finds .wav files."""
        f = tmp_path / "song.wav"
        _write_wav(f)
        result = scan_library(tmp_path)
        assert f in result

    def test_finds_multiple_extensions(self, tmp_path: Path) -> None:
        """scan_library finds .mp3 and .flac stubs."""
        (tmp_path / "a.mp3").write_bytes(b"fake mp3")
        (tmp_path / "b.flac").write_bytes(b"fake flac")
        result = scan_library(tmp_path)
        names = {p.name for p in result}
        assert "a.mp3" in names
        assert "b.flac" in names

    def test_ignores_non_audio(self, tmp_path: Path) -> None:
        """scan_library ignores non-audio files."""
        (tmp_path / "cover.jpg").write_bytes(b"fake image")
        (tmp_path / "info.txt").write_text("info")
        (tmp_path / "song.wav").write_bytes(b"fake wav")
        result = scan_library(tmp_path)
        names = {p.name for p in result}
        assert "cover.jpg" not in names
        assert "info.txt" not in names

    def test_recursive_scan(self, tmp_path: Path) -> None:
        """scan_library finds files in subdirectories."""
        sub = tmp_path / "artist" / "album"
        sub.mkdir(parents=True)
        f = sub / "track.wav"
        _write_wav(f)
        result = scan_library(tmp_path)
        assert f in result

    def test_respects_max_depth(self, tmp_path: Path) -> None:
        """scan_library does not recurse beyond max_depth."""
        deep = tmp_path
        for i in range(5):
            deep = deep / f"level{i}"
        deep.mkdir(parents=True)
        (deep / "song.wav").write_bytes(b"fake")

        # max_depth=2 should NOT find the file 5 levels deep
        result = scan_library(tmp_path, max_depth=2)
        assert not any("song.wav" == p.name for p in result)

        # max_depth=10 SHOULD find it
        result2 = scan_library(tmp_path, max_depth=10)
        assert any("song.wav" == p.name for p in result2)

    def test_case_insensitive_extension(self, tmp_path: Path) -> None:
        """scan_library matches extensions case-insensitively."""
        (tmp_path / "SONG.MP3").write_bytes(b"fake")
        (tmp_path / "track.WAV").write_bytes(b"fake")
        result = scan_library(tmp_path)
        names = {p.name for p in result}
        assert "SONG.MP3" in names
        assert "track.WAV" in names

    def test_returns_sorted(self, tmp_path: Path) -> None:
        """scan_library returns paths in sorted order."""
        for name in ["z.wav", "a.mp3", "m.flac"]:
            (tmp_path / name).write_bytes(b"fake")
        result = scan_library(tmp_path)
        assert result == sorted(result)


class TestContentId:
    """Tests for content_id()."""

    def test_deterministic(self, tmp_path: Path) -> None:
        """content_id returns the same value for the same file."""
        f = tmp_path / "test.wav"
        _write_wav(f)
        assert content_id(f) == content_id(f)

    def test_differs_for_different_files(self, tmp_path: Path) -> None:
        """content_id differs for files with different content."""
        f1 = tmp_path / "a.wav"
        f2 = tmp_path / "b.wav"
        _write_wav(f1, freq=440.0)
        _write_wav(f2, freq=880.0)
        assert content_id(f1) != content_id(f2)

    def test_stable_across_rename(self, tmp_path: Path) -> None:
        """content_id is stable when a file is renamed."""
        f = tmp_path / "original.wav"
        _write_wav(f)
        cid_before = content_id(f)
        renamed = tmp_path / "renamed.wav"
        f.rename(renamed)
        cid_after = content_id(renamed)
        assert cid_before == cid_after

    def test_returns_24_hex_chars(self, tmp_path: Path) -> None:
        """content_id returns exactly 24 hex characters."""
        f = tmp_path / "x.wav"
        _write_wav(f)
        cid = content_id(f)
        assert len(cid) == 24
        assert all(c in "0123456789abcdef" for c in cid)

    def test_missing_file_raises(self) -> None:
        """content_id raises LibraryError for a non-existent file."""
        from simil.core.exceptions import LibraryError

        with pytest.raises(LibraryError):
            content_id(Path("/no/such/file.wav"))
