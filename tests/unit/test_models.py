"""Unit tests for simil.core.models."""

from __future__ import annotations

from pathlib import Path

import pytest

from simil.core.models import AudioConfig, IndexerResult, SearchResult, Track


class TestTrack:
    """Tests for the Track dataclass."""

    def test_basic_construction(self) -> None:
        """Track can be constructed with minimal required fields."""
        t = Track(id="abc123", path=Path("/music/song.mp3"))
        assert t.id == "abc123"
        assert t.path == Path("/music/song.mp3")
        assert t.title is None
        assert t.artist is None
        assert t.album is None
        assert t.duration_seconds is None
        assert t.mtime is None
        assert t.extra == {}

    def test_full_construction(self) -> None:
        """Track can be constructed with all fields."""
        t = Track(
            id="deadbeef",
            path=Path("/music/artist/album/song.flac"),
            title="Song",
            artist="Artist",
            album="Album",
            duration_seconds=240.5,
            mtime=1700000000.0,
            extra={"genre": "jazz"},
        )
        assert t.title == "Song"
        assert t.artist == "Artist"
        assert t.album == "Album"
        assert t.duration_seconds == pytest.approx(240.5)
        assert t.mtime == pytest.approx(1700000000.0)
        assert t.extra == {"genre": "jazz"}

    def test_as_dict_posix_path(self) -> None:
        """as_dict() stores path as posix string."""
        t = Track(id="abc", path=Path("/some/path/file.mp3"))
        d = t.as_dict()
        assert d["path"] == "/some/path/file.mp3"
        assert isinstance(d["path"], str)

    def test_round_trip(self) -> None:
        """Track → as_dict() → from_dict() round-trips correctly."""
        t = Track(
            id="xyz789",
            path=Path("/a/b/c.wav"),
            title="T",
            artist="A",
            album="Al",
            duration_seconds=60.0,
            mtime=12345.0,
            extra={"bpm": 120},
        )
        d = t.as_dict()
        t2 = Track.from_dict(d)
        assert t2.id == t.id
        assert t2.path == t.path
        assert t2.title == t.title
        assert t2.artist == t.artist
        assert t2.album == t.album
        assert t2.duration_seconds == pytest.approx(t.duration_seconds)
        assert t2.mtime == pytest.approx(t.mtime)
        assert t2.extra == t.extra


class TestSearchResult:
    """Tests for the SearchResult dataclass."""

    def _make_track(self) -> Track:
        return Track(id="t1", path=Path("/music/t1.mp3"), title="T1", artist="A")

    def test_as_dict_structure(self) -> None:
        """as_dict() returns expected keys."""
        sr = SearchResult(
            track=self._make_track(),
            raw_score=0.9,
            score=0.95,
            rank=1,
        )
        d = sr.as_dict()
        assert "rank" in d
        assert "score" in d
        assert "raw_score" in d
        assert "track" in d
        assert d["rank"] == 1
        assert d["score"] == pytest.approx(0.95)
        assert d["raw_score"] == pytest.approx(0.9)

    def test_as_dict_json_safe(self) -> None:
        """as_dict() output can be serialised to JSON."""
        import json

        sr = SearchResult(
            track=self._make_track(),
            raw_score=0.5,
            score=0.75,
            rank=2,
        )
        # Should not raise
        json.dumps(sr.as_dict())


class TestAudioConfig:
    """Tests for the AudioConfig dataclass."""

    def test_defaults(self) -> None:
        """AudioConfig has sensible defaults."""
        ac = AudioConfig()
        assert ac.sample_rate == 22050
        assert ac.n_mels == 0
        assert ac.hop_length == 512
        assert ac.mono is True

    def test_as_dict(self) -> None:
        """as_dict() returns correct keys and values."""
        ac = AudioConfig(sample_rate=16000, n_mels=128, hop_length=256, mono=True)
        d = ac.as_dict()
        assert d["sample_rate"] == 16000
        assert d["n_mels"] == 128
        assert d["hop_length"] == 256
        assert d["mono"] is True

    def test_round_trip(self) -> None:
        """AudioConfig → as_dict() → from_dict() round-trips correctly."""
        ac = AudioConfig(sample_rate=44100, n_mels=64, hop_length=1024, mono=False)
        d = ac.as_dict()
        ac2 = AudioConfig.from_dict(d)
        assert ac2.sample_rate == ac.sample_rate
        assert ac2.n_mels == ac.n_mels
        assert ac2.hop_length == ac.hop_length
        assert ac2.mono == ac.mono


class TestIndexerResult:
    """Tests for the IndexerResult dataclass."""

    def test_construction(self) -> None:
        """IndexerResult stores all fields."""
        r = IndexerResult(
            indexed=10,
            skipped=3,
            failed=[Path("/a/b.mp3")],
            duration_seconds=5.2,
        )
        assert r.indexed == 10
        assert r.skipped == 3
        assert len(r.failed) == 1
        assert r.duration_seconds == pytest.approx(5.2)
