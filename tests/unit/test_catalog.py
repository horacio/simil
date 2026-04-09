"""Unit tests for simil.catalog.TrackCatalog."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from simil.catalog import TrackCatalog
from simil.core.exceptions import CatalogSchemaError
from simil.core.models import Track


def _make_track(tid: str, path_str: str = "/music/song.mp3") -> Track:
    return Track(id=tid, path=Path(path_str), title=f"Title {tid}", mtime=1234.0)


class TestCatalogMutation:
    """Tests for add / get / contains / remove."""

    def test_add_and_get(self) -> None:
        """Added track can be retrieved by ID."""
        cat = TrackCatalog(library_id="lib1")
        t = _make_track("abc")
        cat.add(t)
        retrieved = cat.get("abc")
        assert retrieved is not None
        assert retrieved.id == "abc"

    def test_contains(self) -> None:
        """contains() returns True after add, False before."""
        cat = TrackCatalog()
        assert not cat.contains("x")
        cat.add(_make_track("x"))
        assert cat.contains("x")

    def test_size(self) -> None:
        """size property reflects the number of tracks."""
        cat = TrackCatalog()
        assert cat.size == 0
        cat.add(_make_track("a"))
        cat.add(_make_track("b"))
        assert cat.size == 2

    def test_remove(self) -> None:
        """Removed track is no longer accessible."""
        cat = TrackCatalog()
        cat.add(_make_track("r"))
        cat.remove("r")
        assert not cat.contains("r")
        assert cat.get("r") is None
        assert cat.size == 0

    def test_all_ids(self) -> None:
        """all_ids() returns list of all IDs."""
        cat = TrackCatalog()
        cat.add(_make_track("p"))
        cat.add(_make_track("q"))
        ids = cat.all_ids()
        assert set(ids) == {"p", "q"}

    def test_all_tracks(self) -> None:
        """all_tracks() returns list of Track objects."""
        cat = TrackCatalog()
        cat.add(_make_track("m"))
        tracks = cat.all_tracks()
        assert len(tracks) == 1
        assert tracks[0].id == "m"

    def test_add_updates_existing(self) -> None:
        """Adding a track with the same ID replaces the existing entry."""
        cat = TrackCatalog()
        cat.add(_make_track("id1"))
        updated = Track(id="id1", path=Path("/new/path.mp3"), title="New Title")
        cat.add(updated)
        assert cat.size == 1
        assert cat.get("id1").title == "New Title"


class TestCatalogPersistence:
    """Tests for save() / load() round-trip."""

    def test_save_and_load(self, tmp_path: Path) -> None:
        """save() + load() round-trips track data."""
        cat = TrackCatalog(library_id="lib_xyz")
        cat.add(Track(id="t1", path=Path("/a/b.mp3"), title="T1", mtime=999.0))
        cat.add(Track(id="t2", path=Path("/c/d.flac"), artist="A2", mtime=1000.0))

        catalog_path = tmp_path / "catalog.json"
        cat.save(catalog_path)

        cat2 = TrackCatalog.load(catalog_path, library_id="lib_xyz")
        assert cat2.size == 2
        assert cat2.contains("t1")
        assert cat2.contains("t2")

    def test_path_stored_as_posix(self, tmp_path: Path) -> None:
        """Paths in catalog.json are stored as posix strings."""
        cat = TrackCatalog(library_id="lib1")
        cat.add(Track(id="t1", path=Path("/a/b/c.mp3")))
        cat.save(tmp_path / "catalog.json")

        raw = json.loads((tmp_path / "catalog.json").read_text())
        stored_path = raw["tracks"][0]["path"]
        assert stored_path == "/a/b/c.mp3"
        assert "\\" not in stored_path

    def test_load_reconstructs_path(self, tmp_path: Path) -> None:
        """Loaded Track.path is a Path object."""
        cat = TrackCatalog(library_id="lib1")
        cat.add(Track(id="t1", path=Path("/x/y.wav")))
        cat.save(tmp_path / "catalog.json")

        cat2 = TrackCatalog.load(tmp_path / "catalog.json")
        t = cat2.get("t1")
        assert isinstance(t.path, Path)

    def test_schema_version_mismatch_raises(self, tmp_path: Path) -> None:
        """Loading a catalog with wrong schema_version raises CatalogSchemaError."""
        cat = TrackCatalog()
        cat.add(_make_track("t1"))
        p = tmp_path / "catalog.json"
        cat.save(p)

        raw = json.loads(p.read_text())
        raw["schema_version"] = 999
        p.write_text(json.dumps(raw))

        with pytest.raises(CatalogSchemaError):
            TrackCatalog.load(p)
