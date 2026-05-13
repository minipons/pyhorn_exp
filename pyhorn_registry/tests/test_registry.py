"""Tests for pyhorn_registry."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from pyhorn_registry import Registry, RegistryEntry, registry


class TestRegistryEntry:
    def test_to_dict_roundtrip(self):
        entry = RegistryEntry(
            name="test_driver",
            kind="driver",
            description="A test driver",
            tags=["8ohm", "fullrange"],
            created="2026-01-01T00:00:00Z",
            modified="2026-01-02T00:00:00Z",
            file_path="drivers/test_driver.yaml",
        )
        d = entry.to_dict()
        restored = RegistryEntry.from_dict(d)
        assert restored.name == entry.name
        assert restored.kind == entry.kind
        assert restored.description == entry.description
        assert restored.tags == entry.tags
        assert restored.file_path == entry.file_path

    def test_tags_deduped_on_to_dict(self):
        entry = RegistryEntry(name="x", kind="driver", tags=["a", "a", "b"])
        d = entry.to_dict()
        assert d["tags"] == ["a", "b"]


class TestRegistry:
    def setup_method(self):
        self.tmpdir = TemporaryDirectory()
        self.base = Path(self.tmpdir.name)

    def teardown_method(self):
        self.tmpdir.cleanup()

    def _reg(self) -> Registry:
        return Registry(base=self.base)

    def test_empty_registry_list(self):
        reg = self._reg()
        assert reg.list() == []

    def test_add_driver(self):
        reg = self._reg()
        # create a dummy source file
        driver_file = self.base / "driver.yaml"
        driver_file.write_text("sd: 0.013\n")

        entry = reg.add("my_driver", "driver", driver_file, copy=True)
        assert entry.name == "my_driver"
        assert entry.kind == "driver"
        assert (self.base / "drivers" / "my_driver.yaml").exists()

    def test_add_project_no_copy(self):
        reg = self._reg()
        project_file = self.base / "project.yaml"
        project_file.write_text("geometry: {}\n")

        entry = reg.add("my_project", "project", project_file, copy=False)
        assert entry.name == "my_project"
        assert entry.kind == "project"

    def test_list_filter_by_kind(self):
        reg = self._reg()
        d1 = self.base / "d1.yaml"
        d1.write_text("sd: 0.01\n")
        p1 = self.base / "p1.yaml"
        p1.write_text("geometry: {}\n")

        reg.add("d1", "driver", d1, copy=False)
        reg.add("p1", "project", p1, copy=False)

        drivers = reg.list(kind="driver")
        projects = reg.list(kind="project")
        assert all(e.kind == "driver" for e in drivers)
        assert all(e.kind == "project" for e in projects)

    def test_get_exists(self):
        reg = self._reg()
        f = self.base / "f.yaml"
        f.write_text("x: 1\n")
        reg.add("foo", "driver", f, copy=False)
        assert reg.get("foo") is not None
        assert reg.get("missing") is None

    def test_exists(self):
        reg = self._reg()
        f = self.base / "f.yaml"
        f.write_text("x: 1\n")
        reg.add("foo", "driver", f, copy=False)
        assert reg.exists("foo") is True
        assert reg.exists("bar") is False

    def test_remove_without_file(self):
        reg = self._reg()
        f = self.base / "f.yaml"
        f.write_text("x: 1\n")
        reg.add("foo", "driver", f, copy=False)
        reg.remove("foo")
        assert reg.exists("foo") is False

    def test_remove_with_file(self):
        reg = self._reg()
        f = self.base / "f.yaml"
        f.write_text("x: 1\n")
        reg.add("foo", "driver", f, copy=True)
        assert (self.base / "drivers" / "foo.yaml").exists()
        reg.remove("foo", delete_file=True)
        assert not (self.base / "drivers" / "foo.yaml").exists()

    def test_remove_unknown_raises(self):
        reg = self._reg()
        with pytest.raises(KeyError):
            reg.remove("does_not_exist")

    def test_add_duplicate_raises(self):
        reg = self._reg()
        f = self.base / "f.yaml"
        f.write_text("x: 1\n")
        reg.add("foo", "driver", f, copy=False)
        with pytest.raises(ValueError, match="already exists"):
            reg.add("foo", "driver", f, copy=False)

    def test_add_invalid_kind_raises(self):
        reg = self._reg()
        f = self.base / "f.yaml"
        f.write_text("x: 1\n")
        with pytest.raises(ValueError, match="driver.*project"):
            reg.add("foo", "invalid", f, copy=False)

    def test_update_metadata_description(self):
        reg = self._reg()
        f = self.base / "f.yaml"
        f.write_text("x: 1\n")
        reg.add("foo", "driver", f, copy=False)
        entry = reg.update_metadata("foo", description="updated desc")
        assert entry.description == "updated desc"
        # persisted
        reg2 = self._reg()
        assert reg2.get("foo").description == "updated desc"

    def test_update_metadata_tags(self):
        reg = self._reg()
        f = self.base / "f.yaml"
        f.write_text("x: 1\n")
        reg.add("foo", "driver", f, copy=False, tags=["a"])
        entry = reg.update_metadata("foo", tags=["b", "c"])
        assert entry.tags == ["b", "c"]
        # persisted
        reg2 = self._reg()
        assert reg2.get("foo").tags == ["b", "c"]

    def test_resolve_path(self):
        reg = self._reg()
        f = self.base / "f.yaml"
        f.write_text("x: 1\n")
        reg.add("foo", "driver", f, copy=True)
        path = reg.resolve_path("foo")
        assert path is not None
        assert path.exists()
        assert path.name == "foo.yaml"

    def test_resolve_path_missing(self):
        reg = self._reg()
        assert reg.resolve_path("does_not_exist") is None

    def test_load_yaml(self):
        reg = self._reg()
        f = self.base / "f.yaml"
        f.write_text("sd: 0.013\nre: 7.8\n")
        reg.add("foo", "driver", f, copy=True)
        data = reg.load_yaml("foo")
        assert data["sd"] == 0.013
        assert data["re"] == 7.8

    def test_save_yaml(self):
        reg = self._reg()
        f = self.base / "f.yaml"
        f.write_text("sd: 0.013\n")
        reg.add("foo", "driver", f, copy=True)
        reg.save_yaml("foo", {"sd": 0.015, "re": 8.0})
        data = reg.load_yaml("foo")
        assert data["sd"] == 0.015
        assert data["re"] == 8.0


class TestConvenienceRegistry:
    def setup_method(self):
        self.tmpdir = TemporaryDirectory()
        self.base = Path(self.tmpdir.name)

    def teardown_method(self):
        self.tmpdir.cleanup()

    def test_registry_convenience_function(self):
        """registry() returns a Registry instance for the given base."""
        reg = registry(base=self.base)
        assert isinstance(reg, Registry)
        assert reg.base == self.base
