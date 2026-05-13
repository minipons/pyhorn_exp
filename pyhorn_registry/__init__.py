"""pyhorn_registry — Flat-file driver and project registry.

Data is stored in ~/.pyhorn/ as YAML files under drivers/ and projects/,
with a registry.json index for fast lookup by name.

Registry structure:
    ~/.pyhorn/
        registry.json   ← index (name → metadata)
        drivers/
            FE166NV2.yaml
            Peerless_5inch.yaml
        projects/
            living_room_bhl.yaml
            studio_monitor_flh.yaml

Extracted from pyhorn_core/registry.py as part of the May 2 2026 midnight sprint
architecture refactoring.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import yaml


REGISTRY_VERSION = "1.0"
REGISTRY_FILENAME = "registry.json"


@dataclass
class RegistryEntry:
    """A registered driver or project."""

    name: str
    kind: str  # "driver" | "project"
    description: str = ""
    tags: list[str] = field(default_factory=list)
    created: str = ""
    modified: str = ""
    file_path: str = ""  # relative to registry base

    def to_dict(self) -> dict:
        d = asdict(self)
        d["tags"] = sorted(set(d["tags"]))  # dedupe
        return d

    @classmethod
    def from_dict(cls, d: dict) -> RegistryEntry:
        return cls(
            name=d["name"],
            kind=d["kind"],
            description=d.get("description", ""),
            tags=list(d.get("tags", [])),
            created=d.get("created", ""),
            modified=d.get("modified", ""),
            file_path=d.get("file_path", ""),
        )


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _default_base() -> Path:
    """Default registry base (~/.pyhorn)."""
    return Path.home() / ".pyhorn"


def _subdir(kind: str) -> str:
    return {"driver": "drivers", "project": "projects"}[kind]


# ─── Core registry operations ─────────────────────────────────────────────────


class Registry:
    """Flat-file registry for drivers and projects."""

    def __init__(self, base: Optional[Path] = None):
        self.base = base or _default_base()
        self._entries: dict[str, RegistryEntry] = {}
        self._load()

    # ── Persistence ──────────────────────────────────────────────────────────

    def _registry_path(self) -> Path:
        return self.base / REGISTRY_FILENAME

    def _load(self) -> None:
        p = self._registry_path()
        if not p.exists():
            return
        try:
            with open(p) as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError):
            return
        self._entries = {
            name: RegistryEntry.from_dict(d)
            for name, d in data.get("entries", {}).items()
        }

    def _save(self) -> None:
        self.base.mkdir(parents=True, exist_ok=True)
        data = {
            "version": REGISTRY_VERSION,
            "base": str(self.base),
            "entries": {name: e.to_dict() for name, e in self._entries.items()},
        }
        tmp = self._registry_path().with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
        tmp.replace(self._registry_path())

    # ── Query ────────────────────────────────────────────────────────────────

    def list(self, kind: Optional[str] = None) -> list[RegistryEntry]:
        """All entries, optionally filtered by kind."""
        if kind is None:
            return sorted(self._entries.values(), key=lambda e: e.name)
        return sorted(
            (e for e in self._entries.values() if e.kind == kind),
            key=lambda e: e.name,
        )

    def get(self, name: str) -> Optional[RegistryEntry]:
        """Get entry by name, or None."""
        return self._entries.get(name)

    def resolve_path(self, name: str) -> Optional[Path]:
        """Return absolute path to the file for a given entry, or None."""
        entry = self._entries.get(name)
        if not entry:
            return None
        p = self.base / entry.file_path
        return p if p.exists() else None

    def exists(self, name: str) -> bool:
        """Check if an entry with this name exists."""
        return name in self._entries

    # ── Mutate ───────────────────────────────────────────────────────────────

    def _ensure_unique(self, name: str) -> None:
        if name in self._entries:
            raise ValueError(f"Entry already exists: {name}")

    def add(
        self,
        name: str,
        kind: str,
        source_path: Path,
        description: str = "",
        tags: Optional[list[str]] = None,
        copy: bool = True,
    ) -> RegistryEntry:
        """
        Add an entry to the registry.

        If copy=True, the file is copied into the registry directory.
        If copy=False, a reference is stored (file stays in place).
        """
        if kind not in ("driver", "project"):
            raise ValueError(f"kind must be 'driver' or 'project', got {kind}")
        self._ensure_unique(name)

        subdir = _subdir(kind)
        dest_dir = self.base / subdir
        dest_dir.mkdir(parents=True, exist_ok=True)

        if copy:
            dest = dest_dir / f"{name}.yaml"
            shutil.copy2(source_path, dest)
            rel_path = f"{subdir}/{name}.yaml"
        else:
            rel_path = str(source_path)  # absolute or relative as given

        entry = RegistryEntry(
            name=name,
            kind=kind,
            description=description,
            tags=tags or [],
            created=_now(),
            modified=_now(),
            file_path=rel_path,
        )
        self._entries[name] = entry
        self._save()
        return entry

    def remove(self, name: str, delete_file: bool = False) -> None:
        """Remove an entry. If delete_file=True, also delete the underlying file."""
        entry = self._entries.pop(name, None)
        if entry is None:
            raise KeyError(f"No entry named {name}")
        if delete_file:
            p = self.base / entry.file_path
            if p.exists():
                p.unlink()
        self._save()

    def update_metadata(
        self,
        name: str,
        description: Optional[str] = None,
        tags: Optional[list[str]] = None,
    ) -> RegistryEntry:
        """Update description and/or tags of an existing entry."""
        entry = self._entries.get(name)
        if entry is None:
            raise KeyError(f"No entry named {name}")
        if description is not None:
            entry.description = description
        if tags is not None:
            entry.tags = sorted(set(tags))
        entry.modified = _now()
        self._save()
        return entry

    # ── File helpers ─────────────────────────────────────────────────────────

    def load_yaml(self, name: str) -> dict:
        """Load and return the YAML file contents for an entry."""
        path = self.resolve_path(name)
        if path is None:
            raise FileNotFoundError(f"File not found for {name}: {path}")
        with open(path) as f:
            return yaml.safe_load(f)

    def save_yaml(self, name: str, data: dict) -> None:
        """Write updated YAML data back to the entry's file."""
        path = self.resolve_path(name)
        if path is None:
            raise FileNotFoundError(f"No entry named {name}")
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        # touch modified
        entry = self._entries[name]
        entry.modified = _now()
        self._save()


# ─── Convenience ──────────────────────────────────────────────────────────────

def registry(base: Optional[Path] = None) -> Registry:
    """Get the global registry instance."""
    return Registry(base=base)
