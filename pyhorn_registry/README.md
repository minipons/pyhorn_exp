# pyhorn_registry

Flat-file registry for pyhorn drivers and projects.

## Overview

`pyhorn_registry` provides a simple, file-based registry for managing acoustic driver definitions and speaker project files. Data is stored as YAML files under a registry base directory with a `registry.json` index for fast lookup.

## Registry Structure

```
~/.pyhorn/
    registry.json   ← index (name → metadata)
    drivers/
        FE166NV2.yaml
        Peerless_5inch.yaml
    projects/
        living_room_bhl.yaml
        studio_monitor_flh.yaml
```

## Usage

```python
from pyhorn_registry import Registry, RegistryEntry, registry

# Get the global registry (~/.pyhorn)
reg = registry()

# List all entries
for entry in reg.list(kind="driver"):
    print(entry.name, entry.description)

# Add a new entry
reg.add("my_driver", "driver", Path("/path/to/driver.yaml"), copy=True)

# Load the YAML for an entry
data = reg.load_yaml("my_driver")

# Resolve the file path for an entry
path = reg.resolve_path("my_driver")
```

## Backward Compatibility

`pyhorn_core.registry` is kept as a backward-compat shim that re-exports from `pyhorn_registry`. New code should import directly from `pyhorn_registry`.

## Extracted

This package was extracted from `pyhorn_core/registry.py` as part of the May 2 2026 midnight sprint architecture refactoring.
