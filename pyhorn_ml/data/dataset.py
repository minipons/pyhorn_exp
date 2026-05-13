"""Store and retrieve evaluated designs."""
from __future__ import annotations
import json
import os
from pathlib import Path
from typing import List, Optional, Iterator

import numpy as np
import joblib

from pyhorn_ml.core.design_point import DesignPoint


class DesignDataset:
    """Append-only log of evaluated designs.

    Stores designs as individual JSON lines (JSONL) so we can append
    without rewriting the whole file. Each entry contains the full
    DesignPoint data needed to retrain surrogates or regenerate plots.

    Directory structure:
        dataset_dir/
          designs.jsonl       — one JSON dict per line
          metadata.json       — creation time, target, etc.
          surrogates/         — saved surrogate model files
            gp_v1.pkl
            mlp_v1.pkl
    """

    def __init__(self, dataset_dir: str):
        self.dir = Path(dataset_dir)
        self.designs_file = self.dir / "designs.jsonl"
        self.metadata_file = self.dir / "metadata.json"
        self.surrogate_dir = self.dir / "surrogates"
        self._cache: Optional[List[dict]] = None

    def init(self, target: dict) -> None:
        """Create the dataset directory and write metadata."""
        self.dir.mkdir(parents=True, exist_ok=True)
        self.surrogate_dir.mkdir(exist_ok=True)
        self._write_metadata(target)
        self._cache = []

    @property
    def n(self) -> int:
        """Number of designs in the dataset."""
        if self._cache is not None:
            return len(self._cache)
        if not self.designs_file.exists():
            return 0
        with open(self.designs_file) as f:
            return sum(1 for _ in f)

    def add(self, design: DesignPoint) -> None:
        """Append a design to the dataset."""
        entry = self._serialise(design)
        mode = "a" if self.designs_file.exists() else "w"
        with open(self.designs_file, mode) as f:
            f.write(json.dumps(entry) + "\n")
        if self._cache is not None:
            self._cache.append(entry)

    def all(self, rebuild: bool = False) -> List[DesignPoint]:
        """Load all designs as DesignPoint objects."""
        if not self.designs_file.exists():
            return []

        if self._cache is not None and not rebuild:
            entries = self._cache
        else:
            entries = []
            with open(self.designs_file) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        entries.append(json.loads(line))
            self._cache = entries

        return [self._deserialise(e) for e in entries]

    def iter_batches(self, batch_size: int = 50) -> Iterator[List[DesignPoint]]:
        """Iterate over designs in batches (for streaming retraining)."""
        batch = []
        for dp in self.all():
            batch.append(dp)
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    def X_y(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (params_matrix, spl_matrix) for surrogate training.

        Returns:
            X: N designs × D params
            y: N designs × F frequency points
        """
        dps = self.all()
        X = np.array([dp.params for dp in dps])
        y = np.array([dp.real_spl for dp in dps], dtype=np.float64)
        return X, y

    def save_surrogate(self, name: str, model) -> str:
        """Save a surrogate model to disk."""
        path = self.surrogate_dir / f"{name}.pkl"
        joblib.dump(model, path)
        return str(path)

    def load_surrogate(self, name: str):
        """Load a surrogate model from disk."""
        path = self.surrogate_dir / f"{name}.pkl"
        return joblib.load(path)

    def _write_metadata(self, target: dict) -> None:
        import time
        meta = {"created_at": time.time(), "target": target}
        with open(self.metadata_file, "w") as f:
            json.dump(meta, f, indent=2)

    def _serialise(self, dp: DesignPoint) -> dict:
        def _arr_to_list(a):
            """Convert numpy array to list, handling complex values."""
            if a is None:
                return None
            a = np.asarray(a)
            if np.iscomplexobj(a):
                return [float(x.real) for x in a]
            return a.tolist()

        return {
            "params": _arr_to_list(dp.params),
            "geometry_params": dp.geometry_params,
            "real_spl": _arr_to_list(dp.real_spl),
            "real_z": _arr_to_list(dp.real_z),
            "freq": _arr_to_list(dp.freq),
            "score": float(dp.score),
            "flatness_score": float(dp.flatness_score),
            "sensitivity_score": float(dp.sensitivity_score),
            "bass_score": float(dp.bass_score),
            "dominated": bool(dp.dominated),
            "pareto_rank": int(dp.pareto_rank),
        }

    def _deserialise(self, entry: dict) -> DesignPoint:
        dp = DesignPoint(
            params=np.array(entry["params"]) if entry["params"] else np.array([]),
            geometry_params=entry["geometry_params"],
            real_spl=np.array(entry["real_spl"]) if entry["real_spl"] else None,
            real_z=np.array(entry["real_z"]) if entry["real_z"] else None,
            freq=np.array(entry["freq"]) if entry["freq"] else None,
            score=entry["score"],
            flatness_score=entry["flatness_score"],
            sensitivity_score=entry["sensitivity_score"],
            bass_score=entry["bass_score"],
            dominated=entry["dominated"],
            pareto_rank=entry["pareto_rank"],
        )
        return dp
