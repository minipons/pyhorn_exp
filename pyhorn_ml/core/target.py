"""What the horn should do — acoustic targets for optimization."""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TargetResponse:
    """Acoustic targets for a horn design.

    All weights are 0.0–1.0. The optimizer maximizes a weighted sum
    of these objectives (or finds the Pareto front for conflicting targets).

    Example:
        TargetResponse(
            f_min=80.0,
            f_max=5000.0,
            flatness=0.8,        # heavily reward flat response
            sensitivity=0.5,     # moderate sensitivity weight
            bass_extension=0.6,  # moderate bass weight
            max_size_m=1.2,      # physical constraint
            target_spl=95.0,     # target mid-band SPL in dB
        )
    """

    f_min: float = 80.0
    f_max: float = 5000.0

    # Objective weights (0.0 = ignore, 1.0 = maximize)
    flatness: float = 1.0       # minimize ripple in [f_min, f_max]
    sensitivity: float = 1.0     # maximize mid-band SPL
    bass_extension: float = 1.0  # reward response close to f_min

    # Physical constraints
    max_size_m: float = 2.0      # max path length in metres

    # Reference
    target_spl: float = 95.0    # mid-band SPL target in dB

    @classmethod
    def from_str(cls, s: str) -> "TargetResponse":
        """Parse key=val,key=val string into TargetResponse.

        Example: "flatness=0.8,sensitivity=0.5,bass_extension=0.6,f_min=60"
        Only specified keys are set; others use defaults.
        """
        defaults = cls()
        updates = {}
        for pair in s.split(","):
            if "=" not in pair:
                raise ValueError(f"Expected key=val, got: {pair!r}")
            key, val = pair.strip().split("=", 1)
            if not hasattr(defaults, key):
                raise ValueError(f"Unknown target key: {key!r}")
            updates[key] = float(val) if "." in val else int(val) if val.isdigit() else val
        return cls(**{**dataclass_fields_dict(defaults), **updates})

    def __post_init__(self) -> None:
        for name in ("flatness", "sensitivity", "bass_extension"):
            v = getattr(self, name)
            if not 0.0 <= v <= 1.0:
                raise ValueError(f"{name} must be in [0.0, 1.0], got {v}")


def dataclass_fields_dict(obj) -> dict:
    """Return {field: value} for a dataclass instance."""
    import dataclasses
    return {f.name: getattr(obj, f.name) for f in dataclasses.fields(obj)}
