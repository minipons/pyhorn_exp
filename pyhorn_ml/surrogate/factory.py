"""Surrogate model factory and auto-switcher."""
from pyhorn_ml.surrogate.base import SurrogateModel
from pyhorn_ml.surrogate.gp import GPSurrogate
from pyhorn_ml.surrogate.mlp import MLPSurrogate


SWITCH_THRESHOLD = 500  # switch to MLP when dataset exceeds this many points


def make_surrogate(model_type: str = "gp", n_samples: int = 0) -> SurrogateModel:
    """Factory: return the right surrogate based on model type and dataset size.

    Args:
        model_type: "gp" or "mlp"
        n_samples: number of designs in training set
    """
    if model_type == "gp":
        return GPSurrogate()
    elif model_type == "mlp":
        return MLPSurrogate()
    else:
        raise ValueError(f"Unknown model type: {model_type!r}")


def auto_surrogate(n_samples: int = 0) -> SurrogateModel:
    """Automatically choose GP or MLP based on dataset size.

    GP:  better with < 500 samples, gives uncertainty
    MLP: faster at prediction, needs 500+ samples
    """
    if n_samples < SWITCH_THRESHOLD:
        return GPSurrogate()
    return MLPSurrogate()
