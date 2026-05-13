"""Train a surrogate model from existing optimisation data."""
import os
import joblib

from pyhorn_ml.data.dataset import DesignDataset
from pyhorn_ml.surrogate.factory import make_surrogate


def run_train(data_dir: str, model_type: str, save_as: str) -> None:
    """Load a dataset and train a GP or MLP surrogate model."""
    print(f"[pyhorn-ml] Loading dataset: {data_dir}")
    dataset = DesignDataset(data_dir)
    X, y = dataset.X_y()
    print(f"  {len(X)} designs × {y.shape[1]} frequency points")

    model = make_surrogate(model_type=model_type, n_samples=len(X))
    print(f"  Training {model.name}...")
    model.fit(X, y)

    # Auto-name if not given
    if save_as == "model.pkl":
        save_path = os.path.join(data_dir, f"{model_type}_surrogate.pkl")
    else:
        save_path = save_as

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    joblib.dump(model, save_path)
    print(f"  Saved: {save_path}")
