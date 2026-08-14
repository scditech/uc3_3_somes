from __future__ import annotations

from pathlib import Path
from typing import Any


def load_baseline_model(model_path: str) -> Any:
    """
    Load a baseline PVOUT prediction model produced by an upstream trainer.

    Mirrors `InferencePiece.utils.loader.load_model_object` so the same
    checkpoint that powers `InferencePiece(mode=pvout_correction)` downstream
    can be used here to generate the baseline prediction column needed for
    error-correction training.
    """
    import joblib

    p = Path(model_path)
    suffix = p.suffix.lower()

    if suffix == ".pkl":
        obj = joblib.load(p)
        if isinstance(obj, dict) and "trained_model_object" in obj:
            return obj["trained_model_object"]
        return obj

    if suffix in {".json", ".ubj", ".bin", ".model"}:
        import xgboost as xgb

        return xgb.Booster(model_file=str(p))

    raise ValueError(
        f"Unsupported baseline_model_path extension '{suffix}'. "
        "Expected .pkl (joblib) or xgboost booster file."
    )


def predict_baseline(model: Any, X) -> Any:
    """
    Predict with a baseline model regardless of whether it is a `PredictionModel`
    wrapper (e.g. `XGBRegressorModel`) or a raw `xgb.Booster`. Returns a 1D
    numpy array aligned with the input rows.
    """
    import numpy as np

    if hasattr(model, "predict"):
        try:
            pred = model.predict(X)
            return np.asarray(pred).ravel()
        except Exception:
            import xgboost as xgb

            pred = model.predict(xgb.DMatrix(X))
            return np.asarray(pred).ravel()

    raise ValueError("Baseline model does not support predict()")
