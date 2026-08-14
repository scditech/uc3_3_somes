from __future__ import annotations

import numpy as np
import pandas as pd


def _predict_with_model(model, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict"):
        try:
            pred = model.predict(X)
            return np.asarray(pred).ravel()
        except Exception:
            import xgboost as xgb

            pred = model.predict(xgb.DMatrix(X))
            return np.asarray(pred).ravel()
    raise ValueError("Model object does not support predict()")


def run_pvout_correction(
    model,
    source_df: pd.DataFrame,
    X: pd.DataFrame,
    base_forecast_column: str,
) -> pd.DataFrame:
    if base_forecast_column not in source_df.columns:
        raise ValueError(f"Missing base forecast column '{base_forecast_column}'")

    correction = _predict_with_model(model, X)
    base = pd.to_numeric(source_df[base_forecast_column], errors="coerce").values
    final = base + correction

    out = source_df.copy()
    out["base_forecast"] = base
    out["correction"] = correction
    out["final_forecast"] = final
    return out


def run_price_level(
    model,
    source_df: pd.DataFrame,
    X: pd.DataFrame,
) -> pd.DataFrame:
    """
    Direct regression: final_forecast is the model output (e.g. EUR/MWh).
    base_forecast is zero; correction equals prediction for schema compatibility.
    """
    level = _predict_with_model(model, X)
    out = source_df.copy()
    out["base_forecast"] = 0.0
    out["correction"] = level
    out["final_forecast"] = level
    return out


def run_price_ahead(
    model,
    source_df: pd.DataFrame,
    X: pd.DataFrame,
    baseline_column: str,
) -> pd.DataFrame:
    if baseline_column not in source_df.columns:
        raise ValueError(f"Missing baseline column '{baseline_column}'")

    correction = _predict_with_model(model, X)
    base = pd.to_numeric(source_df[baseline_column], errors="coerce").values
    final = base + correction

    out = source_df.copy()
    out["base_forecast"] = base
    out["correction"] = correction
    out["final_forecast"] = final
    return out
