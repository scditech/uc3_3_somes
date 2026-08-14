from __future__ import annotations

from pathlib import Path
from typing import Any


def load_model_object(payload: dict) -> Any:
    """
    Load a model from `payload['model_path']`.

    Mirrors `InferencePiece.utils.loader.load_model_object` so this piece can
    consume the same checkpoint format that the training pieces produce:
    - `.pkl` files written by the trainers contain
      `{"metadata": {...}, "trained_model_object": <PredictionModel>}` and the
      wrapper is unwrapped here.
    - Raw xgboost booster files (`.json`, `.ubj`, `.bin`, `.model`) are loaded
      via `xgb.Booster(model_file=...)`.
    """
    import joblib

    model_path = Path(payload["model_path"])
    suffix = model_path.suffix.lower()

    if suffix == ".pkl":
        obj = joblib.load(model_path)
        if isinstance(obj, dict) and "trained_model_object" in obj:
            return obj["trained_model_object"]
        return obj

    if suffix in {".json", ".ubj", ".bin", ".model"}:
        import xgboost as xgb

        return xgb.Booster(model_file=str(model_path))

    raise ValueError(
        f"Unsupported model_path extension '{suffix}'. "
        "Expected .pkl (joblib) or xgboost booster file."
    )


def load_input_dataframe(payload: dict):
    """
    Load a DataFrame from `payload['data_path']` (CSV/parquet) or
    `payload['tabular_data']` (list[dict] | dict[str, list]).

    Returns `None` if neither is provided, so callers can fall back to the
    in-memory `payload['data']` path.
    """
    import pandas as pd

    tabular_data = payload.get("tabular_data") or payload.get("dataframe")
    data_path = payload.get("data_path") or payload.get("csv_path")

    if tabular_data is not None:
        return pd.DataFrame(tabular_data)

    if data_path:
        p = Path(data_path)
        if p.suffix.lower() == ".csv":
            return pd.read_csv(p)
        if p.suffix.lower() in {".parquet", ".pq"}:
            return pd.read_parquet(p)
        raise ValueError(f"Unsupported input format: {p.suffix}")

    return None
