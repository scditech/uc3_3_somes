from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def load_input_dataframe(payload: dict) -> pd.DataFrame:
    input_cfg = payload.get("input") or {}
    tabular_data = input_cfg.get("tabular_data") or payload.get("tabular_data")
    data_path = input_cfg.get("data_path") or payload.get("data_path")

    if tabular_data is not None:
        return pd.DataFrame(tabular_data)

    if data_path:
        p = Path(data_path)
        if p.suffix.lower() == ".csv":
            return pd.read_csv(p)
        if p.suffix.lower() in {".parquet", ".pq"}:
            return pd.read_parquet(p)
        raise ValueError(f"Unsupported input format: {p.suffix}")

    raise ValueError(
        "Provide either `data_path` / `tabular_data` (top-level) or "
        "`payload.input.data_path` / `payload.input.tabular_data`."
    )


def load_preprocessing_metadata(payload: dict) -> dict:
    explicit = payload.get("preprocessing_metadata_path")
    model_path = Path(payload["model_path"])

    if explicit:
        meta_path = Path(explicit)
    else:
        base_dir = model_path if model_path.is_dir() else model_path.parent
        meta_path = base_dir / "preprocessing_metadata.json"

    if not meta_path.exists():
        return {}
    return json.loads(meta_path.read_text(encoding="utf-8"))


def load_model_object(payload: dict) -> Any:
    """
    MVP loaders:
    - .pkl -> joblib.load
    - .json / .ubj / .bin / .model -> xgboost.Booster
    """
    model_path = Path(payload["model_path"])
    suffix = model_path.suffix.lower()

    if suffix in {".pkl", ".joblib"}:
        import joblib

        obj = joblib.load(model_path)
        # Trainer pickles `{"metadata": ..., "trained_model_object": model}`.
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
