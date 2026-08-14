from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def build_forecast_table(
    df: pd.DataFrame,
    datetime_column: str,
    horizon_column: str = "pred_sequence_id",
    target_column: str | None = None,
) -> pd.DataFrame:
    cols = [datetime_column]
    if horizon_column in df.columns:
        cols.append(horizon_column)
    cols += ["base_forecast", "correction", "final_forecast"]
    # Keep the ground-truth column alongside the prediction so downstream evaluators
    # can compute MAE/RMSE/MAPE directly from the saved forecast CSV.
    if target_column and target_column in df.columns and target_column not in cols:
        cols.append(target_column)

    out = df[cols].copy()
    out = out.sort_values(datetime_column).reset_index(drop=True)
    return out


def build_per_horizon_outputs(
    forecast_df: pd.DataFrame,
    horizon_column: str = "pred_sequence_id",
) -> dict[str, list[dict[str, Any]]]:
    if horizon_column not in forecast_df.columns:
        return {}

    out: dict[str, list[dict[str, Any]]] = {}
    for h, g in forecast_df.groupby(horizon_column):
        try:
            key = str(int(float(h)))
        except (TypeError, ValueError):
            key = str(h)
        out[key] = g.to_dict(orient="records")
    return out


def serialize_forecast_if_requested(
    forecast_df: pd.DataFrame,
    csv_path: str | None,
) -> str | None:
    if not csv_path:
        return None
    p = Path(csv_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    forecast_df.to_csv(p, index=False)
    return str(p)
