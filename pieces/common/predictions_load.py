"""Convert PredictPiece predictions CSV into load CSV for downstream sim pieces."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def predictions_to_load_csv(predictions_csv: str | Path, target_csv: str | Path) -> Path:
    """Create load_csv expected by MRK pieces from predictions output."""
    src = Path(predictions_csv)
    dst = Path(target_csv)
    if not src.is_file():
        raise FileNotFoundError(f"Predictions CSV missing: {src}")
    df = pd.read_csv(src, parse_dates=["datetime"])
    if "prediction_load_kw" not in df.columns:
        raise ValueError("predictions CSV must contain prediction_load_kw")
    if "price_eur_kwh" in df.columns:
        price = pd.to_numeric(df["price_eur_kwh"], errors="coerce")
    elif "price_eur_per_kwh" in df.columns:
        price = pd.to_numeric(df["price_eur_per_kwh"], errors="coerce")
    elif "price_eur_mwh" in df.columns:
        price = pd.to_numeric(df["price_eur_mwh"], errors="coerce") / 1000.0
    else:
        raise ValueError(
            "predictions CSV must contain one of: "
            "price_eur_kwh, price_eur_per_kwh, price_eur_mwh"
        )
    out = pd.DataFrame(
        {
            "datetime": pd.to_datetime(df["datetime"]),
            "load_kw": pd.to_numeric(df["prediction_load_kw"], errors="coerce").fillna(0.0),
            "price_eur_per_kwh": price,
        }
    )
    out["price_eur_per_kwh"] = out["price_eur_per_kwh"].interpolate(limit_direction="both")
    if out["price_eur_per_kwh"].isna().all():
        raise ValueError("All price values in predictions are NaN after normalization")
    dst.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dst, index=False)
    return dst
