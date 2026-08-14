"""Map UC3.4 InferencePiece forecast CSV to SoMES virtual_solar.csv for BatterySim."""
from __future__ import annotations

try:
    from domino.base_piece import BasePiece
except ModuleNotFoundError:
    from local_compat.base_piece import BasePiece

from .models import InputModel, OutputModel

from pathlib import Path

import pandas as pd


class PvoutToVirtualSolarPiece(BasePiece):
    def piece_function(self, input_data: InputModel, secrets_data=None) -> OutputModel:
        src = Path(str(input_data.forecast_csv_path))
        if not src.is_file():
            raise FileNotFoundError(f"UC3.4 forecast CSV not found: {src}")

        df = pd.read_csv(src)
        # InferencePiece typically emits datetime + final_forecast (+ base/correction).
        value_col = None
        for c in ("final_forecast", "PVOUT", "pv_kw", "prediction", "y_pred"):
            if c in df.columns:
                value_col = c
                break
        if value_col is None:
            numeric = [c for c in df.columns if c.lower() not in {"datetime", "date", "time", "horizon"}]
            if not numeric:
                raise ValueError(f"No forecast column in {src}; columns={list(df.columns)}")
            value_col = numeric[0]

        if "datetime" in df.columns:
            dt = pd.to_datetime(df["datetime"], errors="coerce")
        elif "Date" in df.columns and "Time" in df.columns:
            dt = pd.to_datetime(
                df["Date"].astype(str).str.strip() + " " + df["Time"].astype(str).str.strip(),
                dayfirst=True,
                errors="coerce",
            )
        else:
            raise ValueError(f"Forecast CSV missing datetime/Date+Time columns: {list(df.columns)}")

        out = pd.DataFrame(
            {
                "datetime": dt,
                "pv_kw": pd.to_numeric(df[value_col], errors="coerce").fillna(0.0).clip(lower=0.0),
            }
        ).dropna(subset=["datetime"])

        out_dir = Path(self.results_path or ".")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "virtual_solar.csv"
        out.to_csv(out_path, index=False)

        return OutputModel(
            message=f"Mapped UC3.4 column '{value_col}' → virtual_solar.csv ({len(out)} rows)",
            virtual_solar_csv=str(out_path),
        )
