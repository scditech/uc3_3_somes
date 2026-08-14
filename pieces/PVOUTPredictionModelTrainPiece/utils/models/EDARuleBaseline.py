from __future__ import annotations

import numpy as np
import pandas as pd
import os
from datetime import datetime
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from ..base import PredictionModel


class EDARuleBaseline(PredictionModel):
    """
    Very simple rule-based baseline derived from EDA insights.

    - Uses only a few strongly correlated features (GHI, TEMP, solar_elevation_sin).
    - Does not fit a complex model; instead, it estimates a single scaling
      coefficient alpha from the training data and applies a set of `if` rules.

    This is intentionally lightweight and interpretable, to serve as a
    human-readable baseline against which more advanced models are compared.
    """

    def __init__(self, params: dict | None = None):
        # Configuration parameters with sensible defaults
        params = params or {}
        self.min_ghi = float(params.get("min_ghi", 10.0))
        self.min_solar_elev = float(params.get("min_solar_elev", 0.05))
        self.max_pvout = params.get("max_pvout", None)

        # Learned from train()
        self.alpha = 0.0  # scaling factor between GHI and PVOUT
        self.fallback_mean = 0.0

    def train(self, X: pd.DataFrame, y: pd.Series):
        """
        Estimate a single scaling coefficient alpha from train data.

        We approximate the relation PVOUT ≈ alpha * GHI for daytime samples
        with sufficient irradiance and positive PV output. Alpha is chosen as
        the median of PVOUT / GHI over valid points to be robust to outliers.
        """
        df = X.copy()
        df["PVOUT"] = y.values

        ghi = df.get("GHI")
        if ghi is None:
            raise ValueError(
                "EDARuleBaseline expects feature 'GHI' to be present in X."
            )

        solar_elev = df.get("solar_elevation_sin")

        mask = ghi > self.min_ghi
        mask &= df["PVOUT"] > 0
        if solar_elev is not None:
            mask &= solar_elev > self.min_solar_elev

        valid = df.loc[mask]
        if not valid.empty:
            ratios = valid["PVOUT"] / valid["GHI"]
            self.alpha = float(ratios.median())
        else:
            # Fallback: simple mean-based scaling if filters remove all rows
            eps = 1e-6
            self.alpha = float(df["PVOUT"].mean() / (ghi.mean() + eps))

        self.fallback_mean = float(df["PVOUT"].mean())

    def _predict_row(self, row: pd.Series) -> float:
        ghi = row.get("GHI", np.nan)
        temp = row.get("TEMP", np.nan)
        solar_elev = row.get("solar_elevation_sin", np.nan)

        # Night or very low irradiance -> essentially zero PV output
        if pd.isna(ghi) or ghi <= self.min_ghi:
            return 0.0
        if not pd.isna(solar_elev) and solar_elev <= self.min_solar_elev:
            return 0.0

        # Base linear relation from EDA: PVOUT grows roughly linearly with GHI
        pv = self.alpha * ghi

        # Simple temperature derating: at higher temperatures panel efficiency drops.
        # We apply a mild penalty above ~25°C based on EDA ranges.
        if not pd.isna(temp) and temp > 25:
            # Reduce output by up to ~10% at very high temperatures
            pv *= max(0.9, 1.0 - 0.01 * (temp - 25))

        # Optional clipping to physically plausible range if max_pvout provided
        if self.max_pvout is not None:
            pv = float(np.clip(pv, 0.0, self.max_pvout))

        return float(pv)

    def predict(self, X: pd.DataFrame):
        """
        Apply the hand-crafted rules row-wise.

        Any NaN predictions (e.g. if required features are missing) are
        replaced by the global mean PVOUT observed in training.
        """
        preds = X.apply(self._predict_row, axis=1).astype(float)
        preds = preds.fillna(self.fallback_mean)
        return preds.values

    def evaluate(self, X: pd.DataFrame, y: pd.Series):
        y_pred = self.predict(X)
        return {
            "mean_squared_error": mean_squared_error(y, y_pred),
            "mean_absolute_error": mean_absolute_error(y, y_pred),
            "r2_score": r2_score(y, y_pred),
        }

    def save_model(self, model_path: str):
        os.makedirs(model_path, exist_ok=True)
        joblib.dump(
            self.model,
            f"{model_path}/{self.model.__class__.__name__}{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl",
        )

    def load_model(self, model_path: str):
        self.model = joblib.load(
            f"{model_path}/{self.model.__class__.__name__}{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        )
