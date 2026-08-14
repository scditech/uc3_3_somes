"""
ErrorCorrectionDifficultyWeightedXGBRegressorModel: two-pass error correction with
difficulty-aware sample weights.

1. First pass: fit XGBoost(s) on pvout_error (no sample weights).
2. Compute in-sample residuals and derive difficulty per row (e.g. |residual| or squared).
3. Map difficulty to sample weights (high difficulty → lower weight) in a principled way.
4. Second pass: retrain the same model structure with sample_weight = difficulty weights.

This down-weights "hard" rows so the model focuses more on typical errors while
still seeing hard examples with reduced influence.
"""

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from ..base import PredictionModel


class ErrorCorrectionDifficultyWeightedXGBRegressorModel(PredictionModel):
    def __init__(
        self,
        per_horizon: bool = False,
        difficulty_weight_alpha: float = 1.0,
        difficulty_metric: str = "abs_residual",
        normalize_per_horizon: bool = True,
        weight_formula: str = "inverse",
        min_weight: float = 0.1,
        **params,
    ):
        """
        Args:
            per_horizon: If True, one XGBoost per pred_sequence_id (both passes).
            difficulty_weight_alpha: Scale for difficulty in weight; higher = more
                aggressive down-weighting of hard rows. Used as w = 1/(1+alpha*d)
                or w = exp(-alpha*d) depending on weight_formula.
            difficulty_metric: "abs_residual" (d = |r|) or "squared_residual" (d = r^2).
            normalize_per_horizon: If True, scale difficulty within each horizon
                by median difficulty in that horizon so weights are comparable.
            weight_formula: "inverse" -> w = 1/(1+alpha*d), "exp_decay" -> w = exp(-alpha*d).
            min_weight: Minimum sample weight (avoid zero weight for numerical stability).
            **params: Passed to underlying XGBRegressor(s).
        """
        self.params = params or {}
        self.per_horizon = per_horizon
        self.difficulty_weight_alpha = difficulty_weight_alpha
        self.difficulty_metric = difficulty_metric
        self.normalize_per_horizon = normalize_per_horizon
        self.weight_formula = weight_formula
        self.min_weight = min_weight

        self.model = XGBRegressor()
        if self.params:
            self.model.set_params(**self.params)
        self._horizon_models: dict[int, XGBRegressor] = {}
        self._horizon_col = "pred_sequence_id"
        # self._days_ahead_col = "days_ahead"

    @staticmethod
    def compute_difficulty_weights(
        residuals: np.ndarray,
        horizon: np.ndarray | None = None,
        alpha: float = 1.0,
        difficulty_metric: str = "abs_residual",
        normalize_per_horizon: bool = True,
        weight_formula: str = "inverse",
        min_weight: float = 0.1,
    ) -> np.ndarray:
        """
        Compute difficulty-based sample weights from residuals (e.g. for use with
        another model like residual meta). High |residual| -> lower weight.

        Args:
            residuals: 1D array of residuals (e.g. pvout_error - first_pass_pred).
            horizon: Optional 1D array of horizon id per row; used if normalize_per_horizon.
            alpha, difficulty_metric, normalize_per_horizon, weight_formula, min_weight:
                Same meaning as in __init__.
        Returns:
            Weights array of shape (len(residuals),), values in [min_weight, 1].
        """
        if difficulty_metric == "squared_residual":
            d = np.square(residuals)
        else:
            d = np.abs(residuals)

        if normalize_per_horizon and horizon is not None:
            d = np.asarray(d, dtype=float)
            h_flat = np.asarray(horizon, dtype=int)
            for h in np.unique(h_flat):
                mask = h_flat == h
                med = np.median(d[mask])
                if med > 0:
                    d[mask] = d[mask] / med

        if weight_formula == "exp_decay":
            w = np.exp(-alpha * d)
        else:
            w = 1.0 / (1.0 + alpha * d)

        return np.clip(w, min_weight, 1.0)

    def _preprocess_X(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.drop(columns=["datetime"], errors="ignore")
        # if self._horizon_col in X.columns and self._days_ahead_col not in X.columns:
        #     X = X.copy()
        #     X[self._days_ahead_col] = X[self._horizon_col] - 1
        return X

    def _difficulty_weights(
        self,
        residuals: np.ndarray,
        horizon: np.ndarray | None,
    ) -> np.ndarray:
        """Compute sample weights from residuals: high |residual| -> lower weight."""
        if self.difficulty_metric == "squared_residual":
            d = np.square(residuals)
        else:
            d = np.abs(residuals)

        if self.normalize_per_horizon and horizon is not None:
            d = np.asarray(d, dtype=float)
            h_flat = np.asarray(horizon, dtype=int)
            for h in np.unique(h_flat):
                mask = h_flat == h
                med = np.median(d[mask])
                if med > 0:
                    d[mask] = d[mask] / med
                # else leave as-is

        if self.weight_formula == "exp_decay":
            w = np.exp(-self.difficulty_weight_alpha * d)
        else:
            w = 1.0 / (1.0 + self.difficulty_weight_alpha * d)

        w = np.clip(w, self.min_weight, 1.0)
        return w

    def train(
        self,
        train_x: pd.DataFrame,
        pred_pvout: pd.Series,
        true_pvout: pd.Series,
        error_matrix=None,
        sample_weight=None,
        eval_set=None,
        early_stopping_rounds: int | None = None,
    ):
        """
        Two-pass train: first fit without weights, compute difficulty from residuals,
        then refit with difficulty-aware sample weights.
        """
        pvout_error = true_pvout - pred_pvout
        train_x_proc = self._preprocess_X(train_x)
        horizon = (
            train_x_proc[self._horizon_col].values
            if self._horizon_col in train_x_proc.columns
            else None
        )

        # ----- Per-horizon: first pass -----
        if self.per_horizon and self._horizon_col in train_x_proc.columns:
            self._horizon_models.clear()
            first_pass_pred = np.full(len(train_x_proc), np.nan, dtype=float)

            for h in sorted(
                train_x_proc[self._horizon_col].dropna().unique().astype(int)
            ):
                mask = train_x_proc[self._horizon_col].astype(int) == h
                X_h = train_x_proc.loc[mask].drop(
                    columns=[self._horizon_col], errors="ignore"
                )
                y_h = pvout_error.loc[mask]
                m = XGBRegressor()
                if self.params:
                    m.set_params(**self.params)
                m.fit(X_h, y_h)
                self._horizon_models[int(h)] = m
                first_pass_pred[mask.values] = m.predict(X_h)

            np.nan_to_num(first_pass_pred, copy=False, nan=0.0)
            residuals = (pvout_error - first_pass_pred).values
            dw = self._difficulty_weights(residuals, horizon)
            if sample_weight is not None:
                dw = dw * np.asarray(sample_weight)

            # Second pass: refit with difficulty weights
            self._horizon_models.clear()
            for h in sorted(
                train_x_proc[self._horizon_col].dropna().unique().astype(int)
            ):
                mask = train_x_proc[self._horizon_col].astype(int) == h
                X_h = train_x_proc.loc[mask].drop(
                    columns=[self._horizon_col], errors="ignore"
                )
                y_h = pvout_error.loc[mask]
                w_h = dw[mask]
                m = XGBRegressor()
                if self.params:
                    m.set_params(**self.params)
                m.fit(X_h, y_h, sample_weight=w_h)
                self._horizon_models[int(h)] = m
            return

        # ----- Global: first pass -----
        fit_kwargs = {}
        if eval_set is not None and len(eval_set) > 0:
            X_val, y_val = eval_set[0]
            X_val_proc = self._preprocess_X(X_val)
            fit_kwargs["eval_set"] = [(X_val_proc, y_val)]
        if early_stopping_rounds is not None:
            self.model.set_params(early_stopping_rounds=early_stopping_rounds)

        self.model.fit(train_x_proc, pvout_error, **fit_kwargs)
        first_pass_pred = self.model.predict(train_x_proc)
        residuals = (pvout_error - first_pass_pred).values
        dw = self._difficulty_weights(residuals, horizon)
        if sample_weight is not None:
            dw = dw * np.asarray(sample_weight)

        # Second pass: refit with difficulty weights
        fit_kwargs["sample_weight"] = dw
        self.model = XGBRegressor()
        if self.params:
            self.model.set_params(**self.params)
        if early_stopping_rounds is not None:
            self.model.set_params(early_stopping_rounds=early_stopping_rounds)
        self.model.fit(train_x_proc, pvout_error, **fit_kwargs)

    def _predict_per_horizon(self, X: pd.DataFrame) -> np.ndarray:
        preds = np.full(len(X), np.nan, dtype=float)
        if self._horizon_col not in X.columns:
            return np.zeros(len(X), dtype=float)
        for h, m in self._horizon_models.items():
            mask = X[self._horizon_col].astype(int) == h
            if not mask.any():
                continue
            X_h = X.loc[mask].drop(columns=[self._horizon_col], errors="ignore")
            preds[mask.values] = m.predict(X_h)
        np.nan_to_num(preds, copy=False, nan=0.0)
        return preds

    def predict(self, test_x: pd.DataFrame) -> np.ndarray:
        test_x_proc = self._preprocess_X(test_x)
        if self._horizon_models:
            return self._predict_per_horizon(test_x_proc)
        return self.model.predict(test_x_proc)

    def evaluate(self, test_x: pd.DataFrame, test_y: pd.Series) -> float:
        preds = self.predict(test_x)
        if len(preds) != len(test_y):
            return 0.0
        return float(np.corrcoef(test_y, preds)[0, 1] ** 2)

    def save_model(self, model_path: str) -> None:
        self.model.save_model(model_path)

    def load_model(self, model_path: str) -> None:
        self.model = XGBRegressor()
        self.model.load_model(model_path)
