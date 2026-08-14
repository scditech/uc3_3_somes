import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost import XGBRegressor

from ..base import PredictionModel
from ..HorizonFeatureMetaLayer import HorizonFeatureMetaLayer
from ..losses import (
    compute_sigma_per_horizon,
    hal_xgb_objective,
    derive_regime,
    derive_uncertainty,
    diagnostic_xgb_objective,
    diagnostic_weighted_grad_hess,
)


class ErrorCorrectionResidualMetaXGBRegressorModel(PredictionModel):
    """
    Two-stage error-correction model with optional intelligent feature weighting:

    1. Base XGBoost learns the usual correction term:
         pvout_error = true_pvout - pred_pvout
    2. A second XGBoost (meta model) learns the residual of the base model:
         residual = pvout_error - base_prediction

       Final correction = base_prediction + meta_prediction.

    Optional: pass a HorizonFeatureMetaLayer (and optionally summary_df in train()) to
    - add horizon-aware reliability features (*_meta_weight, meta_horizon_importance)
      so the model can learn relations between error and feature/horizon reliability,
    - optionally use sample weights from the meta layer (noisier horizons/features down-weighted).
    """

    def __init__(
        self,
        per_horizon: bool = False,
        meta_layer: HorizonFeatureMetaLayer | None = None,
        use_hal: bool = False,
        hal_alpha: float = 1.0,
        hal_beta: float = 0.5,
        hal_weight_mode: str = "floor",
        hal_weight_floor: float = 0.4,
        use_diagnostic_loss: bool = False,
        diag_w_diff_alpha: float = 0.5,
        diag_w_horizon_beta: float = 0.3,
        diag_alpha_under: float = 1.5,
        diag_huber_delta: float = 1.5,
        **params,
    ):
        # Shared params for all underlying XGBRegressor instances
        self.params = params or {}
        self.per_horizon = per_horizon
        self.meta_layer = meta_layer
        self.use_hal = use_hal
        self.use_diagnostic_loss = use_diagnostic_loss
        self.hal_alpha = hal_alpha
        self.hal_beta = hal_beta
        self.hal_weight_mode = hal_weight_mode
        self.hal_weight_floor = hal_weight_floor
        self.diag_w_diff_alpha = diag_w_diff_alpha
        self.diag_w_horizon_beta = diag_w_horizon_beta
        self.diag_alpha_under = diag_alpha_under
        self.diag_huber_delta = diag_huber_delta

        # Global (non per-horizon) base and meta models
        self.model = XGBRegressor()
        self.meta_model: XGBRegressor | None = None

        if self.params:
            self.model.set_params(**self.params)

        # When per_horizon=True, maintain separate models per pred_sequence_id
        self._horizon_models: dict[int, XGBRegressor] = {}
        self._meta_horizon_models: dict[int, XGBRegressor] = {}

        self._horizon_col = "pred_sequence_id"
        self._is_hal_booster: bool = False
        # When per_horizon + use_diagnostic_loss, base models are xgb.Booster
        self._horizon_boosters: bool = False
        # self._days_ahead_col = "days_ahead"

    # ---------- Internal helpers ----------

    def _preprocess_X(self, X: pd.DataFrame) -> pd.DataFrame:
        """Drop datetime, add days_ahead, and optionally apply meta_layer (feature weighting + reliability features)."""
        X = X.drop(columns=["datetime"], errors="ignore")
        # if self._horizon_col in X.columns and self._days_ahead_col not in X.columns:
        #     X = X.copy()
        #     X[self._days_ahead_col] = X[self._horizon_col] - 1
        if self.meta_layer is not None and getattr(self.meta_layer, "_fitted", False):
            X = self.meta_layer.transform(X)
        return X

    def _predict_per_horizon(
        self,
        X: pd.DataFrame,
        use_meta: bool = True,
    ) -> np.ndarray:
        """Predict using per-horizon base (+ optional meta) models."""
        preds = np.full(len(X), np.nan, dtype=float)
        if self._horizon_col not in X.columns:
            return np.zeros(len(X), dtype=float)

        for h, base_m in self._horizon_models.items():
            mask = X[self._horizon_col].astype(int) == h
            if not mask.any():
                continue
            X_h = X.loc[mask].drop(columns=[self._horizon_col], errors="ignore")
            if (
                self._horizon_boosters
                and hasattr(base_m, "predict")
                and not hasattr(base_m, "fit")
            ):
                base_pred = base_m.predict(xgb.DMatrix(X_h))
            else:
                base_pred = base_m.predict(X_h)
            if use_meta and h in self._meta_horizon_models:
                meta_pred = self._meta_horizon_models[h].predict(X_h)
                base_pred = base_pred + meta_pred
            preds[mask.values] = base_pred

        np.nan_to_num(preds, copy=False, nan=0.0)
        return preds

    def _base_predict(self, X: pd.DataFrame) -> np.ndarray:
        """Base model prediction; handles HAL Booster (DMatrix) or XGBRegressor."""
        if self._is_hal_booster:
            d = xgb.DMatrix(X)
            return self.model.predict(d)
        return self.model.predict(X)

    def _train_base_hal(
        self,
        train_x: pd.DataFrame,
        pvout_error: pd.Series,
        sample_weight=None,
        eval_set=None,
        early_stopping_rounds: int | None = None,
    ):
        """Train base model with Horizon-Aware Asymmetric Loss (HAL)."""
        train_horizons = np.asarray(train_x[self._horizon_col].values, dtype=int)
        y_corr = np.asarray(pvout_error).ravel()
        sigma_per_horizon = compute_sigma_per_horizon(y_corr, train_horizons)

        dtrain = xgb.DMatrix(
            train_x,
            label=y_corr,
            weight=np.asarray(sample_weight) if sample_weight is not None else None,
        )

        params = dict(self.params)
        params.pop("objective", None)
        params.setdefault("tree_method", "hist")
        num_boost_round = int(params.pop("n_estimators", 100))

        obj = hal_xgb_objective(
            train_horizons,
            sigma_per_horizon,
            alpha=self.hal_alpha,
            beta=self.hal_beta,
            weight_mode=self.hal_weight_mode,
            weight_floor=self.hal_weight_floor,
        )

        evals = []
        if eval_set is not None and len(eval_set) > 0:
            X_val, y_val = eval_set[0]
            X_val = self._preprocess_X(X_val)
            dval = xgb.DMatrix(X_val, label=np.asarray(y_val).ravel())
            evals = [(dtrain, "train"), (dval, "eval")]

        train_kwargs = dict(
            params=params,
            dtrain=dtrain,
            num_boost_round=num_boost_round,
            obj=obj,
            evals=evals if evals else None,
        )
        if early_stopping_rounds is not None and evals:
            train_kwargs["early_stopping_rounds"] = early_stopping_rounds

        self.model = xgb.train(**train_kwargs)
        self._is_hal_booster = True

    def _train_base_diagnostic(
        self,
        train_x: pd.DataFrame,
        pvout_error: pd.Series,
        sample_weight=None,
        eval_set=None,
        early_stopping_rounds: int | None = None,
    ):
        """
        Train global base model with the Diagnostic-Weighted Error Correction Loss.
        """
        train_horizons = np.asarray(train_x[self._horizon_col].values, dtype=int)
        if "diffuse_fraction" not in train_x.columns:
            raise ValueError(
                "Diagnostic-weighted loss requires 'diffuse_fraction' in features "
                "to derive regimes."
            )
        train_regimes = derive_regime(train_x["diffuse_fraction"].values)
        unc_col = (
            train_x["PVOUT_UNC_LOW"].values
            if "PVOUT_UNC_LOW" in train_x.columns
            else None
        )
        train_unc = derive_uncertainty(train_horizons, unc_column=unc_col)

        y_corr = np.asarray(pvout_error).ravel()
        residual_scale = float(np.median(np.abs(y_corr))) or 1.0

        dtrain = xgb.DMatrix(
            train_x,
            label=y_corr,
            weight=np.asarray(sample_weight) if sample_weight is not None else None,
        )

        params = dict(self.params)
        params.pop("objective", None)
        params.setdefault("tree_method", "hist")
        num_boost_round = int(params.pop("n_estimators", 100))
        early_stopping_rounds = early_stopping_rounds or params.pop(
            "early_stopping_rounds", None
        )

        obj = diagnostic_xgb_objective(
            train_horizons=train_horizons,
            train_regimes=train_regimes,
            train_unc=train_unc,
            residual_scale=residual_scale,
            w_diff_alpha=self.diag_w_diff_alpha,
            w_horizon_beta=self.diag_w_horizon_beta,
            alpha_under=self.diag_alpha_under,
            huber_delta=self.diag_huber_delta,
        )

        evals = []
        if eval_set is not None and len(eval_set) > 0:
            X_val, y_val = eval_set[0]
            X_val = self._preprocess_X(X_val)
            dval = xgb.DMatrix(X_val, label=np.asarray(y_val).ravel())
            evals = [(dtrain, "train"), (dval, "eval")]

        train_kwargs = dict(
            params=params,
            dtrain=dtrain,
            num_boost_round=num_boost_round,
            obj=obj,
            evals=evals if evals else None,
        )
        if early_stopping_rounds is not None and evals:
            train_kwargs["early_stopping_rounds"] = early_stopping_rounds

        self.model = xgb.train(**train_kwargs)
        self._is_hal_booster = True

    def _train_per_horizon_base_diagnostic(
        self,
        train_x: pd.DataFrame,
        pvout_error: pd.Series,
        sample_weight=None,
    ):
        """Train per-horizon base models with Diagnostic-Weighted Loss; meta stays standard XGBRegressor."""
        self._horizon_models.clear()
        self._meta_horizon_models.clear()
        self._horizon_boosters = True

        unc_col = (
            train_x["PVOUT_UNC_LOW"].values
            if "PVOUT_UNC_LOW" in train_x.columns
            else None
        )
        params = dict(self.params)
        params.pop("objective", None)
        params.setdefault("tree_method", "hist")
        num_boost_round = int(params.pop("n_estimators", 100))

        for h in sorted(train_x[self._horizon_col].dropna().unique().astype(int)):
            mask = (train_x[self._horizon_col].astype(int) == h).values
            X_h = train_x.loc[mask].drop(columns=[self._horizon_col], errors="ignore")
            y_h = np.asarray(pvout_error.loc[mask]).ravel()
            w_h = np.asarray(sample_weight[mask]) if sample_weight is not None else None
            n_h = len(y_h)
            train_horizons_h = np.full(n_h, h, dtype=np.int32)
            train_regimes_h = derive_regime(
                train_x.loc[mask, "diffuse_fraction"].values
            )
            unc_h = (
                derive_uncertainty(
                    train_horizons_h,
                    unc_column=unc_col[mask] if unc_col is not None else None,
                )
                if unc_col is not None
                else derive_uncertainty(train_horizons_h)
            )
            residual_scale_h = float(np.median(np.abs(y_h))) or 1.0

            dtrain = xgb.DMatrix(X_h, label=y_h, weight=w_h)
            obj = diagnostic_xgb_objective(
                train_horizons=train_horizons_h,
                train_regimes=train_regimes_h,
                train_unc=unc_h,
                residual_scale=residual_scale_h,
                w_diff_alpha=self.diag_w_diff_alpha,
                w_horizon_beta=self.diag_w_horizon_beta,
                alpha_under=self.diag_alpha_under,
                huber_delta=self.diag_huber_delta,
            )
            booster = xgb.train(
                params=params,
                dtrain=dtrain,
                num_boost_round=num_boost_round,
                obj=obj,
            )
            self._horizon_models[int(h)] = booster

            # Meta on residual of base (labels = y_h - base_pred)
            base_pred_h = booster.predict(xgb.DMatrix(X_h))
            resid_h = y_h - base_pred_h
            meta_m = XGBRegressor()
            if self.params:
                meta_m.set_params(**self.params)
            if w_h is not None:
                meta_m.fit(X_h, resid_h, sample_weight=w_h)
            else:
                meta_m.fit(X_h, resid_h)
            self._meta_horizon_models[int(h)] = meta_m

    def get_diagnostic_arrays(
        self,
        train_x: pd.DataFrame,
        y_correction: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """
        Return (grad, hess, w_i) for the base-stage diagnostic loss in train_x row order.
        For heatmap visualization. Returns None if base does not use diagnostic loss.
        """
        if not self.use_diagnostic_loss:
            return None
        train_x = self._preprocess_X(train_x)
        y_correction = np.asarray(y_correction).ravel()
        if len(y_correction) != len(train_x):
            return None
        if (
            self._horizon_col not in train_x.columns
            or "diffuse_fraction" not in train_x.columns
        ):
            return None

        horizons = np.asarray(train_x[self._horizon_col].values, dtype=int)
        regimes = derive_regime(train_x["diffuse_fraction"].values)
        unc_col = (
            train_x["PVOUT_UNC_LOW"].values
            if "PVOUT_UNC_LOW" in train_x.columns
            else None
        )
        unc = derive_uncertainty(horizons, unc_column=unc_col)

        if self._horizon_models and self._horizon_boosters:
            n = len(y_correction)
            grad_full = np.empty(n)
            hess_full = np.empty(n)
            w_full = np.empty(n)
            for h in sorted(self._horizon_models.keys()):
                mask = horizons == h
                if not np.any(mask):
                    continue
                X_h = train_x.loc[mask].drop(
                    columns=[self._horizon_col], errors="ignore"
                )
                booster = self._horizon_models[h]
                base_pred_h = booster.predict(xgb.DMatrix(X_h))
                labels_h = y_correction[mask]
                h_arr = np.full(np.sum(mask), h, dtype=np.int32)
                r_h = regimes[mask]
                u_h = unc[mask]
                scale_h = float(np.median(np.abs(labels_h - base_pred_h))) or 1.0
                g, he, wi = diagnostic_weighted_grad_hess(
                    labels_h,
                    base_pred_h,
                    h_arr,
                    r_h,
                    u_h,
                    residual_scale=scale_h,
                    w_diff_alpha=self.diag_w_diff_alpha,
                    w_horizon_beta=self.diag_w_horizon_beta,
                    alpha_under=self.diag_alpha_under,
                    huber_delta=self.diag_huber_delta,
                    return_weights=True,
                )
                grad_full[mask] = g
                hess_full[mask] = he
                w_full[mask] = wi
            return grad_full, hess_full, w_full

        if self.model is not None:
            base_pred = self._base_predict(train_x)
            scale = float(np.median(np.abs(y_correction - base_pred))) or 1.0
            grad, hess, w_i = diagnostic_weighted_grad_hess(
                y_correction,
                base_pred,
                horizons,
                regimes,
                unc,
                residual_scale=scale,
                w_diff_alpha=self.diag_w_diff_alpha,
                w_horizon_beta=self.diag_w_horizon_beta,
                alpha_under=self.diag_alpha_under,
                huber_delta=self.diag_huber_delta,
                return_weights=True,
            )
            return grad, hess, w_i
        return None

    # ---------- Public API ----------

    def train(
        self,
        train_x: pd.DataFrame,
        pred_pvout: pd.Series,
        true_pvout: pd.Series,
        error_matrix=None,
        sample_weight=None,
        eval_set=None,
        early_stopping_rounds: int | None = None,
        summary_df=None,
        metric_col: str = "abs_mean_diff",
    ):
        """
        Train base and residual meta models.

        pred_pvout - predicted PV output (sequence id != 1)
        true_pvout - true PV output (sequence id 1)
        We train on target = true_pvout - pred_pvout so that corrected = pred_pvout + model.predict(X) ≈ true_pvout.
        error_matrix - optional, unused (kept for API compatibility).
        sample_weight - optional 1D array/Series for XGBoost sample_weight. If None and meta_layer is fitted,
            sample weights from meta_layer.get_sample_weights(train_x) are used when meta_layer is set.
        eval_set - optional list of (X_val, y_val) for early stopping (applied to the base model only).
        early_stopping_rounds - e.g. 10; use with eval_set to reduce overfitting.
        summary_df - optional; if meta_layer is set, fit it from this ErrorEvaluator-style summary
            (pred_sequence_id x variable → metric) to add intelligent feature weighting and reliability features.
        metric_col - column name in summary_df for the error metric (e.g. "abs_mean_diff").
        """
        # Fit meta layer from error summary if provided (captures relations between error and features/horizons)
        if self.meta_layer is not None and summary_df is not None:
            self.meta_layer.fit_from_diff_summary(summary_df, metric_col=metric_col)
        if (
            sample_weight is None
            and self.meta_layer is not None
            and getattr(self.meta_layer, "_fitted", False)
        ):
            sample_weight = self.meta_layer.get_sample_weights(train_x, aggregate="min")

        pvout_error = true_pvout - pred_pvout
        train_x_proc = self._preprocess_X(train_x)

        # ----- Per-horizon case -----
        if self.per_horizon and self._horizon_col in train_x_proc.columns:
            if self.use_diagnostic_loss and "diffuse_fraction" in train_x_proc.columns:
                self._train_per_horizon_base_diagnostic(
                    train_x_proc,
                    pvout_error,
                    sample_weight=sample_weight,
                )
            else:
                self._horizon_models.clear()
                self._meta_horizon_models.clear()
                self._horizon_boosters = False

                for h in sorted(
                    train_x_proc[self._horizon_col].dropna().unique().astype(int)
                ):
                    mask = train_x_proc[self._horizon_col].astype(int) == h
                    X_h = train_x_proc.loc[mask].drop(
                        columns=[self._horizon_col], errors="ignore"
                    )
                    y_h = pvout_error.loc[mask]
                    w_h = sample_weight[mask] if sample_weight is not None else None

                    # 1) Train base model for this horizon
                    base_m = XGBRegressor()
                    if self.params:
                        base_m.set_params(**self.params)
                    if w_h is not None:
                        base_m.fit(X_h, y_h, sample_weight=np.asarray(w_h))
                    else:
                        base_m.fit(X_h, y_h)

                    # 2) Train residual meta model on the remaining error
                    base_pred_h = base_m.predict(X_h)
                    resid_h = y_h - base_pred_h
                    meta_m = XGBRegressor()
                    if self.params:
                        meta_m.set_params(**self.params)
                    if w_h is not None:
                        meta_m.fit(X_h, resid_h, sample_weight=np.asarray(w_h))
                    else:
                        meta_m.fit(X_h, resid_h)

                    self._horizon_models[int(h)] = base_m
                    self._meta_horizon_models[int(h)] = meta_m

            return

        # ----- Global (non per-horizon) case -----
        if self.use_hal and self._horizon_col in train_x_proc.columns:
            self._train_base_hal(
                train_x_proc,
                pvout_error,
                sample_weight=sample_weight,
                eval_set=eval_set,
                early_stopping_rounds=early_stopping_rounds,
            )
        elif self.use_diagnostic_loss and self._horizon_col in train_x_proc.columns:
            self._train_base_diagnostic(
                train_x_proc,
                pvout_error,
                sample_weight=sample_weight,
                eval_set=eval_set,
                early_stopping_rounds=early_stopping_rounds,
            )
        else:
            fit_kwargs = {}
            if sample_weight is not None:
                fit_kwargs["sample_weight"] = np.asarray(sample_weight)
            if eval_set is not None and len(eval_set) > 0:
                X_val, y_val = eval_set[0]
                X_val_proc = self._preprocess_X(X_val)
                fit_kwargs["eval_set"] = [(X_val_proc, y_val)]
            if early_stopping_rounds is not None:
                self.model.set_params(early_stopping_rounds=early_stopping_rounds)
            self.model.fit(train_x_proc, pvout_error, **fit_kwargs)

        # 2) Train residual meta model on remaining error
        base_pred = self._base_predict(train_x_proc)
        resid = pvout_error - base_pred
        self.meta_model = XGBRegressor()
        if self.params:
            self.meta_model.set_params(**self.params)
        if sample_weight is not None:
            self.meta_model.fit(
                train_x_proc, resid, sample_weight=np.asarray(sample_weight)
            )
        else:
            self.meta_model.fit(train_x_proc, resid)

    def predict(self, test_x: pd.DataFrame):
        test_x_proc = self._preprocess_X(test_x)

        # Per-horizon prediction path
        if self.per_horizon and self._horizon_models:
            return self._predict_per_horizon(test_x_proc, use_meta=True)

        # Global prediction path
        base_pred = self._base_predict(test_x_proc)
        if self.meta_model is None:
            return base_pred
        meta_pred = self.meta_model.predict(test_x_proc)
        return base_pred + meta_pred

    def evaluate(self, test_x: pd.DataFrame, test_y: pd.Series):
        preds = self.predict(test_x)
        if len(preds) != len(test_y):
            return 0.0
        # Return R^2, mirroring ErrorCorrectionXGBRegressorModel
        return float(np.corrcoef(test_y, preds)[0, 1] ** 2)

    def save_model(self, model_path: str):
        # Only saves the global base model; extend if you need full persistence.
        self.model.save_model(model_path)

    def load_model(self, model_path: str):
        self.model = xgb.Booster(model_file=model_path)
        self._is_hal_booster = True
