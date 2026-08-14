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


class ErrorCorrectionXGBRegressorModel(PredictionModel):
    def __init__(
        self,
        meta_layer=None,
        per_horizon: bool = False,
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
        self.params = params or {}
        self.per_horizon = per_horizon
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
        self.model = XGBRegressor()
        if self.params:
            self.model.set_params(**self.params)
        self.meta_layer: HorizonFeatureMetaLayer | None = meta_layer
        if self.meta_layer is None and self.params.get("weighten_features", False):
            self.meta_layer = HorizonFeatureMetaLayer(
                horizon_col="pred_sequence_id",
                weight_formula="inverse_error",
                inverse_scale=self.params.get("meta_layer_smoothing", 1.0),
            )
        # When per_horizon=True, one XGBoost per pred_sequence_id
        self._horizon_models: dict[int, XGBRegressor] = {}
        self._horizon_col = "pred_sequence_id"
        # When use_hal=True and global model, self.model may be an xgb.Booster
        self._is_hal_booster: bool = False
        # When per_horizon=True and use_diagnostic_loss=True, _horizon_models stores xgb.Booster
        self._horizon_boosters: bool = False

    def _preprocess_X(self, X: pd.DataFrame) -> pd.DataFrame:
        """Drop datetime, add days_ahead if horizon present, optionally apply meta_layer transform."""
        X = X.drop(columns=["datetime"], errors="ignore")
        # if self._horizon_col in X.columns and self._days_ahead_col not in X.columns:
        #     X = X.copy()
        #     X[self._days_ahead_col] = X[self._horizon_col] - 1
        if self.meta_layer is not None and self._horizon_col in X.columns:
            if not getattr(self.meta_layer, "_fitted", False):
                self.meta_layer.fit(X)
            X = self.meta_layer.transform(X)
        return X

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
        pred_pvout - predicted PV output (sequence id != 1)
        true_pvout - true PV output (sequence id 1)
        We train on target = true_pvout - pred_pvout so that corrected = pred_pvout + model.predict(X) ≈ true_pvout.
        error_matrix - optional, unused (kept for API compatibility).
        sample_weight - optional 1D array/Series for XGBoost sample_weight (e.g. from meta_layer.get_sample_weights).
        eval_set - optional list of (X_val, y_val) for early stopping; X_val gets same preprocessing as train_x.
        early_stopping_rounds - e.g. 10; use with eval_set to reduce overfitting (set on model before fit).
        """
        pvout_error = (
            true_pvout - pred_pvout
        )  # correction term: so corrected = pred + model_output ≈ true
        train_x = self._preprocess_X(train_x)

        if self.use_hal and self.use_diagnostic_loss:
            raise ValueError("use_hal and use_diagnostic_loss cannot both be True.")

        if self.per_horizon and self._horizon_col in train_x.columns:
            if self.use_diagnostic_loss:
                self._train_per_horizon_diagnostic(
                    train_x,
                    pvout_error,
                    sample_weight=sample_weight,
                )
            else:
                self._horizon_models.clear()
                self._horizon_boosters = False
                for h in sorted(
                    train_x[self._horizon_col].dropna().unique().astype(int)
                ):
                    mask = train_x[self._horizon_col] == h
                    X_h = train_x.loc[mask].drop(
                        columns=[self._horizon_col], errors="ignore"
                    )
                    y_h = pvout_error.loc[mask]
                    w_h = sample_weight[mask] if sample_weight is not None else None
                    m = XGBRegressor()
                    if self.params:
                        m.set_params(**self.params)
                    if w_h is not None:
                        m.fit(X_h, y_h, sample_weight=np.asarray(w_h))
                    else:
                        m.fit(X_h, y_h)
                    self._horizon_models[int(h)] = m
            return

        # ----- Global model: custom objectives (HAL / diagnostic) or standard fit -----
        if self.use_hal and self._horizon_col in train_x.columns:
            self._train_hal(
                train_x,
                pvout_error,
                sample_weight=sample_weight,
                eval_set=eval_set,
                early_stopping_rounds=early_stopping_rounds,
            )
            return

        if self.use_diagnostic_loss and self._horizon_col in train_x.columns:
            self._train_diagnostic(
                train_x,
                pvout_error,
                sample_weight=sample_weight,
                eval_set=eval_set,
                early_stopping_rounds=early_stopping_rounds,
            )
            return

        fit_kwargs = {}
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = np.asarray(sample_weight)
        if eval_set is not None and len(eval_set) > 0:
            X_val, y_val = eval_set[0]
            X_val = self._preprocess_X(X_val)
            fit_kwargs["eval_set"] = [(X_val, y_val)]
        if early_stopping_rounds is not None:
            self.model.set_params(early_stopping_rounds=early_stopping_rounds)
        self.model.fit(train_x, pvout_error, **fit_kwargs)

    def _train_hal(
        self,
        train_x: pd.DataFrame,
        pvout_error: pd.Series,
        sample_weight=None,
        eval_set=None,
        early_stopping_rounds: int | None = None,
    ):
        """Train global model with Horizon-Aware Asymmetric Loss (HAL) via xgb.train."""
        train_horizons = np.asarray(train_x[self._horizon_col].values, dtype=int)
        y_corr = np.asarray(pvout_error).ravel()
        sigma_per_horizon = compute_sigma_per_horizon(y_corr, train_horizons)

        # Keep horizon in features so the model can learn horizon-specific corrections (fair comparison with normal)
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

    def _train_diagnostic(
        self,
        train_x: pd.DataFrame,
        pvout_error: pd.Series,
        sample_weight=None,
        eval_set=None,
        early_stopping_rounds: int | None = None,
    ):
        """
        Train a global model with the Diagnostic-Weighted Error Correction Loss
        via xgb.train and diagnostic_xgb_objective.
        """
        # Meta features for the loss (same row order as train_x / pvout_error)
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

    def _train_per_horizon_diagnostic(
        self,
        train_x: pd.DataFrame,
        pvout_error: pd.Series,
        sample_weight=None,
    ):
        """Train one XGBoost per horizon with Diagnostic-Weighted Loss (xgb.Booster per horizon)."""
        if "diffuse_fraction" not in train_x.columns:
            raise ValueError(
                "Per-horizon diagnostic loss requires 'diffuse_fraction' in features."
            )
        self._horizon_models.clear()
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
            w_h = (
                np.asarray(sample_weight.loc[mask])
                if sample_weight is not None
                else None
            )
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

            dtrain = xgb.DMatrix(
                X_h,
                label=y_h,
                weight=w_h,
            )
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

    def get_diagnostic_arrays(
        self,
        train_x: pd.DataFrame,
        y_correction: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """
        Return (grad, hess, w_i) for the diagnostic loss in train_x row order,
        for heatmap visualization. Returns None if this model does not use diagnostic loss.
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
            # Per-horizon: get predictions per horizon, then grad/hess/w_i per slice and assemble
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
                preds_h = booster.predict(xgb.DMatrix(X_h))
                labels_h = y_correction[mask]
                h_arr = np.full(np.sum(mask), h, dtype=np.int32)
                r_h = regimes[mask]
                u_h = unc[mask]
                scale_h = float(np.median(np.abs(labels_h - preds_h))) or 1.0
                g, he, wi = diagnostic_weighted_grad_hess(
                    labels_h,
                    preds_h,
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
            if getattr(self, "_is_hal_booster", False):
                preds = self.model.predict(xgb.DMatrix(train_x))
            else:
                preds = self.model.predict(train_x)
            scale = float(np.median(np.abs(y_correction - preds))) or 1.0
            grad, hess, w_i = diagnostic_weighted_grad_hess(
                y_correction,
                preds,
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

    def evaluate(self, test_x: pd.DataFrame, test_y: pd.Series):
        preds = self.predict(test_x)
        if len(preds) != len(test_y):
            return 0.0
        return float(np.corrcoef(test_y, preds)[0, 1] ** 2)

    def _predict_per_horizon(self, test_x: pd.DataFrame) -> np.ndarray:
        preds = np.full(len(test_x), np.nan, dtype=float)
        if self._horizon_col not in test_x.columns:
            return np.zeros(len(test_x))
        for h, m in self._horizon_models.items():
            mask = test_x[self._horizon_col].astype(int) == h
            if not mask.any():
                continue
            X_h = test_x.loc[mask].drop(columns=[self._horizon_col], errors="ignore")
            if (
                self._horizon_boosters
                and hasattr(m, "predict")
                and not hasattr(m, "fit")
            ):
                preds[mask.values] = m.predict(xgb.DMatrix(X_h))
            else:
                preds[mask.values] = m.predict(X_h)
        np.nan_to_num(preds, copy=False, nan=0.0)
        return preds

    def predict(self, test_x: pd.DataFrame):
        test_x = self._preprocess_X(test_x)
        if self._horizon_models:
            return self._predict_per_horizon(test_x)
        if self._is_hal_booster:
            dtest = xgb.DMatrix(test_x)
            return self.model.predict(dtest)
        return self.model.predict(test_x)

    def save_model(self, model_path: str):
        self.model.save_model(model_path)

    def load_model(self, model_path: str):
        self.model = xgb.Booster(model_file=model_path)
        self._is_hal_booster = True
