from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict
import inspect


@dataclass
class TrainedModel:
    model_type: str
    feature_columns: list[str]
    target_column: str
    coefficients: list[float]
    intercept: float
    setup: dict

    def predict(self, X):
        import numpy as np  # type: ignore

        coef = np.asarray(self.coefficients, dtype=float)
        return X @ coef + self.intercept

    def to_dict(self) -> dict:
        return {
            "model_type": self.model_type,
            "feature_columns": self.feature_columns,
            "target_column": self.target_column,
            "coefficients": list(self.coefficients),
            "intercept": float(self.intercept),
            "setup": self.setup,
        }


MODEL_TYPES = {
    "linear_regression",
    "ridge_regression",
    "error_correction_xgb_regressor_model",
    "error_correction_residual_meta_xgb_regressor_model",
    "error_correction_difficulty_weighted_xgb_regressor_model",
}


def _fit_native_model(model_type: str, model_params: dict):
    mt = model_type.lower()
    if mt == "error_correction_xgb_regressor_model":
        from .models.ErrorCorrectionXGBoostRegressor import (
            ErrorCorrectionXGBRegressorModel,
        )

        return ErrorCorrectionXGBRegressorModel(**model_params)
    if mt == "error_correction_residual_meta_xgb_regressor_model":
        from .models.ErrorCorrectionResidualMetaXGBRegressor import (
            ErrorCorrectionResidualMetaXGBRegressorModel,
        )

        return ErrorCorrectionResidualMetaXGBRegressorModel(**model_params)
    if mt == "error_correction_difficulty_weighted_xgb_regressor_model":
        from .models.ErrorCorrectionDifficultyWeightedXGBRegressor import (
            ErrorCorrectionDifficultyWeightedXGBRegressorModel,
        )

        return ErrorCorrectionDifficultyWeightedXGBRegressorModel(**model_params)
    raise ValueError(f"Unsupported model_type: {model_type}")


def _fit_linear_regression(X, y):
    import numpy as np  # type: ignore

    """
    Closed-form OLS using lstsq with an explicit bias column.
    """
    Xb = np.column_stack([X, np.ones(len(X))])
    beta, *_ = np.linalg.lstsq(Xb, y, rcond=None)
    coef = beta[:-1]
    intercept = float(beta[-1])
    return coef, intercept


def _fit_ridge_regression(X, y, alpha: float):
    import numpy as np  # type: ignore

    """
    Closed-form ridge regression with an explicit bias term not regularized.
    """
    Xb = np.column_stack([X, np.ones(len(X))])
    n_features_plus_bias = Xb.shape[1]
    reg = np.eye(n_features_plus_bias) * alpha
    reg[-1, -1] = 0.0  # don't penalize intercept
    beta = np.linalg.solve(Xb.T @ Xb + reg, Xb.T @ y)
    coef = beta[:-1]
    intercept = float(beta[-1])
    return coef, intercept


def train_model(
    model_type: str,
    X,
    y,
    setup: Dict,
    model_params: Dict | None = None,
    full_df: Any = None,
):
    """
    Train and return a model instance chosen by `model_type`.
    """
    model_type = str(model_type).lower()
    feature_columns = list(setup.get("feature_columns", []))
    target_column = str(setup.get("target_column", "PVOUT"))

    model_params = model_params or {}

    if model_type == "linear_regression":
        coef, intercept = _fit_linear_regression(X, y)
        return TrainedModel(
            model_type=model_type,
            feature_columns=feature_columns,
            target_column=target_column,
            coefficients=coef.tolist(),
            intercept=intercept,
            setup=setup,
        )

    if model_type == "ridge_regression":
        alpha = float(setup.get("ridge_alpha", 1.0))
        coef, intercept = _fit_ridge_regression(X, y, alpha=alpha)
        return TrainedModel(
            model_type=model_type,
            feature_columns=feature_columns,
            target_column=target_column,
            coefficients=coef.tolist(),
            intercept=intercept,
            setup=setup,
        )

    native_model = _fit_native_model(model_type=model_type, model_params=model_params)
    train_sig = inspect.signature(native_model.train)
    train_params = set(train_sig.parameters.keys())

    if {"train_x", "pred_pvout", "true_pvout"}.issubset(train_params):
        pred_column = setup.get("pred_column")
        if not pred_column:
            raise ValueError(
                f"Model `{model_type}` requires `model_setup.pred_column` "
                "for baseline predicted PVOUT values."
            )
        native_model.train(
            train_x=full_df[feature_columns],
            pred_pvout=full_df[pred_column],
            true_pvout=full_df[target_column],
        )
    else:
        native_model.train(full_df[feature_columns], full_df[target_column])

    return native_model
