from __future__ import annotations

from typing import Dict


MODEL_TYPES = {
    "linear_regression_model",
    "xgb_regressor_model",
    "interval_xgb_regressor_model",
    "eda_rule_baseline",
    "tabpfn_regressor_model",
}


def create_model(model_type: str, model_params: Dict | None = None):
    model_params = model_params or {}
    model_type = str(model_type).lower()

    if model_type == "linear_regression_model":
        from .models.LinearRegression import LinearRegressionModel

        return LinearRegressionModel(model_params)

    if model_type == "xgb_regressor_model":
        from .models.XGBRegressor import XGBRegressorModel

        return XGBRegressorModel(model_params)

    if model_type == "interval_xgb_regressor_model":
        from .models.IntervalXGBRegressor import IntervalXGBRegressorModel

        return IntervalXGBRegressorModel(model_params)

    if model_type == "eda_rule_baseline":
        from .models.EDARuleBaseline import EDARuleBaseline

        return EDARuleBaseline(model_params)

    if model_type == "tabpfn_regressor_model":
        from .models.TabPFNRegressor import TabPFNRegressorModel

        return TabPFNRegressorModel(model_params)

    raise ValueError(
        f"Unsupported model_type: {model_type}. Available: {sorted(MODEL_TYPES)}"
    )
