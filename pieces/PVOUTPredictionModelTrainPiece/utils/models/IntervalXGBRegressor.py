import pandas as pd
import os
from datetime import datetime
import joblib

from xgboost import XGBRegressor
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

from ..base import PredictionModel


class IntervalXGBRegressorModel(PredictionModel):
    def __init__(self, params: dict = None):

        params_lower = params.copy()
        params_upper = params.copy()

        quantile_alpha_lower = params_lower.pop("quantile_alpha_lower", 0.05)
        quantile_alpha_upper = params_upper.pop("quantile_alpha_upper", 0.95)

        params_lower.pop("quantile_alpha_upper", None)  # Remove from lower model
        params_upper.pop("quantile_alpha_lower", None)  # Remove from upper model

        params_lower["quantile_alpha"] = quantile_alpha_lower
        params_upper["quantile_alpha"] = quantile_alpha_upper

        self.model_lower = XGBRegressor(**params_lower)
        self.model_upper = XGBRegressor(**params_upper)

    def train(self, X: pd.DataFrame, y: pd.Series):
        self.model_lower.fit(X, y)
        self.model_upper.fit(X, y)

    def predict(self, X: pd.DataFrame):
        return self.model_lower.predict(X), self.model_upper.predict(X)

    def evaluate(self, X: pd.DataFrame, y: pd.Series):
        y_pred_lower = self.model_lower.predict(X)
        y_pred_upper = self.model_upper.predict(X)
        return {
            "mean_squared_error_lower": mean_squared_error(y, y_pred_lower),
            "mean_absolute_error_lower": mean_absolute_error(y, y_pred_lower),
            "r2_score_lower": r2_score(y, y_pred_lower),
            "mean_squared_error_upper": mean_squared_error(y, y_pred_upper),
            "mean_absolute_error_upper": mean_absolute_error(y, y_pred_upper),
            "r2_score_upper": r2_score(y, y_pred_upper),
        }

    def save_model(self, model_path: str):
        # Create directory if it doesn't exist
        os.makedirs(model_path, exist_ok=True)

        # Save models with unique filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        lower_path = f"{model_path}/IntervalXGBRegressor_lower_{timestamp}.pkl"
        upper_path = f"{model_path}/IntervalXGBRegressor_upper_{timestamp}.pkl"

        # Use joblib to save (consistent with XGBRegressorModel)
        joblib.dump(self.model_lower, lower_path)
        joblib.dump(self.model_upper, upper_path)

    def load_model(self, model_path: str):
        # This would need to be updated to load both models
        # For now, using joblib like XGBRegressorModel
        # Note: You may need to specify which model files to load
        raise NotImplementedError(
            "load_model not implemented for IntervalXGBRegressorModel"
        )
