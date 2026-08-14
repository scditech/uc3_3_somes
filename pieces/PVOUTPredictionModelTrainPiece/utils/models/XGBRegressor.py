import pandas as pd
import joblib
from xgboost import XGBRegressor
import os
from datetime import datetime
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from ..base import PredictionModel


class XGBRegressorModel(PredictionModel):
    def __init__(self, params: dict = None):
        self.params = params
        self.horizon_weights_active = self.params.pop("horizon_weights_active", False)
        print(f"[INFO] Horizon weights active: {self.horizon_weights_active}")
        self.model = XGBRegressor()
        if self.params:
            self.model.set_params(**self.params)

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        horizon_weights: np.array = None,
        horizon_weights_deactivated: bool = False,
    ):
        if "datetime" in X.columns:
            X = X.drop(columns=["datetime"])
        if self.horizon_weights_active and not horizon_weights_deactivated:
            print("[INFO] Using horizon weights")
            sequential_ids = X["pred_sequence_id"].values
            horizon_weights = np.exp(-0.1 * (sequential_ids - 1))
            self.model.fit(X, y, sample_weight=horizon_weights)
        else:
            self.model.fit(X, y)

    def predict(self, data: pd.DataFrame):
        if "datetime" in data.columns:
            data = data.drop(columns=["datetime"])
        return self.model.predict(data)

    def evaluate(self, X: pd.DataFrame, y_true: pd.Series):
        y_pred = self.predict(X)
        return {
            "mean_squared_error": mean_squared_error(y_true, y_pred),
            "mean_absolute_error": mean_absolute_error(y_true, y_pred),
            "r2_score": r2_score(y_true, y_pred),
        }

    def save_model(self, model_path: str):
        os.makedirs(model_path, exist_ok=True)
        joblib.dump(
            self.model,
            f"{model_path}/{self.model.__class__.__name__}{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl",
        )

    def load_model(self, model_path: str):
        self.model = joblib.load(
            f"{model_path}/{self.model.__class__.__name__}{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl",
        )
