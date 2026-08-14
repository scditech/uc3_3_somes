import pandas as pd
import os
from datetime import datetime
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression

from ..base import PredictionModel


class LinearRegressionModel(PredictionModel):
    def __init__(self, params: dict = None):
        # Ensure we always pass a mapping to sklearn's LinearRegression
        self.params = params or {}
        if self.params != {}:
            self.model = LinearRegression(**self.params)
        else:
            self.model = LinearRegression()

    def train(self, X: pd.DataFrame, y: pd.Series):
        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame):
        return self.model.predict(X)

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
