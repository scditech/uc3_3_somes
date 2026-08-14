# https://huggingface.co/Prior-Labs/tabpfn_2_5

import pandas as pd
import joblib
from tabpfn import TabPFNRegressor
import os
from datetime import datetime
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

from ..base import PredictionModel
from ..utils import decide_device


class TabPFNRegressorModel(PredictionModel):
    def __init__(self, params: dict = None):
        params = dict(params or {})
        # Allow >50k samples when dataset exceeds TabPFN's official limit
        model_kwargs = {"device": decide_device()}
        if params.get("ignore_pretraining_limits") is not None:
            model_kwargs["ignore_pretraining_limits"] = params.pop(
                "ignore_pretraining_limits"
            )
        self.model = TabPFNRegressor(**model_kwargs)
        if params:
            self.model.set_params(**params)

    def train(self, X: pd.DataFrame, y: pd.Series):
        self.model.fit(X, y)

    def predict(self, data: pd.DataFrame):
        return self.model.predict(data)

    def evaluate(self, X: pd.DataFrame, y_true: pd.Series):
        y_pred = self.predict(X)
        # y_pred_proba = self.model.predict_proba(X)
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
