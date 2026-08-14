from abc import ABC, abstractmethod
import pandas as pd


class PredictionModel(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series):
        raise NotImplementedError("train method not implemented")

    @abstractmethod
    def save_model(self, model_path: str):
        raise NotImplementedError("save_model method not implemented")

    @abstractmethod
    def load_model(self, model_path: str):
        raise NotImplementedError("load_model method not implemented")

    @abstractmethod
    def predict(self, X: pd.DataFrame):
        raise NotImplementedError("predict method not implemented")

    @abstractmethod
    def evaluate(self, X: pd.DataFrame, y: pd.Series):
        raise NotImplementedError("evaluate method not implemented")
