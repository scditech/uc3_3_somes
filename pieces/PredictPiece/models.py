from typing import Optional

try:
    from common.onedata_models import OneDataSecretsModel, RunIdInputMixin
except ModuleNotFoundError:
    from pieces.common.onedata_models import OneDataSecretsModel, RunIdInputMixin


from pydantic import BaseModel, ConfigDict, Field


class InputModel(RunIdInputMixin):
    model_config = ConfigDict(protected_namespaces=())

    model_path: str = Field(description="Path to trained XGBoost model")
    load_csv: str = Field(
        description="Historical load CSV used to build the future prediction time grid",
    )
    prediction_days: int = Field(
        default=30,
        ge=1,
        le=366,
        description="How many days ahead to predict (overridden by scenario.yaml / workflow JSON when present)",
    )
    timestep_minutes: int = Field(
        default=15,
        ge=1,
        le=60,
        description="Time step for the prediction grid (must match training data)",
    )
    use_rolling_prediction: bool = Field(
        default=True,
        description="True (default): bridge_rows of real load_kw, then lags from prior predictions.",
    )
    bridge_rows: int = Field(default=4, ge=1)


class SecretsModel(OneDataSecretsModel):
    pass


class OutputModel(BaseModel):
    message: str
    prediction_file_path: str
    runtime_load_csv: str = Field(
        description="Load CSV for MRK sizing/simulation (load_kw from predictions)",
    )
