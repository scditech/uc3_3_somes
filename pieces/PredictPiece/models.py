from pydantic import BaseModel, ConfigDict, Field


class InputModel(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_path: str = Field(description="Path to trained XGBoost model")
    data_path: str = Field(description="Path to prediction dataset (15min)")
    use_rolling_prediction: bool = Field(
        default=False,
        description="True: use bridge_rows of real load_kw, then compute lags from prior predictions.",
    )
    bridge_rows: int = Field(default=4, ge=1)


class OutputModel(BaseModel):
    message: str
    prediction_file_path: str
