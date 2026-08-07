from pydantic import BaseModel, Field


class InputModel(BaseModel):
    input_path: str = Field(
        description="Path to merged energy parquet file"
    )

    forecast_hours: int = Field(
        default=24,
        description="Ignored when generate_predict_dataset is False",
    )

    generate_predict_dataset: bool = Field(
        default=False,
        description="If False (default): only train_dataset.parquet. Predict uses separate CSV in PredictPiece.",
    )


class OutputModel(BaseModel):
    message: str
    train_file_path: str
    predict_file_path: str = Field(
        default="",
        description="Empty if generate_predict_dataset is False",
    )
    data_quality_report_json: str = Field(default="", description="Quality indicators + correction recommendations")
    data_quality_indicators_csv: str = Field(default="", description="One row per ingested dataset")
    missing_data_report_csv: str = Field(default="", description="Gaps and NaN counts found before correction")
    data_quality_severity: str = Field(default="ok", description="ok | info | warning | critical")
