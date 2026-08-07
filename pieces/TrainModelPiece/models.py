from pydantic import BaseModel, ConfigDict, Field


class InputModel(BaseModel):
    data_path: str = Field(
        title="Training dataset path",
        description="Path to preprocessed parquet or CSV dataset"
    )


class OutputModel(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    message: str = Field(
        description="Training result message"
    )
    model_file_path: str = Field(
        description="Path to trained model file"
    )
    train_log_path: str = Field(
        description="Path to training log file"
    )
