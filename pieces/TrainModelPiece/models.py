from pydantic import BaseModel, Field

try:
    from common.onedata_models import OneDataSecretsModel, RunIdInputMixin
except ModuleNotFoundError:
    from pieces.common.onedata_models import OneDataSecretsModel, RunIdInputMixin

from pydantic import BaseModel, ConfigDict, Field


class InputModel(RunIdInputMixin):
    data_path: str = Field(
        title="Training dataset path",
        description="Path to preprocessed parquet or CSV dataset"
    )


class SecretsModel(OneDataSecretsModel):
    pass


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
