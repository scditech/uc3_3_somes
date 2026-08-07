from pydantic import BaseModel, Field

try:
    from common.onedata_models import OneDataSecretsModel, RunIdInputMixin
except ModuleNotFoundError:
    from pieces.common.onedata_models import OneDataSecretsModel, RunIdInputMixin

from pydantic import BaseModel, ConfigDict, Field


class InputModel(RunIdInputMixin):
    model_config = ConfigDict(protected_namespaces=())
    history_csv: str = Field(description="Merged history CSV")
    models_index_json: str = Field(description="JSON map department -> model path")
    model_registry_dir: str = Field(description="Allowed model registry root for secure loading")
    horizon_hours: int = Field(default=24, ge=1, description="Forecast horizon in hours")


class SecretsModel(OneDataSecretsModel):
    pass


class OutputModel(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    message: str
    forecast_csv: str
    forecast_confidence_json: str = Field(default="", description="Forecast confidence metrics per department")
    peak_demand_json: str = Field(default="", description="Predicted peak demand values per day and department")
    site_forecast_csv: str = Field(
        default="", description="Site-level D+1 load profile with prices, consumed by the dispatch chain"
    )
