from typing import Optional

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
    lookback_rows: int = Field(default=672, ge=96)
    z_threshold: float = Field(default=3.5, gt=0)
    min_abs_delta_kw: float = Field(default=2.0, ge=0.0)
    cooldown_minutes: int = Field(default=60, ge=0)
    warn_kw_threshold: float = Field(default=10.0, ge=0.0)
    critical_kw_threshold: float = Field(default=25.0, ge=0.0)


class SecretsModel(OneDataSecretsModel):
    pass



class OutputModel(BaseModel):
    message: str
    alerts_csv: str
    drift_report_json: str
