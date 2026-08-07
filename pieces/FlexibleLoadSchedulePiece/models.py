from pydantic import BaseModel, Field

try:
    from common.onedata_models import OneDataSecretsModel, RunIdInputMixin
except ModuleNotFoundError:
    from pieces.common.onedata_models import OneDataSecretsModel, RunIdInputMixin



class InputModel(RunIdInputMixin):
    load_csv: str = Field(description="Load / horizon CSV with datetime")
    flexible_loads_json: str = Field(default="", description="Flexible load definitions JSON")
    price_forecast_csv: str = Field(default="", description="Optional price forecast CSV")
    scenario_yaml: str = Field(default="", description="Optional scenario YAML for timestep")


class SecretsModel(OneDataSecretsModel):
    pass


class OutputModel(BaseModel):
    message: str
    flexible_load_schedule_csv: str
    flexible_load_activation_csv: str
