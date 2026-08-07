from pydantic import BaseModel, Field

try:
    from common.onedata_models import OneDataSecretsModel, RunIdInputMixin
except ModuleNotFoundError:
    from pieces.common.onedata_models import OneDataSecretsModel, RunIdInputMixin



class InputModel(RunIdInputMixin):
    prices_csv: str = Field(description="Historical electricity prices CSV")
    horizon_steps: int = Field(default=96, ge=1, description="Forecast horizon steps (15-min default)")


class SecretsModel(OneDataSecretsModel):
    pass


class OutputModel(BaseModel):
    message: str
    price_forecast_csv: str
    tariff_scenarios_csv: str = Field(default="", description="Long-format low/base/high tariff scenarios")
