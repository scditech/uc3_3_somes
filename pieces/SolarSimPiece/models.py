from pydantic import BaseModel, Field

try:
    from common.onedata_models import OneDataSecretsModel, RunIdInputMixin
except ModuleNotFoundError:
    from pieces.common.onedata_models import OneDataSecretsModel, RunIdInputMixin



class InputModel(RunIdInputMixin):
    load_csv: str = Field(description="Path to historical load CSV")
    scenario_yaml: str = Field(description="Path to sized scenario YAML")
    weather_forecast_csv: str = Field(default="", description="Optional weather/irradiance forecast CSV")
    pv_production_csv: str = Field(default="", description="Optional measured PV production CSV for calibration")
    solargis_csv: str = Field(
        default="",
        description=(
            "Optional SolarGIS irradiance history to train PVOUT. Leave empty for ops "
            "(SAV/UMMS): model trains from seed/synthetic data and calibrates with "
            "pv_production_csv + weather_forecast_csv."
        ),
    )


class SecretsModel(OneDataSecretsModel):
    pass


class OutputModel(BaseModel):
    message: str
    virtual_solar_csv: str
    pv_forecast_uncertainty_json: str = Field(default="", description="Production uncertainty indicators")
