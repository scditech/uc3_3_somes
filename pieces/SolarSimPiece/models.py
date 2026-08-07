from pydantic import BaseModel, Field


class InputModel(BaseModel):
    load_csv: str = Field(description="Path to historical load CSV")
    scenario_yaml: str = Field(description="Path to sized scenario YAML")
    weather_forecast_csv: str = Field(default="", description="Optional weather/irradiance forecast CSV")
    pv_production_csv: str = Field(default="", description="Optional measured PV production CSV for calibration")
    solargis_csv: str = Field(default="", description="Optional SolarGIS irradiance history used to train PVOUT")


class OutputModel(BaseModel):
    message: str
    virtual_solar_csv: str
    pv_forecast_uncertainty_json: str = Field(default="", description="Production uncertainty indicators")
