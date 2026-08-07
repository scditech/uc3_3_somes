from pydantic import BaseModel, Field


class InputModel(BaseModel):
    connectors_dir: str = Field(
        default="",
        description="Directory with SoMES connector files (created with demos if empty/missing)",
    )
    weather_api_enabled: bool = Field(
        default=False, description="Pull live irradiance/temperature forecast from Open-Meteo"
    )
    latitude: float = Field(default=48.148, description="Site latitude for the weather pull")
    longitude: float = Field(default=17.107, description="Site longitude for the weather pull")
    forecast_days: int = Field(default=2, description="Weather forecast horizon in days")
    prices_api_enabled: bool = Field(
        default=False, description="Pull day-ahead spot prices from the OKTE ISOT public API"
    )
    prices_lookback_days: int = Field(
        default=30, description="How many past days of OKTE prices to pull together with D+1"
    )
    weather_url: str = Field(default="", description="Optional CSV endpoint overriding the weather provider")
    prices_url: str = Field(default="", description="Optional CSV endpoint overriding the price provider")
    pv_production_url: str = Field(default="", description="Optional CSV endpoint with measured PV production")
    bess_telemetry_url: str = Field(default="", description="Optional CSV endpoint with BESS telemetry")
    allow_demo_fallback: bool = Field(
        default=True, description="If False the piece fails instead of generating synthetic fixtures"
    )


class OutputModel(BaseModel):
    message: str
    connectors_manifest_json: str
    weather_forecast_csv: str
    pv_production_csv: str
    bess_telemetry_csv: str
    prices_csv: str = ""
    grid_constraints_json: str
    flexible_loads_json: str
    data_quality_report_json: str = ""
    data_quality_indicators_csv: str = ""
    missing_data_report_csv: str = ""
    live_sources_used: int = 0
