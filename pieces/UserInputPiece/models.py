from typing import Optional

from pydantic import BaseModel, Field

try:
    from common.onedata_models import OneDataSecretsModel, RunIdInputMixin
except ModuleNotFoundError:
    from pieces.common.onedata_models import OneDataSecretsModel, RunIdInputMixin


class InputModel(RunIdInputMixin):
    load_csv: str = Field(
        description=(
            "Historical load CSV. Needs datetime plus load_kw, or department columns "
            "that will be summed to the site total. Prices are optional."
        )
    )
    prices_csv: str = Field(
        default="",
        description=(
            "Optional prices CSV (datetime + price_eur_per_kwh). Leave empty to use "
            "prices already in the load file, or to download OKTE DAM for the same window."
        ),
    )
    scenario_yaml: str = Field(
        description="SoMES scenario YAML (site, PV, battery, grid, dispatch, MRK)."
    )
    site_name: Optional[str] = Field(
        default=None,
        description="Plant / offtake name shown on the form and dashboard.",
    )
    latitude: Optional[float] = Field(
        default=None,
        description="Site latitude in decimal degrees. Used by Open-Meteo and SHMÚ.",
    )
    longitude: Optional[float] = Field(
        default=None,
        description="Site longitude in decimal degrees. Used by Open-Meteo and SHMÚ.",
    )
    timezone: Optional[str] = Field(
        default=None,
        description="IANA timezone of the offtake, typically Europe/Bratislava.",
    )
    installed_kwp: Optional[float] = Field(
        default=None,
        description="Installed PV capacity in kWp used for the physical PVOUT model.",
    )
    tilt_deg: Optional[float] = Field(
        default=None,
        description="Array tilt in degrees from horizontal.",
    )
    azimuth_deg: Optional[float] = Field(
        default=None,
        description="Array azimuth in degrees (180 = south).",
    )
    energy_kwh: Optional[float] = Field(
        default=None,
        description="Usable battery energy in kWh.",
    )
    max_c_rate: Optional[float] = Field(
        default=None,
        description="Battery C-rate (charge/discharge power as a fraction of energy).",
    )
    initial_soc_pct: Optional[float] = Field(
        default=None,
        description="Starting state of charge in percent if no BMS telemetry is present.",
    )
    import_limit_kw: Optional[float] = Field(
        default=None,
        description="Maximum grid import at the point of connection (kW).",
    )
    export_limit_kw: Optional[float] = Field(
        default=None,
        description="Maximum grid export at the point of connection (kW).",
    )
    inverter_limit_kw: Optional[float] = Field(
        default=None,
        description="AC inverter / PCS limit (kW).",
    )
    dispatch_method: Optional[str] = Field(
        default=None,
        description="Dispatch solver: auto (LP then greedy), lp, or greedy.",
    )
    degradation_cost_eur_per_kwh: Optional[float] = Field(
        default=None,
        description="Battery throughput cost in the dispatch objective (€/kWh).",
    )
    contract_kw: Optional[float] = Field(
        default=None,
        description="Contracted reserved capacity MRK (kW) at the DSO.",
    )
    horizon_hours: Optional[int] = Field(
        default=None,
        description="D+1 load-forecast horizon in hours (typically 24).",
    )
    enable_open_meteo: bool = Field(
        default=True,
        description="Fetch Open-Meteo irradiance and compute physical PVOUT for the load window.",
    )
    enable_shmu: bool = Field(
        default=True,
        description="Calibrate Open-Meteo GHI against the nearest SHMÚ station when overlap exists.",
    )
    enable_okte: bool = Field(
        default=True,
        description="If the load file has no prices, download OKTE DAM for the same timestamps.",
    )
    ems_endpoint_url: Optional[str] = Field(
        default=None,
        description="Optional HTTP endpoint for the next-day plan. Empty keeps a file-only EMS payload.",
    )
    ui_language: str = Field(
        default="sk",
        description="Default language of the generated HTML form (sk or en). The page still has a switch.",
    )


class SecretsModel(OneDataSecretsModel):
    pass


class OutputModel(BaseModel):
    message: str
    load_csv: str
    scenario_yaml: str
    run_id: str = ""
    enable_shmu: bool = Field(
        default=True,
        description="Whether SHMÚ calibration is enabled for this run (wired to ShmuCalibrationPiece).",
    )
    ems_endpoint_url: str = Field(
        default="",
        description="Optional EMS HTTP endpoint (wired to EmsBemsOutputPiece). Empty = file-only payload.",
    )
    user_input_summary_json: str = Field(
        default="",
        description="JSON with price source, merge mode, datetime horizon and applied form overrides.",
    )
    user_input_html: str = Field(
        default="",
        description="Bilingual HTML form filled with the values used for this run.",
    )
