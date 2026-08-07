from pydantic import BaseModel, Field

try:
    from common.onedata_models import OneDataSecretsModel, RunIdInputMixin
except ModuleNotFoundError:
    from pieces.common.onedata_models import OneDataSecretsModel, RunIdInputMixin



class InputModel(RunIdInputMixin):
    load_csv: str = Field(description="Path to historical load CSV")
    scenario_yaml: str = Field(description="Path to sized scenario YAML")
    virtual_solar_csv: str = Field(description="Path to virtual_solar.csv")
    battery_strategy_json: str = Field(default="", description="Optional battery strategy recommendation JSON")
    grid_constraints_json: str = Field(default="", description="Optional SoMES grid constraints JSON")
    flexible_load_schedule_csv: str = Field(default="", description="Optional flexible load schedule CSV")
    price_forecast_csv: str = Field(default="", description="Optional price/tariff forecast CSV")
    bess_telemetry_csv: str = Field(
        default="", description="BESS telemetry CSV; last record initialises SOC and available power"
    )
    dispatch_method: str = Field(
        default="auto", description="auto (LP with greedy fallback) | lp | greedy"
    )
    export_price_eur_kwh: float = Field(default=0.05, description="Feed-in remuneration used by the optimiser")
    forecast_accuracy_json: str = Field(default="", description="Forecast error statistics folded into the KPI set")
    degradation_cost_eur_kwh: float = Field(default=0.01, description="Battery throughput cost in the objective")
    peak_price_eur_kw: float = Field(default=0.0, description="Peak import charge; >0 enables peak shaving objective")
    terminal_soc_pct: float = Field(default=-1.0, description="Required SOC at horizon end; <0 disables")


class SecretsModel(OneDataSecretsModel):
    pass


class OutputModel(BaseModel):
    message: str
    virtual_battery_soc_csv: str
    battery_summary_csv: str
    battery_charge_discharge_plan_csv: str
    grid_import_export_plan_csv: str
    next_day_dispatch_plan_csv: str
    technical_validation_json: str
    operational_kpis_json: str
    dispatch_method_used: str = ""
    bess_state_json: str = ""
