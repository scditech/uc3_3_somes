from __future__ import annotations

from pydantic import BaseModel, Field


class InputModel(BaseModel):
    predictions_csv: str = Field(description="CSV with prediction_load_kw and optionally load_kw")
    pv_forecast_csv: str = Field(default="", description="virtual_solar.csv with pv_kw forecast")
    pv_actual_csv: str = Field(default="", description="pv_production_measurements.csv with pv_kw_measured")
    planned_dispatch_csv: str = Field(default="", description="next_day_dispatch_plan.csv issued to the EMS")
    bess_telemetry_csv: str = Field(default="", description="BESS telemetry with the executed battery power")
    timestep_hours: float = Field(default=0.25, description="Dispatch timestep used for energy totals")
    history_csv: str = Field(default="", description="Measured history used to label the retraining dataset")


class OutputModel(BaseModel):
    report_json: str
    daily_csv: str
    message: str
    forecast_vs_actual_json: str = ""
    closed_loop_ok: bool = True
    retraining_dataset_csv: str = Field(default="", description="Labelled dataset for the next incremental training run")
