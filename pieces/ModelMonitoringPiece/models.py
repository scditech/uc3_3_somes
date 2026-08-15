from __future__ import annotations

from pydantic import BaseModel, Field

try:
    from common.onedata_models import OneDataSecretsModel, RunIdInputMixin
except ModuleNotFoundError:
    from pieces.common.onedata_models import OneDataSecretsModel, RunIdInputMixin



class InputModel(RunIdInputMixin):
    predictions_csv: str = Field(description="CSV with prediction_load_kw and optionally load_kw")
    pv_forecast_csv: str = Field(default="", description="virtual_solar.csv with pv_kw forecast")
    pv_actual_csv: str = Field(default="", description="pv_production_measurements.csv with pv_kw_measured")
    pv_model_forecast_csv: str = Field(
        default="",
        description="Inference forecast CSV (final_forecast) for PVOUT model quality when measurements do not overlap",
    )
    pv_model_target_csv: str = Field(
        default="",
        description="Feature/normalized CSV with datetime+PVOUT used as reference for PVOUT model MAE/RMSE",
    )
    planned_dispatch_csv: str = Field(default="", description="next_day_dispatch_plan.csv issued to the EMS")
    bess_telemetry_csv: str = Field(default="", description="BESS telemetry with the executed battery power")
    timestep_hours: float = Field(default=0.25, description="Dispatch timestep used for energy totals")
    history_csv: str = Field(default="", description="Measured history used to label the retraining dataset")


class SecretsModel(OneDataSecretsModel):
    pass


class OutputModel(BaseModel):
    report_json: str
    daily_csv: str
    message: str
    forecast_vs_actual_json: str = ""
    closed_loop_ok: bool = True
    retraining_dataset_csv: str = Field(default="", description="Labelled dataset for the next incremental training run")
