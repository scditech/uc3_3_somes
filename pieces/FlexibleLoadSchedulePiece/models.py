from pydantic import BaseModel, Field


class InputModel(BaseModel):
    load_csv: str = Field(description="Load / horizon CSV with datetime")
    flexible_loads_json: str = Field(default="", description="Flexible load definitions JSON")
    price_forecast_csv: str = Field(default="", description="Optional price forecast CSV")
    scenario_yaml: str = Field(default="", description="Optional scenario YAML for timestep")


class OutputModel(BaseModel):
    message: str
    flexible_load_schedule_csv: str
    flexible_load_activation_csv: str
