from pydantic import BaseModel, Field


class InputModel(BaseModel):
    next_day_dispatch_plan_csv: str = Field(description="Next-day dispatch plan CSV")
    scenario_yaml: str = Field(default="", description="Scenario YAML with battery/grid params")
    grid_constraints_json: str = Field(default="", description="Grid constraints JSON from connectors")


class OutputModel(BaseModel):
    message: str
    technical_validation_json: str
    approved_next_day_plan_csv: str
    feasibility_ok: bool
    corrected_operating_plan_csv: str = Field(default="", description="Schedule clipped back inside technical limits")
    corrected_operating_recommendations_csv: str = Field(default="", description="Per-step correction recommendations")
