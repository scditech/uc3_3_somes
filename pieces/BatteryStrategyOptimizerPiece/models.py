from pydantic import BaseModel, Field


class InputModel(BaseModel):
    load_csv: str = Field(description="Path to historical load CSV")
    scenario_yaml: str = Field(description="Path to sized scenario YAML")


class OutputModel(BaseModel):
    message: str
    battery_strategy_recommendation_json: str
