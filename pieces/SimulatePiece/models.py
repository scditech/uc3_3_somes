from pydantic import BaseModel, Field


class InputModel(BaseModel):
    load_csv: str = Field(description="Path to historical load CSV")
    scenario_yaml: str = Field(description="Path to scenario YAML")
    output_dir: str = Field(default="", description="Optional output dir when results_path is not set")
    battery_strategy_recommendation_json: str = Field(
        default="",
        description="Optional battery strategy recommendation JSON",
    )
    ranked_catalog_json: str = Field(default="", description="Optional ranked catalog recommendation JSON")
    inverter_catalog_json: str = Field(default="", description="Optional inverter catalog JSON")
    battery_catalog_json: str = Field(default="", description="Optional battery catalog JSON")
    catalog_manifest_json: str = Field(default="", description="Optional catalog sync manifest JSON")


class OutputModel(BaseModel):
    message: str
    report_json: str
