from pydantic import BaseModel, ConfigDict, Field


class InputModel(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    history_csv: str = Field(description="Merged long-term history CSV")
    model_registry_dir: str = Field(description="Directory for per-department model artifacts")
    incremental_window_days: int = Field(default=30, ge=1, description="Train update only on recent window in days")
    full_retrain_every_n_updates: int = Field(
        default=20, ge=1, description="Run full retrain after this many incremental updates"
    )
    incremental_trees: int = Field(default=50, ge=10, description="Number of trees appended in incremental update")


class OutputModel(BaseModel):
    message: str
    models_index_json: str
    training_summary_json: str
