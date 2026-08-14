from pydantic import BaseModel, ConfigDict, Field, model_validator


class ForecastEntry(BaseModel):
    """A single per-model forecast to include in the aggregated CSV.

    Mirrors `InferencePiece.OutputModel.forecasts[*]` so the two pieces wire
    directly without any reshape on the orchestration side.
    """

    model_config = ConfigDict(extra="allow", protected_namespaces=())

    model_id: str | None = Field(
        default=None,
        description="Identifier used to suffix the aggregated `pred_<model_id>` column.",
    )
    model_path: str | None = Field(default=None, description="Source model checkpoint (informational).")
    mode: str | None = Field(default=None, description="Inference mode used (informational).")
    forecast_csv_path: str | None = Field(
        default=None,
        description="Path to this model's forecast CSV (produced by InferencePiece).",
    )
    feature_columns: list[str] = Field(default_factory=list)
    target_column: str | None = Field(default=None)


class InputModel(BaseModel):
    model_config = ConfigDict(extra="allow", protected_namespaces=())

    forecasts: list[ForecastEntry] | None = Field(
        default=None,
        description=(
            "Per-model forecast entries. Typically wired from "
            "`InferencePiece.OutputModel.forecasts`."
        ),
    )
    actual_csv_path: str | None = Field(
        default=None,
        description=(
            "Optional CSV containing the ground-truth target column "
            "(e.g. preprocessed dataset). When supplied, an `actual_<target>` column is added. "
            "The target column name is taken from the first forecast entry's `target_column`."
        ),
    )
    datetime_column: str = Field(
        default="datetime", description="Join key for datetime."
    )
    horizon_column: str = Field(
        default="pred_sequence_id", description="Optional join key for horizon/id."
    )
    forecast_column: str = Field(
        default="final_forecast",
        description="Column in each forecast CSV that holds the model's prediction.",
    )
    include_diff: bool = Field(
        default=False,
        description="If true and an actual column is present, also emit `diff_<model_id>` columns.",
    )
    output_csv_name: str = Field(
        default="aggregated_forecast.csv",
        description="Filename of the aggregated CSV under `results_path`.",
    )

    @model_validator(mode="before")
    @classmethod
    def _unwrap_payload(cls, data):
        if isinstance(data, dict) and isinstance(data.get("payload"), dict):
            merged = dict(data["payload"])
            for key, value in data.items():
                if key != "payload":
                    merged[key] = value
            return merged
        return data

    def to_payload_dict(self) -> dict:
        return self.model_dump(exclude_none=True, exclude_defaults=True)

    def payload_as_dict(self) -> dict:
        return self.to_payload_dict()


class OutputModel(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    message: str = Field(description="Human-readable status message.")
    aggregated_csv_path: str | None = Field(
        default=None,
        description="Path to the produced aggregated CSV (None on no-op).",
    )
    model_ids: list[str] = Field(
        default_factory=list,
        description="Identifiers of models present as `pred_<id>` columns.",
    )
    n_rows: int = Field(default=0, description="Number of rows in the aggregated CSV.")
    artifacts: dict = Field(
        default_factory=dict,
        description="`input_payload` on no-op; otherwise `{columns, model_ids, n_rows}`.",
    )
