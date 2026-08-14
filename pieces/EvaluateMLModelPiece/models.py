from pydantic import BaseModel, ConfigDict, Field, model_validator


class ForecastBinding(BaseModel):
    """Structural mirror of `InferencePiece.OutputModel.forecasts[*]`.

    Lets a single upstream edge `EvaluateMLModel.forecasts ← Inference.forecasts`
    auto-populate per-model evaluations without manual per-entry wiring.
    """

    model_config = ConfigDict(extra="allow", protected_namespaces=())

    model_id: str | None = Field(default=None)
    model_path: str | None = Field(default=None)
    mode: str | None = Field(default=None)
    forecast_csv_path: str | None = Field(default=None)
    data_path: str | None = Field(default=None)
    feature_columns: list[str] = Field(default_factory=list)
    target_column: str | None = Field(default=None)


class InputModel(BaseModel):
    model_config = ConfigDict(extra="allow", protected_namespaces=())

    forecasts: list[ForecastBinding] | None = Field(
        default=None,
        description=(
            "Per-model forecast entries — wire in one click from "
            "`InferencePiece.OutputModel.forecasts`. Drives evaluation: one "
            "metrics.json per forecast, with `pred_df_path` / `target_column` / "
            "`forecast_column` auto-derived from each entry (mode-aware: "
            "`correction` for `pvout_correction`, `final_forecast` otherwise)."
        ),
    )
    evaluation_option: str = Field(
        default="normal",
        description="Evaluation mode: `normal` or `errorcorrection`.",
    )
    baseline_id: int = Field(default=1, description="Baseline horizon id.")
    plot: bool = Field(default=False, description="Whether to generate plots/heatmaps.")

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
        # `exclude_unset` keeps explicitly-set defaults (e.g. evaluation_option="normal")
        # so the piece can distinguish "no input provided" from "user asked for the default".
        return self.model_dump(exclude_none=True, exclude_unset=True)

    def payload_as_dict(self) -> dict:
        return self.to_payload_dict()


class MetricsEntry(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_id: str = Field(description="Identifier for this evaluation entry.")
    evaluation_option: str = Field(description="Evaluation mode used.")
    metrics_path: str = Field(description="Path to this entry's metrics JSON.")
    metrics: dict = Field(
        default_factory=dict, description="Computed metrics for this entry."
    )


class OutputModel(BaseModel):
    message: str = Field(description="Human-readable status message.")
    metrics: list[MetricsEntry] = Field(
        default_factory=list,
        description="Per-entry metrics. Length matches the number of evaluations.",
    )
    artifacts: dict = Field(
        default_factory=dict,
        description=(
            "`input_payload` on no-op. On run: aggregated `per_model = {model_id: metrics}` "
            "plus first-entry `metrics`, `metrics_path`, `evaluation_option` at top level for back-compat."
        ),
    )
