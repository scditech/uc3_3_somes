from pydantic import BaseModel, ConfigDict, Field, model_validator


class ForecastBinding(BaseModel):
    """Structural mirror of `InferencePiece.OutputModel.forecasts[*]`.

    Lets a single upstream edge `ExplainablePrediction.forecasts ← Inference.forecasts`
    auto-populate per-model explanations without manual per-entry wiring.
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
            "`InferencePiece.OutputModel.forecasts`. Each entry's `model_path`, "
            "`data_path`, `feature_columns`, `target_column` drive the explanation "
            "run for that model."
        ),
    )
    explain: bool = Field(default=False, description="Enable explainability run.")
    explain_method: str | None = Field(
        default=None,
        description="`lime` or `shap`. Defaults to `shap` when `explain=True`.",
    )
    use_diagnostic_loss: bool = Field(
        default=False, description="Enable diagnostic heatmap artifacts."
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
    artifacts: dict = Field(
        default_factory=dict,
        description=(
            "`input_payload` on no-op. On run: aggregated `per_model = {model_id: {...}}` "
            "plus first-entry top-level keys for back-compat."
        ),
    )
