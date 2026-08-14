from pydantic import BaseModel, ConfigDict, Field, model_validator


class InputModel(BaseModel):
    model_config = ConfigDict(extra="allow")

    preprocessing_option: str | None = Field(
        default=None,
        description="One of: `none`, `prediction`, `correction`.",
    )
    data_path: str | None = Field(
        default=None,
        description=(
            "Single-source input path (back-compat alias for `data_path_solargis`). "
            "Use the more specific `data_path_solargis` / `data_path_okte` fields below "
            "to merge Solargis + OKTE datasets into one preprocessed CSV."
        ),
    )
    data_path_solargis: str | None = Field(
        default=None,
        description=(
            "Path to Solargis-style CSV (weather / PVOUT). "
            "Wire upstream from `SyntheticDataGeneratorPiece.File Path`."
        ),
    )
    data_path_okte: str | None = Field(
        default=None,
        description=(
            "Path to OKTE-style CSV (electricity market). "
            "Wire upstream from `OKTEDataGeneratorPiece.File Path`. "
            "When both Solargis and OKTE paths are provided, the piece inner-joins "
            "them on `datetime` and emits one merged dataset with all features."
        ),
    )
    save_data_path: str | None = Field(
        default=None, description="Optional output path."
    )
    test_size: float | None = Field(default=None, description="Optional split size.")
    keep_datetime: bool | None = Field(
        default=None, description="Keep datetime column if supported."
    )
    target_column: str | None = Field(
        default=None,
        description="Target column for prediction mode. Defaults to `PVOUT` (SolarGIS). "
        "Set to e.g. `spot_price_eur_mwh` for OKTE data.",
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
    message: str = Field(description="Human-readable status message.")
    data_path: str | None = Field(
        default=None,
        description="Path to preprocessed CSV (consumable upstream → normalization / trainer / inference).",
    )
    feature_columns: list[str] = Field(
        default_factory=list,
        description="Resolved feature columns (consumable upstream → trainer / inference).",
    )
    target_column: str = Field(
        default="PVOUT",
        description="Target column used (consumable upstream → trainer).",
    )
    artifacts: dict = Field(
        default_factory=dict,
        description="Optional outputs (e.g., cleaned dataset URI, dropped rows stats).",
    )
