from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator
from typing import Optional
from common.models import SecretsModel

DATASET_TYPE_ALIASES: dict[str, str] = {
    "soalrgis": "solargis",
    "soalrgis dataset": "solargis",
    "solargis dataset": "solargis",
    "solar_gis": "solargis",
    "microstep meteorological data": "microstep",
    "microstep_meteorological_data": "microstep",
    "slovak hydrometeorological institute data": "shmu",
    "slovak_hydrometeorological_institute_data": "shmu",
    "dataset of battery parameters": "battery",
    "dataset_of_battery_parameters": "battery",
    "real time machine data": "machine",
    "real_time_machine_data": "machine",
    "shmi": "shmu",
}

OUTPUT_FORMAT_ALIASES: dict[str, str] = {
    "json": "json",
    "csv": "csv",
}


class InputModel(BaseModel):
    model_config = ConfigDict(extra="allow")

    dataset_type: str | None = Field(
        default=None,
        title="Dataset Type",
        description=(
            "One of: `solargis`, `microstep`, `shmu`, `okte`, `battery`, `machine` "
            "(aliases like `SoalrGIS Dataset` and `shmi` are also accepted)."
        ),
    )
    output_mode: str | None = Field(
        default=None,
        title="Output Mode",
        description="One of: `batch_sample`, `realtime_stream`.",
    )
    output_format: str | None = Field(
        default=None,
        title="Output Format",
        description="One of: `json`, `csv`.",
        validation_alias=AliasChoices(
            "output_format",
            "outputFormat",
            "Output format",
            "Output Format",
            "export_format",
            "file_format",
        ),
    )
    records_count: int = Field(
        default=20,
        description="Number of records to generate.",
    )
    time_step_minutes: int = Field(
        default=15,
        description="Time step between generated records in minutes.",
    )
    interval_ms: int = Field(
        default=1000,
        description="Realtime interval hint in milliseconds.",
    )
    start_at: str | None = Field(
        default=None,
        description="Optional start datetime in ISO format.",
    )
    seed: int | None = Field(
        default=None,
        description="Optional random seed for reproducible synthetic data.",
    )
    timezone_offset_hours: float = Field(
        default=1.0,
        description="Timezone offset used by SolarGIS-like records.",
    )
    output_file_path: str | None = Field(
        default=None,
        description="Optional full path for the generated file (e.g. onedata:///…/dataset.csv).",
    )

    @model_validator(mode="before")
    @classmethod
    def _unwrap_payload(cls, data):
        if not isinstance(data, dict):
            return data

        if isinstance(data.get("payload"), dict):
            merged = dict(data["payload"])
            for key, value in data.items():
                if key != "payload":
                    merged[key] = value
            data = merged

        # Domino / JSON-schema UIs may send the title-cased property key instead of snake_case.
        if not data.get("output_format"):
            for alias in (
                "outputFormat",
                "Output format",
                "Output Format",
                "export_format",
                "file_format",
            ):
                if alias in data and data.get(alias) is not None:
                    data["output_format"] = data.pop(alias)
                    break

        dataset_type = data.get("dataset_type")
        print(f"dataset_type: {dataset_type}")
        if dataset_type is not None:
            normalized = str(dataset_type).strip().lower()
            data["dataset_type"] = DATASET_TYPE_ALIASES.get(normalized, normalized)

        output_mode = data.get("output_mode")
        if output_mode is not None:
            data["output_mode"] = str(output_mode).strip().lower()

        output_format = data.get("output_format")
        if output_format is not None:
            normalized_format = str(output_format).strip().lower()
            data["output_format"] = OUTPUT_FORMAT_ALIASES.get(
                normalized_format, normalized_format
            )

        return data

    def to_payload_dict(self) -> dict:
        return self.model_dump(mode="json", exclude_none=True, exclude_unset=True)


DATASET_DEFAULT_TARGET: dict[str, str] = {
    "solargis": "PVOUT",
    "microstep": "temperature_c",
    "shmu": "temp_2m_c",
    "okte": "spot_price_eur_mwh",
    "battery": "soc_pct",
    "machine": "power_kw",
}


class OutputModel(BaseModel):
    file_path: Optional[str] = Field(default=None, title="Dataset file path")
    target_column: Optional[str] = Field(
        default=None,
        title="Target column",
        description="Suggested target column for the generated dataset (pass downstream via fromUpstream).",
    )
