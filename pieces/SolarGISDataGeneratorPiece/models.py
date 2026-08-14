from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator
from typing import Optional

OUTPUT_FORMAT_ALIASES: dict[str, str] = {
    "json": "json",
    "csv": "csv",
}

TARGET_COLUMN = "PVOUT"

SOLARGIS_CSV_FIELDNAMES = [
    "Date",
    "Time",
    "GHI",
    "DNI",
    "DIF",
    "GTI",
    "SE",
    "SA",
    "PVOUT",
    "TEMP",
    "WS",
    "WG",
    "WD",
    "RH",
    "AP",
    "PVOUT_UNC_LOW",
    "PVOUT_UNC_HIGH",
]

# Open-Meteo archive endpoint — no API key required
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"


class InputModel(BaseModel):
    model_config = ConfigDict(extra="allow")

    latitude: float = Field(
        default=48.15,
        title="Latitude",
        description="Location latitude in decimal degrees (e.g. 48.15 for Bratislava).",
    )
    longitude: float = Field(
        default=17.11,
        title="Longitude",
        description="Location longitude in decimal degrees (e.g. 17.11 for Bratislava).",
    )
    start_date: str = Field(
        default="2026-01-01",
        title="Start Date",
        description="First day of the requested period in YYYY-MM-DD format.",
    )
    end_date: str = Field(
        default="2026-01-07",
        title="End Date",
        description="Last day of the requested period in YYYY-MM-DD format.",
    )
    pvout_peak_kw: float = Field(
        default=5.2,
        title="PV System Peak Power (kWp)",
        description="Installed peak power of the PV system used to estimate PVOUT.",
    )
    panel_tilt: float = Field(
        default=30.0,
        title="Panel Tilt (degrees)",
        description="Tilt of the PV panels in degrees from horizontal for GTI calculation.",
    )
    output_mode: str | None = Field(
        default="batch_sample",
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

        output_mode = data.get("output_mode")
        if output_mode is not None:
            data["output_mode"] = str(output_mode).strip().lower()

        output_format = data.get("output_format")
        if output_format is not None:
            normalized = str(output_format).strip().lower()
            data["output_format"] = OUTPUT_FORMAT_ALIASES.get(normalized, normalized)

        return data

    def to_payload_dict(self) -> dict:
        return self.model_dump(mode="json", exclude_none=True, exclude_unset=True)


class OutputModel(BaseModel):
    file_path: Optional[str] = Field(default=None, title="Dataset file path")
    target_column: Optional[str] = Field(
        default=None,
        title="Target column",
        description="Suggested target column for the generated dataset (pass downstream via fromUpstream).",
    )
