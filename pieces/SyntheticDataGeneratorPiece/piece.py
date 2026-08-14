import math
import random
import csv
from datetime import datetime, timedelta, timezone
from typing import Any, Callable
from pathlib import Path
import json

import fsspec
from domino.base_piece import BasePiece
from common.utils import ensure_parent_dir, fsspec_config_init
from .models import DATASET_DEFAULT_TARGET, InputModel, OutputModel

RecordFactory = Callable[[datetime], dict[str, Any]]


class TimeSeriesDatasetSynthesizer:
    def __init__(
        self,
        factory: RecordFactory,
        start_at: datetime | None = None,
        step_minutes: int = 15,
    ) -> None:
        self._factory = factory
        self._current = start_at or datetime.now(tz=timezone.utc)
        self._step = timedelta(minutes=step_minutes)

    def next_sample(self) -> dict[str, Any]:
        record = self._factory(self._current)
        self._current += self._step
        return record


def _solargis_record(ts: datetime, tz_offset_hours: float = 1.0) -> dict[str, Any]:
    local_ts = ts + timedelta(hours=tz_offset_hours)
    hour = local_ts.hour + local_ts.minute / 60.0
    day_curve = max(math.sin(math.pi * (hour - 6) / 12), 0.0)

    ghi = max(0.0, 1000 * day_curve + random.uniform(-40, 40))
    dni = max(0.0, 920 * day_curve + random.uniform(-50, 50))
    dif = max(0.0, 250 * day_curve + random.uniform(-25, 25))
    gti = max(0.0, ghi * 1.07 + random.uniform(-20, 20))
    pvout = max(0.0, 5200 * day_curve * random.uniform(0.86, 0.97))
    pvout_kw = pvout / 1000.0
    return {
        "Date": local_ts.strftime("%d.%m.%Y"),
        "Time": local_ts.strftime("%H:%M"),
        "GHI": round(ghi, 2),
        "DNI": round(dni, 2),
        "DIF": round(dif, 2),
        "GTI": round(gti, 2),
        "SE": round(day_curve * 75, 2),
        "SA": round((hour / 24.0) * 360, 2),
        "PVOUT": round(pvout_kw, 3),
        "TEMP": round(random.uniform(-8, 36), 2),
        "WS": round(random.uniform(0, 15), 2),
        "WG": round(random.uniform(0, 22), 2),
        "WD": round(random.uniform(0, 360), 2),
        "RH": round(random.uniform(18, 95), 2),
        "AP": round(random.uniform(980, 1035), 2),
        "PVOUT_UNC_LOW": round(pvout_kw * random.uniform(0.88, 0.96), 3),
        "PVOUT_UNC_HIGH": round(pvout_kw * random.uniform(1.03, 1.14), 3),
    }


def _microstep_record(ts: datetime) -> dict[str, Any]:
    return {
        "timestamp_utc": ts.isoformat(),
        "station_id": "MS-001",
        "temperature_c": round(random.uniform(-12, 37), 2),
        "humidity_pct": round(random.uniform(20, 98), 2),
        "pressure_hpa": round(random.uniform(975, 1038), 2),
        "wind_speed_ms": round(random.uniform(0, 19), 2),
        "rain_rate_mmh": round(max(0.0, random.gauss(0.6, 1.2)), 2),
    }


def _shmu_record(ts: datetime) -> dict[str, Any]:
    return {
        "timestamp_utc": ts.isoformat(),
        "station_code": "SHMU-BA",
        "temp_2m_c": round(random.uniform(-15, 35), 2),
        "cloud_cover_okta": random.randint(0, 8),
        "snow_cm": round(max(0.0, random.gauss(2.0, 4.5)), 2),
        "precip_total_mm": round(max(0.0, random.gauss(0.9, 1.8)), 2),
        "wind_direction_deg": round(random.uniform(0, 360), 2),
    }


def _okte_record(ts: datetime, tz_offset_hours: float = 1.0) -> dict[str, Any]:
    local_ts = ts + timedelta(hours=tz_offset_hours)

    return {
        "Date": local_ts.strftime("%d.%m.%Y"),
        "Time": local_ts.strftime("%H:%M"),
        "market_area": "SK",
        "imbalance_mw": round(random.uniform(-280, 260), 3),
        "spot_price_eur_mwh": round(random.uniform(25, 220), 2),
        "scheduled_generation_mw": round(random.uniform(1200, 4200), 2),
        "actual_generation_mw": round(random.uniform(1200, 4200), 2),
    }


def _battery_record(ts: datetime) -> dict[str, Any]:
    soc = random.uniform(8, 98)
    return {
        "timestamp_utc": ts.isoformat(),
        "battery_id": "BATT-01",
        "soc_pct": round(soc, 2),
        "soh_pct": round(random.uniform(84, 100), 2),
        "voltage_v": round(random.uniform(320, 890), 2),
        "current_a": round(random.uniform(-210, 210), 2),
        "temperature_c": round(random.uniform(11, 43), 2),
        "cycles_count": random.randint(25, 12000),
    }


def _machine_record(ts: datetime) -> dict[str, Any]:
    vibration = max(0.0, random.gauss(1.8, 0.65))
    return {
        "timestamp_utc": ts.isoformat(),
        "machine_id": "MC-100",
        "state": random.choice(["idle", "running", "maintenance"]),
        "rpm": random.randint(0, 4800),
        "temperature_c": round(random.uniform(22, 96), 2),
        "vibration_mm_s": round(vibration, 3),
        "power_kw": round(random.uniform(0, 480), 2),
        "alarm_code": random.choice(["none", "A12", "B03", "C21"]),
    }


class SyntheticDataGeneratorPiece(BasePiece):
    DATASET_ALIASES = {
        "solargis": "solargis",
        "soalrgis": "solargis",
        "soalrgis dataset": "solargis",
        "solargis dataset": "solargis",
        "solar_gis": "solargis",
        "microstep": "microstep",
        "microstep meteorological data": "microstep",
        "microstep_meteorological_data": "microstep",
        "shmu": "shmu",
        "shmi": "shmu",
        "slovak hydrometeorological institute data": "shmu",
        "slovak_hydrometeorological_institute_data": "shmu",
        "okte": "okte",
        "dataset of battery parameters": "battery",
        "dataset_of_battery_parameters": "battery",
        "battery": "battery",
        "real time machine data": "machine",
        "real_time_machine_data": "machine",
        "machine": "machine",
    }

    def piece_function(self, input_data, secrets_data: Any):
        fsspec_config_init(secrets_data)
        payload = input_data.to_payload_dict()
        extra = getattr(input_data, "model_extra", None) or {}
        if not payload.get("output_format"):
            for key in (
                "output_format",
                "outputFormat",
                "Output format",
                "Output Format",
                "export_format",
                "file_format",
            ):
                val = extra.get(key)
                if val is not None and str(val).strip() != "":
                    payload["output_format"] = val
                    break
        self.logger.info("Running SyntheticDataGeneratorPiece.")

        try:
            if not payload.get("dataset_type"):
                self.logger.warning("No dataset_type provided. Returning empty output.")
                return OutputModel(
                    file_path=None,
                )

            raw_dataset_type = str(payload.get("dataset_type")).strip().lower()
            dataset_type = self.DATASET_ALIASES.get(raw_dataset_type)
            if dataset_type is None:
                raise ValueError(
                    "Invalid dataset_type. Choose one of: "
                    "SolarGIS Dataset, MicroStep Meteorological Data, "
                    "Slovak Hydrometeorological Institute Data, OKTE, "
                    "Dataset of Battery Parameters, Real Time Machine Data."
                )

            output_mode = (
                str(payload.get("output_mode", "batch_sample")).strip().lower()
            )
            if output_mode not in {"batch_sample", "realtime_stream"}:
                raise ValueError(
                    "output_mode must be `batch_sample` or `realtime_stream`."
                )
            output_format = str(payload.get("output_format", "json")).strip().lower()
            self.logger.info("Output format: %s", output_format)
            if output_format not in {"json", "csv"}:
                raise ValueError("output_format must be `json` or `csv`.")

            self.logger.info(
                "Setup of configuration completed with output_mode=%s and output_format=%s",
                output_mode,
                output_format,
            )

            records_count = int(payload.get("records_count", 20))
            if records_count <= 0:
                raise ValueError("records_count must be > 0")

            time_step_minutes = int(payload.get("time_step_minutes", 15))
            if time_step_minutes <= 0:
                raise ValueError("time_step_minutes must be > 0")

            interval_ms = int(payload.get("interval_ms", 1000))
            if interval_ms <= 0:
                raise ValueError("interval_ms must be > 0")

            seed = payload.get("seed")
            if seed is not None:
                random.seed(int(seed))

            start_at_value = payload.get("start_at")
            if start_at_value:
                start_at = datetime.fromisoformat(str(start_at_value))
                if start_at.tzinfo is None:
                    start_at = start_at.replace(tzinfo=timezone.utc)
            else:
                start_at = datetime.now(tz=timezone.utc)

            tz_offset_hours = float(payload.get("timezone_offset_hours", 1.0))

            def _factory(ts: datetime) -> dict[str, Any]:
                if dataset_type == "solargis":
                    return _solargis_record(ts, tz_offset_hours=tz_offset_hours)
                if dataset_type == "microstep":
                    return _microstep_record(ts)
                if dataset_type == "shmu":
                    return _shmu_record(ts)
                if dataset_type == "okte":
                    return _okte_record(ts, tz_offset_hours=tz_offset_hours)
                if dataset_type == "battery":
                    return _battery_record(ts)
                return _machine_record(ts)

            self.logger.info("Setup of configuration completed.")

            self.logger.info("Starting data generation.")
            synthesizer = TimeSeriesDatasetSynthesizer(
                factory=_factory, start_at=start_at, step_minutes=time_step_minutes
            )
            self.logger.info("Data generation completed.")

            records = [synthesizer.next_sample() for _ in range(records_count)]

            self.logger.info("Saving dataset to file.")
            file_suffix = "stream" if output_mode == "realtime_stream" else "batch"
            file_name = f"dataset_{file_suffix}.{output_format}"
            custom_output = payload.get("output_file_path")
            if custom_output:
                file_path = str(custom_output).strip()
            else:
                file_path = str(Path(self.results_path) / file_name)

            if custom_output:
                ensure_parent_dir(file_path)

            if output_format == "json":
                with fsspec.open(file_path, "w", encoding="utf-8") as f:
                    f.write(json.dumps(records, indent=4))
            else:
                if dataset_type == "solargis":
                    fieldnames = [
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
                else:
                    fieldnames = list(records[0].keys()) if records else []

                with fsspec.open(file_path, "w", encoding="utf-8", newline="") as csvfile:
                    writer = csv.DictWriter(
                        csvfile,
                        fieldnames=fieldnames,
                        delimiter=";",
                        extrasaction="ignore",
                    )
                    writer.writeheader()
                    writer.writerows(records)
            self.logger.info("Dataset saved to file at %s", file_path)

            # Display records in a Domino GUI
            self.display_result = {"file_type": "txt", "file_path": file_path}

            return OutputModel(
                file_path=file_path,
                target_column=DATASET_DEFAULT_TARGET.get(dataset_type),
            )
        except Exception:
            self.logger.exception(
                "SyntheticDataGeneratorPiece failed. " "input_payload=%s",
                payload,
            )
            raise
