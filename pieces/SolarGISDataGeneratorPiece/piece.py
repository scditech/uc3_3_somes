import math
import csv
from datetime import datetime
from typing import Any
from pathlib import Path
import json

from domino.base_piece import BasePiece

from .models import (
    InputModel,
    OutputModel,
    TARGET_COLUMN,
    SOLARGIS_CSV_FIELDNAMES,
    OPEN_METEO_ARCHIVE_URL,
)

_OPEN_METEO_HOURLY_VARS = ",".join([
    "shortwave_radiation",
    "direct_normal_irradiance",
    "diffuse_radiation",
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "wind_gusts_10m",
    "wind_direction_10m",
    "surface_pressure",
])

_PERFORMANCE_RATIO = 0.75  # typical real-world PV performance ratio


def _solar_position(dt: datetime, lat: float, lon: float) -> tuple[float, float]:
    """Return (elevation_deg, azimuth_deg) using simplified solar geometry."""
    doy = dt.timetuple().tm_yday
    B = math.radians((360.0 / 365.0) * (doy - 81))
    decl = math.radians(23.45 * math.sin(B))
    # Equation of time correction in minutes
    eot = 9.87 * math.sin(2 * B) - 7.53 * math.cos(B) - 1.5 * math.sin(B)
    solar_noon = 12.0 - lon / 15.0 - eot / 60.0
    hour = dt.hour + dt.minute / 60.0
    ha = math.radians(15.0 * (hour - solar_noon))
    lat_r = math.radians(lat)

    sin_el = (
        math.sin(lat_r) * math.sin(decl)
        + math.cos(lat_r) * math.cos(decl) * math.cos(ha)
    )
    elevation = math.degrees(math.asin(max(-1.0, min(1.0, sin_el))))
    if elevation <= 0.0:
        return 0.0, 0.0

    cos_az = (math.sin(decl) * math.cos(lat_r) - math.cos(decl) * math.sin(lat_r) * math.cos(ha))
    cos_az /= math.cos(math.radians(elevation)) + 1e-10
    azimuth = math.degrees(math.acos(max(-1.0, min(1.0, cos_az))))
    if ha > 0:
        azimuth = 360.0 - azimuth

    return round(elevation, 2), round(azimuth, 2)


def _gti_from_ghi(ghi: float, panel_tilt: float) -> float:
    """Approximate Global Tilted Irradiance from GHI for a south-facing panel."""
    tilt_rad = math.radians(panel_tilt)
    return max(0.0, ghi * (1.0 + math.sin(tilt_rad) * 0.07))


def _fetch_open_meteo(
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
    timeout: int = 30,
) -> dict[str, Any]:
    
    import requests 

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": _OPEN_METEO_HOURLY_VARS,
        "wind_speed_unit": "ms",
        "timezone": "auto",
    }
    resp = requests.get(OPEN_METEO_ARCHIVE_URL, params=params, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def _build_records(
    hourly: dict[str, Any],
    lat: float,
    lon: float,
    pvout_peak_kw: float,
    panel_tilt: float,
) -> list[dict[str, Any]]:
    times = hourly.get("time", [])
    n = len(times)

    def _col(key: str) -> list:
        return hourly.get(key) or [None] * n

    ghi_col = _col("shortwave_radiation")
    dni_col = _col("direct_normal_irradiance")
    dif_col = _col("diffuse_radiation")
    temp_col = _col("temperature_2m")
    rh_col = _col("relative_humidity_2m")
    ws_col = _col("wind_speed_10m")
    wg_col = _col("wind_gusts_10m")
    wd_col = _col("wind_direction_10m")
    ap_col = _col("surface_pressure")

    records = []
    for i, ts_str in enumerate(times):
        dt = datetime.fromisoformat(ts_str)
        ghi = max(0.0, ghi_col[i] or 0.0)
        dni = max(0.0, dni_col[i] or 0.0)
        dif = max(0.0, dif_col[i] or 0.0)
        gti = _gti_from_ghi(ghi, panel_tilt)
        pvout = max(0.0, pvout_peak_kw * (ghi / 1000.0) * _PERFORMANCE_RATIO)
        se, sa = _solar_position(dt, lat, lon)

        records.append({
            "Date": dt.strftime("%d.%m.%Y"),
            "Time": dt.strftime("%H:%M"),
            "GHI": round(ghi, 2),
            "DNI": round(dni, 2),
            "DIF": round(dif, 2),
            "GTI": round(gti, 2),
            "SE": se,
            "SA": sa,
            "PVOUT": round(pvout, 3),
            "TEMP": round(temp_col[i] or 0.0, 2),
            "WS": round(ws_col[i] or 0.0, 2),
            "WG": round(wg_col[i] or 0.0, 2),
            "WD": round(wd_col[i] or 0.0, 2),
            "RH": round(rh_col[i] or 0.0, 2),
            "AP": round(ap_col[i] or 0.0, 2),
            "PVOUT_UNC_LOW": round(pvout * 0.92, 3),
            "PVOUT_UNC_HIGH": round(pvout * 1.08, 3),
        })
    return records


class SolarGISDataGeneratorPiece(BasePiece):
    def piece_function(self, input_data: InputModel):
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
        self.logger.info("Running SolarGISDataGeneratorPiece.")

        try:
            output_mode = str(payload.get("output_mode", "batch_sample")).strip().lower()
            if output_mode not in {"batch_sample", "realtime_stream"}:
                raise ValueError("output_mode must be `batch_sample` or `realtime_stream`.")

            output_format = str(payload.get("output_format", "json")).strip().lower()
            if output_format not in {"json", "csv"}:
                raise ValueError("output_format must be `json` or `csv`.")

            latitude = float(payload["latitude"])
            longitude = float(payload["longitude"])
            start_date = str(payload["start_date"])
            end_date = str(payload["end_date"])
            pvout_peak_kw = float(payload.get("pvout_peak_kw", 5.2))
            panel_tilt = float(payload.get("panel_tilt", 30.0))

            self.logger.info(
                "Fetching Open-Meteo data for lat=%.4f lon=%.4f from %s to %s",
                latitude, longitude, start_date, end_date,
            )
            response_json = _fetch_open_meteo(latitude, longitude, start_date, end_date)
            hourly = response_json.get("hourly", {})

            records = _build_records(hourly, latitude, longitude, pvout_peak_kw, panel_tilt)
            self.logger.info("Built %d records from API response.", len(records))

            if not records:
                self.logger.warning("No records returned for the requested date range.")
                return OutputModel(file_path=None)

            file_suffix = "stream" if output_mode == "realtime_stream" else "batch"
            file_name = f"solargis_{file_suffix}.{output_format}"
            file_path = str(Path(self.results_path) / file_name)

            if output_format == "json":
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(json.dumps(records, indent=4))
            else:
                with open(file_path, "w", encoding="utf-8", newline="") as csvfile:
                    writer = csv.DictWriter(
                        csvfile,
                        fieldnames=SOLARGIS_CSV_FIELDNAMES,
                        delimiter=";",
                        extrasaction="ignore",
                    )
                    writer.writeheader()
                    writer.writerows(records)

            self.logger.info("Dataset saved to %s", file_path)
            self.display_result = {"file_type": "txt", "file_path": file_path}

            return OutputModel(
                file_path=file_path,
                target_column=TARGET_COLUMN,
            )
        except Exception:
            self.logger.exception(
                "SolarGISDataGeneratorPiece failed. input_payload=%s", payload
            )
            raise
