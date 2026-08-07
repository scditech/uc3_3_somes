"""SoMES data connectors — weather, PV production, BESS telemetry, grid, flexible loads."""
from __future__ import annotations

import io
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib import error as urlerror
from urllib import parse as urlparse
from urllib import request as urlrequest

import numpy as np
import pandas as pd

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
OKTE_DAM_URL = "https://isot.okte.sk/api/v1/dam/results"
SITE_TIMEZONE = "Europe/Bratislava"


CONNECTOR_FILES = {
    "weather_forecast": "weather_forecast.csv",
    "pv_production": "pv_production_measurements.csv",
    "bess_telemetry": "bess_telemetry.csv",
    "grid_constraints": "grid_constraints.json",
    "flexible_loads": "flexible_loads.json",
    "prices_history": "prices.csv",
}


DEMO_MARKER = ".demo_generated.json"


def ensure_connector_dir(root: Path) -> Path:
    d = Path(root)
    d.mkdir(parents=True, exist_ok=True)
    return d


def demo_generated_keys(root: Path) -> set[str]:
    """Keys whose file on disk is a synthetic fixture, not data from a real source."""
    marker = Path(root) / DEMO_MARKER
    if not marker.is_file():
        return set()
    try:
        return set(json.loads(marker.read_text(encoding="utf-8")).get("keys", []))
    except (json.JSONDecodeError, OSError):
        return set()


def mark_demo_generated(root: Path, keys: set[str]) -> None:
    marker = Path(root) / DEMO_MARKER
    existing = demo_generated_keys(root)
    payload = {"keys": sorted(existing | set(keys)), "updated_at_utc": datetime.now(timezone.utc).isoformat()}
    marker.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def clear_demo_mark(root: Path, key: str) -> None:
    remaining = demo_generated_keys(root) - {key}
    marker = Path(root) / DEMO_MARKER
    marker.write_text(
        json.dumps({"keys": sorted(remaining), "updated_at_utc": datetime.now(timezone.utc).isoformat()}, indent=2),
        encoding="utf-8",
    )


def _synth_index(n: int = 96, start: str = "2025-01-15 00:00:00", freq: str = "15min") -> pd.DatetimeIndex:
    return pd.date_range(start=start, periods=n, freq=freq)


def generate_demo_connectors(
    out_dir: Path,
    *,
    n: int = 96,
    seed: int = 42,
    datetimes: pd.Series | None = None,
    overwrite_existing: bool = True,
) -> dict[str, Path]:
    """Create demo connector artefacts if missing (audit-safe local fixtures).

    With ``overwrite_existing=False`` files already delivered by a real source
    are left untouched, so a production input folder is never clobbered.
    """
    out_dir = ensure_connector_dir(out_dir)
    synthesized: set[str] = set()
    rng = np.random.default_rng(seed)
    if datetimes is not None and len(datetimes) > 0:
        idx = pd.DatetimeIndex(pd.to_datetime(datetimes))
        n = len(idx)
    else:
        idx = _synth_index(n)
    hour = idx.hour + idx.minute / 60.0
    written: dict[str, Path] = {}

    weather_path = out_dir / CONNECTOR_FILES["weather_forecast"]
    # Refresh weather/PV/BESS when datetimes are provided so the fixtures align
    # with the horizon — unless the caller supplied real measurements.
    force = datetimes is not None and overwrite_existing
    if force or not weather_path.is_file():
        ghi = np.clip(900.0 * np.sin(np.pi * (hour - 6) / 12.0), 0.0, None)
        ghi = np.where((hour >= 6) & (hour <= 18), ghi, 0.0) * (0.85 + 0.15 * rng.random(n))
        pd.DataFrame(
            {
                "datetime": idx,
                "ghi_wm2": np.round(ghi, 2),
                "temp_c": np.round(5.0 + 10.0 * np.sin(2 * np.pi * hour / 24.0) + rng.normal(0, 0.4, n), 2),
                "wind_ms": np.round(2.0 + rng.random(n) * 3.0, 2),
                "cloud_cover_pct": np.round(np.clip(30 + rng.normal(0, 15, n), 0, 100), 1),
            }
        ).to_csv(weather_path, index=False)
        synthesized.add("weather_forecast")
    written["weather_forecast"] = weather_path

    pv_path = out_dir / CONNECTOR_FILES["pv_production"]
    if force or not pv_path.is_file():
        weather = pd.read_csv(weather_path, parse_dates=["datetime"])
        pv_kw = (weather["ghi_wm2"] / 1000.0) * 400.0 * 0.85
        pd.DataFrame({"datetime": weather["datetime"], "pv_kw_measured": np.round(pv_kw, 3)}).to_csv(
            pv_path, index=False
        )
        synthesized.add("pv_production")
    written["pv_production"] = pv_path

    bess_path = out_dir / CONNECTOR_FILES["bess_telemetry"]
    if force or not bess_path.is_file():
        soc = 50.0 + np.cumsum(rng.normal(0, 0.8, n))
        soc = np.clip(soc, 15.0, 95.0)
        pd.DataFrame(
            {
                "datetime": idx,
                "soc_pct": np.round(soc, 2),
                "power_kw": np.round(rng.normal(0, 40, n), 2),
                "available_charge_kw": 200.0,
                "available_discharge_kw": 200.0,
                "temp_c": np.round(22 + rng.normal(0, 1.5, n), 2),
            }
        ).to_csv(bess_path, index=False)
        synthesized.add("bess_telemetry")
    written["bess_telemetry"] = bess_path

    grid_path = out_dir / CONNECTOR_FILES["grid_constraints"]
    if not grid_path.is_file():
        grid_path.write_text(
            json.dumps(
                {
                    "format": "somes_grid_constraints_v1",
                    "import_limit_kw": 420.0,
                    "export_limit_kw": 350.0,
                    "inverter_limit_kw": 400.0,
                    "connection_capacity_kw": 500.0,
                    "voltage_band_pu": [0.95, 1.05],
                    "short_circuit_power_mva": 25.0,
                    "grid_r_x_ratio": 0.5,
                    "nominal_voltage_kv": 22.0,
                    "notes": "Demo industrial connection limits for SoMES technical validation.",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    written["grid_constraints"] = grid_path

    flex_path = out_dir / CONNECTOR_FILES["flexible_loads"]
    if not flex_path.is_file():
        flex_path.write_text(
            json.dumps(
                {
                    "format": "somes_flexible_loads_v1",
                    "loads": [
                        {
                            "load_id": "flex_hvac_shift",
                            "name": "HVAC peak shift",
                            "flexible_kw": 80.0,
                            "earliest_start": "06:00",
                            "latest_end": "22:00",
                            "duration_hours": 2.0,
                            "priority": 1,
                        },
                        {
                            "load_id": "flex_process_batch",
                            "name": "Process batch window",
                            "flexible_kw": 120.0,
                            "earliest_start": "10:00",
                            "latest_end": "18:00",
                            "duration_hours": 1.5,
                            "priority": 2,
                        },
                    ],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    written["flexible_loads"] = flex_path

    meta = {
        "format": "somes_connectors_manifest_v1",
        "files": {k: str(v.name) for k, v in written.items()},
        "categories": [
            "load_profiles",
            "pv_res_production_measurements",
            "bess_telemetry",
            "weather_forecasts",
            "electricity_prices",
            "grid_constraints",
            "flexible_load_definitions",
        ],
    }
    (out_dir / "connectors_manifest.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    if synthesized:
        mark_demo_generated(out_dir, synthesized)
    return written


def fetch_url_csv(url: str, *, timeout: int = 20) -> pd.DataFrame:
    """Pull a CSV endpoint (SCADA/historian export, OneData share, plain HTTP)."""
    req = urlrequest.Request(url, headers={"User-Agent": "somes-connector/1.0"})
    with urlrequest.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    df = pd.read_csv(io.StringIO(raw))
    if "datetime" not in df.columns:
        for alias in ("timestamp", "time", "date_time"):
            if alias in df.columns:
                df = df.rename(columns={alias: "datetime"})
                break
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    return df.dropna(subset=["datetime"]).sort_values("datetime")


def fetch_weather_open_meteo(
    *,
    latitude: float,
    longitude: float,
    forecast_days: int = 2,
    timeout: int = 20,
    freq: str = "15min",
) -> pd.DataFrame:
    """Live irradiance/temperature forecast (Open-Meteo, keyless) resampled to the dispatch grid."""
    query = urlparse.urlencode(
        {
            "latitude": f"{latitude:.4f}",
            "longitude": f"{longitude:.4f}",
            "hourly": "shortwave_radiation,temperature_2m,wind_speed_10m,cloud_cover",
            "forecast_days": int(forecast_days),
            "timezone": "UTC",
        }
    )
    req = urlrequest.Request(f"{OPEN_METEO_URL}?{query}", headers={"User-Agent": "somes-connector/1.0"})
    with urlrequest.urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    hourly = payload.get("hourly") or {}
    if not hourly.get("time"):
        raise ValueError("Open-Meteo returned no hourly data")
    df = pd.DataFrame(
        {
            "datetime": pd.to_datetime(hourly["time"]),
            "ghi_wm2": pd.to_numeric(hourly.get("shortwave_radiation"), errors="coerce"),
            "temp_c": pd.to_numeric(hourly.get("temperature_2m"), errors="coerce"),
            "wind_ms": pd.to_numeric(hourly.get("wind_speed_10m"), errors="coerce"),
            "cloud_cover_pct": pd.to_numeric(hourly.get("cloud_cover"), errors="coerce"),
        }
    )
    resampled = df.set_index("datetime").resample(freq).interpolate(limit_direction="both")
    return resampled.reset_index()


def fetch_prices_okte(
    *,
    lookback_days: int = 30,
    forward_days: int = 1,
    timeout: int = 30,
    tz: str = SITE_TIMEZONE,
) -> pd.DataFrame:
    """Day-ahead spot prices from the OKTE ISOT public API (keyless), EUR/MWh -> EUR/kWh.

    Rows are stamped with the *end* of each delivery period in local site time,
    which is the 15-min convention the rest of the pipeline uses. D+1 prices only
    exist after the day-ahead auction is published (~12:45 CET).
    """
    today = pd.Timestamp.now(tz=tz).date()
    day_from = today - timedelta(days=max(int(lookback_days), 0))
    day_to = today + timedelta(days=max(int(forward_days), 0))
    query = urlparse.urlencode(
        {"deliveryDayFrom": day_from.isoformat(), "deliveryDayTo": day_to.isoformat()}
    )
    req = urlrequest.Request(f"{OKTE_DAM_URL}?{query}", headers={"User-Agent": "somes-connector/1.0"})
    with urlrequest.urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    rows = payload.get("data") if isinstance(payload, dict) else payload
    if not rows:
        raise ValueError("OKTE returned no day-ahead results")
    raw = pd.DataFrame(rows)
    if "price" not in raw.columns or "deliveryEnd" not in raw.columns:
        raise ValueError("OKTE response is missing price/deliveryEnd fields")
    price_mwh = pd.to_numeric(raw["price"], errors="coerce")
    end = pd.to_datetime(raw["deliveryEnd"], utc=True, errors="coerce").dt.tz_convert(tz).dt.tz_localize(None)
    out = pd.DataFrame(
        {
            "datetime": end,
            "price_eur_per_kwh": (price_mwh / 1000.0).round(6),
            "price_eur_kwh": (price_mwh / 1000.0).round(6),
            "price_eur_mwh": price_mwh.round(3),
        }
    ).dropna(subset=["datetime", "price_eur_mwh"])
    if out.empty:
        raise ValueError("OKTE returned no published prices for the requested window")
    return out.drop_duplicates(subset=["datetime"], keep="last").sort_values("datetime").reset_index(drop=True)


def resolve_connector(
    key: str,
    *,
    out_dir: Path,
    url: str = "",
    local_path: Path | None = None,
    live_fetch=None,
    allow_demo: bool = True,
) -> dict[str, Any]:
    """Resolve one connector by priority: explicit file -> URL -> live API -> demo fixture.

    Returns provenance so the manifest can prove where each dataset came from.
    """
    target = Path(out_dir) / CONNECTOR_FILES[key]
    attempts: list[str] = []

    if local_path and Path(local_path).is_file():
        df = pd.read_csv(local_path)
        df.to_csv(target, index=False)
        clear_demo_mark(out_dir, key)
        return {"key": key, "path": target, "source": "file", "detail": str(local_path), "attempts": attempts}

    if url:
        try:
            df = fetch_url_csv(url)
            df.to_csv(target, index=False)
            clear_demo_mark(out_dir, key)
            return {"key": key, "path": target, "source": "url", "detail": url, "attempts": attempts}
        except (urlerror.URLError, OSError, ValueError) as exc:
            attempts.append(f"url failed: {exc}")

    if live_fetch is not None:
        try:
            df = live_fetch()
            df.to_csv(target, index=False)
            clear_demo_mark(out_dir, key)
            return {"key": key, "path": target, "source": "live_api", "detail": "provider pull", "attempts": attempts}
        except Exception as exc:  # noqa: BLE001 - provider errors must degrade, not crash ingestion
            attempts.append(f"live api failed: {exc}")

    if target.is_file() and key not in demo_generated_keys(out_dir):
        return {"key": key, "path": target, "source": "cached_file", "detail": str(target), "attempts": attempts}

    if not allow_demo:
        raise FileNotFoundError(f"Connector '{key}' has no source and demo fallback is disabled ({attempts})")
    return {"key": key, "path": target, "source": "demo", "detail": "synthetic fixture", "attempts": attempts}


def read_bess_state(path: Path | None, *, scenario_battery: dict[str, Any] | None = None) -> dict[str, Any]:
    """Latest BESS telemetry record -> dispatch initial state. Falls back to scenario defaults."""
    scenario_battery = scenario_battery or {}
    fallback_energy = float(scenario_battery.get("energy_kwh", scenario_battery.get("capacity_kWh", 0.0)) or 0.0)
    fallback_c_rate = float(scenario_battery.get("max_c_rate", 0.5) or 0.5)
    state = {
        "source": "scenario_yaml",
        "initial_soc_pct": float(scenario_battery.get("initial_soc_pct", 50.0) or 50.0),
        "energy_kwh": fallback_energy,
        "available_charge_kw": fallback_energy * fallback_c_rate,
        "available_discharge_kw": fallback_energy * fallback_c_rate,
        "soc_min_pct": float(scenario_battery.get("soc_min_pct", 5.0) or 5.0),
        "soc_max_pct": float(scenario_battery.get("soc_max_pct", 95.0) or 95.0),
        "telemetry_timestamp": None,
        "temp_c": None,
        "warnings": [],
    }
    if not path or not Path(path).is_file():
        state["warnings"].append("bess telemetry unavailable, using scenario defaults")
        return state

    df = pd.read_csv(path)
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        df = df.dropna(subset=["datetime"]).sort_values("datetime")
    if df.empty:
        state["warnings"].append("bess telemetry file empty, using scenario defaults")
        return state

    last = df.iloc[-1]
    state["source"] = str(path)
    state["telemetry_timestamp"] = str(last.get("datetime")) if "datetime" in df.columns else None

    soc = pd.to_numeric(pd.Series([last.get("soc_pct")]), errors="coerce").iloc[0]
    if pd.notna(soc):
        state["initial_soc_pct"] = float(np.clip(float(soc), 0.0, 100.0))
    else:
        state["warnings"].append("soc_pct missing in telemetry, using scenario default")

    for tele_col, key in (("available_charge_kw", "available_charge_kw"), ("available_discharge_kw", "available_discharge_kw")):
        val = pd.to_numeric(pd.Series([last.get(tele_col)]), errors="coerce").iloc[0]
        if pd.notna(val) and float(val) > 0:
            state[key] = float(val)

    cap = pd.to_numeric(pd.Series([last.get("capacity_kwh")]), errors="coerce").iloc[0]
    if pd.notna(cap) and float(cap) > 0:
        state["energy_kwh"] = float(cap)

    temp = pd.to_numeric(pd.Series([last.get("temp_c")]), errors="coerce").iloc[0]
    if pd.notna(temp):
        state["temp_c"] = float(temp)
        if float(temp) > 40.0:
            state["warnings"].append(f"BESS temperature {float(temp):.1f} °C — derating odporúčaný")

    if state["telemetry_timestamp"]:
        age_h = (pd.Timestamp.utcnow().tz_localize(None) - pd.Timestamp(state["telemetry_timestamp"])).total_seconds() / 3600.0
        state["telemetry_age_hours"] = round(float(age_h), 2)
        if age_h > 24.0:
            state["warnings"].append(f"telemetria je stará {age_h:.1f} h — overiť zber dát")
    return state


def load_grid_constraints(path: Path | None, scenario_cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    scenario_cfg = scenario_cfg or {}
    grid = scenario_cfg.get("grid") or {}
    mrk = scenario_cfg.get("mrk") or {}
    defaults = {
        "import_limit_kw": float(mrk.get("contract_kw") or grid.get("import_limit_kw") or 420.0),
        "export_limit_kw": float(grid.get("export_limit_kw") or 350.0),
        "inverter_limit_kw": float(
            grid.get("inverter_limit_kw") or (scenario_cfg.get("pv") or {}).get("installed_kwp") or 400.0
        ),
        "connection_capacity_kw": float(grid.get("connection_capacity_kw") or 500.0),
        "short_circuit_power_mva": float(grid.get("short_circuit_power_mva") or 25.0),
        "grid_r_x_ratio": float(grid.get("grid_r_x_ratio") or 0.5),
        "nominal_voltage_kv": float(grid.get("nominal_voltage_kv") or 22.0),
        "power_factor": float(grid.get("power_factor") or 1.0),
    }
    band = grid.get("voltage_band_pu") or [0.95, 1.05]
    if path and Path(path).is_file():
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        for k in defaults:
            if raw.get(k) is not None:
                defaults[k] = float(raw[k])
        if raw.get("voltage_band_pu"):
            band = raw["voltage_band_pu"]
        defaults["source"] = str(path)
    else:
        defaults["source"] = "scenario_fallback"
    defaults["voltage_band_pu"] = [float(band[0]), float(band[1])]
    return defaults


def load_flexible_loads(path: Path | None) -> list[dict[str, Any]]:
    if not path or not Path(path).is_file():
        return []
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    loads = raw.get("loads") if isinstance(raw, dict) else raw
    return loads if isinstance(loads, list) else []


def align_series_to_index(df: pd.DataFrame, index: pd.DatetimeIndex, value_col: str, fill: float = 0.0) -> np.ndarray:
    if df is None or df.empty or value_col not in df.columns:
        return np.full(len(index), fill, dtype=float)
    tmp = df.copy()
    tmp["datetime"] = pd.to_datetime(tmp["datetime"], errors="coerce")
    tmp = tmp.dropna(subset=["datetime"]).sort_values("datetime")
    tmp[value_col] = pd.to_numeric(tmp[value_col], errors="coerce").fillna(fill)
    s = tmp.set_index("datetime")[value_col].reindex(index).interpolate(limit_direction="both").fillna(fill)
    return s.to_numpy(dtype=float)
