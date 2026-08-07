"""Forecast vs actual feedback loop (module 18) — load, PV and executed dispatch."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _align(forecast: pd.DataFrame, actual: pd.DataFrame, fcol: str, acol: str) -> pd.DataFrame:
    f = forecast[["datetime", fcol]].copy()
    a = actual[["datetime", acol]].copy()
    for frame in (f, a):
        frame["datetime"] = pd.to_datetime(frame["datetime"], errors="coerce")
    f = f.dropna(subset=["datetime"]).drop_duplicates(subset=["datetime"], keep="last")
    a = a.dropna(subset=["datetime"]).drop_duplicates(subset=["datetime"], keep="last")
    merged = f.merge(a, on="datetime", how="inner")
    merged[fcol] = pd.to_numeric(merged[fcol], errors="coerce")
    merged[acol] = pd.to_numeric(merged[acol], errors="coerce")
    return merged.dropna(subset=[fcol, acol])


def error_metrics(forecast: np.ndarray, actual: np.ndarray, *, capacity: float | None = None) -> dict[str, Any]:
    if len(forecast) == 0:
        return {"samples": 0}
    err = forecast - actual
    denom = np.where(np.abs(actual) > 1e-6, np.abs(actual), np.nan)
    mape = float(np.nanmean(np.abs(err) / denom) * 100.0) if np.isfinite(denom).any() else None
    scale = float(capacity) if capacity and capacity > 0 else float(np.max(np.abs(actual)) or 1.0)
    return {
        "samples": int(len(forecast)),
        "mae": round(float(np.mean(np.abs(err))), 4),
        "rmse": round(float(np.sqrt(np.mean(err**2))), 4),
        "bias": round(float(np.mean(err)), 4),
        "mape_pct": round(mape, 3) if mape is not None else None,
        "nrmse_pct": round(float(np.sqrt(np.mean(err**2)) / scale * 100.0), 3),
        "r2": round(
            float(1.0 - np.sum(err**2) / max(np.sum((actual - np.mean(actual)) ** 2), 1e-9)),
            4,
        ),
    }


def pv_forecast_vs_actual(
    forecast_csv: Path | str,
    actual_csv: Path | str,
    *,
    forecast_col: str = "pv_kw",
    actual_col: str = "pv_kw_measured",
) -> dict[str, Any]:
    fpath, apath = Path(forecast_csv), Path(actual_csv)
    if not fpath.is_file() or not apath.is_file():
        return {"available": False, "reason": "pv forecast or measurement file missing"}
    merged = _align(pd.read_csv(fpath), pd.read_csv(apath), forecast_col, actual_col)
    if merged.empty:
        return {"available": False, "reason": "no overlapping timestamps between PV forecast and measurement"}
    metrics = error_metrics(merged[forecast_col].to_numpy(float), merged[actual_col].to_numpy(float))
    daylight = merged.loc[merged[actual_col] > 0.05 * max(merged[actual_col].max(), 1e-9)]
    return {
        "available": True,
        "series": "pv_generation_kw",
        "overall": metrics,
        "daylight_only": error_metrics(
            daylight[forecast_col].to_numpy(float), daylight[actual_col].to_numpy(float)
        ),
        "retrain_recommended": bool(metrics.get("nrmse_pct", 0) > 15.0 or abs(metrics.get("bias", 0.0)) > 0.05 * max(merged[actual_col].mean(), 1e-9)),
    }


def dispatch_plan_vs_actual(
    planned_csv: Path | str,
    telemetry_csv: Path | str,
    *,
    dt_h: float = 0.25,
) -> dict[str, Any]:
    ppath, tpath = Path(planned_csv), Path(telemetry_csv)
    if not ppath.is_file() or not tpath.is_file():
        return {"available": False, "reason": "planned dispatch or BESS telemetry missing"}
    plan = pd.read_csv(ppath)
    tele = pd.read_csv(tpath)
    if "battery_kw" not in plan.columns or "power_kw" not in tele.columns:
        return {"available": False, "reason": "battery_kw / power_kw columns not found"}

    power = _align(plan, tele, "battery_kw", "power_kw")
    if power.empty:
        return {"available": False, "reason": "no overlapping timestamps between plan and telemetry"}
    power_metrics = error_metrics(power["battery_kw"].to_numpy(float), power["power_kw"].to_numpy(float))

    soc_metrics: dict[str, Any] = {"samples": 0}
    if "soc_pct" in plan.columns and "soc_pct" in tele.columns:
        soc = _align(plan.rename(columns={"soc_pct": "soc_plan"}), tele.rename(columns={"soc_pct": "soc_actual"}), "soc_plan", "soc_actual")
        if not soc.empty:
            soc_metrics = error_metrics(soc["soc_plan"].to_numpy(float), soc["soc_actual"].to_numpy(float))

    planned_energy = float(np.abs(power["battery_kw"]).sum() * dt_h)
    actual_energy = float(np.abs(power["power_kw"]).sum() * dt_h)
    follow = 1.0 - min(1.0, abs(planned_energy - actual_energy) / max(planned_energy, 1e-9))
    deviations = power.loc[(power["battery_kw"] - power["power_kw"]).abs() > 0.1 * max(np.abs(power["battery_kw"]).max(), 1e-9)]
    return {
        "available": True,
        "battery_power": power_metrics,
        "soc_tracking": soc_metrics,
        "planned_throughput_kwh": round(planned_energy, 3),
        "actual_throughput_kwh": round(actual_energy, 3),
        "schedule_following_ratio": round(float(follow), 4),
        "deviation_steps": int(len(deviations)),
        "worst_deviation_kw": round(float((power["battery_kw"] - power["power_kw"]).abs().max()), 3),
    }


def build_feedback_bundle(
    *,
    load_metrics: dict[str, Any] | None = None,
    pv: dict[str, Any] | None = None,
    dispatch: dict[str, Any] | None = None,
) -> dict[str, Any]:
    actions: list[str] = []
    if load_metrics and load_metrics.get("mape_pct") and load_metrics["mape_pct"] > 15.0:
        actions.append("load model: MAPE > 15 % — spustiť IncrementalTrainPiece")
    if pv and pv.get("retrain_recommended"):
        actions.append("PV model: nRMSE/bias mimo tolerancie — rekalibrovať PVOUT na namerané dáta")
    if dispatch and dispatch.get("available") and dispatch.get("schedule_following_ratio", 1.0) < 0.9:
        actions.append("dispatch: EMS nedodržal plán (<90 %) — overiť doručenie a lokálne blokovania")
    return {
        "format": "somes_forecast_vs_actual_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "load_forecast": load_metrics or {"available": False},
        "pv_forecast": pv or {"available": False},
        "dispatch_execution": dispatch or {"available": False},
        "recommended_actions": actions,
        "closed_loop_ok": not actions,
    }
