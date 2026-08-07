"""SoMES architecture helpers — technical validation, ops KPIs, EMS/BEMS, ops dashboard."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def validate_battery_plan(
    plan: pd.DataFrame,
    *,
    energy_kwh: float,
    max_c_rate: float,
    soc_min: float = 5.0,
    soc_max: float = 95.0,
) -> dict[str, Any]:
    violations: list[str] = []
    pmax = float(energy_kwh) * float(max_c_rate) if energy_kwh > 0 else 0.0
    if "battery_kw" in plan.columns:
        excess = plan.loc[plan["battery_kw"].abs() > pmax + 1e-6]
        for _, row in excess.head(20).iterrows():
            violations.append(f"battery_kw {row['battery_kw']:.2f} exceeds pmax {pmax:.2f} at {row.get('datetime')}")
    if "soc_pct" in plan.columns:
        bad = plan.loc[(plan["soc_pct"] < soc_min - 1e-6) | (plan["soc_pct"] > soc_max + 1e-6)]
        for _, row in bad.head(20).iterrows():
            violations.append(f"soc {row['soc_pct']:.2f} outside [{soc_min},{soc_max}] at {row.get('datetime')}")
    return {
        "module": "battery_constraints_check",
        "ok": len(violations) == 0,
        "pmax_kw": pmax,
        "soc_band_pct": [soc_min, soc_max],
        "violations": violations,
        "n_violations": len(violations),
    }


def corrected_operating_plan(
    plan: pd.DataFrame,
    *,
    energy_kwh: float,
    max_c_rate: float,
    grid: dict[str, Any],
    soc_min: float = 5.0,
    soc_max: float = 95.0,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Clip an infeasible schedule back inside the technical envelope.

    Returns the corrected schedule plus one recommendation record per corrected
    step, so an operator can see exactly which setpoint was changed and why.
    """
    out = plan.copy()
    recs: list[dict[str, Any]] = []
    pmax = float(energy_kwh) * float(max_c_rate) if energy_kwh > 0 else 0.0
    inverter = float(grid.get("inverter_limit_kw") or 0.0)
    connection = float(grid.get("connection_capacity_kw") or 0.0)
    import_limit = float(grid.get("import_limit_kw") or 0.0)
    export_limit = float(grid.get("export_limit_kw") or 0.0)

    def _clip(column: str, limit: float, reason: str, *, symmetric: bool = False) -> None:
        if column not in out.columns or limit <= 0:
            return
        original = out[column].to_numpy(dtype=float)
        clipped = np.clip(original, -limit, limit) if symmetric else np.clip(original, 0.0, limit)
        changed = np.abs(clipped - original) > 1e-6
        for pos in np.flatnonzero(changed):
            recs.append(
                {
                    "datetime": str(out["datetime"].iloc[pos]) if "datetime" in out.columns else str(pos),
                    "variable": column,
                    "original_kw": round(float(original[pos]), 3),
                    "corrected_kw": round(float(clipped[pos]), 3),
                    "limit_kw": round(limit, 3),
                    "reason": reason,
                }
            )
        out[column] = clipped

    _clip("battery_kw", pmax, "battery power limit (C-rate x usable energy)", symmetric=True)
    _clip("grid_export_kw", min(v for v in (inverter, export_limit, connection) if v > 0) if any(
        v > 0 for v in (inverter, export_limit, connection)
    ) else 0.0, "inverter / export / connection limit")
    _clip("grid_import_kw", min(v for v in (import_limit, connection) if v > 0) if any(
        v > 0 for v in (import_limit, connection)
    ) else 0.0, "import / connection limit")

    if "soc_pct" in out.columns:
        original = out["soc_pct"].to_numpy(dtype=float)
        clipped = np.clip(original, soc_min, soc_max)
        for pos in np.flatnonzero(np.abs(clipped - original) > 1e-6):
            recs.append(
                {
                    "datetime": str(out["datetime"].iloc[pos]) if "datetime" in out.columns else str(pos),
                    "variable": "soc_pct",
                    "original_kw": round(float(original[pos]), 3),
                    "corrected_kw": round(float(clipped[pos]), 3),
                    "limit_kw": soc_max,
                    "reason": f"SOC band [{soc_min}, {soc_max}] %",
                }
            )
        out["soc_pct"] = clipped

    return out, recs


def validate_inverter_connection(
    plan: pd.DataFrame,
    *,
    inverter_limit_kw: float,
    connection_capacity_kw: float,
) -> dict[str, Any]:
    violations: list[str] = []
    if "grid_export_kw" in plan.columns and inverter_limit_kw > 0:
        bad = plan.loc[plan["grid_export_kw"] > inverter_limit_kw + 1e-6]
        for _, row in bad.head(20).iterrows():
            violations.append(
                f"export {row['grid_export_kw']:.2f} > inverter {inverter_limit_kw:.2f} at {row.get('datetime')}"
            )
    if "grid_import_kw" in plan.columns and connection_capacity_kw > 0:
        bad = plan.loc[plan["grid_import_kw"] > connection_capacity_kw + 1e-6]
        for _, row in bad.head(20).iterrows():
            violations.append(
                f"import {row['grid_import_kw']:.2f} > connection {connection_capacity_kw:.2f} at {row.get('datetime')}"
            )
    util = 0.0
    if "grid_export_kw" in plan.columns and inverter_limit_kw > 0:
        util = float(plan["grid_export_kw"].max() / inverter_limit_kw)
    return {
        "module": "inverter_connection_limits_check",
        "ok": len(violations) == 0,
        "inverter_limit_kw": inverter_limit_kw,
        "connection_capacity_kw": connection_capacity_kw,
        "inverter_peak_utilization": round(util, 4),
        "violations": violations,
        "n_violations": len(violations),
    }


def validate_grid_feasibility(
    plan: pd.DataFrame,
    *,
    import_limit_kw: float,
    export_limit_kw: float,
    connection_capacity_kw: float,
) -> dict[str, Any]:
    violations: list[str] = []
    if "grid_import_kw" in plan.columns and import_limit_kw > 0:
        bad = plan.loc[plan["grid_import_kw"] > import_limit_kw + 1e-6]
        for _, row in bad.head(20).iterrows():
            violations.append(f"import {row['grid_import_kw']:.2f} > limit {import_limit_kw:.2f}")
    if "grid_export_kw" in plan.columns and export_limit_kw > 0:
        bad = plan.loc[plan["grid_export_kw"] > export_limit_kw + 1e-6]
        for _, row in bad.head(20).iterrows():
            violations.append(f"export {row['grid_export_kw']:.2f} > limit {export_limit_kw:.2f}")
    peak_flow = 0.0
    if "grid_import_kw" in plan.columns:
        peak_flow = max(peak_flow, float(plan["grid_import_kw"].max()))
    if "grid_export_kw" in plan.columns:
        peak_flow = max(peak_flow, float(plan["grid_export_kw"].max()))
    ok = len(violations) == 0 and peak_flow <= connection_capacity_kw + 1e-6
    if peak_flow > connection_capacity_kw + 1e-6:
        violations.append(f"peak grid flow {peak_flow:.2f} > connection capacity {connection_capacity_kw:.2f}")
    return {
        "module": "grid_feasibility_check",
        "ok": ok,
        "import_limit_kw": import_limit_kw,
        "export_limit_kw": export_limit_kw,
        "connection_capacity_kw": connection_capacity_kw,
        "peak_grid_flow_kw": round(peak_flow, 3),
        "approved_operational_schedule": ok,
        "violations": violations,
        "n_violations": len(violations),
    }


def estimate_pcc_voltage(
    plan: pd.DataFrame,
    *,
    short_circuit_power_mva: float,
    r_x_ratio: float = 0.5,
    power_factor: float = 1.0,
    nominal_voltage_pu: float = 1.0,
) -> np.ndarray:
    """Linearised voltage at the point of common coupling.

    Uses the standard connection-point approximation dU/Un = (P*R + Q*X) / Un^2
    with the equivalent impedance derived from the short-circuit power. It is not
    a load-flow of the downstream network, but it captures how the dispatch plan
    itself moves the voltage at the PCC.
    """
    n = len(plan)
    if short_circuit_power_mva <= 0:
        return np.full(n, nominal_voltage_pu, dtype=float)
    imp_mw = (plan["grid_import_kw"].to_numpy(dtype=float) if "grid_import_kw" in plan.columns else np.zeros(n)) / 1000.0
    exp_mw = (plan["grid_export_kw"].to_numpy(dtype=float) if "grid_export_kw" in plan.columns else np.zeros(n)) / 1000.0
    net_mw = imp_mw - exp_mw
    pf = float(np.clip(power_factor, 0.1, 1.0))
    tan_phi = float(np.sqrt(max(0.0, 1.0 - pf**2)) / pf)
    rx = float(max(r_x_ratio, 1e-6))
    scale = (rx + tan_phi) / (short_circuit_power_mva * float(np.sqrt(1.0 + rx**2)))
    return nominal_voltage_pu - net_mw * scale


def validate_voltage_band(
    plan: pd.DataFrame,
    *,
    short_circuit_power_mva: float,
    voltage_band_pu: list[float] | tuple[float, float] = (0.95, 1.05),
    r_x_ratio: float = 0.5,
    power_factor: float = 1.0,
) -> dict[str, Any]:
    v_min, v_max = float(voltage_band_pu[0]), float(voltage_band_pu[1])
    v_pu = estimate_pcc_voltage(
        plan,
        short_circuit_power_mva=short_circuit_power_mva,
        r_x_ratio=r_x_ratio,
        power_factor=power_factor,
    )
    violations: list[str] = []
    bad = np.flatnonzero((v_pu < v_min - 1e-9) | (v_pu > v_max + 1e-9))
    stamps = plan["datetime"].astype(str).tolist() if "datetime" in plan.columns else [str(i) for i in range(len(plan))]
    for i in bad[:20]:
        violations.append(f"voltage {v_pu[i]:.4f} pu outside [{v_min},{v_max}] at {stamps[int(i)]}")
    headroom = float(min(v_max - v_pu.max(), v_pu.min() - v_min)) if len(v_pu) else 0.0
    return {
        "module": "voltage_band_check",
        "ok": len(violations) == 0,
        "method": "linearised_pcc_estimate",
        "short_circuit_power_mva": short_circuit_power_mva,
        "voltage_band_pu": [v_min, v_max],
        "min_voltage_pu": round(float(v_pu.min()), 5) if len(v_pu) else None,
        "max_voltage_pu": round(float(v_pu.max()), 5) if len(v_pu) else None,
        "band_headroom_pu": round(headroom, 5),
        "violations": violations,
        "n_violations": len(violations),
    }


def build_technical_validation_bundle(
    plan: pd.DataFrame,
    *,
    energy_kwh: float,
    max_c_rate: float,
    grid: dict[str, Any],
) -> dict[str, Any]:
    battery = validate_battery_plan(plan, energy_kwh=energy_kwh, max_c_rate=max_c_rate)
    inverter = validate_inverter_connection(
        plan,
        inverter_limit_kw=float(grid.get("inverter_limit_kw") or 0.0),
        connection_capacity_kw=float(grid.get("connection_capacity_kw") or 0.0),
    )
    feasibility = validate_grid_feasibility(
        plan,
        import_limit_kw=float(grid.get("import_limit_kw") or 0.0),
        export_limit_kw=float(grid.get("export_limit_kw") or 0.0),
        connection_capacity_kw=float(grid.get("connection_capacity_kw") or 0.0),
    )
    voltage = validate_voltage_band(
        plan,
        short_circuit_power_mva=float(grid.get("short_circuit_power_mva") or 0.0),
        voltage_band_pu=grid.get("voltage_band_pu") or (0.95, 1.05),
        r_x_ratio=float(grid.get("grid_r_x_ratio") or 0.5),
        power_factor=float(grid.get("power_factor") or 1.0),
    )
    feasibility["voltage_check"] = voltage
    feasibility["ok"] = bool(feasibility["ok"] and voltage["ok"])
    feasibility["approved_operational_schedule"] = feasibility["ok"]
    if not voltage["ok"]:
        feasibility["violations"] = list(feasibility["violations"]) + voltage["violations"]
        feasibility["n_violations"] = len(feasibility["violations"])
    ok = bool(battery["ok"] and inverter["ok"] and feasibility["ok"])
    _, recommendations = corrected_operating_plan(plan, energy_kwh=energy_kwh, max_c_rate=max_c_rate, grid=grid)
    battery["corrected_operating_recommendations"] = [r for r in recommendations if r["variable"] in ("battery_kw", "soc_pct")]
    return {
        "format": "somes_technical_validation_v3",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "overall_ok": ok,
        "battery_constraints_check": battery,
        "inverter_connection_limits_check": inverter,
        "grid_feasibility_check": feasibility,
        "voltage_band_check": voltage,
        "approved_next_day_plan": ok,
        "corrected_operating_recommendations": recommendations,
        "n_corrections": len(recommendations),
    }


def schedule_flexible_loads(
    datetimes: pd.Series,
    price: np.ndarray,
    loads: list[dict[str, Any]],
    *,
    dt_h: float = 0.25,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Shift flexible loads into cheapest windows inside allowed time bands."""
    idx = pd.DatetimeIndex(pd.to_datetime(datetimes))
    n = len(idx)
    schedule_kw = np.zeros(n, dtype=float)
    rows: list[dict[str, Any]] = []
    for load in loads:
        flex_kw = float(load.get("flexible_kw") or 0.0)
        if flex_kw <= 0:
            continue
        duration_h = float(load.get("duration_hours") or 1.0)
        steps = max(1, int(round(duration_h / max(dt_h, 1e-9))))
        earliest = str(load.get("earliest_start") or "00:00")
        latest = str(load.get("latest_end") or "23:59")
        e_h, e_m = [int(x) for x in earliest.split(":")[:2]]
        l_h, l_m = [int(x) for x in latest.split(":")[:2]]
        e_min = e_h * 60 + e_m
        l_min = l_h * 60 + l_m
        mins = (idx.hour * 60 + idx.minute).astype(int)
        eligible = [i for i in range(n - steps + 1) if e_min <= int(mins[i]) <= l_min]
        if not eligible:
            eligible = list(range(max(1, n - steps + 1)))
        best_i = min(eligible, key=lambda i: float(np.mean(price[i : i + steps])))
        schedule_kw[best_i : best_i + steps] += flex_kw
        rows.append(
            {
                "load_id": load.get("load_id"),
                "name": load.get("name"),
                "flexible_kw": flex_kw,
                "start_datetime": str(idx[best_i]),
                "end_datetime": str(idx[min(best_i + steps - 1, n - 1)]),
                "duration_hours": duration_h,
                "mean_price_eur_kwh": float(np.mean(price[best_i : best_i + steps])),
                "priority": load.get("priority"),
            }
        )
    plan = pd.DataFrame({"datetime": idx, "flexible_load_kw": schedule_kw})
    return plan, pd.DataFrame(rows)


def forecast_prices(history: pd.DataFrame, horizon_steps: int) -> pd.DataFrame:
    """Tariff/price forecast: time-of-day profile plus low/base/high tariff scenarios."""
    df = history.copy()
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    price_col = None
    for c in ("price_eur_kwh", "price_eur_per_kwh", "price"):
        if c in df.columns:
            price_col = c
            break
    if price_col is None:
        raise ValueError("price history must contain price_eur_kwh")
    df = df.dropna(subset=["datetime"]).sort_values("datetime")
    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
    df = df.dropna(subset=[price_col])
    if df.empty:
        raise ValueError("empty price history")
    last_dt = df["datetime"].iloc[-1]
    step = df["datetime"].diff().median()
    if pd.isna(step) or step <= pd.Timedelta(0):
        step = pd.Timedelta(minutes=15)
    # typical day profile by time-of-day
    df["tod"] = df["datetime"].dt.strftime("%H:%M")
    profile = df.groupby("tod")[price_col].mean()
    spread = df.groupby("tod")[price_col].std(ddof=0).fillna(0.0)
    overall = float(df[price_col].tail(96).mean())
    overall_spread = float(df[price_col].std(ddof=0) or 0.0)
    rows = []
    for i in range(1, horizon_steps + 1):
        dt = last_dt + step * i
        tod = dt.strftime("%H:%M")
        base = float(profile.get(tod, overall))
        sigma = float(spread.get(tod, overall_spread))
        # 80 % interval from the historical dispersion at the same time of day
        rows.append(
            {
                "datetime": dt,
                "price_eur_kwh_forecast": round(base, 6),
                "price_low_eur_kwh": round(max(0.0, base - 1.2816 * sigma), 6),
                "price_high_eur_kwh": round(base + 1.2816 * sigma, 6),
                "price_sigma_eur_kwh": round(sigma, 6),
                "tariff_scenario": "base",
            }
        )
    return pd.DataFrame(rows)


def tariff_scenarios(forecast: pd.DataFrame) -> pd.DataFrame:
    """Expand the price forecast into explicit low/base/high tariff scenarios."""
    frames = []
    mapping = {
        "low": "price_low_eur_kwh",
        "base": "price_eur_kwh_forecast",
        "high": "price_high_eur_kwh",
    }
    for scenario, col in mapping.items():
        if col not in forecast.columns:
            continue
        frame = forecast[["datetime", col]].rename(columns={col: "price_eur_kwh"}).copy()
        frame["tariff_scenario"] = scenario
        frames.append(frame)
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return out.sort_values(["tariff_scenario", "datetime"]).reset_index(drop=True)


def operational_kpis_from_dispatch(
    plan: pd.DataFrame,
    *,
    dt_h: float,
    price: np.ndarray | None = None,
    pv_kw: np.ndarray | None = None,
    load_kw: np.ndarray | None = None,
    export_price_eur_kwh: float = 0.05,
    forecast_accuracy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    imp = plan["grid_import_kw"].to_numpy(dtype=float) if "grid_import_kw" in plan.columns else np.zeros(len(plan))
    exp = plan["grid_export_kw"].to_numpy(dtype=float) if "grid_export_kw" in plan.columns else np.zeros(len(plan))
    bat = plan["battery_kw"].to_numpy(dtype=float) if "battery_kw" in plan.columns else np.zeros(len(plan))
    soc = plan["soc_pct"].to_numpy(dtype=float) if "soc_pct" in plan.columns else np.zeros(len(plan))
    price_arr = price if price is not None else np.full(len(plan), 0.12)
    cost = float(np.sum(imp * price_arr * dt_h))
    revenue = float(np.sum(exp * export_price_eur_kwh * dt_h))
    pv_sum = float(np.sum(pv_kw) * dt_h) if pv_kw is not None else 0.0
    load_sum = float(np.sum(load_kw) * dt_h) if load_kw is not None else 0.0
    self_cons = 0.0
    if pv_sum > 1e-9 and load_kw is not None and pv_kw is not None:
        self_cons = float(np.minimum(load_kw, pv_kw).sum() * dt_h / pv_sum)

    savings = _cost_savings(
        optimized_cost=cost - revenue,
        price=price_arr,
        pv_kw=pv_kw,
        load_kw=load_kw,
        dt_h=dt_h,
        export_price_eur_kwh=export_price_eur_kwh,
    )

    return {
        "format": "somes_operational_kpis_v2",
        "grid_import_kwh": round(float(imp.sum() * dt_h), 3),
        "grid_export_kwh": round(float(exp.sum() * dt_h), 3),
        "peak_import_kw": round(float(imp.max()) if len(imp) else 0.0, 3),
        "peak_export_kw": round(float(exp.max()) if len(exp) else 0.0, 3),
        "battery_charge_kwh": round(float(np.clip(-bat, 0, None).sum() * dt_h), 3),
        "battery_discharge_kwh": round(float(np.clip(bat, 0, None).sum() * dt_h), 3),
        "mean_soc_pct": round(float(soc.mean()) if len(soc) else 0.0, 2),
        "operating_cost_eur_period": round(cost - revenue, 2),
        "renewable_generation_kwh": round(pv_sum, 3),
        "load_kwh": round(load_sum, 3),
        "self_consumption_ratio": round(self_cons, 4),
        "renewable_utilization_ratio": round(min(1.0, self_cons), 4),
        "battery_utilization_ratio": round(
            float(np.abs(bat).sum() * dt_h / max(float(np.abs(bat).max() * dt_h * len(bat)), 1e-9)), 4
        )
        if len(bat)
        else 0.0,
        **savings,
        "forecast_accuracy": forecast_accuracy or {},
    }


def _cost_savings(
    *,
    optimized_cost: float,
    price: np.ndarray,
    pv_kw: np.ndarray | None,
    load_kw: np.ndarray | None,
    dt_h: float,
    export_price_eur_kwh: float,
) -> dict[str, Any]:
    """Cost of the optimized schedule against two counterfactuals.

    `grid_only` is the site with no PV and no battery; `pv_no_battery` isolates
    how much of the saving comes from the battery and the dispatch optimizer
    rather than from the PV plant itself.
    """
    if load_kw is None:
        return {
            "baseline_grid_only_cost_eur": None,
            "baseline_pv_no_battery_cost_eur": None,
            "cost_savings_vs_grid_only_eur": None,
            "cost_savings_vs_grid_only_pct": None,
            "cost_savings_from_battery_eur": None,
        }

    load = np.asarray(load_kw, dtype=float)
    grid_only = float(np.sum(load * price * dt_h))

    if pv_kw is None:
        pv_no_batt = grid_only
    else:
        pv = np.asarray(pv_kw, dtype=float)
        net = load - pv
        pv_no_batt = float(
            np.sum(np.clip(net, 0, None) * price * dt_h)
            - np.sum(np.clip(-net, 0, None) * export_price_eur_kwh * dt_h)
        )

    return {
        "baseline_grid_only_cost_eur": round(grid_only, 2),
        "baseline_pv_no_battery_cost_eur": round(pv_no_batt, 2),
        "cost_savings_vs_grid_only_eur": round(grid_only - optimized_cost, 2),
        "cost_savings_vs_grid_only_pct": round((grid_only - optimized_cost) / grid_only * 100, 2)
        if abs(grid_only) > 1e-9
        else 0.0,
        "cost_savings_from_battery_eur": round(pv_no_batt - optimized_cost, 2),
    }


def build_ems_bems_payload(
    *,
    next_day_plan: pd.DataFrame,
    technical_validation: dict[str, Any],
    operational_kpis: dict[str, Any],
    flexible_schedule: pd.DataFrame | None = None,
) -> dict[str, Any]:
    instructions = []
    for _, row in next_day_plan.iterrows():
        instructions.append(
            {
                "datetime": str(row.get("datetime")),
                "battery_setpoint_kw": float(row.get("battery_kw", 0.0)),
                "soc_target_pct": float(row.get("soc_pct", 0.0)),
                "grid_import_kw": float(row.get("grid_import_kw", 0.0)),
                "grid_export_kw": float(row.get("grid_export_kw", 0.0)),
            }
        )
    flex = []
    if flexible_schedule is not None and not flexible_schedule.empty:
        flex = flexible_schedule.to_dict(orient="records")
    return {
        "format": "somes_ems_bems_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "approved": bool(technical_validation.get("approved_next_day_plan")),
        "dispatch_instructions": instructions,
        "flexible_load_activation_plan": flex,
        "operational_kpis": operational_kpis,
        "technical_validation_summary": {
            "overall_ok": technical_validation.get("overall_ok"),
            "modules": [
                technical_validation.get("battery_constraints_check", {}).get("module"),
                technical_validation.get("inverter_connection_limits_check", {}).get("module"),
                technical_validation.get("grid_feasibility_check", {}).get("module"),
            ],
        },
    }


def build_ops_dashboard(
    *,
    next_day_plan: pd.DataFrame,
    operational_kpis: dict[str, Any],
    technical_validation: dict[str, Any],
    alerts_summary: dict[str, Any] | None = None,
    forecast_vs_actual: dict[str, Any] | None = None,
    seed_appendix: dict[str, Any] | None = None,
    data_quality: dict[str, Any] | None = None,
) -> dict[str, Any]:
    plan = next_day_plan.copy()
    if len(plan) > 1200:
        step = max(1, len(plan) // 800)
        plan = plan.iloc[::step].copy()
    chart = {
        "title": "Next-day dispatch: battery + grid",
        "x": plan["datetime"].astype(str).tolist() if "datetime" in plan.columns else [],
        "series": [
            {
                "name": "Battery kW (+dis/-chg)",
                "unit": "kW",
                "values": pd.to_numeric(plan.get("battery_kw"), errors="coerce").fillna(0).round(3).tolist()
                if "battery_kw" in plan.columns
                else [],
            },
            {
                "name": "Grid import kW",
                "unit": "kW",
                "values": pd.to_numeric(plan.get("grid_import_kw"), errors="coerce").fillna(0).round(3).tolist()
                if "grid_import_kw" in plan.columns
                else [],
            },
            {
                "name": "Grid export kW",
                "unit": "kW",
                "values": pd.to_numeric(plan.get("grid_export_kw"), errors="coerce").fillna(0).round(3).tolist()
                if "grid_export_kw" in plan.columns
                else [],
            },
            {
                "name": "SOC %",
                "unit": "%",
                "values": pd.to_numeric(plan.get("soc_pct"), errors="coerce").fillna(0).round(2).tolist()
                if "soc_pct" in plan.columns
                else [],
            },
        ],
    }
    return {
        "format": "somes_ops_dashboard_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "workflow_type": "SoMES",
        "decision_kpis": {
            "operating_cost_eur_period": operational_kpis.get("operating_cost_eur_period"),
            "self_consumption_ratio": operational_kpis.get("self_consumption_ratio"),
            "renewable_utilization_ratio": operational_kpis.get("renewable_utilization_ratio"),
            "peak_import_kw": operational_kpis.get("peak_import_kw"),
            "grid_import_kwh": operational_kpis.get("grid_import_kwh"),
            "grid_export_kwh": operational_kpis.get("grid_export_kwh"),
            "battery_charge_kwh": operational_kpis.get("battery_charge_kwh"),
            "battery_discharge_kwh": operational_kpis.get("battery_discharge_kwh"),
            "mean_soc_pct": operational_kpis.get("mean_soc_pct"),
            "dispatch_approved": technical_validation.get("approved_next_day_plan"),
        },
        "dispatch_status": {
            "approved": technical_validation.get("approved_next_day_plan"),
            "overall_ok": technical_validation.get("overall_ok"),
        },
        "single_chart": chart,
        "alerts_and_drift": alerts_summary or {"summary": {"total": 0, "critical": 0, "warning": 0, "info": 0}},
        "forecast_vs_actual": forecast_vs_actual or {},
        "data_quality": data_quality or {},
        "technical_validation": technical_validation,
        "seed_appendix": seed_appendix or {"note": "Investment CAPEX/NPV/payback belongs to UC3.2 SEED."},
    }


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
