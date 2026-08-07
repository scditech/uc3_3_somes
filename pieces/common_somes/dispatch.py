"""Operational dispatch helpers for SoMES (battery plan + grid import/export)."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

# Penalties that keep the LP feasible under any input while making the slacks
# economically unattractive, so the solver only uses them when physics forces it.
UNSERVED_PENALTY_EUR_KWH = 500.0
CURTAILMENT_PENALTY_EUR_KWH = 50.0


def lp_battery_dispatch(
    *,
    load_kw: np.ndarray,
    pv_kw: np.ndarray,
    price: np.ndarray,
    dt_h: float,
    energy_kwh: float,
    max_charge_kw: float,
    max_discharge_kw: float,
    eta_c: float,
    eta_d: float,
    initial_soc_pct: float,
    import_limit_kw: float | None = None,
    export_limit_kw: float | None = None,
    soc_min_pct: float = 5.0,
    soc_max_pct: float = 95.0,
    export_price: np.ndarray | float = 0.05,
    degradation_cost_eur_kwh: float = 0.01,
    peak_price_eur_kw: float = 0.0,
    terminal_soc_pct: float | None = None,
) -> dict[str, Any]:
    """Cost-optimal dispatch as a linear program solved with HiGHS.

    Decision variables per step: charge, discharge, grid import, grid export,
    plus unserved-load and curtailment slacks and one peak-import variable.
    The degradation cost makes simultaneous charge/discharge strictly worse than
    idling, so the complementarity condition holds without binary variables.
    """
    from scipy.optimize import linprog
    from scipy.sparse import csr_matrix, hstack, vstack

    n = int(len(load_kw))
    if n == 0 or energy_kwh <= 0:
        raise ValueError("LP dispatch needs a non-empty horizon and positive battery energy")

    load_kw = np.asarray(load_kw, dtype=float)
    pv_kw = np.asarray(pv_kw, dtype=float)
    price = np.asarray(price, dtype=float)
    exp_price = np.full(n, float(export_price)) if np.isscalar(export_price) else np.asarray(export_price, dtype=float)

    e0 = float(np.clip(initial_soc_pct, soc_min_pct, soc_max_pct)) / 100.0 * energy_kwh
    e_min = soc_min_pct / 100.0 * energy_kwh
    e_max = soc_max_pct / 100.0 * energy_kwh

    # x = [charge, discharge, import, export, unserved, curtail, peak]
    zeros = csr_matrix((n, n))
    eye = csr_matrix(np.eye(n))
    tril = csr_matrix(np.tril(np.ones((n, n))))
    col0 = csr_matrix((n, 1))

    cost = np.concatenate(
        [
            np.full(n, degradation_cost_eur_kwh * dt_h),
            np.full(n, degradation_cost_eur_kwh * dt_h),
            price * dt_h,
            -exp_price * dt_h,
            np.full(n, UNSERVED_PENALTY_EUR_KWH * dt_h),
            np.full(n, CURTAILMENT_PENALTY_EUR_KWH * dt_h),
            np.array([float(peak_price_eur_kw)]),
        ]
    )

    # Power balance: pv + discharge + import + unserved - charge - export - curtail = load
    a_eq = hstack([-eye, eye, eye, -eye, eye, -eye, col0], format="csr")
    b_eq = load_kw - pv_kw

    # Cumulative energy: E_t = e0 + dt * sum(eta_c*charge - discharge/eta_d)
    energy_gain = tril.multiply(dt_h * eta_c)
    energy_loss = tril.multiply(-dt_h / eta_d)
    soc_expr = hstack([energy_gain, energy_loss, zeros, zeros, zeros, zeros, col0], format="csr")

    blocks = [soc_expr, -soc_expr]
    rhs = [np.full(n, e_max - e0), np.full(n, e0 - e_min)]

    # Peak import coupling: import_t <= peak
    peak_block = hstack([zeros, zeros, eye, zeros, zeros, zeros, csr_matrix(-np.ones((n, 1)))], format="csr")
    blocks.append(peak_block)
    rhs.append(np.zeros(n))

    if terminal_soc_pct is not None:
        target = float(terminal_soc_pct) / 100.0 * energy_kwh
        last = csr_matrix(soc_expr.toarray()[-1:, :])
        blocks.append(-last)
        rhs.append(np.array([e0 - target]))

    a_ub = vstack(blocks, format="csr")
    b_ub = np.concatenate(rhs)

    big = float(max(np.max(load_kw), np.max(pv_kw), 1.0)) * 10.0
    bounds = (
        [(0.0, float(max_charge_kw))] * n
        + [(0.0, float(max_discharge_kw))] * n
        + [(0.0, float(import_limit_kw) if import_limit_kw else None)] * n
        + [(0.0, float(export_limit_kw) if export_limit_kw else None)] * n
        + [(0.0, big)] * n
        + [(0.0, big)] * n
        + [(0.0, None)]
    )

    res = linprog(cost, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"LP dispatch infeasible: {res.message}")

    x = res.x
    charge = np.clip(x[0:n], 0.0, None)
    discharge = np.clip(x[n : 2 * n], 0.0, None)
    grid_import = np.clip(x[2 * n : 3 * n], 0.0, None)
    grid_export = np.clip(x[3 * n : 4 * n], 0.0, None)
    unserved = np.clip(x[4 * n : 5 * n], 0.0, None)
    curtail = np.clip(x[5 * n : 6 * n], 0.0, None)

    energy = e0 + np.cumsum(dt_h * (eta_c * charge - discharge / eta_d))
    soc_trace = np.clip(energy / energy_kwh * 100.0, soc_min_pct, soc_max_pct)
    battery_kw = discharge - charge

    violations: list[str] = []
    for i in np.flatnonzero(unserved > 1e-6)[:20]:
        violations.append(f"t={int(i)}: unserved load {unserved[i]:.2f} kW (import limit binding)")
    for i in np.flatnonzero(curtail > 1e-6)[:20]:
        violations.append(f"t={int(i)}: PV curtailment {curtail[i]:.2f} kW (export limit binding)")

    return {
        "soc_pct": soc_trace,
        "battery_kw": battery_kw,
        "grid_import_kw": grid_import,
        "grid_export_kw": grid_export,
        "unserved_load_kw": unserved,
        "curtailed_pv_kw": curtail,
        "violations": violations[:50],
        "n_violations": len(violations),
        "method": "lp_highs",
        "objective_eur": round(float(res.fun), 4),
        "peak_import_kw": round(float(x[-1]), 3),
        "solver_status": res.message,
    }


def plan_dispatch(*, method: str = "auto", **kwargs: Any) -> dict[str, Any]:
    """Run the LP optimiser, degrading to the greedy heuristic when it is unavailable.

    The heuristic keeps the workflow runnable on images without scipy; the
    fallback reason is reported so operators know which engine produced the plan.
    """
    greedy_only = {"charge_below", "discharge_above"}
    lp_only = {"max_charge_kw", "max_discharge_kw", "export_price", "degradation_cost_eur_kwh",
               "peak_price_eur_kw", "terminal_soc_pct"}

    if method in ("auto", "lp"):
        lp_kwargs = {k: v for k, v in kwargs.items() if k not in greedy_only}
        lp_kwargs.setdefault("max_charge_kw", kwargs.get("max_charge_kw") or _pmax(kwargs))
        lp_kwargs.setdefault("max_discharge_kw", kwargs.get("max_discharge_kw") or _pmax(kwargs))
        lp_kwargs.pop("max_c_rate", None)
        try:
            result = lp_battery_dispatch(**lp_kwargs)
            result["fallback_reason"] = None
            return result
        except Exception as exc:  # noqa: BLE001 - any solver problem must degrade, not stop dispatch
            if method == "lp":
                raise
            reason = f"{type(exc).__name__}: {exc}"
    else:
        reason = "greedy explicitly requested"

    greedy_kwargs = {k: v for k, v in kwargs.items() if k not in lp_only}
    greedy_kwargs.setdefault("max_c_rate", kwargs.get("max_c_rate", 0.5))
    result = operational_battery_dispatch(**greedy_kwargs)
    result["method"] = "greedy_threshold"
    result["fallback_reason"] = reason
    return result


def _pmax(kwargs: dict[str, Any]) -> float:
    return float(kwargs.get("energy_kwh", 0.0)) * float(kwargs.get("max_c_rate", 0.5))


def operational_battery_dispatch(
    *,
    load_kw: np.ndarray,
    pv_kw: np.ndarray,
    price: np.ndarray,
    dt_h: float,
    energy_kwh: float,
    max_c_rate: float,
    eta_c: float,
    eta_d: float,
    initial_soc_pct: float,
    charge_below: float,
    discharge_above: float,
    import_limit_kw: float | None = None,
    export_limit_kw: float | None = None,
    soc_min_pct: float = 5.0,
    soc_max_pct: float = 95.0,
) -> dict[str, Any]:
    """
    Next-horizon operational schedule:
    - charge from PV excess (and cheap grid if price < charge_below),
    - discharge when net load positive and price >= discharge_above,
    - emit explicit grid import/export after battery action.
    """
    n = len(load_kw)
    pmax = float(energy_kwh) * float(max_c_rate) if energy_kwh > 0 else 0.0
    soc = float(np.clip(initial_soc_pct, soc_min_pct, soc_max_pct))
    soc_trace = np.zeros(n, dtype=float)
    bat_kw = np.zeros(n, dtype=float)  # + discharge to load, - charge
    grid_import = np.zeros(n, dtype=float)
    grid_export = np.zeros(n, dtype=float)
    violations: list[str] = []

    for i in range(n):
        load = float(load_kw[i])
        pv = float(pv_kw[i])
        pr = float(price[i])
        net = load - pv
        action = 0.0

        e_free = max(0.0, (soc_max_pct - soc) / 100.0 * energy_kwh)
        e_avail = max(0.0, (soc - soc_min_pct) / 100.0 * energy_kwh * eta_d)
        max_charge = min(pmax, e_free / max(dt_h * eta_c, 1e-9) if dt_h > 0 else 0.0)
        max_discharge = min(pmax, e_avail / max(dt_h, 1e-9) if dt_h > 0 else 0.0)

        if net < -1e-9:
            # PV excess → charge battery, export remainder
            charge = min(max_charge, -net)
            action = -charge
            leftover_export = -net - charge
            grid_export[i] = leftover_export
            grid_import[i] = 0.0
        elif net > 1e-9:
            if pr >= discharge_above and max_discharge > 0:
                discharge = min(max_discharge, net)
                action = discharge
                need = net - discharge
            else:
                need = net
            # Optional cheap-grid charge when load already covered? skip when net>0
            grid_import[i] = max(0.0, need)
            grid_export[i] = 0.0
            if pr < charge_below and max_charge > 0 and need <= 1e-9:
                # rare: already balanced; allow idle charge from grid
                pass
        else:
            if pr < charge_below and max_charge > 0:
                charge = min(max_charge, pmax * 0.25)
                action = -charge
                grid_import[i] = charge
            grid_export[i] = 0.0

        if import_limit_kw is not None and grid_import[i] > import_limit_kw + 1e-6:
            violations.append(f"t={i}: import {grid_import[i]:.2f} > limit {import_limit_kw:.2f}")
            grid_import[i] = float(import_limit_kw)
        if export_limit_kw is not None and grid_export[i] > export_limit_kw + 1e-6:
            violations.append(f"t={i}: export {grid_export[i]:.2f} > limit {export_limit_kw:.2f}")
            grid_export[i] = float(export_limit_kw)

        if action < 0:
            # charging
            soc += (-action) * dt_h * eta_c / max(energy_kwh, 1e-9) * 100.0
        elif action > 0:
            soc -= action * dt_h / max(eta_d * energy_kwh, 1e-9) * 100.0
        soc = float(np.clip(soc, soc_min_pct, soc_max_pct))
        soc_trace[i] = soc
        bat_kw[i] = action

    return {
        "soc_pct": soc_trace,
        "battery_kw": bat_kw,
        "grid_import_kw": grid_import,
        "grid_export_kw": grid_export,
        "violations": violations[:50],
        "n_violations": len(violations),
    }


def plans_to_frames(datetimes: pd.Series, result: dict[str, Any]) -> dict[str, pd.DataFrame]:
    base = pd.DataFrame(
        {
            "datetime": pd.to_datetime(datetimes),
            "battery_kw": result["battery_kw"],
            "soc_pct": result["soc_pct"],
            "grid_import_kw": result["grid_import_kw"],
            "grid_export_kw": result["grid_export_kw"],
        }
    )
    for optional in ("unserved_load_kw", "curtailed_pv_kw"):
        if optional in result:
            base[optional] = result[optional]
    battery_plan = base[["datetime", "battery_kw", "soc_pct"]].copy()
    battery_plan["charge_kw"] = np.clip(-battery_plan["battery_kw"], 0.0, None)
    battery_plan["discharge_kw"] = np.clip(battery_plan["battery_kw"], 0.0, None)
    grid_cols = ["datetime", "grid_import_kw", "grid_export_kw"]
    grid_cols += [c for c in ("unserved_load_kw", "curtailed_pv_kw") if c in base.columns]
    grid_plan = base[grid_cols].copy()
    dispatch = base.copy()
    return {
        "battery_charge_discharge_plan": battery_plan,
        "grid_import_export_plan": grid_plan,
        "next_day_dispatch_plan": dispatch,
    }
