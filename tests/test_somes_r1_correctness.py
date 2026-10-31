"""Guards for the R1 correctness fixes vs SoMES 0.2.7 gaps."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from pieces.common_somes.architecture import operational_kpis_from_dispatch
from pieces.common_somes.connectors import generate_demo_connectors


def test_self_consumption_includes_pv_stored_in_battery(tmp_path: Path):
    n = 4
    plan = pd.DataFrame(
        {
            "datetime": pd.date_range("2026-10-26", periods=n, freq="15min"),
            "grid_import_kw": [0, 0, 0, 0],
            "grid_export_kw": [0, 0, 10, 0],
            "battery_kw": [-20, 0, 0, 10],
            "soc_pct": [50, 55, 55, 52],
        }
    )
    pv = np.array([40.0, 40.0, 40.0, 0.0])
    load = np.array([20.0, 40.0, 30.0, 10.0])
    kpis = operational_kpis_from_dispatch(
        plan, dt_h=0.25, pv_kw=pv, load_kw=load, energy_kwh=100.0
    )
    # PV energy = 30 kWh, export = 2.5 kWh → self-cons = (30-2.5)/30
    assert kpis["self_consumption_ratio"] == round(27.5 / 30.0, 4)
    assert kpis["battery_full_cycles"] == round(10.0 * 0.25 / 100.0, 4)


def test_no_demo_grid_when_fallback_off(tmp_path: Path):
    written = generate_demo_connectors(tmp_path, allow_demo=False, overwrite_existing=False)
    assert "grid_constraints" not in written
    assert "flexible_loads" not in written
    assert not (tmp_path / "grid_constraints.json").exists()


def test_demo_grid_when_fallback_on(tmp_path: Path):
    written = generate_demo_connectors(tmp_path, allow_demo=True, overwrite_existing=False)
    raw = json.loads(Path(written["grid_constraints"]).read_text(encoding="utf-8"))
    assert raw["import_limit_kw"] == 420.0


def test_load_grid_constraints_reads_nested_grid(tmp_path: Path):
    from pieces.common_somes.connectors import load_grid_constraints

    path = tmp_path / "grid_constraints.json"
    path.write_text(
        json.dumps(
            {
                "format": "somes_grid_constraints_v1",
                "grid": {
                    "export_limit_kw": 0.0,
                    "import_limit_kw": 500.0,
                    "connection_capacity_kw": 500.0,
                    "inverter_limit_kw": 400.0,
                },
            }
        ),
        encoding="utf-8",
    )
    grid = load_grid_constraints(path, {"grid": {"import_limit_kw": 420.0, "export_limit_kw": 350.0}})
    assert grid["import_limit_kw"] == 500.0
    assert grid["export_limit_kw"] == 0.0


def test_lp_dispatch_does_not_circulate_import_and_export():
    from pieces.common_somes.dispatch import lp_battery_dispatch

    n = 8
    result = lp_battery_dispatch(
        load_kw=np.full(n, 70.0),
        pv_kw=np.zeros(n),
        price=np.full(n, 0.04),
        dt_h=0.25,
        energy_kwh=400.0,
        max_charge_kw=200.0,
        max_discharge_kw=200.0,
        eta_c=0.95,
        eta_d=0.95,
        initial_soc_pct=50.0,
        import_limit_kw=420.0,
        export_limit_kw=350.0,
        export_price=0.05,
    )
    both = (result["grid_import_kw"] > 1e-3) & (result["grid_export_kw"] > 1e-3)
    assert not both.any()
