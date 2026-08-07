from __future__ import annotations

import importlib
import json
from pathlib import Path
import sys
import traceback

import numpy as np
import pandas as pd
import yaml

try:
    from domino.base_piece import BasePiece
except ModuleNotFoundError:
    from local_compat.base_piece import BasePiece

from .models import InputModel, OutputModel


def _load_simulate_module():
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    return importlib.import_module("pieces.SimulatePiece.piece")


class BatterySimPiece(BasePiece):
    """SoMES battery charge/discharge plan + SOC trajectory for the optimisation horizon."""

    def piece_function(self, input_data: InputModel) -> OutputModel:
        csv_path = Path(input_data.load_csv)
        scenario_path = Path(input_data.scenario_yaml)
        solar_path = Path(input_data.virtual_solar_csv)
        out_dir = Path(self.results_path or scenario_path.parent)
        out_dir.mkdir(parents=True, exist_ok=True)
        log_path = out_dir / "battery_sim.log"
        repo_root = Path(__file__).resolve().parents[2]
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))

        def _log(msg: str) -> None:
            text = f"[BatterySimPiece] {msg}"
            print(text, flush=True)
            with log_path.open("a", encoding="utf-8") as f:
                f.write(text + "\n")

        _log(f"Input load_csv={csv_path}")
        _log(f"Input scenario_yaml={scenario_path}")
        _log(f"Input virtual_solar_csv={solar_path}")
        if not csv_path.is_file():
            raise FileNotFoundError(f"Load CSV not found: {csv_path}")
        if not scenario_path.is_file():
            raise FileNotFoundError(f"Scenario YAML not found: {scenario_path}")
        if not solar_path.is_file():
            raise FileNotFoundError(f"Virtual solar CSV not found: {solar_path}")

        try:
            from pieces.common_somes.architecture import (
                build_technical_validation_bundle,
                operational_kpis_from_dispatch,
                write_json,
            )
            from pieces.common_somes.connectors import load_grid_constraints, read_bess_state
            from pieces.common_somes.dispatch import plan_dispatch, plans_to_frames

            sim = _load_simulate_module()
            cfg = yaml.safe_load(scenario_path.read_text(encoding="utf-8")) or {}
            df = sim.load_consumption_csv(csv_path)
            solar_df = pd.read_csv(solar_path)
            if "pv_kw" not in solar_df.columns:
                raise ValueError("virtual_solar_csv must contain pv_kw column")

            dt_h = sim.infer_timestep_hours(df)
            price = sim.build_price_series(df, cfg).values.astype(float)
            if input_data.price_forecast_csv and Path(input_data.price_forecast_csv).is_file():
                pfc = pd.read_csv(input_data.price_forecast_csv, parse_dates=["datetime"])
                col = "price_eur_kwh_forecast" if "price_eur_kwh_forecast" in pfc.columns else "price_eur_kwh"
                merged = df[["datetime"]].merge(pfc[["datetime", col]], on="datetime", how="left")
                alt = pd.to_numeric(merged[col], errors="coerce")
                if alt.notna().any():
                    price = alt.fillna(pd.Series(price)).to_numpy(dtype=float)
                    _log("Using price forecast series")

            bat = cfg.get("battery") or {}
            charge_below = 0.08
            discharge_above = 0.15
            candidates = []
            if input_data.battery_strategy_json:
                candidates.append(Path(input_data.battery_strategy_json))
            candidates.extend(
                [
                    repo_root / "tests" / "BatteryStrategyOptimizerPiece_Outputs" / "battery_strategy_recommendation.json",
                    out_dir.parent / "BatteryStrategyOptimizerPiece_Outputs" / "battery_strategy_recommendation.json",
                ]
            )
            for candidate in candidates:
                if candidate.is_file():
                    rec = json.loads(candidate.read_text(encoding="utf-8"))
                    charge_below = float(rec.get("charge_below_eur_per_kwh", charge_below))
                    discharge_above = float(rec.get("discharge_above_eur_per_kwh", discharge_above))
                    _log(f"Loaded strategy thresholds from {candidate}")
                    break

            grid = load_grid_constraints(
                Path(input_data.grid_constraints_json) if input_data.grid_constraints_json else None,
                cfg,
            )
            import_limit = float(grid.get("import_limit_kw") or 0.0) or None
            export_limit = float(grid.get("export_limit_kw") or 0.0) or None

            load_kw = df["load_kw"].astype(float).values.copy()
            if input_data.flexible_load_schedule_csv and Path(input_data.flexible_load_schedule_csv).is_file():
                flex = pd.read_csv(input_data.flexible_load_schedule_csv, parse_dates=["datetime"])
                if "flexible_load_kw" in flex.columns:
                    flex = flex.drop_duplicates(subset=["datetime"], keep="last")
                    merged = df[["datetime"]].merge(flex[["datetime", "flexible_load_kw"]], on="datetime", how="left")
                    add = pd.to_numeric(merged["flexible_load_kw"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
                    if len(add) != len(load_kw):
                        add = np.resize(add, len(load_kw))
                    load_kw = load_kw + add
                    _log(f"Applied flexible load schedule, extra energy={float(add.sum() * dt_h):.2f} kWh")

            pv_kw = solar_df["pv_kw"].astype(float).values[: len(df)]
            if len(pv_kw) < len(df):
                pv_kw = np.pad(pv_kw, (0, len(df) - len(pv_kw)))
            elif len(pv_kw) > len(df):
                pv_kw = pv_kw[: len(df)]

            bess_state = read_bess_state(
                Path(input_data.bess_telemetry_csv) if input_data.bess_telemetry_csv else None,
                scenario_battery=bat,
            )
            state_path = write_json(out_dir / "bess_state.json", bess_state)
            energy_kwh = float(bess_state["energy_kwh"]) or float(bat.get("energy_kwh", bat.get("capacity_kWh", 0.0)))
            max_c_rate = float(bat.get("max_c_rate", 0.5))
            pmax = energy_kwh * max_c_rate
            max_charge_kw = min(pmax, float(bess_state["available_charge_kw"]) or pmax)
            max_discharge_kw = min(pmax, float(bess_state["available_discharge_kw"]) or pmax)
            _log(
                f"BESS state source={bess_state['source']} soc={bess_state['initial_soc_pct']:.1f}% "
                f"charge<= {max_charge_kw:.1f} kW discharge<= {max_discharge_kw:.1f} kW"
            )
            for warn in bess_state.get("warnings", []):
                _log(f"BESS warning: {warn}")

            result = plan_dispatch(
                method=str(input_data.dispatch_method or "auto"),
                load_kw=load_kw,
                pv_kw=pv_kw,
                price=price,
                dt_h=dt_h,
                energy_kwh=energy_kwh,
                max_c_rate=max_c_rate,
                max_charge_kw=max_charge_kw,
                max_discharge_kw=max_discharge_kw,
                eta_c=float(bat.get("charge_efficiency", 0.95)),
                eta_d=float(bat.get("discharge_efficiency", 0.95)),
                initial_soc_pct=float(bess_state["initial_soc_pct"]),
                soc_min_pct=float(bess_state["soc_min_pct"]),
                soc_max_pct=float(bess_state["soc_max_pct"]),
                charge_below=charge_below,
                discharge_above=discharge_above,
                import_limit_kw=import_limit,
                export_limit_kw=export_limit,
                export_price=float(input_data.export_price_eur_kwh),
                degradation_cost_eur_kwh=float(input_data.degradation_cost_eur_kwh),
                peak_price_eur_kw=float(input_data.peak_price_eur_kw),
                terminal_soc_pct=(
                    float(input_data.terminal_soc_pct) if float(input_data.terminal_soc_pct) >= 0 else None
                ),
            )
            method_used = str(result.get("method", "greedy_threshold"))
            if result.get("fallback_reason"):
                _log(f"Dispatch fell back to {method_used}: {result['fallback_reason']}")
            else:
                _log(f"Dispatch engine={method_used} objective={result.get('objective_eur')} EUR")
            frames = plans_to_frames(df["datetime"], result)
            soc_df = pd.DataFrame({"datetime": df["datetime"], "soc_pct": result["soc_pct"]})
            summary = pd.DataFrame(
                [
                    {
                        "energy_kwh": energy_kwh,
                        "max_c_rate": max_c_rate,
                        "mean_soc_pct": float(soc_df["soc_pct"].mean()),
                        "min_soc_pct": float(soc_df["soc_pct"].min()),
                        "max_soc_pct": float(soc_df["soc_pct"].max()),
                        "charge_kwh": float(frames["battery_charge_discharge_plan"]["charge_kw"].sum() * dt_h),
                        "discharge_kwh": float(frames["battery_charge_discharge_plan"]["discharge_kw"].sum() * dt_h),
                        "grid_import_kwh": float(result["grid_import_kw"].sum() * dt_h),
                        "grid_export_kwh": float(result["grid_export_kw"].sum() * dt_h),
                        "n_constraint_violations": int(result["n_violations"]),
                        "dispatch_method": method_used,
                        "objective_eur": result.get("objective_eur"),
                        "initial_soc_source": bess_state["source"],
                        "mode": "somes_operational_dispatch",
                    }
                ]
            )
            for name, frame in frames.items():
                frame.to_csv(out_dir / f"{name}.csv", index=False)

            dispatch = frames["next_day_dispatch_plan"]
            validation = build_technical_validation_bundle(
                dispatch,
                energy_kwh=energy_kwh,
                max_c_rate=max_c_rate,
                grid=grid,
            )
            validation["dispatch_runtime_violations"] = result["violations"]
            validation["dispatch_engine"] = {
                "method": method_used,
                "objective_eur": result.get("objective_eur"),
                "fallback_reason": result.get("fallback_reason"),
                "bess_state": bess_state,
            }
            tech_path = write_json(out_dir / "technical_validation.json", validation)
            forecast_accuracy = {}
            if input_data.forecast_accuracy_json and Path(input_data.forecast_accuracy_json).is_file():
                forecast_accuracy = json.loads(
                    Path(input_data.forecast_accuracy_json).read_text(encoding="utf-8")
                )
            kpis = operational_kpis_from_dispatch(
                dispatch,
                dt_h=dt_h,
                price=price,
                pv_kw=pv_kw,
                load_kw=load_kw,
                export_price_eur_kwh=float(input_data.export_price_eur_kwh),
                forecast_accuracy=forecast_accuracy,
            )
            kpi_path = write_json(out_dir / "operational_kpis.json", kpis)
            _log(f"Operational battery plan rows={len(soc_df)} approved={validation['overall_ok']}")
        except Exception as exc:
            (out_dir / "battery_sim_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
            _log(f"ERROR during battery operational plan: {exc}")
            raise

        out_soc = out_dir / "virtual_battery_soc.csv"
        out_sum = out_dir / "battery_summary.csv"
        soc_df.to_csv(out_soc, index=False)
        summary.to_csv(out_sum, index=False)
        _log(f"Wrote outputs: {out_soc}, {out_sum}")
        return OutputModel(
            message="SoMES battery charge/discharge plan finished",
            virtual_battery_soc_csv=str(out_soc),
            battery_summary_csv=str(out_sum),
            battery_charge_discharge_plan_csv=str(out_dir / "battery_charge_discharge_plan.csv"),
            grid_import_export_plan_csv=str(out_dir / "grid_import_export_plan.csv"),
            next_day_dispatch_plan_csv=str(out_dir / "next_day_dispatch_plan.csv"),
            technical_validation_json=str(tech_path),
            operational_kpis_json=str(kpi_path),
            dispatch_method_used=method_used,
            bess_state_json=str(state_path),
        )
