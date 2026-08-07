"""SoMES technical validation — battery, inverter/connection, grid feasibility (modules 11-13)."""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

import pandas as pd
import yaml

try:
    from domino.base_piece import BasePiece
except ModuleNotFoundError:
    from local_compat.base_piece import BasePiece

from .models import InputModel, OutputModel

try:
    from common import onedata_io as od
except ModuleNotFoundError:
    try:
        from pieces.common import onedata_io as od
    except ModuleNotFoundError:
        od = None



class GridFeasibilityPiece(BasePiece):
    """Modules 11–13: battery constraints, inverter/connection limits, grid feasibility."""

    def piece_function(self, input_data: InputModel, secrets_data=None) -> OutputModel:
        _stage = None
        _run_id = None
        if od is not None:
            input_data, _stage = od.stage_inputs(input_data, secrets_data)
            _run_id = od.resolve_run_id(
                input_data, secrets_data, generate=False, results_path=getattr(self, "results_path", None)
            )
            if hasattr(input_data, "run_id") and _run_id and not getattr(input_data, "run_id", ""):
                try:
                    input_data.run_id = _run_id
                except Exception:
                    pass
        repo_root = Path(__file__).resolve().parents[2]
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        from pieces.common_somes.architecture import (
            build_technical_validation_bundle,
            corrected_operating_plan,
            write_json,
        )
        from pieces.common_somes.connectors import load_grid_constraints

        out_dir = Path(self.results_path or ".")
        out_dir.mkdir(parents=True, exist_ok=True)
        log_path = out_dir / "grid_feasibility.log"

        def _log(msg: str) -> None:
            text = f"[GridFeasibilityPiece] {msg}"
            print(text, flush=True)
            with log_path.open("a", encoding="utf-8") as f:
                f.write(text + "\n")

        try:
            plan = pd.read_csv(input_data.next_day_dispatch_plan_csv, parse_dates=["datetime"])
            cfg = {}
            if input_data.scenario_yaml and Path(input_data.scenario_yaml).is_file():
                cfg = yaml.safe_load(Path(input_data.scenario_yaml).read_text(encoding="utf-8")) or {}
            bat = cfg.get("battery") or {}
            grid = load_grid_constraints(
                Path(input_data.grid_constraints_json) if input_data.grid_constraints_json else None,
                cfg,
            )
            energy_kwh = float(bat.get("energy_kwh", bat.get("capacity_kWh", 0.0)))
            max_c_rate = float(bat.get("max_c_rate", 0.5))
            bundle = build_technical_validation_bundle(
                plan,
                energy_kwh=energy_kwh,
                max_c_rate=max_c_rate,
                grid=grid,
            )
            out = write_json(out_dir / "technical_validation_bundle.json", bundle)

            corrected, recommendations = corrected_operating_plan(
                plan, energy_kwh=energy_kwh, max_c_rate=max_c_rate, grid=grid
            )
            corrected_path = out_dir / "corrected_operating_plan.csv"
            corrected.to_csv(corrected_path, index=False)
            rec_path = out_dir / "corrected_operating_recommendations.csv"
            pd.DataFrame(
                recommendations
                or [{"datetime": "", "variable": "", "original_kw": "", "corrected_kw": "", "limit_kw": "", "reason": "no correction required"}]
            ).to_csv(rec_path, index=False)

            # An operator must never receive a schedule that breaches a hard limit,
            # so the published plan is the corrected one when validation fails.
            approved = out_dir / "approved_next_day_operating_plan.csv"
            (plan if bundle["overall_ok"] else corrected).to_csv(approved, index=False)
            status = "APPROVED" if bundle["overall_ok"] else "REJECTED"
            _log(f"Validation {status}, corrections={len(recommendations)}")
            _piece_out = OutputModel(
                message=f"Technical validation {status}",
                technical_validation_json=str(out),
                approved_next_day_plan_csv=str(approved),
                feasibility_ok=bool(bundle["overall_ok"]),
                corrected_operating_plan_csv=str(corrected_path),
                corrected_operating_recommendations_csv=str(rec_path),
            )
            if od is not None:
                if hasattr(_piece_out, 'run_id') and _run_id and not getattr(_piece_out, 'run_id', ''):
                    try:
                        _piece_out.run_id = _run_id
                    except Exception:
                        pass
                return od.finish_piece(
                    _piece_out, self.results_path, secrets_data, "GridFeasibilityPiece", _stage, run_id=_run_id
                )
            if _stage is not None:
                _stage.cleanup()
            return _piece_out
        except Exception as exc:
            (out_dir / "grid_feasibility_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
            _log(f"ERROR: {exc}")
            raise
