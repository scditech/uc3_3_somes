from __future__ import annotations

import importlib
import json
from pathlib import Path
import sys
import traceback

import numpy as np
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



def _load_simulate_module():
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    return importlib.import_module("pieces.SimulatePiece.piece")


class BatteryStrategyOptimizerPiece(BasePiece):
    """Build simple price-driven strategy thresholds for battery operation."""

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
        csv_path = Path(input_data.load_csv)
        scenario_path = Path(input_data.scenario_yaml)
        out_dir = Path(self.results_path or scenario_path.parent)
        out_dir.mkdir(parents=True, exist_ok=True)
        log_path = out_dir / "battery_strategy_optimizer.log"

        def _log(msg: str) -> None:
            text = f"[BatteryStrategyOptimizerPiece] {msg}"
            print(text, flush=True)
            with log_path.open("a", encoding="utf-8") as f:
                f.write(text + "\n")

        _log(f"Input load_csv={csv_path}")
        _log(f"Input scenario_yaml={scenario_path}")
        if not csv_path.is_file():
            raise FileNotFoundError(f"Load CSV not found: {csv_path}")
        if not scenario_path.is_file():
            raise FileNotFoundError(f"Scenario YAML not found: {scenario_path}")

        try:
            sim = _load_simulate_module()
            cfg = yaml.safe_load(scenario_path.read_text(encoding="utf-8")) or {}
            df = sim.load_consumption_csv(csv_path)
            price = sim.build_price_series(df, cfg).values.astype(float)
            rec = {
                "charge_below_eur_per_kwh": round(float(np.quantile(price, 0.30)), 6),
                "discharge_above_eur_per_kwh": round(float(np.quantile(price, 0.75)), 6),
                "expensive_hour_threshold_eur_per_kwh": round(float(np.percentile(price, 70.0)), 6),
                "strategy_note": (
                    "SoMES operational thresholds for next-horizon battery dispatch "
                    "(charge on cheap/PV excess, discharge on expensive net load)."
                ),
                "workflow_type": "SoMES",
                "objective": "min_operational_cost_next_horizon",
            }
            _log(f"Computed thresholds from rows={len(df)}")
        except Exception as exc:
            (out_dir / "battery_strategy_optimizer_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
            _log(f"ERROR during strategy optimization: {exc}")
            raise

        out_json = out_dir / "battery_strategy_recommendation.json"
        out_json.write_text(json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8")
        _log(f"Wrote output: {out_json}")
        _piece_out = OutputModel(message="Battery strategy optimized", battery_strategy_recommendation_json=str(out_json))
        if od is not None:
            if hasattr(_piece_out, 'run_id') and _run_id and not getattr(_piece_out, 'run_id', ''):
                try:
                    _piece_out.run_id = _run_id
                except Exception:
                    pass
            return od.finish_piece(
                _piece_out, self.results_path, secrets_data, "BatteryStrategyOptimizerPiece", _stage, run_id=_run_id
            )
        if _stage is not None:
            _stage.cleanup()
        return _piece_out
