"""SoMES Flexible Load Schedule (module 10)."""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

import numpy as np
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



class FlexibleLoadSchedulePiece(BasePiece):
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
        from pieces.common_somes.architecture import schedule_flexible_loads
        from pieces.common_somes.connectors import load_flexible_loads

        out_dir = Path(self.results_path or ".")
        out_dir.mkdir(parents=True, exist_ok=True)
        log_path = out_dir / "flexible_load_schedule.log"

        def _log(msg: str) -> None:
            text = f"[FlexibleLoadSchedulePiece] {msg}"
            print(text, flush=True)
            with log_path.open("a", encoding="utf-8") as f:
                f.write(text + "\n")

        try:
            load_df = pd.read_csv(input_data.load_csv, parse_dates=["datetime"])
            loads = load_flexible_loads(Path(input_data.flexible_loads_json) if input_data.flexible_loads_json else None)
            if input_data.price_forecast_csv and Path(input_data.price_forecast_csv).is_file():
                prices = pd.read_csv(input_data.price_forecast_csv, parse_dates=["datetime"])
                col = "price_eur_kwh_forecast" if "price_eur_kwh_forecast" in prices.columns else "price_eur_kwh"
                merged = load_df.merge(prices[["datetime", col]], on="datetime", how="left")
                price = pd.to_numeric(merged[col], errors="coerce").fillna(0.12).to_numpy(dtype=float)
            else:
                price = np.full(len(load_df), 0.12, dtype=float)

            cfg = {}
            if input_data.scenario_yaml and Path(input_data.scenario_yaml).is_file():
                cfg = yaml.safe_load(Path(input_data.scenario_yaml).read_text(encoding="utf-8")) or {}
            dt_h = float((cfg.get("timestep_minutes") or 15)) / 60.0

            plan, activation = schedule_flexible_loads(load_df["datetime"], price, loads, dt_h=dt_h)
            plan_path = out_dir / "flexible_load_schedule.csv"
            act_path = out_dir / "flexible_load_activation_plan.csv"
            plan.to_csv(plan_path, index=False)
            activation.to_csv(act_path, index=False)
            meta = {"n_loads": len(loads), "n_activations": len(activation), "workflow_type": "SoMES"}
            (out_dir / "flexible_load_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
            _log(f"Scheduled {len(activation)} flexible loads")
            _piece_out = OutputModel(
                message=f"Flexible load schedule activations={len(activation)}",
                flexible_load_schedule_csv=str(plan_path),
                flexible_load_activation_csv=str(act_path),
            )
            if od is not None:
                if hasattr(_piece_out, 'run_id') and _run_id and not getattr(_piece_out, 'run_id', ''):
                    try:
                        _piece_out.run_id = _run_id
                    except Exception:
                        pass
                return od.finish_piece(
                    _piece_out, self.results_path, secrets_data, "FlexibleLoadSchedulePiece", _stage, run_id=_run_id
                )
            if _stage is not None:
                _stage.cleanup()
            return _piece_out
        except Exception as exc:
            (out_dir / "flexible_load_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
            _log(f"ERROR: {exc}")
            raise
