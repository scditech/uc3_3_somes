"""SoMES Price / Tariff Forecast (module 6)."""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

import pandas as pd

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



class PriceForecastPiece(BasePiece):
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
        from pieces.common_somes.architecture import forecast_prices, tariff_scenarios

        out_dir = Path(self.results_path or ".")
        out_dir.mkdir(parents=True, exist_ok=True)
        log_path = out_dir / "price_forecast.log"

        def _log(msg: str) -> None:
            text = f"[PriceForecastPiece] {msg}"
            print(text, flush=True)
            with log_path.open("a", encoding="utf-8") as f:
                f.write(text + "\n")

        try:
            hist = pd.read_csv(input_data.prices_csv)
            if "datetime" not in hist.columns and "timestamp" in hist.columns:
                hist = hist.rename(columns={"timestamp": "datetime"})
            fc = forecast_prices(hist, horizon_steps=int(input_data.horizon_steps))
            out = out_dir / "price_tariff_forecast.csv"
            fc.to_csv(out, index=False)
            scenarios = tariff_scenarios(fc)
            scen_path = out_dir / "tariff_scenarios.csv"
            scenarios.to_csv(scen_path, index=False)
            _log(f"Wrote {len(fc)} forecast rows and {scenarios['tariff_scenario'].nunique()} tariff scenarios")
            _piece_out = OutputModel(
                message=f"Price forecast rows={len(fc)}",
                price_forecast_csv=str(out),
                tariff_scenarios_csv=str(scen_path),
            )
            if od is not None:
                if hasattr(_piece_out, 'run_id') and _run_id and not getattr(_piece_out, 'run_id', ''):
                    try:
                        _piece_out.run_id = _run_id
                    except Exception:
                        pass
                return od.finish_piece(
                    _piece_out, self.results_path, secrets_data, "PriceForecastPiece", _stage, run_id=_run_id
                )
            if _stage is not None:
                _stage.cleanup()
            return _piece_out
        except Exception as exc:
            (out_dir / "price_forecast_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
            _log(f"ERROR: {exc}")
            raise
