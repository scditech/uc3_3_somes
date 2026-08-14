from __future__ import annotations

import importlib
import json
import os
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


def _cloud_cover_series(weather_csv: str, index: pd.Series) -> np.ndarray | None:
    if not weather_csv or not Path(weather_csv).is_file():
        return None
    weather = pd.read_csv(weather_csv, parse_dates=["datetime"])
    if "cloud_cover_pct" not in weather.columns:
        return None
    weather = weather.drop_duplicates(subset=["datetime"], keep="last")
    merged = pd.DataFrame({"datetime": pd.to_datetime(index)}).merge(
        weather[["datetime", "cloud_cover_pct"]], on="datetime", how="left"
    )
    return pd.to_numeric(merged["cloud_cover_pct"], errors="coerce").interpolate(limit_direction="both").fillna(50.0).to_numpy(float)


def _production_uncertainty(
    pv_kw: np.ndarray,
    *,
    installed_kwp: float,
    model_mae: float,
    cloud_cover: np.ndarray | None,
) -> dict:
    """Band around the PV forecast: model error plus a cloud-driven variability term.

    Uncertainty scales with output because a clear-sky night has no forecast risk,
    while a partly clouded midday peak has the most.
    """
    pv_kw = np.asarray(pv_kw, dtype=float)
    n = len(pv_kw)
    capacity = max(float(installed_kwp), 1.0)
    base_sigma = max(float(model_mae), 0.01 * capacity)
    output_share = np.clip(pv_kw / capacity, 0.0, 1.0)

    if cloud_cover is not None and len(cloud_cover) == n:
        cover = np.clip(np.asarray(cloud_cover, dtype=float) / 100.0, 0.0, 1.0)
        # variability peaks at broken cloud (≈50 %), not at clear or fully overcast sky
        cloud_factor = 1.0 + 2.0 * (1.0 - np.abs(cover - 0.5) * 2.0)
    else:
        cloud_factor = np.full(n, 1.5)

    sigma = base_sigma + 0.12 * capacity * output_share * cloud_factor
    sigma = np.where(pv_kw <= 1e-6, 0.0, sigma)
    p10 = np.clip(pv_kw - 1.2816 * sigma, 0.0, None)
    p90 = np.clip(pv_kw + 1.2816 * sigma, 0.0, capacity * 1.2)
    daylight = pv_kw > 1e-6
    return {
        "sigma": np.round(sigma, 4),
        "p10": np.round(p10, 4),
        "p90": np.round(p90, 4),
        "summary": {
            "format": "somes_pv_uncertainty_v1",
            "method": "model_mae + cloud_variability_scaled_by_output",
            "installed_kwp": capacity,
            "model_mae_kw": round(float(model_mae), 4),
            "mean_sigma_kw": round(float(sigma[daylight].mean()) if daylight.any() else 0.0, 4),
            "max_sigma_kw": round(float(sigma.max()) if n else 0.0, 4),
            "mean_relative_band_pct": round(
                float((2 * 1.2816 * sigma[daylight] / np.maximum(pv_kw[daylight], 1e-6)).mean() * 100)
                if daylight.any()
                else 0.0,
                2,
            ),
            "daylight_steps": int(daylight.sum()),
            "cloud_cover_used": cloud_cover is not None,
        },
    }


class SolarSimPiece(BasePiece):
    """SoMES PV/RES forecast via UC3.4-style XGBoost PVOUT (fallback: synthetic)."""

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
        log_path = out_dir / "solar_sim.log"
        repo_root = Path(__file__).resolve().parents[2]
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))

        def _log(msg: str) -> None:
            text = f"[SolarSimPiece] {msg}"
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
            from pieces.common_somes.pvout_ai import ensure_pvout_model, predict_pvout_kw

            sim = _load_simulate_module()
            cfg = yaml.safe_load(scenario_path.read_text(encoding="utf-8")) or {}
            pv = cfg.get("pv") or {}
            installed_kwp = float(pv.get("installed_kwp", 0.0) or (cfg.get("solar") or {}).get("capacity_kWp", 0.0))
            yield_kwp = float(pv.get("yield_kwh_per_kwp_year", 1000.0))
            df = sim.load_consumption_csv(csv_path)

            # SolarGIS is optional and must be explicit. Never fall back to bundled
            # tests/SolarGIS files — ops sites (e.g. SAV/UMMS) use measured PV + weather.
            solargis_raw = (input_data.solargis_csv or os.environ.get("SOMES_SOLARGIS_CSV") or "").strip()
            solargis = Path(solargis_raw) if solargis_raw else None
            train_csv = repo_root / "seed_inputs" / "pvout_train.csv"
            model_dir = out_dir / "pvout_model"
            model_path, meta = ensure_pvout_model(
                model_dir=model_dir,
                installed_kwp=max(installed_kwp, 1.0),
                yield_kwh_per_kwp_year=yield_kwp,
                train_csv=train_csv if train_csv.is_file() else None,
                solargis_csv=solargis if (solargis is not None and solargis.is_file()) else None,
            )
            _log(
                f"PVOUT AI trained source={meta.get('train_source')} "
                f"mae={meta.get('train_mae'):.4f} rows={meta.get('n_train_rows')}"
            )
            pv_kw = predict_pvout_kw(df["datetime"], installed_kwp=max(installed_kwp, 1.0), model_path=model_path)
            mode = "uc3.4_xgb_pvout"
            if input_data.weather_forecast_csv and Path(input_data.weather_forecast_csv).is_file():
                weather = pd.read_csv(input_data.weather_forecast_csv, parse_dates=["datetime"])
                if "ghi_wm2" in weather.columns:
                    weather = weather.drop_duplicates(subset=["datetime"], keep="last")
                    merged = df[["datetime"]].merge(weather[["datetime", "ghi_wm2"]], on="datetime", how="left")
                    ghi = pd.to_numeric(merged["ghi_wm2"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
                    if len(ghi) != len(pv_kw):
                        ghi = np.resize(ghi, len(pv_kw))
                    weather_pv = (ghi / 1000.0) * max(installed_kwp, 0.0) * 0.85
                    blended = 0.55 * np.asarray(pv_kw, dtype=float) + 0.45 * weather_pv
                    pv_kw = pd.Series(blended, name="pv_kw")
                    mode = "uc3.4_xgb_pvout+weather_irradiance"
                    _log("Blended PVOUT AI with weather GHI forecast")
            if input_data.pv_production_csv and Path(input_data.pv_production_csv).is_file():
                meas = pd.read_csv(input_data.pv_production_csv, parse_dates=["datetime"])
                col = "pv_kw_measured" if "pv_kw_measured" in meas.columns else "pv_kw"
                if col in meas.columns:
                    meas = meas.drop_duplicates(subset=["datetime"], keep="last")
                    m = df[["datetime"]].merge(meas[["datetime", col]], on="datetime", how="left")
                    measured = pd.to_numeric(m[col], errors="coerce").to_numpy(dtype=float)
                    base = np.asarray(pv_kw, dtype=float)
                    if len(measured) != len(base):
                        measured = np.resize(measured, len(base))
                    mask = np.isfinite(measured) & (base > 1e-6)
                    if int(mask.sum()) > 10:
                        ratio = measured[mask] / base[mask]
                        scale = float(np.median(ratio))
                        scale = float(max(0.5, min(1.5, scale)))
                        pv_kw = pd.Series(base * scale, name="pv_kw")
                        mode = mode + "+measured_calibration"
                        _log(f"Calibrated PV forecast with measured production scale={scale:.3f}")
            if installed_kwp <= 0:
                pv_kw = pv_kw * 0.0
            base_pv = np.asarray(pv_kw, dtype=float)
            uncertainty = _production_uncertainty(
                base_pv,
                installed_kwp=installed_kwp,
                model_mae=float(meta.get("train_mae") or 0.0),
                cloud_cover=_cloud_cover_series(input_data.weather_forecast_csv, df["datetime"]),
            )
            out_df = pd.DataFrame(
                {
                    "datetime": df["datetime"],
                    "pv_kw": base_pv,
                    "pv_kw_p10": uncertainty["p10"],
                    "pv_kw_p90": uncertainty["p90"],
                    "pv_sigma_kw": uncertainty["sigma"],
                }
            )
            (out_dir / "pv_res_forecast_meta.json").write_text(
                json.dumps(
                    {
                        "mode": mode,
                        "installed_kwp": installed_kwp,
                        "model_meta": meta,
                        "weather_forecast_csv": input_data.weather_forecast_csv or None,
                        "pv_production_csv": input_data.pv_production_csv or None,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            (out_dir / "pv_forecast_uncertainty.json").write_text(
                json.dumps(uncertainty["summary"], indent=2, ensure_ascii=False), encoding="utf-8"
            )
            _log(
                f"Computed AI PV forecast rows={len(out_df)}, installed_kwp={installed_kwp}, "
                f"mean sigma={uncertainty['summary']['mean_sigma_kw']} kW"
            )
        except Exception as exc:
            (out_dir / "solar_sim_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
            _log(f"ERROR during PVOUT AI — falling back to synthetic: {exc}")
            sim = _load_simulate_module()
            cfg = yaml.safe_load(scenario_path.read_text(encoding="utf-8")) or {}
            pv = cfg.get("pv") or {}
            installed_kwp = float(pv.get("installed_kwp", 0.0))
            yield_kwp = float(pv.get("yield_kwh_per_kwp_year", 1000.0))
            df = sim.load_consumption_csv(csv_path)
            pv_ser = sim.synthetic_pv_kw(df["datetime"], installed_kwp, yield_kwh_per_kwp_year=yield_kwp)
            out_df = pd.DataFrame({"datetime": df["datetime"], "pv_kw": pv_ser})
            (out_dir / "pv_res_forecast_meta.json").write_text(
                json.dumps({"mode": "synthetic_fallback", "error": str(exc)}, indent=2),
                encoding="utf-8",
            )

        out_csv = out_dir / "virtual_solar.csv"
        out_df.to_csv(out_csv, index=False)
        uncertainty_path = out_dir / "pv_forecast_uncertainty.json"
        _log(f"Wrote output: {out_csv}")
        _piece_out = OutputModel(
            message="SoMES PV/RES forecast finished (UC3.4-style AI)",
            virtual_solar_csv=str(out_csv),
            pv_forecast_uncertainty_json=str(uncertainty_path) if uncertainty_path.is_file() else "",
        )
        if od is not None:
            if hasattr(_piece_out, 'run_id') and _run_id and not getattr(_piece_out, 'run_id', ''):
                try:
                    _piece_out.run_id = _run_id
                except Exception:
                    pass
            return od.finish_piece(
                _piece_out, self.results_path, secrets_data, "SolarSimPiece", _stage, run_id=_run_id
            )
        if _stage is not None:
            _stage.cleanup()
        return _piece_out
