from __future__ import annotations

import json
import traceback
from pathlib import Path

import joblib
import numpy as np
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


def _safe_load_model(model_path_raw: str, registry_root_raw: str):
    root = Path(registry_root_raw).resolve()
    model_path = Path(model_path_raw).resolve()
    try:
        model_path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Model path outside registry root: {model_path}") from exc
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return joblib.load(model_path)


def _features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values("datetime").reset_index(drop=True).copy()
    out["hour"] = out["datetime"].dt.hour
    out["dayofweek"] = out["datetime"].dt.dayofweek
    out["month"] = out["datetime"].dt.month
    out["is_weekend"] = (out["dayofweek"] >= 5).astype(int)
    for lag in (1, 4, 96, 192):
        out[f"lag_{lag}"] = out["load_kw"].shift(lag)
    prev = out["load_kw"].shift(1)
    for w in (4, 16, 96):
        out[f"roll_mean_{w}"] = prev.rolling(w).mean()
        out[f"roll_std_{w}"] = prev.rolling(w).std(ddof=0)
    out["price_eur_kwh"] = pd.to_numeric(out.get("price_eur_kwh"), errors="coerce").interpolate(limit_direction="both")
    out["price_eur_kwh"] = out["price_eur_kwh"].fillna(0.1)
    return out.dropna().reset_index(drop=True)


def _fcols() -> list[str]:
    return [
        "hour",
        "dayofweek",
        "month",
        "is_weekend",
        "lag_1",
        "lag_4",
        "lag_96",
        "lag_192",
        "roll_mean_4",
        "roll_std_4",
        "roll_mean_16",
        "roll_std_16",
        "roll_mean_96",
        "roll_std_96",
        "price_eur_kwh",
    ]


def _psi(ref: np.ndarray, cur: np.ndarray, bins: int = 10) -> float:
    ref = np.asarray(ref, dtype=float)
    cur = np.asarray(cur, dtype=float)
    if ref.size < 10 or cur.size < 10:
        return 0.0
    edges = np.quantile(ref, np.linspace(0, 1, bins + 1))
    edges = np.unique(edges)
    if edges.size < 3:
        return 0.0
    ref_hist, _ = np.histogram(ref, bins=edges)
    cur_hist, _ = np.histogram(cur, bins=edges)
    ref_pct = np.clip(ref_hist / max(1, ref_hist.sum()), 1e-6, None)
    cur_pct = np.clip(cur_hist / max(1, cur_hist.sum()), 1e-6, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


class AnomalyAlertPiece(BasePiece):
    def piece_function(self, input_data: InputModel, secrets_data=None) -> OutputModel:
        _stage = None
        _piece_out = None
        _run_id = None
        if od is not None:
            input_data, _stage = od.stage_inputs(input_data, secrets_data)
            _run_id = od.resolve_run_id(input_data, secrets_data, generate=False)
        log_path = Path(self.results_path) / "anomaly_alert.log"
        err_path = Path(self.results_path) / "anomaly_alert_error.txt"
        try:
            hist = pd.read_csv(input_data.history_csv, parse_dates=["datetime"])
            models = json.loads(Path(input_data.models_index_json).read_text(encoding="utf-8"))

            alerts: list[dict] = []
            drift_rows: list[dict] = []
            for dept, model_path in models.items():
                g = hist[hist["department_id"].astype(str) == str(dept)].sort_values("datetime").reset_index(drop=True)
                work = _features(g).tail(int(input_data.lookback_rows))
                if work.empty:
                    continue
                resolved_model_path = str(Path(input_data.model_registry_dir) / Path(str(model_path)).name)
                model = _safe_load_model(resolved_model_path, input_data.model_registry_dir)
                expected = model.predict(work[_fcols()])
                actual = work["load_kw"].astype(float).values
                resid = actual - expected
                abs_resid = np.abs(resid)
                med = float(np.median(resid))
                mad = float(np.median(np.abs(resid - med))) + 1e-9
                z = 0.6745 * (resid - med) / mad
                split = max(10, len(work) // 2)
                ref = work.iloc[:split]
                cur = work.iloc[split:]
                ref_pred = expected[:split]
                cur_pred = expected[split:]
                ref_resid = ref["load_kw"].astype(float).values - ref_pred
                cur_resid = cur["load_kw"].astype(float).values - cur_pred
                psi_load = _psi(ref["load_kw"].astype(float).values, cur["load_kw"].astype(float).values)
                psi_resid = _psi(ref_resid, cur_resid)
                resid_mean_shift = float(abs(np.mean(cur_resid) - np.mean(ref_resid)))
                if psi_load >= 0.25 or psi_resid >= 0.25 or resid_mean_shift >= float(input_data.critical_kw_threshold):
                    drift_state = "critical"
                elif psi_load >= 0.10 or psi_resid >= 0.10 or resid_mean_shift >= float(input_data.warn_kw_threshold):
                    drift_state = "warning"
                else:
                    drift_state = "ok"
                drift_rows.append(
                    {
                        "department_id": str(dept),
                        "psi_load": round(float(psi_load), 4),
                        "psi_residual": round(float(psi_resid), 4),
                        "residual_mean_shift_kw": round(resid_mean_shift, 4),
                        "state": drift_state,
                    }
                )
                last_alert_dt = None
                for i, zv in enumerate(z):
                    az = abs(float(zv))
                    if az < float(input_data.z_threshold):
                        continue
                    delta_kw = float(abs_resid[i])
                    if delta_kw < float(input_data.min_abs_delta_kw):
                        continue
                    ts = pd.to_datetime(work.iloc[i]["datetime"])
                    if last_alert_dt is not None:
                        if (ts - last_alert_dt).total_seconds() < int(input_data.cooldown_minutes) * 60:
                            continue
                    if delta_kw >= float(input_data.critical_kw_threshold) or az >= float(input_data.z_threshold) * 2.2:
                        severity = "critical"
                    elif delta_kw >= float(input_data.warn_kw_threshold) or az >= float(input_data.z_threshold) * 1.5:
                        severity = "warning"
                    else:
                        severity = "info"
                    alerts.append(
                        {
                            "datetime": ts,
                            "department_id": str(dept),
                            "severity": severity,
                            "reason": "spike_up" if resid[i] > 0 else "drop_down",
                            "actual_kw": round(float(actual[i]), 3),
                            "expected_kw": round(float(expected[i]), 3),
                            "delta_kw": round(delta_kw, 3),
                            "robust_z": round(float(zv), 3),
                        }
                    )
                    last_alert_dt = ts

            out_dir = Path(self.results_path)
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / "anomaly_alerts.csv"
            drift_path = out_dir / "drift_report.json"
            _alert_cols = [
                "datetime", "department_id", "severity", "reason",
                "actual_kw", "expected_kw", "delta_kw", "robust_z",
            ]
            pd.DataFrame(alerts, columns=_alert_cols).to_csv(out_path, index=False)
            drift_summary = {
                "departments": drift_rows,
                "critical_count": int(sum(1 for r in drift_rows if r["state"] == "critical")),
                "warning_count": int(sum(1 for r in drift_rows if r["state"] == "warning")),
                "ok_count": int(sum(1 for r in drift_rows if r["state"] == "ok")),
            }
            drift_path.write_text(json.dumps(drift_summary, indent=2, ensure_ascii=False), encoding="utf-8")
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"[INFO] alerts_rows={len(alerts)}\n")
                f.write(
                    f"[INFO] drift_states: critical={drift_summary['critical_count']}, "
                    f"warning={drift_summary['warning_count']}, ok={drift_summary['ok_count']}\n"
                )
            _piece_out = OutputModel(
                message=f"Anomaly alerts generated, rows={len(alerts)}",
                alerts_csv=str(out_path),
                drift_report_json=str(drift_path),
            )
        except Exception:
            err = traceback.format_exc()
            with open(log_path, "a", encoding="utf-8") as f:
                f.write("[ERROR] AnomalyAlertPiece failed\n")
                f.write(err + "\n")
            with open(err_path, "w", encoding="utf-8") as f:
                f.write(err)
            raise
        finally:
            if od is not None and _piece_out is None:
                od.cleanup_on_error(self.results_path, secrets_data, "AnomalyAlertPiece", _stage, run_id=_run_id)
            elif _stage is not None:
                _stage.cleanup()
        if od is not None and _piece_out is not None:
            return od.finish_piece(_piece_out, self.results_path, secrets_data, "AnomalyAlertPiece", _stage, run_id=_run_id)
        return _piece_out
