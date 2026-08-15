from __future__ import annotations

import json
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



def _actual_load(predictions: pd.DataFrame, history_csv: str) -> pd.Series:
    """Realised load aligned to the prediction rows.

    Measured history wins over the ``load_kw`` column carried inside the
    predictions file: the forecast grid ships that column filled with zero
    placeholders, and scoring against those zeros produces meaningless errors.
    """
    from_file = pd.to_numeric(predictions.get("load_kw"), errors="coerce") if "load_kw" in predictions.columns else pd.Series(
        [pd.NA] * len(predictions), index=predictions.index, dtype="Float64"
    )
    if not history_csv or not Path(history_csv).is_file():
        return from_file.astype(float)

    hist = pd.read_csv(history_csv, parse_dates=["datetime"])
    if "load_kw" not in hist.columns:
        return from_file.astype(float)
    keys = ["datetime"]
    if "department_id" in predictions.columns and "department_id" in hist.columns:
        keys.append("department_id")
        hist["department_id"] = hist["department_id"].astype(str)
    left = predictions[keys].copy()
    if "department_id" in keys:
        left["department_id"] = left["department_id"].astype(str)
    merged = left.merge(
        hist[keys + ["load_kw"]].drop_duplicates(subset=keys), on=keys, how="left"
    )
    measured = pd.to_numeric(merged["load_kw"], errors="coerce")
    measured.index = predictions.index
    return measured.where(measured.notna(), from_file).astype(float)


def _retraining_dataset(
    predictions: pd.DataFrame, pred: pd.Series, actual: pd.Series, history_csv: str
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Assemble the labelled dataset the next incremental training run consumes.

    Rows carry the realised load next to the prediction that was made for it,
    plus calendar features, so retraining does not have to re-derive the join.
    """
    frame = predictions.copy()
    frame["prediction_load_kw"] = pred.to_numpy()

    if "price_eur_kwh" not in frame.columns and history_csv and Path(history_csv).is_file():
        hist = pd.read_csv(history_csv, parse_dates=["datetime"])
        keys = ["datetime"] + (["department_id"] if "department_id" in frame.columns and "department_id" in hist.columns else [])
        cols = keys + [c for c in ("price_eur_kwh",) if c in hist.columns]
        if len(cols) > len(keys):
            if "department_id" in keys:
                hist["department_id"] = hist["department_id"].astype(str)
                frame["department_id"] = frame["department_id"].astype(str)
            frame = frame.merge(hist[cols].drop_duplicates(subset=keys), on=keys, how="left")

    frame["actual_load_kw"] = pd.to_numeric(actual, errors="coerce").to_numpy()
    frame["residual_kw"] = frame["actual_load_kw"] - frame["prediction_load_kw"]
    frame["hour"] = frame["datetime"].dt.hour
    frame["dayofweek"] = frame["datetime"].dt.dayofweek
    frame["month"] = frame["datetime"].dt.month
    frame["is_weekend"] = (frame["dayofweek"] >= 5).astype(int)
    frame["label_available"] = frame["actual_load_kw"].notna().astype(int)

    keep = [
        c
        for c in (
            "datetime",
            "department_id",
            "actual_load_kw",
            "prediction_load_kw",
            "residual_kw",
            "price_eur_kwh",
            "hour",
            "dayofweek",
            "month",
            "is_weekend",
            "label_available",
        )
        if c in frame.columns
    ]
    out = frame[keep].sort_values("datetime").reset_index(drop=True)
    labelled = int(out["label_available"].sum()) if "label_available" in out.columns else 0
    residuals = out["residual_kw"].dropna() if "residual_kw" in out.columns else pd.Series(dtype=float)
    return out, {
        "format": "somes_model_retraining_dataset_v1",
        "rows": int(len(out)),
        "labelled_rows": labelled,
        "target_column": "actual_load_kw",
        "feature_columns": [c for c in keep if c not in ("datetime", "actual_load_kw", "residual_kw", "label_available")],
        "time_range": [str(out["datetime"].min()), str(out["datetime"].max())] if len(out) else [],
        "residual_mae_kw": round(float(residuals.abs().mean()), 4) if len(residuals) else None,
        "ready_for_incremental_train": labelled >= 96,
    }


class ModelMonitoringPiece(BasePiece):
    """
    Monitors model quality and data drift on 15-minute predictions.
    """

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
        log_path = Path(self.results_path) / "model_monitoring.log"
        err_path = Path(self.results_path) / "model_monitoring_error.txt"
        try:
            pred_path = Path(input_data.predictions_csv)
            if not pred_path.is_file():
                raise FileNotFoundError(f"Predictions not found: {pred_path}")

            df = pd.read_csv(pred_path)
            if "datetime" not in df.columns or "prediction_load_kw" not in df.columns:
                raise ValueError("Predictions CSV must contain datetime and prediction_load_kw.")

            df["datetime"] = pd.to_datetime(df["datetime"])
            df["hour"] = df["datetime"].dt.hour
            pred = pd.to_numeric(df["prediction_load_kw"], errors="coerce").fillna(0.0)

            report: dict[str, object] = {
                "rows": int(len(df)),
                "pred_mean_kw": float(pred.mean()) if len(df) else 0.0,
                "pred_std_kw": float(pred.std(ddof=0)) if len(df) else 0.0,
                "pred_p95_kw": float(pred.quantile(0.95)) if len(df) else 0.0,
            }

            daily = (
                df.assign(prediction_load_kw=pred, date=df["datetime"].dt.date)
                .groupby("date", as_index=False)["prediction_load_kw"]
                .sum()
            )
            daily["prediction_mwh"] = daily["prediction_load_kw"] * 0.25 / 1000.0
            daily = daily.drop(columns=["prediction_load_kw"])

            actual = _actual_load(df, input_data.history_csv)
            mask = actual.notna()
            if mask.any():
                err = pred[mask] - actual[mask]
                denom = actual[mask].replace(0, pd.NA).dropna()
                report.update(
                    {
                        "actual_available": True,
                        "scored_rows": int(mask.sum()),
                        "mae_kw": float(err.abs().mean()),
                        "rmse_kw": float((err.pow(2).mean()) ** 0.5),
                        "mape_pct": (
                            float((err[denom.index].abs() / denom).mean() * 100) if len(denom) else None
                        ),
                    }
                )
            else:
                report["actual_available"] = False

            by_hour = (
                df.assign(prediction_load_kw=pred)
                .groupby("hour", as_index=False)["prediction_load_kw"]
                .mean()
                .sort_values("prediction_load_kw", ascending=False)
                .head(5)
            )
            report["top_consumption_hours"] = [
                {"hour": int(r["hour"]), "avg_pred_kw": round(float(r["prediction_load_kw"]), 2)}
                for _, r in by_hour.iterrows()
            ]

            repo_root = Path(__file__).resolve().parents[2]
            if str(repo_root) not in sys.path:
                sys.path.insert(0, str(repo_root))
            from pieces.common_somes.feedback import (
                build_feedback_bundle,
                dispatch_plan_vs_actual,
                pv_forecast_vs_actual,
                pv_model_vs_target,
            )

            load_metrics = {
                "available": bool(report.get("actual_available")),
                "samples": report.get("scored_rows", 0),
                "mae": report.get("mae_kw"),
                "rmse": report.get("rmse_kw"),
                "mape_pct": report.get("mape_pct"),
            }
            pv_metrics = None
            if input_data.pv_forecast_csv and input_data.pv_actual_csv:
                pv_metrics = pv_forecast_vs_actual(input_data.pv_forecast_csv, input_data.pv_actual_csv)
            # Fallback: score PVOUT model vs labelled feature data (same timestamps).
            if (not pv_metrics or not pv_metrics.get("available")) and (
                input_data.pv_model_forecast_csv and input_data.pv_model_target_csv
            ):
                pv_metrics = pv_model_vs_target(
                    input_data.pv_model_forecast_csv,
                    input_data.pv_model_target_csv,
                )
            dispatch_metrics = None
            if input_data.planned_dispatch_csv and input_data.bess_telemetry_csv:
                dispatch_metrics = dispatch_plan_vs_actual(
                    input_data.planned_dispatch_csv,
                    input_data.bess_telemetry_csv,
                    dt_h=float(input_data.timestep_hours),
                )
            feedback = build_feedback_bundle(
                load_metrics=load_metrics, pv=pv_metrics, dispatch=dispatch_metrics
            )
            report["forecast_vs_actual"] = feedback

            out_dir = Path(self.results_path)
            out_dir.mkdir(parents=True, exist_ok=True)
            report_path = out_dir / "monitoring_report.json"
            daily_path = out_dir / "monitoring_daily.csv"
            feedback_path = out_dir / "forecast_vs_actual.json"
            retrain_path = out_dir / "model_retraining_dataset.csv"
            retrain_meta_path = out_dir / "model_retraining_meta.json"

            retrain_df, retrain_meta = _retraining_dataset(df, pred, actual, input_data.history_csv)
            retrain_df.to_csv(retrain_path, index=False)
            report["model_retraining_dataset"] = retrain_meta
            report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
            feedback_path.write_text(json.dumps(feedback, indent=2, ensure_ascii=False), encoding="utf-8")
            retrain_meta_path.write_text(json.dumps(retrain_meta, indent=2, ensure_ascii=False), encoding="utf-8")
            daily.to_csv(daily_path, index=False)

            with open(log_path, "a", encoding="utf-8") as f:
                f.write("[INFO] ModelMonitoringPiece completed\n")
                f.write(f"[INFO] retraining rows={retrain_meta['rows']} labelled={retrain_meta['labelled_rows']}\n")
                for action in feedback["recommended_actions"]:
                    f.write(f"[ACTION] {action}\n")
            _piece_out = OutputModel(
                report_json=str(report_path),
                daily_csv=str(daily_path),
                message=f"Model monitoring report generated (closed loop ok: {feedback['closed_loop_ok']}).",
                forecast_vs_actual_json=str(feedback_path),
                closed_loop_ok=bool(feedback["closed_loop_ok"]),
                retraining_dataset_csv=str(retrain_path),
            )
            if od is not None:
                if hasattr(_piece_out, 'run_id') and _run_id and not getattr(_piece_out, 'run_id', ''):
                    try:
                        _piece_out.run_id = _run_id
                    except Exception:
                        pass
                return od.finish_piece(
                    _piece_out, self.results_path, secrets_data, "ModelMonitoringPiece", _stage, run_id=_run_id
                )
            if _stage is not None:
                _stage.cleanup()
            return _piece_out
        except Exception:
            err = traceback.format_exc()
            with open(log_path, "a", encoding="utf-8") as f:
                f.write("[ERROR] ModelMonitoringPiece failed\n")
                f.write(err + "\n")
            with open(err_path, "w", encoding="utf-8") as f:
                f.write(err)
            raise
