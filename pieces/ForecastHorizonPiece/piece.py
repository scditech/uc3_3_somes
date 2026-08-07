from __future__ import annotations

import json
import traceback
from pathlib import Path

import joblib
import pandas as pd
try:
    from domino.base_piece import BasePiece
except ModuleNotFoundError:
    from local_compat.base_piece import BasePiece

from .models import InputModel, OutputModel


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
    return out


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


def _backtest_residuals(model, history: pd.DataFrame, n_points: int = 384) -> dict:
    """Residual spread of the model on the most recent history, per hour of day.

    Used as the forecast confidence band; a pure hold-out is not possible here
    because the horizon lies in the future.
    """
    feats = _features(history).dropna(subset=_fcols() + ["load_kw"])
    if feats.empty:
        return {"sigma": 0.0, "by_hour": {}, "n_points": 0, "mae": 0.0}
    tail = feats.tail(n_points)
    pred = model.predict(tail[_fcols()])
    resid = tail["load_kw"].to_numpy(float) - pred
    by_hour = {}
    for hour, group in pd.DataFrame({"hour": tail["hour"].to_numpy(), "resid": resid}).groupby("hour"):
        by_hour[int(hour)] = float(group["resid"].std(ddof=0) or 0.0)
    denom = tail["load_kw"].to_numpy(float)
    mape_mask = denom > 1e-6
    return {
        "sigma": float(resid.std(ddof=0) or 0.0),
        "by_hour": by_hour,
        "n_points": int(len(tail)),
        "mae": float(abs(resid).mean()),
        "bias": float(resid.mean()),
        "mape_pct": float((abs(resid[mape_mask]) / denom[mape_mask]).mean() * 100) if mape_mask.any() else 0.0,
    }


def _peak_demand(forecast: pd.DataFrame) -> dict:
    """Predicted peak demand at site level and per department, per calendar day."""
    if forecast.empty:
        return {"format": "somes_peak_demand_v1", "site": {}, "by_department": {}}

    site = (
        forecast.groupby("datetime", as_index=False)
        .agg(
            prediction_load_kw=("prediction_load_kw", "sum"),
            prediction_p90_kw=("prediction_p90_kw", "sum"),
        )
        .sort_values("datetime")
    )
    site_peak_row = site.loc[site["prediction_load_kw"].idxmax()]
    daily = []
    for day, group in site.groupby(site["datetime"].dt.date):
        row = group.loc[group["prediction_load_kw"].idxmax()]
        daily.append(
            {
                "date": str(day),
                "peak_kw": round(float(row["prediction_load_kw"]), 3),
                "peak_p90_kw": round(float(row["prediction_p90_kw"]), 3),
                "peak_at": str(row["datetime"]),
                "mean_kw": round(float(group["prediction_load_kw"].mean()), 3),
                "load_factor": round(
                    float(group["prediction_load_kw"].mean() / max(float(row["prediction_load_kw"]), 1e-6)), 4
                ),
            }
        )

    by_dept = {}
    for dept, group in forecast.groupby("department_id"):
        row = group.loc[group["prediction_load_kw"].idxmax()]
        by_dept[str(dept)] = {
            "peak_kw": round(float(row["prediction_load_kw"]), 3),
            "peak_at": str(row["datetime"]),
        }

    return {
        "format": "somes_peak_demand_v1",
        "site": {
            "peak_kw": round(float(site_peak_row["prediction_load_kw"]), 3),
            "peak_p90_kw": round(float(site_peak_row["prediction_p90_kw"]), 3),
            "peak_at": str(site_peak_row["datetime"]),
            "daily": daily,
        },
        "by_department": by_dept,
    }


def _site_forecast(forecast: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    """Site-level D+1 profile the dispatch runs on, priced from the history profile.

    Dispatch happens at the connection point, so departments are summed. Prices for
    the forecast day reuse the time-of-day profile of the last week of history;
    module 6 refines them afterwards.
    """
    columns = ["datetime", "load_kw", "load_p10_kw", "load_p90_kw", "price_eur_kwh", "price_eur_per_kwh"]
    if forecast.empty:
        return pd.DataFrame(columns=columns)

    site = (
        forecast.groupby("datetime", as_index=False)
        .agg(
            load_kw=("prediction_load_kw", "sum"),
            load_p10_kw=("prediction_p10_kw", "sum"),
            load_p90_kw=("prediction_p90_kw", "sum"),
        )
        .sort_values("datetime")
        .reset_index(drop=True)
    )

    prices = history.drop_duplicates(subset=["datetime"])[["datetime", "price_eur_kwh"]].copy()
    prices["price_eur_kwh"] = pd.to_numeric(prices["price_eur_kwh"], errors="coerce")
    recent = prices[prices["datetime"] > prices["datetime"].max() - pd.Timedelta(days=7)]
    tod = recent.assign(tod=recent["datetime"].dt.strftime("%H:%M")).groupby("tod")["price_eur_kwh"].mean()
    fallback = float(tod.mean()) if len(tod) and pd.notna(tod.mean()) else 0.12
    site["price_eur_kwh"] = site["datetime"].dt.strftime("%H:%M").map(tod).fillna(fallback)
    site["price_eur_per_kwh"] = site["price_eur_kwh"]
    return site[columns]


class ForecastHorizonPiece(BasePiece):
    def piece_function(self, input_data: InputModel) -> OutputModel:
        log_path = Path(self.results_path) / "forecast_horizon.log"
        err_path = Path(self.results_path) / "forecast_horizon_error.txt"
        try:
            hist = pd.read_csv(input_data.history_csv, parse_dates=["datetime"])
            models = json.loads(Path(input_data.models_index_json).read_text(encoding="utf-8"))
            rows = []
            confidence = {}
            for dept, model_path in models.items():
                g = hist[hist["department_id"].astype(str) == str(dept)].sort_values("datetime").reset_index(drop=True)
                if len(g) < 300:
                    continue
                model = _safe_load_model(model_path, input_data.model_registry_dir)
                residuals = _backtest_residuals(model, g)
                confidence[str(dept)] = residuals
                step_minutes = 15
                last_dt = pd.to_datetime(g["datetime"].iloc[-1])
                steps = max(1, int(round(input_data.horizon_hours * 60 / step_minutes)))
                runtime = g[["datetime", "department_id", "load_kw", "price_eur_kwh"]].copy()
                for i in range(steps):
                    next_dt = last_dt + pd.to_timedelta((i + 1) * step_minutes, unit="m")
                    runtime.loc[len(runtime)] = {
                        "datetime": next_dt,
                        "department_id": dept,
                        "load_kw": float("nan"),
                        "price_eur_kwh": float(runtime["price_eur_kwh"].iloc[-1]),
                    }
                    fx = _features(runtime).iloc[-1]
                    X = pd.DataFrame([{c: fx[c] for c in _fcols()}])
                    pred = float(model.predict(X)[0])
                    runtime.loc[runtime.index[-1], "load_kw"] = max(0.0, pred)
                    value = max(0.0, pred)
                    # uncertainty grows with the horizon because lags become model outputs
                    horizon_growth = (1.0 + i / max(steps, 1)) ** 0.5
                    sigma = residuals["by_hour"].get(int(next_dt.hour), residuals["sigma"]) * horizon_growth
                    rows.append(
                        {
                            "datetime": next_dt,
                            "department_id": str(dept),
                            "prediction_load_kw": value,
                            "prediction_p10_kw": max(0.0, value - 1.2816 * sigma),
                            "prediction_p90_kw": value + 1.2816 * sigma,
                            "prediction_sigma_kw": round(sigma, 4),
                            "horizon_step": i + 1,
                            "horizon_hours": int(input_data.horizon_hours),
                        }
                    )

            out_dir = Path(self.results_path)
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / "forecast_by_department.csv"
            forecast_df = pd.DataFrame(rows)
            forecast_df.to_csv(out_path, index=False)

            conf_path = out_dir / "forecast_confidence.json"
            conf_path.write_text(
                json.dumps(
                    {
                        "format": "somes_load_forecast_confidence_v1",
                        "method": "recent-history residual spread per hour of day, widened over the horizon",
                        "interval": "p10/p90 (80 %)",
                        "horizon_hours": int(input_data.horizon_hours),
                        "by_department": confidence,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

            peak_path = out_dir / "predicted_peak_demand.json"
            peak_path.write_text(json.dumps(_peak_demand(forecast_df), indent=2, default=str), encoding="utf-8")

            site_path = out_dir / "site_load_forecast.csv"
            site_df = _site_forecast(forecast_df, hist)
            site_df.to_csv(site_path, index=False)

            with open(log_path, "a", encoding="utf-8") as f:
                f.write(
                    f"[INFO] forecast_rows={len(rows)} departments={len(confidence)} site_steps={len(site_df)}\n"
                )
            return OutputModel(
                message=f"Forecast completed, rows={len(rows)}",
                forecast_csv=str(out_path),
                forecast_confidence_json=str(conf_path),
                peak_demand_json=str(peak_path),
                site_forecast_csv=str(site_path),
            )
        except Exception:
            err = traceback.format_exc()
            with open(log_path, "a", encoding="utf-8") as f:
                f.write("[ERROR] ForecastHorizonPiece failed\n")
                f.write(err + "\n")
            with open(err_path, "w", encoding="utf-8") as f:
                f.write(err)
            raise
