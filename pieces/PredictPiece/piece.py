
try:
    from domino.base_piece import BasePiece
except ModuleNotFoundError:
    from local_compat.base_piece import BasePiece
from .models import InputModel, OutputModel

import json
import traceback
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
from datetime import datetime
import yaml

try:
    from common import onedata_io as od
except ModuleNotFoundError:
    try:
        from pieces.common import onedata_io as od
    except ModuleNotFoundError:
        od = None

try:
    from common.predictions_load import predictions_to_load_csv
except ModuleNotFoundError:
    try:
        from pieces.common.predictions_load import predictions_to_load_csv
    except ModuleNotFoundError:
        predictions_to_load_csv = None


def _default_shift_profile() -> dict:
    return {"by_dayofweek": {}, "global": {"active_hours": [], "blocks": []}}


def _load_shift_profile(model_path: Path) -> dict:
    p = model_path.with_name("shift_profile.json")
    if not p.is_file():
        return _default_shift_profile()
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return _default_shift_profile()


def _shift_features_for_datetimes(dt: pd.Series, profile: dict) -> pd.DataFrame:
    dts = pd.to_datetime(dt)
    day_map = profile.get("by_dayofweek", {})
    glob = profile.get("global", {})
    rows = []
    for ts in dts:
        dow = int(ts.dayofweek)
        hour = int(ts.hour)
        day_info = day_map.get(str(dow)) or glob or {}
        blocks = day_info.get("blocks") or []
        active = 0
        block_idx = 0
        for i, b in enumerate(blocks, start=1):
            start, end = int(b[0]), int(b[1])
            if start <= hour < end:
                active = 1
                block_idx = i
                break
        rows.append(
            {
                "shift_active": active,
                "shift_block_index": block_idx,
                "shift_block_count": int(len(blocks)),
            }
        )
    return pd.DataFrame(rows, index=dt.index)


def _add_load_features(df: pd.DataFrame, target: str) -> pd.DataFrame:
    out = df.copy()
    if "department_id" in out.columns:
        out["department_id"] = out["department_id"].astype(str)
        grouped = out.groupby("department_id", sort=False)[target]
        for lag in (1, 4, 96, 192, 672):
            out[f"lag_{lag}"] = grouped.shift(lag)
        prev = grouped.shift(1)
        out["_prev"] = prev
        for w in (4, 16, 96):
            out[f"roll_mean_{w}"] = out.groupby("department_id", sort=False)["_prev"].transform(
                lambda s: s.rolling(w).mean()
            )
            out[f"roll_std_{w}"] = out.groupby("department_id", sort=False)["_prev"].transform(
                lambda s: s.rolling(w).std(ddof=0)
            )
        out = out.drop(columns=["_prev"])
        return out
    for lag in (1, 4, 96, 192, 672):
        out[f"lag_{lag}"] = out[target].shift(lag)
    prev = out[target].shift(1)
    for w in (4, 16, 96):
        out[f"roll_mean_{w}"] = prev.rolling(w).mean()
        out[f"roll_std_{w}"] = prev.rolling(w).std(ddof=0)
    return out


def _encode_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "department_id" in out.columns:
        out["department_id"] = out["department_id"].astype(str)
        out = pd.get_dummies(out, columns=["department_id"], dtype=float)
    for col in out.columns:
        if pd.api.types.is_bool_dtype(out[col]):
            out[col] = out[col].astype(int)
        elif pd.api.types.is_object_dtype(out[col]) or pd.api.types.is_string_dtype(out[col]):
            out[col] = pd.to_numeric(out[col], errors="raise")
    return out


def _safe_lag(loads: np.ndarray, i: int, lag: int) -> float:
    j = i - lag
    if j >= 0:
        return float(loads[j])
    return float(loads[0])


def _safe_roll(loads: np.ndarray, i: int, w: int) -> tuple[float, float]:
    start = max(0, i - w)
    hist = loads[start:i]
    if hist.size == 0:
        base = float(loads[max(0, i - 1)])
        return base, 0.0
    return float(hist.mean()), float(hist.std(ddof=0))


def _to_numeric_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace(",", ".", regex=False), errors="coerce")


def _pick_datetime_column(df: pd.DataFrame) -> str | None:
    aliases = {"datetime", "date time", "date_time", "timestamp", "time"}
    for c in df.columns:
        if str(c).replace("\ufeff", "").strip().lower() in aliases:
            return c
    return None


def _read_load_csv(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, sep=None, engine="python")
    if len(raw.columns) == 1 and ";" in str(raw.columns[0]):
        raw = pd.read_csv(path, sep=";")

    dt_col = _pick_datetime_column(raw)
    if dt_col is None:
        raise ValueError(f"{path.name}: missing datetime column")

    df = raw.copy()
    dt_raw = df[dt_col].astype(str).str.strip()
    # ISO first: with dayfirst=True pandas infers %Y-%d-%m from an ISO timestamp
    # whose day is <= 12, which silently swaps day and month for the whole file.
    dt = pd.to_datetime(dt_raw, format="ISO8601", errors="coerce")
    if dt.isna().all():
        dt = pd.to_datetime(dt_raw, format="%d.%m.%y %H:%M", errors="coerce")
    if dt.isna().all():
        dt = pd.to_datetime(dt_raw, dayfirst=True, errors="coerce")
    df["datetime"] = dt
    df = df.dropna(subset=["datetime"])
    if dt_col != "datetime":
        df = df.drop(columns=[dt_col])

    if "load_kw" in df.columns:
        df["load_kw"] = _to_numeric_series(df["load_kw"]).fillna(0.0)
        if "department_id" not in df.columns:
            dept = path.stem.replace("load_", "")
            df["department_id"] = "default" if dept == "load" else dept
        return df[["datetime", "department_id", "load_kw"]]

    reserved = {"department_id", "price_eur_kwh", "price_eur_per_kwh", "price_eur_mwh"}
    value_cols = [c for c in df.columns if c not in reserved]
    if not value_cols:
        raise ValueError(f"{path.name}: no load columns found")
    long = df.melt(
        id_vars=["datetime"],
        value_vars=value_cols,
        var_name="department_id",
        value_name="load_kw",
    )
    long["department_id"] = long["department_id"].astype(str).str.strip().str.replace("prikon ", "", case=False)
    long["load_kw"] = _to_numeric_series(long["load_kw"]).fillna(0.0)
    return long[["datetime", "department_id", "load_kw"]]


def _generate_prediction_input_from_load(
    load_path: Path,
    out_path: Path,
    *,
    prediction_days: int = 7,
    timestep_minutes: int = 15,
) -> Path:
    rows_per_day = max(1, int(24 * 60 / timestep_minutes))
    n_future = max(1, int(prediction_days)) * rows_per_day
    freq = f"{timestep_minutes}min"

    load_all = _read_load_csv(load_path).sort_values(["department_id", "datetime"]).reset_index(drop=True)
    if load_all.empty:
        raise ValueError(f"{load_path.name}: no load rows")
    last_dt = pd.to_datetime(load_all["datetime"].max())
    future_start = last_dt + pd.Timedelta(minutes=timestep_minutes)
    future_dt = pd.date_range(future_start, periods=n_future, freq=freq)
    hours = future_dt.hour.values
    price = 0.07 + 0.035 * ((hours >= 7) & (hours <= 20))
    price = np.clip(price, 0.06, 0.15)
    future_base = pd.DataFrame(
        {
            "datetime": future_dt,
            "load_kw": 0.0,
            "price_eur_per_kwh": price,
            "price_eur_kwh": price,
        }
    )

    parts: list[pd.DataFrame] = []
    for dept, group in load_all.groupby("department_id", sort=False):
        dept_id = str(dept)
        group = group.sort_values("datetime").reset_index(drop=True)
        if len(group) < 4:
            continue
        bridge = group.tail(4)[["datetime", "load_kw"]].copy()
        bridge["datetime"] = pd.date_range(
            end=future_start - pd.Timedelta(minutes=timestep_minutes),
            periods=4,
            freq=freq,
        )
        bridge["price_eur_per_kwh"] = 0.085
        bridge["price_eur_kwh"] = 0.085
        bridge["department_id"] = dept_id

        future = future_base.copy()
        future["department_id"] = dept_id
        parts.append(pd.concat([bridge, future], ignore_index=True))

    if not parts:
        raise RuntimeError(f"No valid department data found for prediction fallback: {load_path}")

    out = pd.concat(parts, ignore_index=True).sort_values(["department_id", "datetime"]).reset_index(drop=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    return out_path


def _read_horizon_from_sidecars(load_csv: Path) -> tuple[int | None, int | None]:
    """Read prediction_days / timestep from scenario.yaml or workflow_user_input.json near load CSV."""
    days: int | None = None
    timestep: int | None = None
    candidates: list[Path] = []
    for name in ("workflow_user_input.json",):
        candidates.append(load_csv.with_name(name))
        candidates.append(load_csv.parent / name)
    for wf_path in candidates:
        if not wf_path.is_file():
            continue
        try:
            wf = json.loads(wf_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        pp = wf.get("PredictPiece") or {}
        if pp.get("prediction_days") is not None:
            days = int(pp["prediction_days"])
        if pp.get("timestep_minutes") is not None:
            timestep = int(pp["timestep_minutes"])
        break

    scen_path: Path | None = None
    for cand in (
        load_csv.with_name("scenario_resolved.yaml"),
        load_csv.with_name("scenario.yaml"),
        load_csv.parent / "scenario_resolved.yaml",
        load_csv.parent / "scenario.yaml",
    ):
        if cand.is_file():
            scen_path = cand
            break
    if scen_path is not None:
        try:
            scen = yaml.safe_load(scen_path.read_text(encoding="utf-8")) or {}
        except Exception:
            scen = {}
        if days is None:
            prod = scen.get("production") or {}
            raw = scen.get("prediction_days") or prod.get("prediction_days")
            if raw is not None:
                days = int(raw)
        if timestep is None and scen.get("timestep_minutes") is not None:
            timestep = int(scen["timestep_minutes"])
    return days, timestep


def _resolve_prediction_horizon(input_data: InputModel, load_path: Path) -> tuple[int, int]:
    file_days, file_ts = _read_horizon_from_sidecars(load_path)
    days = file_days if file_days is not None else int(input_data.prediction_days or 30)
    timestep = file_ts if file_ts is not None else int(input_data.timestep_minutes or 15)
    return days, timestep


def _build_prediction_grid(
    load_path: Path,
    results_dir: Path,
    *,
    prediction_days: int,
    timestep_minutes: int,
) -> Path:
    out_path = results_dir / "prediction_input_generated.csv"
    return _generate_prediction_input_from_load(
        load_path,
        out_path,
        prediction_days=prediction_days,
        timestep_minutes=timestep_minutes,
    )


class PredictPiece(BasePiece):

    def piece_function(self, input_data: InputModel, secrets_data=None) -> OutputModel:
        _stage = None
        _run_id = None
        _orig_model_path = getattr(input_data, "model_path", None)
        if od is not None:
            input_data, _stage = od.stage_inputs(input_data, secrets_data)
            _run_id = od.resolve_run_id(input_data, secrets_data, generate=False)
            if _stage is not None and _stage.active:
                od.fetch_sibling(_orig_model_path, input_data.model_path, "shift_profile.json")
        elif any(
            isinstance(v, str) and str(v).startswith("onedata:")
            for v in (getattr(input_data, "load_csv", None), getattr(input_data, "model_path", None))
        ):
            raise RuntimeError(
                "OneData paths given but onedata_io failed to import "
                "(check pieces/common; predictions_load must not block od import)."
            )
        _piece_out = None
        results_dir = Path(self.results_path or ".")
        results_dir.mkdir(parents=True, exist_ok=True)
        piece_log = results_dir / "predict.log"
        piece_err = results_dir / "predict_error.txt"
        try:
            print("[INFO] PredictPiece started")
            print(f"[INFO] Model path: {input_data.model_path}")
            print(f"[INFO] Load CSV: {input_data.load_csv}")

            model_path = Path(input_data.model_path)
            load_path = Path(input_data.load_csv)
            if str(load_path).startswith("onedata:") or not load_path.is_file():
                raise FileNotFoundError(
                    f"Load CSV not found (staging may have failed): {load_path}"
                )
            prediction_days, timestep_minutes = _resolve_prediction_horizon(input_data, load_path)
            print(
                f"[INFO] Prediction horizon: {prediction_days} days "
                f"({timestep_minutes} min steps)"
            )
            data_path = _build_prediction_grid(
                load_path,
                results_dir,
                prediction_days=prediction_days,
                timestep_minutes=timestep_minutes,
            )

            if not model_path.exists():
                raise FileNotFoundError(f"Model not found: {model_path}")
            print(f"[INFO] Resolved prediction data path: {data_path}")

            model = joblib.load(model_path)
            shift_profile = _load_shift_profile(model_path)

            if data_path.suffix == ".parquet":
                df = pd.read_parquet(data_path)
            else:
                df = pd.read_csv(data_path)

            if "datetime" not in df.columns:
                print("[WARN] datetime column not found, trying index reset")
                df = df.reset_index()

            if "datetime" not in df.columns:
                raise ValueError(
                    f"Prediction dataset must contain datetime column. "
                    f"Columns found: {df.columns.tolist()}"
                )

            df["datetime"] = pd.to_datetime(df["datetime"])
            if "department_id" in df.columns:
                df["department_id"] = df["department_id"].astype(str)
                df = df.sort_values(["department_id", "datetime"]).reset_index(drop=True)
            else:
                df = df.sort_values("datetime").reset_index(drop=True)

            target = "load_kw"

            if target not in df.columns:
                raise ValueError(
                    f"Prediction dataset must contain '{target}'. "
                    f"Columns: {df.columns.tolist()}"
                )

            use_rolling = getattr(input_data, "use_rolling_prediction", True)
            bridge_rows = int(getattr(input_data, "bridge_rows", 4))

            if use_rolling:
                print(f"[INFO] Rolling prediction (bridge_rows={bridge_rows})")
                df_out = self._predict_rolling(model, df, bridge_rows, shift_profile)
            else:
                print("[INFO] Batch prediction (shift na load_kw)")
                df_out = self._predict_batch(model, df, target, shift_profile)

            output_path = results_dir / "predictions_15min.csv"
            df_out.to_csv(output_path, index=False)

            runtime_load_path = results_dir / "runtime_load_for_sim.csv"
            if predictions_to_load_csv is None:
                raise RuntimeError("predictions_load helper not available")
            predictions_to_load_csv(output_path, runtime_load_path)

            feature_names = list(model.get_booster().feature_names)
            log_path = results_dir / "prediction_log.txt"
            with open(log_path, "w") as f:
                f.write(f"Prediction time (UTC): {datetime.utcnow()}\n")
                f.write(f"Rows: {len(df_out)}\n")
                f.write(f"Features used: {feature_names}\n")
                f.write(f"Model: {model_path.name}\n")
                f.write(f"use_rolling_prediction: {use_rolling}\n")

            print("[SUCCESS] Prediction finished")
            print(f"[SUCCESS] Predictions saved to {output_path}")

            _piece_out = OutputModel(
                message="Prediction finished successfully",
                prediction_file_path=str(output_path),
                runtime_load_csv=str(runtime_load_path),
            )
        except Exception:
            err = traceback.format_exc()
            with open(piece_log, "a", encoding="utf-8") as f:
                f.write("[ERROR] PredictPiece failed\n")
                f.write(err + "\n")
            with open(piece_err, "w", encoding="utf-8") as f:
                f.write(err)
            raise
        finally:
            if od is not None and _piece_out is None:
                od.cleanup_on_error(self.results_path, secrets_data, "PredictPiece", _stage, run_id=_run_id)
            elif _stage is not None:
                _stage.cleanup()
        if od is not None and _piece_out is not None:
            return od.finish_piece(_piece_out, self.results_path, secrets_data, "PredictPiece", _stage, run_id=_run_id)
        return _piece_out

    def _predict_batch(self, model, df: pd.DataFrame, target: str, shift_profile: dict) -> pd.DataFrame:
        df = df.copy()
        df["hour"] = df["datetime"].dt.hour
        df["dayofweek"] = df["datetime"].dt.dayofweek
        df["month"] = df["datetime"].dt.month
        shift_df = _shift_features_for_datetimes(df["datetime"], shift_profile)
        for c in shift_df.columns:
            df[c] = shift_df[c]
        df = _add_load_features(df, target)
        df = df.dropna().reset_index(drop=True)
        feature_names = model.get_booster().feature_names
        X = _encode_feature_frame(df.drop(columns=["datetime", target], errors="ignore")).reindex(
            columns=feature_names,
            fill_value=0.0,
        )
        preds = model.predict(X)
        df_out = df.copy()
        df_out["prediction_load_kw"] = preds
        return df_out

    def _predict_rolling(self, model, df: pd.DataFrame, bridge_rows: int, shift_profile: dict) -> pd.DataFrame:
        if "department_id" in df.columns:
            parts: list[pd.DataFrame] = []
            for _, group in df.groupby("department_id", sort=False):
                parts.append(self._predict_rolling_single(model, group.reset_index(drop=True), bridge_rows, shift_profile))
            return pd.concat(parts, ignore_index=True).sort_values(["department_id", "datetime"]).reset_index(drop=True)
        return self._predict_rolling_single(model, df.reset_index(drop=True), bridge_rows, shift_profile)

    def _predict_rolling_single(self, model, df: pd.DataFrame, bridge_rows: int, shift_profile: dict) -> pd.DataFrame:
        n = len(df)
        if n < bridge_rows:
            raise ValueError(f"Need at least {bridge_rows} rows for bridge; got {n}")

        feature_names = list(model.get_booster().feature_names)
        loads = np.zeros(n, dtype=float)

        for i in range(bridge_rows):
            v = df.iloc[i, df.columns.get_loc("load_kw")]
            if pd.isna(v):
                raise ValueError(f"Row {i}: load_kw required for bridge (rolling mode)")
            loads[i] = float(v)

        df_out = df.copy()
        df_out["hour"] = df_out["datetime"].dt.hour
        df_out["dayofweek"] = df_out["datetime"].dt.dayofweek
        df_out["month"] = df_out["datetime"].dt.month
        shift_df = _shift_features_for_datetimes(df_out["datetime"], shift_profile)
        for c in shift_df.columns:
            df_out[c] = shift_df[c]

        for i in range(bridge_rows, n):
            lag_1 = loads[i - 1]
            lag_4 = loads[i - 4]
            row = {
                "hour": int(df_out.iloc[i]["hour"]),
                "dayofweek": int(df_out.iloc[i]["dayofweek"]),
                "month": int(df_out.iloc[i]["month"]),
                "lag_1": lag_1,
                "lag_4": lag_4,
            }
            for lag in (96, 192, 672):
                key = f"lag_{lag}"
                if key in feature_names:
                    row[key] = _safe_lag(loads, i, lag)
            for w in (4, 16, 96):
                m_key = f"roll_mean_{w}"
                s_key = f"roll_std_{w}"
                if m_key in feature_names or s_key in feature_names:
                    m, s = _safe_roll(loads, i, w)
                    if m_key in feature_names:
                        row[m_key] = m
                    if s_key in feature_names:
                        row[s_key] = s
            for c in ("shift_active", "shift_block_index", "shift_block_count"):
                if c in feature_names:
                    row[c] = float(df_out.iloc[i][c])
            if "price_eur_kwh" in feature_names:
                if "price_eur_kwh" in df.columns:
                    row["price_eur_kwh"] = float(df.iloc[i]["price_eur_kwh"])
                elif "price_eur_per_kwh" in df.columns:
                    row["price_eur_kwh"] = float(df.iloc[i]["price_eur_per_kwh"])
                elif "price_eur_mwh" in df.columns:
                    row["price_eur_kwh"] = float(df.iloc[i]["price_eur_mwh"]) / 1000.0
                else:
                    raise ValueError("Missing price_eur_kwh or price_eur_mwh")
            if "price_eur_per_kwh" in feature_names:
                if "price_eur_per_kwh" in df.columns:
                    row["price_eur_per_kwh"] = float(df.iloc[i]["price_eur_per_kwh"])
                elif "price_eur_kwh" in df.columns:
                    row["price_eur_per_kwh"] = float(df.iloc[i]["price_eur_kwh"])
                elif "price_eur_mwh" in df.columns:
                    row["price_eur_per_kwh"] = float(df.iloc[i]["price_eur_mwh"]) / 1000.0
                else:
                    raise ValueError("Missing price_eur_per_kwh or compatible price column")
            if "department_id" in df_out.columns:
                row["department_id"] = str(df_out.iloc[i]["department_id"])

            X_row = _encode_feature_frame(pd.DataFrame([row])).reindex(columns=feature_names, fill_value=0.0)
            pr = float(model.predict(X_row)[0])
            loads[i] = pr

        df_out["prediction_load_kw"] = loads
        return df_out
