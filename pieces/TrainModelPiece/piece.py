
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
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import datetime


def _hours_to_blocks(hours: list[int]) -> list[list[int]]:
    if not hours:
        return []
    hs = sorted(set(int(h) for h in hours))
    blocks: list[list[int]] = []
    start = hs[0]
    prev = hs[0]
    for h in hs[1:]:
        if h == prev + 1:
            prev = h
            continue
        blocks.append([start, prev + 1])
        start = h
        prev = h
    blocks.append([start, prev + 1])
    return blocks


def _build_shift_profile(df: pd.DataFrame) -> dict:
    work = df[["datetime", "load_kw"]].copy()
    work["datetime"] = pd.to_datetime(work["datetime"])
    work = work.dropna(subset=["datetime"])
    work["date"] = work["datetime"].dt.date
    work["dayofweek"] = work["datetime"].dt.dayofweek
    work["hour"] = work["datetime"].dt.hour
    work["load_kw"] = pd.to_numeric(work["load_kw"], errors="coerce").fillna(0.0)

    daily_hour = (
        work.groupby(["date", "dayofweek", "hour"], as_index=False)["load_kw"]
        .mean()
        .pivot_table(index=["date", "dayofweek"], columns="hour", values="load_kw", fill_value=0.0)
        .reindex(columns=list(range(24)), fill_value=0.0)
    )
    if daily_hour.empty:
        return {"by_dayofweek": {}, "global": {"active_hours": [], "blocks": []}}

    lo = daily_hour.min(axis=1)
    hi = daily_hour.max(axis=1)
    thr = lo + (hi - lo) * 0.35
    active = daily_hour.gt(thr, axis=0)

    by_day: dict[str, dict] = {}
    for dow in range(7):
        mask = [idx[1] == dow for idx in active.index]
        day_active = active[mask]
        if day_active.empty:
            by_day[str(dow)] = {"active_hours": [], "blocks": []}
            continue
        probs = day_active.mean(axis=0)
        active_hours = [int(h) for h, p in probs.items() if float(p) >= 0.5]
        by_day[str(dow)] = {
            "active_hours": active_hours,
            "blocks": _hours_to_blocks(active_hours),
        }

    global_probs = active.mean(axis=0)
    global_hours = [int(h) for h, p in global_probs.items() if float(p) >= 0.5]
    return {
        "by_dayofweek": by_day,
        "global": {"active_hours": global_hours, "blocks": _hours_to_blocks(global_hours)},
    }


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
    for lag in (1, 4, 96, 192, 672):
        out[f"lag_{lag}"] = out[target].shift(lag)
    prev = out[target].shift(1)
    for w in (4, 16, 96):
        out[f"roll_mean_{w}"] = prev.rolling(w).mean()
        out[f"roll_std_{w}"] = prev.rolling(w).std(ddof=0)
    return out


class TrainModelPiece(BasePiece):

    def piece_function(self, input_data: InputModel) -> OutputModel:
        piece_log = Path(self.results_path) / "train_model.log"
        piece_err = Path(self.results_path) / "train_model_error.txt"
        try:
            print("[INFO] TrainModelPiece started")
            print(f"[INFO] Using training data: {input_data.data_path}")
            with open(piece_log, "a", encoding="utf-8") as f:
                f.write("[INFO] TrainModelPiece started\n")

            data_path = Path(input_data.data_path)

            if not data_path.exists():
                raise FileNotFoundError(f"Training data not found: {data_path}")

            if data_path.suffix == ".parquet":
                df = pd.read_parquet(data_path)
            else:
                df = pd.read_csv(data_path)

            if "datetime" not in df.columns:
                raise ValueError("Dataset must contain 'datetime' column")

            df["datetime"] = pd.to_datetime(df["datetime"])
            df = df.sort_values("datetime")

            target = "load_kw"
            if target not in df.columns:
                raise ValueError(f"Target column '{target}' not found")

            print("[INFO] Detecting shift profile from historical load")
            shift_profile = _build_shift_profile(df)

            print("[INFO] Creating time features")

            df["hour"] = df["datetime"].dt.hour
            df["dayofweek"] = df["datetime"].dt.dayofweek
            df["month"] = df["datetime"].dt.month
            shift_df = _shift_features_for_datetimes(df["datetime"], shift_profile)
            for c in shift_df.columns:
                df[c] = shift_df[c]

            print("[INFO] Creating lag and rolling features")
            df = _add_load_features(df, target)

            df = df.dropna().reset_index(drop=True)

            split_index = int(len(df) * 0.8)

            train_df = df.iloc[:split_index]
            test_df = df.iloc[split_index:]

            feature_cols = [c for c in df.columns if c not in ["datetime", target]]

            X_train = train_df[feature_cols]
            y_train = train_df[target]

            X_test = test_df[feature_cols]
            y_test = test_df[target]

            print(f"[INFO] Train rows: {len(X_train)}")
            print(f"[INFO] Test rows: {len(X_test)}")

            print("[INFO] Training XGBoost model")

            model = XGBRegressor(
            objective="reg:squarederror",
            learning_rate=0.03,
            max_depth=7,
            n_estimators=500,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_weight=3,
            reg_alpha=0.05,
            reg_lambda=1.5,
        )

            model.fit(X_train, y_train)

            print("[INFO] Evaluating model")

            preds = model.predict(X_test)

            mae = mean_absolute_error(y_test, preds)
            mse = mean_squared_error(y_test, preds)
            rmse = mse ** 0.5

            mean_load_test = float(y_test.mean()) if len(y_test) else 0.0
            mae_pct = (100.0 * mae / mean_load_test) if mean_load_test else 0.0
            rmse_pct = (100.0 * rmse / mean_load_test) if mean_load_test else 0.0
            y_arr = np.asarray(y_test.values if hasattr(y_test, "values") else y_test)
            p_arr = np.asarray(preds).ravel()
            mask = y_arr != 0
            if mask.any():
                mape = float(100.0 * (np.abs(y_arr[mask] - p_arr[mask]) / np.abs(y_arr[mask])).mean())
            else:
                mape = 0.0

            print(f"[METRIC] MAE: {mae:.2f} kW ({mae_pct:.2f}% of test mean)")
            print(f"[METRIC] RMSE: {rmse:.2f} kW ({rmse_pct:.2f}% of test mean)")
            print(f"[METRIC] MAPE: {mape:.2f} %")

            model_path = Path(self.results_path) / "xgboost_model.pkl"
            log_path = Path(self.results_path) / "training_log.txt"
            shift_profile_path = Path(self.results_path) / "shift_profile.json"

            joblib.dump(model, model_path)
            shift_profile_path.write_text(json.dumps(shift_profile, indent=2, ensure_ascii=False), encoding="utf-8")

            with open(log_path, "w") as f:
                f.write(f"Training time (UTC): {datetime.utcnow()}\n")
                f.write(f"Rows total: {len(df)}\n")
                f.write(f"Train rows: {len(train_df)}\n")
                f.write(f"Test rows: {len(test_df)}\n")
                f.write(f"Features: {feature_cols}\n")
                f.write(f"MAE: {mae:.4f} kW\n")
                f.write(f"RMSE: {rmse:.4f} kW\n")
                f.write(f"Mean load (test set): {mean_load_test:.4f} kW\n")
                f.write(f"MAE as % of mean load: {mae_pct:.4f} %\n")
                f.write(f"RMSE as % of mean load: {rmse_pct:.4f} %\n")
                f.write(f"MAPE (mean abs % error): {mape:.4f} %\n")

            print(f"[SUCCESS] Model saved to {model_path}")

            return OutputModel(
                message=(
                    f"Model trained. MAE={mae:.2f} kW ({mae_pct:.2f}%), "
                    f"RMSE={rmse:.2f} kW ({rmse_pct:.2f}%), MAPE={mape:.2f}%"
                ),
                model_file_path=str(model_path),
                train_log_path=str(log_path)
            )
        except Exception:
            err = traceback.format_exc()
            with open(piece_log, "a", encoding="utf-8") as f:
                f.write("[ERROR] TrainModelPiece failed\n")
                f.write(err + "\n")
            with open(piece_err, "w", encoding="utf-8") as f:
                f.write(err)
            raise
