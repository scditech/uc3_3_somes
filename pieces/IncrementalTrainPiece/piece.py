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
from xgboost import XGBRegressor

from .models import InputModel, OutputModel

try:
    from common import onedata_io as od
except ModuleNotFoundError:
    try:
        from pieces.common import onedata_io as od
    except ModuleNotFoundError:
        od = None



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


class IncrementalTrainPiece(BasePiece):
    def piece_function(self, input_data: InputModel, secrets_data=None) -> OutputModel:
        _stage = None
        _run_id = None
        _reg_remote = None
        _reg_local = None
        if od is not None:
            _orig_registry = getattr(input_data, "model_registry_dir", None) or ""
            input_data, _stage = od.stage_inputs(input_data, secrets_data)
            _run_id = od.resolve_run_id(
                input_data, secrets_data, generate=True, results_path=getattr(self, "results_path", None)
            )
            if hasattr(input_data, "run_id") and _run_id and not getattr(input_data, "run_id", ""):
                try:
                    input_data.run_id = _run_id
                except Exception:
                    pass
            if od.has_protocol(_orig_registry):
                _reg_remote = od.normalize_remote_path(_orig_registry)
                _reg_local = str(input_data.model_registry_dir)
        log_path = Path(self.results_path) / "incremental_train.log"
        err_path = Path(self.results_path) / "incremental_train_error.txt"
        try:
            hist = pd.read_csv(input_data.history_csv, parse_dates=["datetime"])
            model_dir = Path(input_data.model_registry_dir)
            model_dir.mkdir(parents=True, exist_ok=True)

            models_index: dict[str, str] = {}
            summary: list[dict] = []
            for dept, g in hist.groupby("department_id"):
                dept_id = str(dept)
                work = _features(g)
                if work.empty:
                    continue
                model_path = model_dir / f"{dept_id}_xgb.pkl"
                meta_path = model_dir / f"{dept_id}_meta.json"
                meta = {"updates_since_full": 0, "trained_until": None}
                if meta_path.is_file():
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))

                do_full_retrain = not model_path.is_file()
                if model_path.is_file():
                    if int(meta.get("updates_since_full", 0)) >= int(input_data.full_retrain_every_n_updates):
                        do_full_retrain = True

                if do_full_retrain:
                    X = work[_fcols()]
                    y = work["load_kw"].astype(float)
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
                    model.fit(X, y, verbose=False)
                    mode = "full_retrain" if model_path.is_file() else "initial_train"
                    meta["updates_since_full"] = 0
                else:
                    trained_until = pd.to_datetime(meta.get("trained_until"), errors="coerce")
                    if pd.isna(trained_until):
                        trained_until = work["datetime"].max() - pd.Timedelta(days=int(input_data.incremental_window_days))
                    recent_cut = max(
                        trained_until,
                        work["datetime"].max() - pd.Timedelta(days=int(input_data.incremental_window_days)),
                    )
                    update = work[work["datetime"] > recent_cut].copy()
                    if update.empty:
                        update = work.tail(96).copy()
                    X = update[_fcols()]
                    y = update["load_kw"].astype(float)
                    try:
                        model = joblib.load(model_path)
                        model.set_params(n_estimators=int(input_data.incremental_trees))
                        model.fit(X, y, xgb_model=model.get_booster(), verbose=False)
                        mode = "incremental_update"
                        meta["updates_since_full"] = int(meta.get("updates_since_full", 0)) + 1
                    except Exception:
                        # Pickle from older XGBoost/sklearn may lack newer attrs (e.g. device).
                        X_full = work[_fcols()]
                        y_full = work["load_kw"].astype(float)
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
                        model.fit(X_full, y_full, verbose=False)
                        mode = "full_retrain_fallback"
                        meta["updates_since_full"] = 0

                joblib.dump(model, model_path)
                meta["trained_until"] = str(work["datetime"].max())
                meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
                models_index[dept_id] = model_path.name
                summary.append({"department_id": dept_id, "rows_used": int(len(X)), "mode": mode})

            if _reg_remote and _reg_local and od is not None:
                od.upload_registry(_reg_local, _reg_remote)
                # Point index entries at durable OneData registry files for downstream pieces.
                models_index = {
                    dept: f"{_reg_remote.rstrip('/')}/{fname}" for dept, fname in models_index.items()
                }

            out_dir = Path(self.results_path)
            out_dir.mkdir(parents=True, exist_ok=True)
            idx_path = out_dir / "models_index.json"
            sum_path = out_dir / "training_summary.json"
            idx_path.write_text(json.dumps(models_index, indent=2, ensure_ascii=False), encoding="utf-8")
            sum_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"[INFO] departments={len(models_index)}\n")
            _piece_out = OutputModel(
                message=f"Incremental training completed for {len(models_index)} departments",
                models_index_json=str(idx_path),
                training_summary_json=str(sum_path),
            )
            if od is not None:
                if hasattr(_piece_out, 'run_id') and _run_id and not getattr(_piece_out, 'run_id', ''):
                    try:
                        _piece_out.run_id = _run_id
                    except Exception:
                        pass
                return od.finish_piece(
                    _piece_out, self.results_path, secrets_data, "IncrementalTrainPiece", _stage, run_id=_run_id
                )
            if _stage is not None:
                _stage.cleanup()
            return _piece_out
        except Exception:
            err = traceback.format_exc()
            with open(log_path, "a", encoding="utf-8") as f:
                f.write("[ERROR] IncrementalTrainPiece failed\n")
                f.write(err + "\n")
            with open(err_path, "w", encoding="utf-8") as f:
                f.write(err)
            raise
