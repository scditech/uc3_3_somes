"""
UC3.4-inspired PVOUT forecast (XGBoost), used by SolarSimPiece.

Aligned with uc3.4_ai_models_battery_optimisation:
  SolarGIS / Open-Meteo style inputs → PVOUTPredictionModelTrainPiece (xgb)
  → InferencePiece style prediction.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "hour_sin",
    "hour_cos",
    "doy_sin",
    "doy_cos",
    "ghi_proxy",
    "temp_proxy",
    "installed_kwp",
]


def build_time_features(
    dt: pd.Series,
    *,
    installed_kwp: float,
    ghi: pd.Series | None = None,
    temp: pd.Series | None = None,
) -> pd.DataFrame:
    t = pd.DatetimeIndex(pd.to_datetime(dt))
    hours = np.asarray(t.hour + t.minute / 60.0, dtype=float)
    doy = np.asarray(t.dayofyear, dtype=float)
    elev = np.clip(np.sin((hours - 6.0) / 12.0 * np.pi), 0.0, 1.0)
    seasonal = 0.85 + 0.15 * np.cos(2 * math.pi * (doy - 172) / 365.0)
    if ghi is None:
        ghi_v = elev**1.2 * seasonal * 1000.0
    else:
        ghi_v = pd.to_numeric(ghi, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    if temp is None:
        temp_v = 8.0 + 12.0 * seasonal + 4.0 * elev
    else:
        temp_v = pd.to_numeric(temp, errors="coerce").fillna(10.0).to_numpy(dtype=float)
    return pd.DataFrame(
        {
            "hour_sin": np.sin(2 * np.pi * hours / 24.0),
            "hour_cos": np.cos(2 * np.pi * hours / 24.0),
            "doy_sin": np.sin(2 * np.pi * doy / 365.0),
            "doy_cos": np.cos(2 * np.pi * doy / 365.0),
            "ghi_proxy": ghi_v,
            "temp_proxy": temp_v,
            "installed_kwp": float(installed_kwp),
        },
        index=t,
    )


def load_solargis_csv(path: Path, *, installed_kwp: float | None = None) -> pd.DataFrame:
    """Parse SolarGIS TS15 export (UC3.4 SolarGISDataGenerator / site file style)."""
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    start = 0
    for i, line in enumerate(lines):
        if line.startswith("Date;") or line.startswith("Date,"):
            start = i
            break
    raw = pd.read_csv(path, sep=";", skiprows=start)
    raw.columns = [c.strip() for c in raw.columns]
    dt = pd.to_datetime(
        raw["Date"].astype(str).str.strip() + " " + raw["Time"].astype(str).str.strip(),
        dayfirst=True,
        errors="coerce",
    )
    ghi = pd.to_numeric(raw.get("GHI"), errors="coerce").fillna(0.0)
    temp = pd.to_numeric(raw.get("TEMP"), errors="coerce").fillna(10.0)
    pvout = pd.to_numeric(raw.get("PVOUT"), errors="coerce").fillna(0.0)
    if installed_kwp is None:
        installed_kwp = float(max(pvout.max() * 1.05, 1.0))
    feat = build_time_features(dt, installed_kwp=installed_kwp, ghi=ghi, temp=temp)
    out = feat.reset_index(names="datetime")
    out["PVOUT"] = pvout.to_numpy()
    out["GHI"] = ghi.to_numpy()
    out["TEMP"] = temp.to_numpy()
    return out.dropna(subset=["datetime"]).reset_index(drop=True)


def synthesize_training_frame(
    *,
    installed_kwp: float,
    yield_kwh_per_kwp_year: float = 1000.0,
    days: int = 60,
    timestep_minutes: int = 15,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = int(days * 24 * 60 / timestep_minutes)
    start = pd.Timestamp("2024-06-01 00:00:00")
    dt = pd.date_range(start, periods=n, freq=f"{timestep_minutes}min")
    feat = build_time_features(pd.Series(dt), installed_kwp=installed_kwp)
    elev = np.clip(np.sin(((dt.hour + dt.minute / 60.0) - 6.0) / 12.0 * np.pi), 0.0, 1.0)
    seasonal = 0.85 + 0.15 * np.cos(2 * math.pi * (dt.dayofyear - 172) / 365.0)
    raw = seasonal * (elev**1.2) * installed_kwp
    noise = rng.normal(0.0, 0.04 * max(installed_kwp, 1.0), size=n)
    cloud = rng.uniform(0.75, 1.05, size=n)
    pv = np.clip(raw * cloud + noise, 0.0, installed_kwp * 1.15)
    dt_h = timestep_minutes / 60.0
    energy = float(np.sum(pv * dt_h))
    target = yield_kwh_per_kwp_year * installed_kwp * (n * dt_h / 8760.0)
    if energy > 1e-6:
        pv = pv * (target / energy)
    out = feat.reset_index(names="datetime")
    out["PVOUT"] = pv
    return out


def train_pvout_xgb(
    train_df: pd.DataFrame,
    *,
    model_path: Path,
    meta_path: Path,
    target_column: str = "PVOUT",
) -> dict[str, Any]:
    from xgboost import XGBRegressor

    missing = [c for c in FEATURE_COLUMNS if c not in train_df.columns]
    if missing:
        raise ValueError(f"PVOUT train CSV missing features: {missing}")
    if target_column not in train_df.columns:
        raise ValueError(f"PVOUT train CSV missing target column {target_column}")

    df = train_df.dropna(subset=FEATURE_COLUMNS + [target_column]).copy()
    # Cap rows for CI/local speed (full year SolarGIS is ~35k).
    if len(df) > 12000:
        df = df.sample(n=12000, random_state=42).sort_index()
    X = df[FEATURE_COLUMNS]
    y = df[target_column].astype(float)
    model = XGBRegressor(
        n_estimators=120,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=2,
    )
    model.fit(X, y)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    pred = model.predict(X)
    meta = {
        "source": "uc3.4_ai_models_battery_optimisation style (xgb_regressor_model)",
        "model_type": "xgb_regressor_model",
        "feature_columns": FEATURE_COLUMNS,
        "target_column": target_column,
        "n_train_rows": int(len(df)),
        "train_mae": float(np.mean(np.abs(pred - y))),
        "train_rmse": float(np.sqrt(np.mean((pred - y) ** 2))),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def predict_pvout_kw(
    datetimes: pd.Series,
    *,
    installed_kwp: float,
    model_path: Path,
    ghi: pd.Series | None = None,
    temp: pd.Series | None = None,
) -> pd.Series:
    model = joblib.load(model_path)
    feat = build_time_features(datetimes, installed_kwp=installed_kwp, ghi=ghi, temp=temp)
    pred = np.asarray(model.predict(feat[FEATURE_COLUMNS]), dtype=float)
    pred = np.clip(pred, 0.0, max(installed_kwp, 0.0) * 1.2)
    return pd.Series(pred, index=feat.index, name="pv_kw")


def ensure_pvout_model(
    *,
    model_dir: Path,
    installed_kwp: float,
    yield_kwh_per_kwp_year: float,
    train_csv: Path | None = None,
    solargis_csv: Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "pvout_xgb.pkl"
    meta_path = model_dir / "pvout_xgb_meta.json"

    train_df: pd.DataFrame | None = None
    source = "synthetic"
    if solargis_csv and solargis_csv.is_file():
        train_df = load_solargis_csv(solargis_csv, installed_kwp=installed_kwp)
        # Scale PVOUT to target plant size vs SolarGIS site peak.
        site_peak = float(train_df["PVOUT"].max()) if len(train_df) else 0.0
        if site_peak > 1e-6:
            train_df["PVOUT"] = train_df["PVOUT"] * (float(installed_kwp) / site_peak)
        train_df["installed_kwp"] = float(installed_kwp)
        source = f"solargis:{solargis_csv.name}"
    elif train_csv and train_csv.is_file():
        train_df = pd.read_csv(train_csv)
        if "installed_kwp" not in train_df.columns:
            train_df["installed_kwp"] = float(installed_kwp)
        source = f"csv:{train_csv.name}"
    else:
        train_df = synthesize_training_frame(
            installed_kwp=installed_kwp,
            yield_kwh_per_kwp_year=yield_kwh_per_kwp_year,
        )
        (model_dir / "pvout_train_synthetic.csv").write_text(
            train_df.to_csv(index=False), encoding="utf-8"
        )

    meta = train_pvout_xgb(train_df, model_path=model_path, meta_path=meta_path)
    meta["train_source"] = source
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return model_path, meta
