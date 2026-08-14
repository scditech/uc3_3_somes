from __future__ import annotations

import pandas as pd


def parse_datetime_column(df: pd.DataFrame, datetime_column: str) -> pd.DataFrame:
    out = df.copy()
    out[datetime_column] = pd.to_datetime(out[datetime_column], errors="coerce")
    out = out.dropna(subset=[datetime_column])
    return out


def apply_optional_feature_derivations(
    df: pd.DataFrame, datetime_column: str = "datetime"
) -> pd.DataFrame:
    """
    Inference-time feature derivations using existing columns only.
    """
    out = df.copy()
    if "diffuse_fraction" not in out.columns and {"DIF", "GHI"}.issubset(out.columns):
        ghi_safe = out["GHI"].replace(0, pd.NA)
        out["diffuse_fraction"] = (out["DIF"] / ghi_safe).fillna(0.0)

    if "hour_of_day" not in out.columns and datetime_column in out.columns:
        out["hour_of_day"] = pd.to_datetime(out[datetime_column]).dt.hour

    if "solar_elevation_sin" not in out.columns and "SE" in out.columns:
        import numpy as np

        out["solar_elevation_sin"] = np.sin(np.radians(out["SE"]))

    return out


def apply_horizon_filter(
    df: pd.DataFrame,
    horizon_column: str,
    max_horizon: int | None,
) -> pd.DataFrame:
    if max_horizon is None or horizon_column not in df.columns:
        return df
    return df[df[horizon_column].astype(float) <= float(max_horizon)].copy()


def ensure_feature_schema(
    df: pd.DataFrame,
    feature_columns: list[str],
    strict_schema: bool = True,
    missing_fill_value: float = 0.0,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    missing = [c for c in feature_columns if c not in df.columns]
    added: list[str] = []

    if missing:
        if strict_schema:
            raise ValueError(f"Missing required feature columns: {missing}")
        out = df.copy()
        for c in missing:
            out[c] = missing_fill_value
            added.append(c)
    else:
        out = df

    return out[feature_columns].copy(), missing, added


def add_weekday_slot_columns(df: pd.DataFrame, datetime_column: str) -> pd.DataFrame:
    out = df.copy()
    dt = pd.to_datetime(out[datetime_column])
    out["dow"] = dt.dt.weekday
    out["slot_15m"] = dt.dt.hour * 4 + dt.dt.minute // 15
    return out


def build_price_baseline_from_profile(
    df: pd.DataFrame,
    datetime_column: str,
    profile_path: str,
    baseline_column: str = "price_baseline",
) -> pd.DataFrame:
    out = add_weekday_slot_columns(df, datetime_column)
    prof = pd.read_csv(profile_path)

    expected = {"dow", "slot_15m", "avg_price_eur_mwh"}
    if not expected.issubset(prof.columns):
        raise ValueError(
            f"Price profile missing required columns {expected}. Found={list(prof.columns)}"
        )

    out = out.merge(
        prof[["dow", "slot_15m", "avg_price_eur_mwh"]],
        on=["dow", "slot_15m"],
        how="left",
    )
    out[baseline_column] = out["avg_price_eur_mwh"]
    return out.drop(columns=["avg_price_eur_mwh"])
