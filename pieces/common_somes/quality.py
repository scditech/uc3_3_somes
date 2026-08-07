"""Data validation & quality checks for SoMES ingestion (module 2).

Produces the artefacts required by the reference architecture:
data quality indicators, missing data report and correction recommendations.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

SEVERITY_ORDER = {"ok": 0, "info": 1, "warning": 2, "critical": 3}


def _worst(*severities: str) -> str:
    return max(severities, key=lambda s: SEVERITY_ORDER.get(s, 0)) if severities else "ok"


def detect_timestep(idx: pd.DatetimeIndex) -> pd.Timedelta:
    if len(idx) < 3:
        return pd.Timedelta(minutes=15)
    step = pd.Series(idx).diff().dropna().mode()
    if step.empty:
        return pd.Timedelta(minutes=15)
    value = step.iloc[0]
    return value if value > pd.Timedelta(0) else pd.Timedelta(minutes=15)


def missing_timestamps(idx: pd.DatetimeIndex, step: pd.Timedelta) -> list[dict[str, Any]]:
    """Return contiguous gaps on the expected regular grid."""
    if len(idx) < 2:
        return []
    full = pd.date_range(idx.min(), idx.max(), freq=step)
    missing = full.difference(idx)
    if len(missing) == 0:
        return []
    gaps: list[dict[str, Any]] = []
    start = prev = missing[0]
    for ts in missing[1:]:
        if ts - prev == step:
            prev = ts
            continue
        gaps.append({"gap_start": str(start), "gap_end": str(prev), "missing_steps": int((prev - start) / step) + 1})
        start = prev = ts
    gaps.append({"gap_start": str(start), "gap_end": str(prev), "missing_steps": int((prev - start) / step) + 1})
    return gaps


def profile_dataset(
    df: pd.DataFrame,
    *,
    name: str,
    datetime_col: str = "datetime",
    value_cols: list[str] | None = None,
    non_negative_cols: tuple[str, ...] = ("load_kw", "pv_kw", "ghi_wm2", "grid_import_kw", "grid_export_kw"),
) -> dict[str, Any]:
    """Compute quality indicators for one ingested dataset."""
    report: dict[str, Any] = {"dataset": name, "rows": int(len(df))}
    recommendations: list[str] = []
    severity = "ok"

    if datetime_col not in df.columns or df.empty:
        return {
            **report,
            "severity": "critical",
            "issues": [f"missing '{datetime_col}' column or empty dataset"],
            "recommendations": ["reject the file and re-request it from the source system"],
        }

    ts = pd.to_datetime(df[datetime_col], errors="coerce")
    unparsable = int(ts.isna().sum())
    valid = ts.dropna().sort_values()
    idx = pd.DatetimeIndex(valid)
    duplicates = int(idx.duplicated().sum())
    step = detect_timestep(pd.DatetimeIndex(idx.unique()))
    gaps = missing_timestamps(pd.DatetimeIndex(idx.unique()), step)
    missing_steps = int(sum(g["missing_steps"] for g in gaps))
    expected = missing_steps + len(idx.unique())
    coverage = float(len(idx.unique()) / expected) if expected else 0.0

    cols = value_cols or [c for c in df.columns if c != datetime_col and pd.api.types.is_numeric_dtype(df[c])]
    columns: dict[str, Any] = {}
    for col in cols:
        s = pd.to_numeric(df[col], errors="coerce")
        nan_count = int(s.isna().sum())
        neg_count = int((s < 0).sum()) if col in non_negative_cols else 0
        std = float(s.std(ddof=0)) if s.notna().any() else 0.0
        mean = float(s.mean()) if s.notna().any() else 0.0
        outliers = 0
        if std > 1e-9:
            outliers = int((np.abs(s - mean) > 5.0 * std).sum())
        flat = 0
        if s.notna().any():
            flat = int((s.diff().fillna(1.0) == 0).sum())
        columns[col] = {
            "nan_count": nan_count,
            "nan_pct": round(100.0 * nan_count / max(len(s), 1), 3),
            "negative_count": neg_count,
            "outlier_count_5sigma": outliers,
            "constant_run_steps": flat,
            "min": None if s.dropna().empty else round(float(s.min()), 4),
            "max": None if s.dropna().empty else round(float(s.max()), 4),
            "mean": None if s.dropna().empty else round(mean, 4),
        }
        if nan_count:
            severity = _worst(severity, "warning" if nan_count / max(len(s), 1) < 0.05 else "critical")
            recommendations.append(f"{name}.{col}: {nan_count} chýbajúcich hodnôt — interpolácia/ffill pred tréningom")
        if neg_count:
            severity = _worst(severity, "warning")
            recommendations.append(f"{name}.{col}: {neg_count} záporných hodnôt v nezápornej veličine — orezať na 0")
        if outliers:
            severity = _worst(severity, "info")
            recommendations.append(f"{name}.{col}: {outliers} hodnôt mimo 5σ — overiť meranie")
        if flat > max(8, int(0.5 * len(s))):
            severity = _worst(severity, "warning")
            recommendations.append(f"{name}.{col}: dlhý konštantný úsek ({flat} krokov) — možný zamrznutý senzor")

    if unparsable:
        severity = _worst(severity, "warning")
        recommendations.append(f"{name}: {unparsable} neparsovateľných timestampov — zjednotiť formát na ISO 8601")
    if duplicates:
        severity = _worst(severity, "warning")
        recommendations.append(f"{name}: {duplicates} duplicitných timestampov — deduplikácia keep=last")
    if missing_steps:
        severity = _worst(severity, "warning" if coverage > 0.95 else "critical")
        recommendations.append(
            f"{name}: {missing_steps} chýbajúcich krokov v {len(gaps)} dierach — resample + interpolácia"
        )

    report.update(
        {
            "time_range": {"start": str(idx.min()) if len(idx) else None, "end": str(idx.max()) if len(idx) else None},
            "expected_timestep_minutes": round(step.total_seconds() / 60.0, 2),
            "unparsable_timestamps": unparsable,
            "duplicate_timestamps": duplicates,
            "missing_steps": missing_steps,
            "gap_count": len(gaps),
            "gaps": gaps[:50],
            "coverage_ratio": round(coverage, 5),
            "columns": columns,
            "severity": severity,
            "recommendations": recommendations,
        }
    )
    return report


def apply_corrections(
    df: pd.DataFrame,
    *,
    datetime_col: str = "datetime",
    freq: str = "15min",
    non_negative_cols: tuple[str, ...] = ("load_kw", "pv_kw", "ghi_wm2"),
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Deduplicate, regularise on the expected grid, interpolate and clip. Returns audit counters."""
    counters = {"dropped_unparsable": 0, "dropped_duplicates": 0, "inserted_steps": 0, "imputed_values": 0, "clipped_negatives": 0}
    out = df.copy()
    ts = pd.to_datetime(out[datetime_col], errors="coerce")
    counters["dropped_unparsable"] = int(ts.isna().sum())
    out = out.loc[ts.notna()].copy()
    out[datetime_col] = ts.dropna().values
    before = len(out)
    out = out.drop_duplicates(subset=[datetime_col], keep="last").sort_values(datetime_col)
    counters["dropped_duplicates"] = before - len(out)
    if out.empty:
        return out, counters

    out = out.set_index(datetime_col)
    numeric = out.select_dtypes(include=[np.number])
    other = out.drop(columns=numeric.columns)
    regular = numeric.resample(freq).mean()
    counters["inserted_steps"] = int(max(0, len(regular) - len(numeric)))
    counters["imputed_values"] = int(regular.isna().sum().sum())
    regular = regular.interpolate(limit_direction="both")
    for col in non_negative_cols:
        if col in regular.columns:
            neg = int((regular[col] < 0).sum())
            if neg:
                counters["clipped_negatives"] += neg
                regular[col] = regular[col].clip(lower=0.0)
    if not other.empty:
        other = other.resample(freq).ffill()
        regular = regular.join(other, how="left").ffill()
    return regular.reset_index().rename(columns={"index": datetime_col}), counters


def build_quality_bundle(
    datasets: dict[str, pd.DataFrame],
    *,
    corrections: dict[str, dict[str, int]] | None = None,
) -> dict[str, Any]:
    reports = {name: profile_dataset(df, name=name) for name, df in datasets.items()}
    severity = _worst(*[r.get("severity", "ok") for r in reports.values()]) if reports else "ok"
    recommendations = [rec for r in reports.values() for rec in r.get("recommendations", [])]
    return {
        "format": "somes_data_quality_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "overall_severity": severity,
        "validated": severity != "critical",
        "datasets": reports,
        "corrections_applied": corrections or {},
        "correction_recommendations": recommendations,
    }


def quality_indicator_frame(bundle: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for name, rep in (bundle.get("datasets") or {}).items():
        rows.append(
            {
                "dataset": name,
                "rows": rep.get("rows"),
                "coverage_ratio": rep.get("coverage_ratio"),
                "missing_steps": rep.get("missing_steps"),
                "gap_count": rep.get("gap_count"),
                "duplicate_timestamps": rep.get("duplicate_timestamps"),
                "unparsable_timestamps": rep.get("unparsable_timestamps"),
                "timestep_minutes": rep.get("expected_timestep_minutes"),
                "severity": rep.get("severity"),
            }
        )
    return pd.DataFrame(rows)


def missing_data_frame(bundle: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for name, rep in (bundle.get("datasets") or {}).items():
        for gap in rep.get("gaps", []):
            rows.append({"dataset": name, **gap})
        for col, stats in (rep.get("columns") or {}).items():
            if stats.get("nan_count"):
                rows.append(
                    {
                        "dataset": name,
                        "gap_start": None,
                        "gap_end": None,
                        "missing_steps": stats["nan_count"],
                        "column": col,
                        "nan_pct": stats.get("nan_pct"),
                    }
                )
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["dataset", "gap_start", "gap_end", "missing_steps"])
