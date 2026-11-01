"""Read load/history tables from CSV or parquet, including OneData staged files."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

_ENCODINGS = ("utf-8-sig", "utf-8", "cp1250", "latin-1")


def read_history_table(path: str | Path) -> pd.DataFrame:
    """Load a datetime-indexed energy table.

    FetchEnergyData writes parquet. IncrementalTrain / Forecast / AnomalyAlert
    used to call ``read_csv`` on that path, which dies with UnicodeDecodeError
    (byte 0x80) on the Harbor 0.3.0 image. Detect parquet by magic or suffix
    and fall back through encodings used by UC3.2.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"History/load file not found: {p}")
    head = p.read_bytes()[:4]
    if head == b"PAR1" or p.suffix.lower() == ".parquet":
        df = pd.read_parquet(p)
    else:
        last: Exception | None = None
        df = None
        for enc in _ENCODINGS:
            try:
                df = pd.read_csv(p, sep=None, engine="python", encoding=enc)
                break
            except UnicodeDecodeError as exc:
                last = exc
        if df is None:
            raise last or UnicodeDecodeError("utf-8", b"", 0, 1, str(p))
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    return df


def ensure_department_id(df: pd.DataFrame, default: str = "default") -> pd.DataFrame:
    """Load series such as measured_last_day.csv have no department column."""
    if "department_id" not in df.columns:
        df = df.copy()
        df["department_id"] = default
    return df
