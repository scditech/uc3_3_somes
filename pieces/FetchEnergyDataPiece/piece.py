try:
    from domino.base_piece import BasePiece
except ModuleNotFoundError:
    from local_compat.base_piece import BasePiece
from .models import InputModel, OutputModel
import pandas as pd
from pathlib import Path
import traceback


def _to_numeric_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace(",", ".", regex=False), errors="coerce")


def _pick_datetime_column(df: pd.DataFrame) -> str | None:
    aliases = {"datetime", "date time", "date_time", "timestamp", "time"}
    for c in df.columns:
        norm = str(c).replace("\ufeff", "").strip().lower()
        if norm in aliases:
            return c
    return None


def _parse_datetimes(values: pd.Series) -> pd.Series:
    """Parse timestamps ISO-first.

    Falling straight through to ``dayfirst=True`` silently rewrites ISO dates
    such as 2025-11-01 into 11 January, which shreds the timeline and shows up
    only later as enormous gaps in the data-quality report.
    """
    text = values.astype(str).str.strip()
    iso = pd.to_datetime(text, format="ISO8601", errors="coerce")
    if iso.notna().any():
        return iso
    european = pd.to_datetime(text, format="%d.%m.%y %H:%M", errors="coerce")
    if european.notna().any():
        return european
    return pd.to_datetime(text, dayfirst=True, errors="coerce")


def _normalize_load_frame(raw: pd.DataFrame, source_name: str) -> pd.DataFrame:
    dt_col = _pick_datetime_column(raw)
    if dt_col is None:
        raise ValueError(f"{source_name}: missing datetime column")

    df = raw.copy()
    df["datetime"] = _parse_datetimes(df[dt_col])
    df = df.dropna(subset=["datetime"])
    if dt_col != "datetime":
        df = df.drop(columns=[dt_col])

    if "load_kw" in df.columns:
        df["load_kw"] = _to_numeric_series(df["load_kw"]).fillna(0.0)
        if "department_id" not in df.columns:
            dept = source_name.replace("load_", "") or "default"
            df["department_id"] = dept if dept != "load" else "default"
        return df[["datetime", "department_id", "load_kw"]]

    reserved = {"department_id", "price_eur_kwh", "price_eur_mwh"}
    value_cols = [c for c in df.columns if c not in reserved]
    if not value_cols:
        raise ValueError(f"{source_name}: no load columns found")

    long = df.melt(
        id_vars=["datetime"],
        value_vars=value_cols,
        var_name="department_id",
        value_name="load_kw",
    )
    long["department_id"] = long["department_id"].astype(str).str.strip().str.replace("prikon ", "", case=False)
    long["load_kw"] = _to_numeric_series(long["load_kw"]).fillna(0.0)
    return long[["datetime", "department_id", "load_kw"]]


def _log(results_path: str | None, message: str) -> None:
    print(message)
    if not results_path:
        return
    p = Path(results_path)
    p.mkdir(parents=True, exist_ok=True)
    with open(p / "fetch_energy_data.log", "a", encoding="utf-8") as f:
        f.write(message + "\n")


class FetchEnergyDataPiece(BasePiece):
    """
    Load and merge energy CSV files from shared storage
    """

    def piece_function(self, input_data: InputModel) -> OutputModel:
        try:
            _log(self.results_path, "[INFO] FetchEnergyDataPiece started")
            _log(self.results_path, f"[INFO] Load CSV: {input_data.load_csv}")
            _log(self.results_path, f"[INFO] Prices CSV: {input_data.prices_csv}")

            load_csv = Path(input_data.load_csv)
            prices_csv = Path(input_data.prices_csv)

            prices_df = pd.DataFrame()
            if prices_csv and prices_csv.is_file():
                prices_df = pd.read_csv(prices_csv, parse_dates=["datetime"])
                prices_df = prices_df.set_index("datetime")
            else:
                _log(self.results_path, "[WARN] Prices CSV not found; continuing without price columns")

            if load_csv.is_dir():
                load_files = sorted(load_csv.glob("load*.csv"))
                if not load_files:
                    load_files = [p for p in sorted(load_csv.glob("*.csv")) if "price" not in p.name.lower()]
            elif load_csv.is_file():
                load_files = [load_csv]
            else:
                load_files = []
            if not load_files:
                message = f"No load CSV files found at: {load_csv}"
                _log(self.results_path, f"[ERROR] {message}")
                return OutputModel(message=message, output_path="")

            _log(self.results_path, "[INFO] Reading CSV files")
            merged_parts = []
            for lf in load_files:
                raw = pd.read_csv(lf, sep=None, engine="python")
                if len(raw.columns) == 1 and ";" in str(raw.columns[0]):
                    raw = pd.read_csv(lf, sep=";")
                load_df = _normalize_load_frame(raw, lf.stem).set_index("datetime")
                if not prices_df.empty:
                    part = load_df.join(prices_df, how="left").reset_index()
                else:
                    part = load_df.reset_index()
                merged_parts.append(part)

            _log(self.results_path, "[INFO] Merging data")
            merged_df = pd.concat(merged_parts, ignore_index=True)
            merged_df = merged_df.sort_values(["department_id", "datetime"]).reset_index(drop=True)

            if "price_eur_mwh" in merged_df.columns:
                merged_df["price_eur_mwh"] = merged_df.groupby("department_id")["price_eur_mwh"].ffill().bfill()
            if "price_eur_kwh" in merged_df.columns:
                merged_df["price_eur_kwh"] = merged_df.groupby("department_id")["price_eur_kwh"].ffill().bfill()

            output_path = Path(self.results_path) / "merged_energy_data.parquet"
            merged_df.to_parquet(output_path, index=False)

            _log(self.results_path, f"[SUCCESS] Data merged, rows: {len(merged_df)}")
            _log(self.results_path, f"[SUCCESS] Output written to {output_path}")

            self.display_result = {"file_type": "parquet", "file_path": str(output_path)}
            return OutputModel(
                message=f"Data merged successfully ({len(merged_df)} rows)",
                output_path=str(output_path),
            )
        except Exception:
            err = traceback.format_exc()
            _log(self.results_path, f"[ERROR] {err}")
            if self.results_path:
                with open(Path(self.results_path) / "fetch_energy_data_error.txt", "w", encoding="utf-8") as f:
                    f.write(err)
            raise
