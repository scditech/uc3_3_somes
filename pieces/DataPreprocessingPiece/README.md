# DataPreprocessingPiece

Clean, validate, and reshape raw smart-grid CSVs into modeling-ready datasets. Supports Open-Meteo PV CSV, OKTE-only, or merged dual-source inputs.

## Input

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `preprocessing_option` | `str` | `none` | `none`, `prediction`, `correction` |
| `data_path` | `str` | — | CSV from **OpenMeteoPVDataPiece** (or compatible irradiance CSV) |
| `data_path_okte` | `str` | — | Optional OKTE-style market CSV (inner-join on datetime when both set) |
| `target_column` | `str` | `PVOUT` | Target for prediction mode |
| `save_data_path` | `str` | auto | Override output path |
| `keep_datetime` | `bool` | — | Retain datetime column in features |
| `flag_each_day` | `bool` | — | Add day-boundary flags |
| `test_size` | `float` | — | Optional train/test split fraction |

## Output

| Mode | Fields |
|------|--------|
| `prediction` | `data_path` → `preprocessed.csv`, `feature_columns`, `target_column` |
| `correction` | `data_path_pred`, `data_path_true` (baseline vs truth), `target_column` = `PVOUT` |
| `none` | Status only |

Auto-infers numeric feature columns and builds `datetime` from `Date`+`Time` or `timestamp_utc`.

## Typical workflow

```text
OpenMeteoPVDataPiece
  → DataPreprocessingPiece
  → ModelDeciderPiece
```

## Running tests

```bash
pytest pieces/DataPreprocessingPiece/test_data_preprocessing_piece.py -v
```
