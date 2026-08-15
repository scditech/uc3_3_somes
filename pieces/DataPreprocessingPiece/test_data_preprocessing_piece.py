from domino.testing import piece_dry_run
import os
import pytest
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.preprocessor_utils import (
    ensure_datetime_column,
    preprocess_solargis_data,
)
from utils.modes import preprocess_prediction


def test_data_preprocessing_piece_smoke():
    output_data = piece_dry_run(
        "DataPreprocessingPiece",
        {"payload": {}},
    )
    assert output_data["message"] is not None


def test_data_preprocessing_piece_none_mode():
    output_data = piece_dry_run(
        "DataPreprocessingPiece",
        {"payload": {"preprocessing_option": "none"}},
    )
    assert output_data["message"].endswith("(none).")


def test_data_preprocessing_piece_none_mode_alias():
    output_data = piece_dry_run(
        "DataPreprocessingPiece",
        {"payload": {"mode": "none"}},
    )
    assert output_data["message"].endswith("(none).")


def test_data_preprocessing_piece_invalid_option_raises():
    if os.environ.get("PIECES_IMAGES_MAP"):
        pytest.skip("Skipping expected-exception assertion in HTTP dry-run mode.")
    with pytest.raises(ValueError, match=r"Invalid preprocessing option"):
        piece_dry_run(
            "DataPreprocessingPiece",
            {"payload": {"preprocessing_option": "does_not_exist"}},
        )


def test_preprocess_prediction_infers_features_when_missing():
    payload = {
        "dataframe": pd.DataFrame(
            {
                "datetime": ["2026-05-07 10:00:00", "2026-05-07 11:00:00"],
                "GHI": [50.0, 60.0],
                "DIF": [10.0, 15.0],
                "SE": [20.0, 25.0],
                "PVOUT": [30.0, 35.0],
            }
        ),
        "preprocessing_option": "prediction",
    }

    result = preprocess_prediction(payload)
    features = result["artifacts"]["features"]

    assert "PVOUT" not in features
    assert "GHI" in features


def test_preprocess_prediction_supports_solargis_date_time_columns():
    payload = {
        "dataframe": pd.DataFrame(
            {
                "Date": ["07.05.2026", "07.05.2026"],
                "Time": ["10:00", "11:00"],
                "GHI": [50.0, 60.0],
                "DIF": [10.0, 15.0],
                "SE": [20.0, 25.0],
                "PVOUT": [30.0, 35.0],
            }
        ),
        "preprocessing_option": "prediction",
        "keep_datetime": True,
    }

    result = preprocess_prediction(payload)
    features = result["artifacts"]["features"]

    assert "datetime" in features
    assert "GHI" in features


def test_ensure_datetime_column_from_datetime_schema():
    data = pd.DataFrame(
        {
            "datetime": ["11.05.2026 13:18"],
            "GHI": [912.81],
            "DIF": [246.47],
            "SE": [70.7],
            "PVOUT": [4.218],
        }
    )
    out = ensure_datetime_column(data)
    assert "datetime" in out.columns
    assert str(out["datetime"].dtype).startswith("datetime64")


def test_ensure_datetime_column_from_date_time_schema():
    data = pd.DataFrame(
        {
            "Date": ["11.05.2026"],
            "Time": ["13:18"],
            "GHI": [912.81],
            "DIF": [246.47],
            "SE": [70.7],
            "PVOUT": [4.218],
        }
    )
    out = ensure_datetime_column(data)
    assert "datetime" in out.columns
    assert str(out["datetime"].dtype).startswith("datetime64")


def test_ensure_datetime_column_missing_schema_raises():
    data = pd.DataFrame({"GHI": [100], "DIF": [10], "SE": [20], "PVOUT": [1.0]})
    with pytest.raises(ValueError, match=r"either a `datetime` column"):
        ensure_datetime_column(data)


def test_preprocess_solargis_data_accepts_date_time_schema():
    data = pd.DataFrame(
        {
            "Date": ["11.05.2026"],
            "Time": ["13:18"],
            "GHI": [912.81],
            "DIF": [246.47],
            "SE": [70.7],
            "PVOUT": [4.218],
        }
    )
    out = preprocess_solargis_data(data)
    assert "datetime" in out.columns
    assert "hour_of_day" in out.columns


def test_preprocess_prediction_merges_open_meteo_and_okte(tmp_path):
    """
    When both `data_path` and `data_path_okte` are wired, the piece
    should inner-join the two on `datetime` and emit one merged dataset whose
    feature pool includes columns from both sources.
    """
    weather_path = tmp_path / "open_meteo.csv"
    weather_df = pd.DataFrame(
        {
            "datetime": [
                "2026-05-07 10:00:00",
                "2026-05-07 10:15:00",
                "2026-05-07 10:30:00",
            ],
            "GHI": [500.0, 520.0, 540.0],
            "DIF": [100.0, 110.0, 120.0],
            "SE": [50.0, 52.0, 54.0],
            "PVOUT": [4.1, 4.3, 4.5],
        }
    )
    weather_df.to_csv(weather_path, index=False)

    okte_path = tmp_path / "okte.csv"
    okte_df = pd.DataFrame(
        {
            "Date": ["07.05.2026", "07.05.2026", "07.05.2026"],
            "Time": ["10:00", "10:15", "10:30"],
            "spot_price_eur_mwh": [80.0, 82.0, 79.0],
            "imbalance_mw": [10.0, -5.0, 3.0],
            "scheduled_generation_mw": [2000.0, 2050.0, 2100.0],
            "actual_generation_mw": [1990.0, 2060.0, 2080.0],
        }
    )
    okte_df.to_csv(okte_path, index=False)

    result = preprocess_prediction(
        {
            "preprocessing_option": "prediction",
            "data_path": str(weather_path),
            "data_path_okte": str(okte_path),
            "target_column": "PVOUT",
        }
    )

    features = result["artifacts"]["features"]
    assert "GHI" in features
    assert "spot_price_eur_mwh" in features
    assert "imbalance_mw" in features
    # The chosen target is excluded from features.
    assert "PVOUT" not in features


def test_preprocess_prediction_data_path_open_meteo(tmp_path):
    """Open-Meteo CSV wired via `data_path` alone is enough for prediction mode."""
    weather_path = tmp_path / "open_meteo_only.csv"
    pd.DataFrame(
        {
            "datetime": ["2026-05-07 10:00:00", "2026-05-07 10:15:00"],
            "GHI": [500.0, 520.0],
            "DIF": [100.0, 110.0],
            "SE": [50.0, 52.0],
            "PVOUT": [4.1, 4.3],
        }
    ).to_csv(weather_path, index=False)

    result = preprocess_prediction(
        {
            "preprocessing_option": "prediction",
            "data_path": str(weather_path),
        }
    )
    assert result["artifacts"]["target_column"] == "PVOUT"
    assert "GHI" in result["artifacts"]["features"]
