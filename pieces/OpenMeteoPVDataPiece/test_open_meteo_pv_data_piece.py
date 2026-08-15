import csv
import json
import os
from pathlib import Path

from domino.testing import piece_dry_run


def test_open_meteo_json_output():
    output = piece_dry_run(
        "OpenMeteoPVDataPiece",
        {
            "latitude": 48.15,
            "longitude": 17.11,
            "start_date": "2024-06-01",
            "end_date": "2024-06-01",
            "pvout_peak_kw": 5.0,
            "panel_tilt": 30.0,
            "output_format": "json",
        },
    )
    file_path = output["file_path"]
    assert file_path is not None
    assert file_path.endswith(".json")

    if os.environ.get("PIECES_IMAGES_MAP"):
        return

    records = json.loads(Path(file_path).read_text(encoding="utf-8"))
    first = records[0]
    for col in (
        "Date",
        "Time",
        "GHI",
        "DNI",
        "DIF",
        "GTI",
        "SE",
        "SA",
        "PVOUT",
        "TEMP",
        "WS",
        "WG",
        "WD",
        "RH",
        "AP",
        "PVOUT_UNC_LOW",
        "PVOUT_UNC_HIGH",
    ):
        assert col in first, f"Missing column: {col}"
