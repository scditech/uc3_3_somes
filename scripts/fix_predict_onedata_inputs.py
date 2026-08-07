"""Fix PredictPiece inputs in onedata customization for _pub_PredictPiece schema."""
from __future__ import annotations

import json
from pathlib import Path

FILES = [
    Path(r"C:\Users\NTB\Desktop\uc33_somes_onedata.customization"),
    Path(r"C:\Users\NTB\Desktop\somes_domino_import_test\uc33_somes_onedata.customization"),
    Path(r"C:\Users\NTB\Desktop\somes_domino_import_test\uc33_somes_onedata.json"),
]

LOAD = "onedata:///SCDI/UC3.3_SOMES/inputs/predict_in/load.csv"


def main() -> None:
    for path in FILES:
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        for nid, piece in data["workflowPieces"].items():
            if piece.get("name") != "PredictPiece":
                continue
            inputs = data["workflowPiecesData"][nid]["inputs"]
            # published Predict expects load_csv, not data_path
            if "data_path" in inputs and "load_csv" not in inputs:
                spec = inputs.pop("data_path")
                if isinstance(spec, dict) and not spec.get("fromUpstream"):
                    spec["value"] = LOAD
                inputs["load_csv"] = spec
            elif "data_path" in inputs:
                inputs.pop("data_path", None)
            if "load_csv" in inputs and not inputs["load_csv"].get("fromUpstream"):
                inputs["load_csv"]["value"] = LOAD
            # ensure published optional fields exist
            for key, default in (
                ("prediction_days", 1),
                ("timestep_minutes", 15),
                ("use_rolling_prediction", True),
                ("bridge_rows", 4),
            ):
                if key not in inputs:
                    inputs[key] = {
                        "fromUpstream": False,
                        "upstreamId": "",
                        "upstreamArgument": "",
                        "upstreamValue": "",
                        "value": default,
                    }
            print(path.name, "Predict inputs", list(inputs.keys()))
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
