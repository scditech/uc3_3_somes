"""Build uc33_somes_onedata.customization from local fixed file + OneData paths."""
from __future__ import annotations

import json
from pathlib import Path

SRC = Path(r"C:\Users\NTB\Desktop\somes_domino_import_test\uc33_somes.customization")
OUT_DIR = Path(r"C:\Users\NTB\Desktop\somes_domino_import_test")
OUT = OUT_DIR / "uc33_somes_onedata.customization"
OUT_DESKTOP = Path(r"C:\Users\NTB\Desktop\uc33_somes_onedata.customization")

BASE = "onedata:///SCDI/UC3.3_SOMES/inputs"
OUT_BASE = "onedata:///SCDI/UC3.3_SOMES/outputs"

# Map piece input field -> OneData path (static seeds). Upstream-wired fields stay.
STATIC_PATHS = {
    "FetchEnergyDataPiece": {
        "load_csv": f"{BASE}/load_input",
        "prices_csv": f"{BASE}/prices.csv",
    },
    "IncrementalTrainPiece": {
        "history_csv": f"{BASE}/history_by_department.csv",
        "model_registry_dir": f"{BASE}/model_registry",
    },
    "ForecastHorizonPiece": {
        "history_csv": f"{BASE}/history_by_department.csv",
        "model_registry_dir": f"{BASE}/model_registry",
    },
    "SomesConnectorsPiece": {
        "connectors_dir": f"{BASE}/connectors",
    },
    "PredictPiece": {
        "load_csv": f"{BASE}/predict_in/load.csv",
    },
    "SolarSimPiece": {
        "load_csv": f"{BASE}/measured_last_day.csv",
        "scenario_yaml": f"{BASE}/scenario.yaml",
        "weather_forecast_csv": f"{BASE}/weather_history.csv",
        "solargis_csv": f"{BASE}/solargis_irradiance.csv",
    },
    "BatteryStrategyOptimizerPiece": {
        "scenario_yaml": f"{BASE}/scenario.yaml",
    },
    "BatterySimPiece": {
        "load_csv": f"{BASE}/measured_last_day.csv",
        "scenario_yaml": f"{BASE}/scenario.yaml",
    },
    "FlexibleLoadSchedulePiece": {
        "scenario_yaml": f"{BASE}/scenario.yaml",
    },
    "GridFeasibilityPiece": {
        "scenario_yaml": f"{BASE}/scenario.yaml",
    },
    "ModelMonitoringPiece": {
        "history_csv": f"{BASE}/measured_last_day.csv",
    },
}

SECRETS_SCHEMA = {
    "title": "SecretsModel",
    "type": "object",
    "properties": {
        "onedata_onezone_host": {
            "anyOf": [{"type": "string"}, {"type": "null"}],
            "default": "data.spice-platform.eu",
            "description": "OneData Onezone host",
            "title": "Onedata Onezone Host",
        },
        "onedata_token": {
            "anyOf": [{"type": "string"}, {"type": "null"}],
            "default": "",
            "description": "OneData access token (optional if piece defaults apply)",
            "title": "Onedata Token",
        },
        "onedata_output_dir": {
            "anyOf": [{"type": "string"}, {"type": "null"}],
            "default": OUT_BASE,
            "description": "Base dir for per-run outputs",
            "title": "Onedata Output Dir",
        },
    },
}


def _spec(value: str) -> dict:
    return {
        "fromUpstream": False,
        "upstreamId": "",
        "upstreamArgument": "",
        "upstreamValue": "",
        "value": value,
    }


def main() -> None:
    data = json.loads(SRC.read_text(encoding="utf-8"))
    pieces = data["workflowPieces"]
    pdata = data["workflowPiecesData"]

    # Find Fetch node for run_id upstream
    fetch_nid = None
    fetch_tid = None
    for nid, piece in pieces.items():
        if piece.get("name") == "FetchEnergyDataPiece":
            fetch_nid = nid
            uuid = nid.split("_", 1)[-1].replace("-", "")
            fetch_tid = f"FetchEnerg_{uuid}"
            break

    for nid, piece in pieces.items():
        name = piece["name"]
        piece["secrets_schema"] = SECRETS_SCHEMA
        # strip SOMES_DISABLE hint if any
        inputs = pdata[nid].setdefault("inputs", {})

        # Ensure run_id field exists
        if name == "FetchEnergyDataPiece":
            inputs.setdefault("run_id", _spec(""))
        elif fetch_tid:
            # Prefer shared run_id from Fetch when edge exists; otherwise empty (results_path fallback)
            if "run_id" not in inputs or not inputs["run_id"].get("fromUpstream"):
                # only set upstream if we don't break nodes that already have values
                inputs["run_id"] = {
                    "fromUpstream": False,
                    "upstreamId": "",
                    "upstreamArgument": "",
                    "upstreamValue": "",
                    "value": "",
                }

        mapping = STATIC_PATHS.get(name, {})
        for field, path in mapping.items():
            if field not in inputs:
                continue
            spec = inputs[field]
            if isinstance(spec, dict) and spec.get("fromUpstream"):
                continue  # keep upstream wiring
            # For SolarSim backtest node, measured_last_day is correct;
            # for forecast SolarSim, load_csv may be fromUpstream site_forecast — skip if upstream
            inputs[field] = _spec(path)

        # Storage access mode stays Read/Write; Domino Settings still Local for results mount
        st = pdata[nid].setdefault("storage", {})
        st["storageAccessMode"] = "Read/Write"

    # Also replace any leftover /home/shared_storage/somes paths in static values
    for nid, block in pdata.items():
        for key, spec in (block.get("inputs") or {}).items():
            if not isinstance(spec, dict) or spec.get("fromUpstream"):
                continue
            val = spec.get("value")
            if isinstance(val, str) and val.startswith("/home/shared_storage/somes"):
                rel = val[len("/home/shared_storage/somes") :].lstrip("/")
                spec["value"] = f"{BASE}/{rel}" if rel else BASE

    OUT.write_text(json.dumps(data, indent=2), encoding="utf-8")
    OUT_DESKTOP.write_text(json.dumps(data, indent=2), encoding="utf-8")
    (OUT_DIR / "uc33_somes_onedata.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
    print("wrote", OUT)
    print("wrote", OUT_DESKTOP)


if __name__ == "__main__":
    main()
