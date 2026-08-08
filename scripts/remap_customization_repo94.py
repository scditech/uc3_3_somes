"""Remap SoMES customization to Domino repo 94 (0.2.1) schemas and fix PredictPiece."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(r"C:\Users\NTB\Domino\uc3_3_somes")
REPO_PIECES = Path(r"C:\Users\NTB\Desktop\somes_domino_import_test\repo94_pieces.json")
SRC = ROOT / "domino_import" / "uc33_somes.customization"
OUTS = [
    ROOT / "domino_import" / "uc33_somes.customization",
    ROOT / "domino_import" / "uc33_somes_onedata.customization",
    ROOT / "domino_import" / "uc33_somes.json",
    Path(r"C:\Users\NTB\Desktop\uc33_somes_onedata.customization"),
    Path(r"C:\Users\NTB\Desktop\uc33_somes_FIXED.customization"),
    Path(r"C:\Users\NTB\Desktop\somes_domino_import_test\uc33_somes_onedata.customization"),
    Path(r"C:\Users\NTB\Desktop\somes_domino_import_test\uc33_somes.customization"),
]

LOAD = "onedata:///SCDI/UC3.3_SOMES/inputs/predict_in/load.csv"


def empty_spec(prop: dict) -> dict:
    typ = prop.get("type")
    if typ is None and isinstance(prop.get("anyOf"), list):
        for opt in prop["anyOf"]:
            if isinstance(opt, dict) and opt.get("type") and opt.get("type") != "null":
                typ = opt["type"]
                break
    default = prop.get("default", None)
    if typ == "boolean":
        value = bool(default) if default is not None else False
    elif typ == "integer":
        value = int(default) if default is not None else 0
    elif typ == "number":
        value = float(default) if default is not None else 0.0
    elif typ == "array":
        value = default if isinstance(default, list) else []
    else:
        value = "" if default is None else default
    return {
        "fromUpstream": False,
        "upstreamId": "",
        "upstreamArgument": "",
        "upstreamValue": "",
        "value": value,
    }


def main() -> None:
    repo = json.loads(REPO_PIECES.read_text(encoding="utf-8"))
    data = json.loads(SRC.read_text(encoding="utf-8"))

    for nid, piece in data["workflowPieces"].items():
        name = piece["name"]
        meta = repo.get(name)
        if not meta:
            print("WARN no repo piece", name)
            continue
        piece["id"] = meta["id"]
        piece["repository_id"] = 94
        piece["source_image"] = meta.get("source_image") or "ghcr.io/scditech/uc3_3_somes:0.2.1-group0"
        if meta.get("input_schema"):
            piece["input_schema"] = meta["input_schema"]
        if meta.get("output_schema"):
            piece["output_schema"] = meta["output_schema"]
        if meta.get("secrets_schema") is not None:
            piece["secrets_schema"] = meta["secrets_schema"]

        inputs = data["workflowPiecesData"][nid].setdefault("inputs", {})
        props = (piece.get("input_schema") or {}).get("properties") or {}

        # Predict: ensure load_csv, drop obsolete data_path
        if name == "PredictPiece":
            inputs.pop("data_path", None)
            if "load_csv" not in inputs or not inputs["load_csv"].get("fromUpstream"):
                inputs["load_csv"] = empty_spec(props.get("load_csv") or {"type": "string"})
                inputs["load_csv"]["value"] = LOAD

        for key, prop in props.items():
            if key in inputs:
                continue
            inputs[key] = empty_spec(prop)

        # drop inputs not in schema (except keep nothing extra that breaks Yup - Yup only checks schema keys)
        extras = [k for k in list(inputs) if k not in props]
        for k in extras:
            # keep nothing outside schema
            del inputs[k]

        # re-apply predict load value after extras cleanup
        if name == "PredictPiece" and "load_csv" in inputs and not inputs["load_csv"].get("fromUpstream"):
            inputs["load_csv"]["value"] = LOAD

    # audit predict
    for nid, piece in data["workflowPieces"].items():
        if piece["name"] != "PredictPiece":
            continue
        props = set((piece.get("input_schema") or {}).get("properties") or {})
        inputs = set(data["workflowPiecesData"][nid]["inputs"])
        print("Predict schema", sorted(props))
        print("Predict inputs", sorted(inputs))
        print("missing", sorted(props - inputs))
        print("id", piece["id"], "repo", piece["repository_id"], piece["source_image"])

    text = json.dumps(data, indent=2)
    for out in OUTS:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print("wrote", out)


if __name__ == "__main__":
    main()
