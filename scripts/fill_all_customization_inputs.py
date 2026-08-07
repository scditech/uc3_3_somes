"""Fill every input_schema property into workflowPiecesData so Domino UI Create passes Yup."""
from __future__ import annotations

import json
from pathlib import Path

OUT_DIR = Path(r"C:\Users\NTB\Desktop\somes_domino_import_test")
PATHS = [
    OUT_DIR / "uc33_somes.customization",
    OUT_DIR / "uc33_somes.json",
    Path(r"C:\Users\NTB\Desktop\uc33_somes.customization"),
]


def default_value(prop: dict):
    if "default" in prop and prop["default"] is not None:
        return prop["default"]
    if "anyOf" in prop:
        # prefer non-null branch
        types = [x.get("type") for x in prop["anyOf"] if isinstance(x, dict)]
        if "string" in types:
            return ""
        if "number" in types or "integer" in types:
            return 0
        if "boolean" in types:
            return False
        return ""
    t = prop.get("type")
    if t == "string":
        return ""
    if t == "integer":
        return 0
    if t == "number":
        return 0.0
    if t == "boolean":
        return False
    return ""


def empty_spec(prop: dict) -> dict:
    return {
        "fromUpstream": False,
        "upstreamId": "",
        "upstreamArgument": "",
        "upstreamValue": "",
        "value": default_value(prop),
    }


def main() -> int:
    src = OUT_DIR / "uc33_somes.customization"
    data = json.loads(src.read_text(encoding="utf-8"))
    filled = 0
    for nid, piece in data["workflowPieces"].items():
        props = (piece.get("input_schema") or {}).get("properties") or {}
        inputs = data["workflowPiecesData"][nid].setdefault("inputs", {})
        for key, prop in props.items():
            if key in inputs:
                spec = inputs[key]
                # normalize nulls
                if spec.get("fromUpstream"):
                    for f in ("upstreamId", "upstreamArgument", "upstreamValue"):
                        if not spec.get(f):
                            # keep failure visible
                            pass
                    if spec.get("value") is None:
                        spec["value"] = ""
                else:
                    if spec.get("value") is None:
                        spec["value"] = default_value(prop if isinstance(prop, dict) else {})
                    spec["upstreamId"] = spec.get("upstreamId") or ""
                    spec["upstreamArgument"] = spec.get("upstreamArgument") or ""
                    spec["upstreamValue"] = spec.get("upstreamValue") or ""
                continue
            inputs[key] = empty_spec(prop if isinstance(prop, dict) else {})
            filled += 1

        # flatten container resources
        cr = data["workflowPiecesData"][nid].get("containerResources") or {}
        if isinstance(cr.get("cpu"), dict):
            cr = {
                "cpu": int(cr["cpu"].get("max", cr["cpu"].get("min", 500))),
                "memory": int(cr["memory"].get("max", cr["memory"].get("min", 2048))),
                "useGpu": bool(cr.get("useGpu", False)),
            }
        cr["cpu"] = max(100, min(10000, int(cr.get("cpu", 500))))
        cr["memory"] = max(128, min(24000, int(cr.get("memory", 2048))))
        cr["useGpu"] = bool(cr.get("useGpu", False))
        data["workflowPiecesData"][nid]["containerResources"] = cr
        data["workflowPiecesData"][nid]["storage"] = {"storageAccessMode": "Read/Write"}

        # clear node validation flags
    for node in data["workflowNodes"]:
        node.setdefault("data", {})
        node["data"]["validationError"] = False
        node["data"]["orientation"] = node["data"].get("orientation") or "horizontal"

    payload = json.dumps(data, indent=2, ensure_ascii=False)
    for path in PATHS:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload, encoding="utf-8")

    # verify no missing keys remain
    missing_nodes = []
    for nid, piece in data["workflowPieces"].items():
        props = set(((piece.get("input_schema") or {}).get("properties") or {}))
        used = set(data["workflowPiecesData"][nid]["inputs"])
        miss = sorted(props - used)
        if miss:
            missing_nodes.append((piece["name"], miss))

    readme = OUT_DIR / "README.txt"
    readme.write_text(
        "\n".join(
            [
                "OPRAVENY import pre Domino UI Create",
                "",
                "Subor:",
                str(OUT_DIR / "uc33_somes.customization"),
                "",
                "Postup:",
                "1. Workspace: test",
                "2. Workflow editor -> Clear (povinne, vyhod stary import z pamate)",
                "3. Import -> from file -> uc33_somes.customization z tohto priecinka",
                "4. Settings: Name len pismena/cisla/podciarkovnik (napr. SoMESFullTest)",
                "5. Create",
                "",
                "Oprava: Domino Yup validuje KAZDE pole zo schema, aj optional.",
                "Doplnene chybajuce inputs (EMS auth, forecast_hours, peak_price, ...).",
                "",
                f"Doplnenych poli: {filled}",
                f"Zostava missing: {missing_nodes}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps({"filled": filled, "missing": missing_nodes, "out": str(OUT_DIR)}, indent=2))
    return 0 if not missing_nodes else 1


if __name__ == "__main__":
    raise SystemExit(main())
