"""Prepare uc3_3_somes publish artifacts for 0.2.1 OneData release."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(r"C:\Users\NTB\Domino\uc3_3_somes")
IMG = "ghcr.io/scditech/uc3_3_somes:0.2.1-group0"


def main() -> None:
    cfg = ROOT / "config.toml"
    text = cfg.read_text(encoding="utf-8")
    text = text.replace('VERSION = "0.2.0"', 'VERSION = "0.2.1"')
    cfg.write_text(text, encoding="utf-8")

    for name in ("uc33_somes.customization", "uc33_somes_onedata.customization"):
        path = ROOT / "domino_import" / name
        data = json.loads(path.read_text(encoding="utf-8"))
        for piece in data["workflowPieces"].values():
            if "source_image" in piece:
                piece["source_image"] = IMG
            props = (piece.get("secrets_schema") or {}).get("properties") or {}
            if "onedata_token" in props:
                props["onedata_token"]["default"] = ""
                props["onedata_token"]["description"] = (
                    "Optional; piece code defaults apply when empty"
                )
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        print("updated", name)

    # json twin for import folder
    custom = ROOT / "domino_import" / "uc33_somes.customization"
    (ROOT / "domino_import" / "uc33_somes.json").write_text(
        custom.read_text(encoding="utf-8"), encoding="utf-8"
    )

    req = ROOT / "dependencies" / "requirements_0.txt"
    rt = req.read_text(encoding="utf-8")
    if "fsspec" not in rt:
        rt = rt.rstrip() + "\nfsspec>=2023.9\nonedatafilerestclient>=25.0.0\n"
        req.write_text(rt, encoding="utf-8")

    imap = ROOT / ".domino" / "images_map.json"
    if imap.is_file():
        data = json.loads(imap.read_text(encoding="utf-8"))
        for k in list(data.keys()):
            data[k] = IMG
        imap.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

    print("done")


if __name__ == "__main__":
    main()
