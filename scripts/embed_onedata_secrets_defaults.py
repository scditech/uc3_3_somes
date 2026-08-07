"""Embed OneData token defaults into onedata customization secrets_schema."""
from __future__ import annotations

import json
import re
from pathlib import Path

defaults = Path(r"C:\Users\NTB\Domino\uc3.3_somes\pieces\common\onedata_defaults.py").read_text(encoding="utf-8")
tok = re.search(r'DEFAULT_ONEDATA_TOKEN = \(\s*"([^"]+)"', defaults, re.S).group(1)
out_dir = "onedata:///SCDI/UC3.3_SOMES/outputs"

for p in [
    Path(r"C:\Users\NTB\Desktop\uc33_somes_onedata.customization"),
    Path(r"C:\Users\NTB\Desktop\somes_domino_import_test\uc33_somes_onedata.customization"),
    Path(r"C:\Users\NTB\Desktop\somes_domino_import_test\uc33_somes_onedata.json"),
]:
    if not p.exists():
        continue
    d = json.loads(p.read_text(encoding="utf-8"))
    for piece in d["workflowPieces"].values():
        props = (piece.get("secrets_schema") or {}).get("properties") or {}
        if "onedata_token" in props:
            props["onedata_token"]["default"] = tok
        if "onedata_output_dir" in props:
            props["onedata_output_dir"]["default"] = out_dir
        if "onedata_onezone_host" in props:
            props["onedata_onezone_host"]["default"] = "data.spice-platform.eu"
    p.write_text(json.dumps(d, indent=2), encoding="utf-8")
    print("updated", p)
