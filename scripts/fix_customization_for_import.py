"""Fix uc33_somes.customization so Domino UI Import accepts it."""
from __future__ import annotations
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
PATHS = [ROOT / "domino_import" / "uc33_somes.customization", ROOT / "domino_import" / "uc33_somes.json"]

def fix(data: dict) -> dict:
    for i, (_nid, piece) in enumerate(data.get("workflowPieces", {}).items(), start=1):
        if not isinstance(piece.get("id"), int):
            piece["id"] = 1000 + i
        if piece.get("repository_id") is None:
            piece["repository_id"] = 0
        if piece.get("secrets_schema") is None:
            piece["secrets_schema"] = {}
        cr = piece.get("container_resources") or {}
        if "use_gpu" not in cr:
            cr["use_gpu"] = False
            piece["container_resources"] = cr
    for pdata in data.get("workflowPiecesData", {}).values():
        for inp in (pdata.get("inputs") or {}).values():
            for k in ("value", "upstreamId", "upstreamArgument", "upstreamValue"):
                if inp.get(k) is None:
                    inp[k] = ""
    for edge in data.get("workflowEdges", []):
        if not str(edge.get("id", "")).startswith("reactflow__edge-"):
            src, tgt = edge["source"], edge["target"]
            edge["id"] = f"reactflow__edge-{src}source-{src}-{tgt}target-{tgt}"
    return data

def main() -> int:
    for path in PATHS:
        data = fix(json.loads(path.read_text(encoding="utf-8")))
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        piece = next(iter(data["workflowPieces"].values()))
        print(path.name, "id=", piece.get("id"))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
