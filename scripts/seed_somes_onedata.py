"""Seed SoMES ops inputs into onedata:///SCDI/UC3.3_SOMES/inputs from local shared_storage."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pieces"))

from common import onedata_io as od  # noqa: E402
from common.onedata_defaults import (  # noqa: E402
    DEFAULT_INPUT_DIR,
    DEFAULT_ONEDATA_TOKEN,
    DEFAULT_ONEZONE_HOST,
    DEFAULT_OUTPUT_DIR,
)

HOST_SHARED = Path(r"C:\Users\NTB\domino_data\somes")
REMOTE_BASE = DEFAULT_INPUT_DIR  # onedata:///SCDI/UC3.3_SOMES/inputs

# local relative -> remote relative under inputs/
MAP = [
    ("load_input", "load_input"),
    ("prices.csv", "prices.csv"),
    ("scenario.yaml", "scenario.yaml"),
    ("history_by_department.csv", "history_by_department.csv"),
    ("measured_last_day.csv", "measured_last_day.csv"),
    ("weather_history.csv", "weather_history.csv"),
    ("solargis_irradiance.csv", "solargis_irradiance.csv"),
    ("predict_in/load.csv", "predict_in/load.csv"),
    ("connectors", "connectors"),
]


def main() -> int:
    secrets = {
        "onedata_onezone_host": DEFAULT_ONEZONE_HOST,
        "onedata_token": DEFAULT_ONEDATA_TOKEN,
        "onedata_output_dir": DEFAULT_OUTPUT_DIR,
    }
    if not od.configure_onedata(secrets, force=True):
        print("FAIL configure_onedata")
        return 1

    od.makedirs(REMOTE_BASE, exist_ok=True)
    od.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)
    od.makedirs(f"{REMOTE_BASE}/model_registry", exist_ok=True)

    for local_rel, remote_rel in MAP:
        src = HOST_SHARED / local_rel
        dst = f"{REMOTE_BASE.rstrip('/')}/{remote_rel}"
        if not src.exists():
            print("SKIP missing", src)
            continue
        if src.is_dir():
            od.makedirs(dst, exist_ok=True)
            for f in sorted(src.rglob("*")):
                if f.is_file():
                    rel = f.relative_to(src).as_posix()
                    target = f"{dst.rstrip('/')}/{rel}"
                    print("UPLOAD", f, "->", target)
                    od.write_bytes(target, f.read_bytes())
        else:
            print("UPLOAD", src, "->", dst)
            od.write_bytes(dst, src.read_bytes())

    print("DONE seed ->", REMOTE_BASE)
    print("outputs ->", DEFAULT_OUTPUT_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
