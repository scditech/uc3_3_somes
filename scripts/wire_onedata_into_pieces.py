"""Wire OneData into SoMES ops pieces (models + piece_function stage/finish)."""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PIECES = ROOT / "pieces"

OPS = [
    "FetchEnergyDataPiece",
    "PreprocessEnergyDataPiece",
    "TrainModelPiece",
    "PredictPiece",
    "IncrementalTrainPiece",
    "ForecastHorizonPiece",
    "SomesConnectorsPiece",
    "PriceForecastPiece",
    "SolarSimPiece",
    "BatteryStrategyOptimizerPiece",
    "BatterySimPiece",
    "FlexibleLoadSchedulePiece",
    "ModelMonitoringPiece",
    "GridFeasibilityPiece",
    "DashboardPiece",
    "EmsBemsOutputPiece",
]

GENERATE_RUN_ID = {
    "FetchEnergyDataPiece",
    "SomesConnectorsPiece",
    "IncrementalTrainPiece",
}

MODELS_IMPORT = """from pydantic import BaseModel, Field

try:
    from common.onedata_models import OneDataSecretsModel, RunIdInputMixin
except ModuleNotFoundError:
    from pieces.common.onedata_models import OneDataSecretsModel, RunIdInputMixin
"""

OD_IMPORT = """
try:
    from common import onedata_io as od
except ModuleNotFoundError:
    try:
        from pieces.common import onedata_io as od
    except ModuleNotFoundError:
        od = None
"""


def _replace_input_base(text: str) -> str:
    if "RunIdInputMixin" not in text:
        if "from pydantic import BaseModel, Field" in text:
            text = text.replace("from pydantic import BaseModel, Field", MODELS_IMPORT, 1)
        else:
            text = MODELS_IMPORT + "\n" + text
    text = re.sub(r"class InputModel\(BaseModel\):", "class InputModel(RunIdInputMixin):", text, count=1)
    if "class SecretsModel" not in text:
        if "class OutputModel" in text:
            text = text.replace(
                "class OutputModel",
                "class SecretsModel(OneDataSecretsModel):\n    pass\n\n\nclass OutputModel",
                1,
            )
        else:
            text += "\n\nclass SecretsModel(OneDataSecretsModel):\n    pass\n"
    return text


def _ensure_output_run_id(text: str, piece: str) -> str:
    if piece not in GENERATE_RUN_ID:
        return text
    out = text.split("class OutputModel", 1)
    if len(out) < 2:
        return text
    if re.search(r"\brun_id\s*:", out[1]):
        return text
    return text.replace(
        "class OutputModel(BaseModel):",
        'class OutputModel(BaseModel):\n    run_id: str = Field(default="", description="OneData run folder id")',
        1,
    )


def _find_matching_paren(s: str, open_idx: int) -> int:
    depth = 0
    i = open_idx
    while i < len(s):
        ch = s[i]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def _rewrite_output_returns(text: str, piece_name: str) -> tuple[str, int]:
    """Rewrite `return OutputModel(...)` (possibly multiline) to finish_piece."""
    count = 0
    out = []
    i = 0
    while True:
        m = re.search(r"^([ \t]+)return OutputModel\(", text[i:], flags=re.M)
        if not m:
            out.append(text[i:])
            break
        abs_start = i + m.start()
        indent = m.group(1)
        open_paren = i + m.end() - 1  # position of '('
        close = _find_matching_paren(text, open_paren)
        if close < 0:
            out.append(text[i:])
            break
        # include trailing whitespace/newline after )
        end = close + 1
        expr = text[abs_start + len(indent) + len("return ") : end]
        out.append(text[i:abs_start])
        block = (
            f"{indent}_piece_out = {expr}\n"
            f"{indent}if od is not None:\n"
            f"{indent}    if hasattr(_piece_out, 'run_id') and _run_id and not getattr(_piece_out, 'run_id', ''):\n"
            f"{indent}        try:\n"
            f"{indent}            _piece_out.run_id = _run_id\n"
            f"{indent}        except Exception:\n"
            f"{indent}            pass\n"
            f"{indent}    return od.finish_piece(\n"
            f"{indent}        _piece_out, self.results_path, secrets_data, \"{piece_name}\", _stage, run_id=_run_id\n"
            f"{indent}    )\n"
            f"{indent}if _stage is not None:\n"
            f"{indent}    _stage.cleanup()\n"
            f"{indent}return _piece_out"
        )
        out.append(block)
        count += 1
        i = end
    return "".join(out), count


def _patch_models(piece_dir: Path) -> None:
    models = piece_dir / "models.py"
    text = models.read_text(encoding="utf-8")
    text = _replace_input_base(text)
    text = _ensure_output_run_id(text, piece_dir.name)
    models.write_text(text, encoding="utf-8")


def _patch_piece(piece_dir: Path) -> None:
    piece = piece_dir / "piece.py"
    text = piece.read_text(encoding="utf-8")
    name = piece_dir.name
    if "stage_inputs" in text and "finish_piece" in text:
        print("already wired", name)
        return

    if "onedata_io as od" not in text:
        if "from .models import InputModel, OutputModel" in text:
            text = text.replace(
                "from .models import InputModel, OutputModel",
                "from .models import InputModel, OutputModel\n" + OD_IMPORT,
                1,
            )
        else:
            text = OD_IMPORT + "\n" + text

    text = re.sub(
        r"def piece_function\(self, input_data: InputModel\) -> OutputModel:",
        "def piece_function(self, input_data: InputModel, secrets_data=None) -> OutputModel:",
        text,
        count=1,
    )

    gen = "True" if name in GENERATE_RUN_ID else "False"
    prologue = f"""        _stage = None
        _run_id = None
        if od is not None:
            input_data, _stage = od.stage_inputs(input_data, secrets_data)
            _run_id = od.resolve_run_id(
                input_data, secrets_data, generate={gen}, results_path=getattr(self, "results_path", None)
            )
            if hasattr(input_data, "run_id") and _run_id and not getattr(input_data, "run_id", ""):
                try:
                    input_data.run_id = _run_id
                except Exception:
                    pass
"""
    m = re.search(
        r"def piece_function\(self, input_data: InputModel, secrets_data=None\) -> OutputModel:\n",
        text,
    )
    if m and "stage_inputs" not in text[m.end() : m.end() + 400]:
        text = text[: m.end()] + prologue + text[m.end() :]

    text, n = _rewrite_output_returns(text, name)
    piece.write_text(text, encoding="utf-8")
    print("patched", name, "returns", n)


def main() -> None:
    for name in OPS:
        d = PIECES / name
        if not d.is_dir():
            print("MISSING", name)
            continue
        _patch_models(d)
        _patch_piece(d)


if __name__ == "__main__":
    main()
