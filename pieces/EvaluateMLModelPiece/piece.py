import json
import re
from pathlib import Path
from typing import Any

from domino.base_piece import BasePiece

from .models import InputModel, MetricsEntry, OutputModel


_INHERITED_KEYS = (
    "evaluation_option",
    "baseline_id",
    "plot",
)


def _slugify_id(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()) or "model"
    return slug.strip("_") or "model"


def _model_id_for(entry: dict, index: int) -> str:
    candidate = entry.get("model_id")
    if candidate:
        return _slugify_id(str(candidate))
    pred_df_path = entry.get("pred_df_path")
    if pred_df_path:
        return _slugify_id(Path(str(pred_df_path)).stem)
    return f"model_{index + 1}"


def _default_forecast_column_for_mode(mode: str | None) -> str:
    """Pick a sensible forecast column for a given inference mode.

    - `pvout_correction` adds `correction = model.predict(X)` and writes
      `final_forecast = base + correction`. With `base_forecast_column=PVOUT`
      (the truth column), `final_forecast - PVOUT` is just `correction`, so
      `correction` is the column that measures actual prediction quality.
    - All other modes (`price_level`, `price_ahead`) put the meaningful
      prediction in `final_forecast`.
    """
    if (mode or "").lower() == "pvout_correction":
        return "correction"
    return "final_forecast"


def _derive_evaluations_from_forecasts(payload: dict) -> list[dict]:
    """Auto-build an evaluations list from an upstream `forecasts` bind.

    When `EvaluateMLModelPiece` is wired to `InferencePiece.forecasts` (at the
    list level), each ForecastEntry carries everything we need — model_id,
    forecast_csv_path, target_column, mode — to evaluate it without manual
    per-entry configuration. Any explicit overrides set on the `evaluations`
    field still take precedence per `model_id`.
    """
    forecasts = payload.get("forecasts")
    if not isinstance(forecasts, list) or not forecasts:
        return []

    overrides_by_id: dict[str, dict] = {}
    explicit = payload.get("evaluations") or []
    if isinstance(explicit, list):
        for entry in explicit:
            if not isinstance(entry, dict):
                continue
            key = entry.get("model_id")
            if key:
                overrides_by_id[str(key)] = dict(entry)

    derived: list[dict] = []
    for index, forecast in enumerate(forecasts):
        if not isinstance(forecast, dict):
            continue
        model_id = forecast.get("model_id") or f"model_{index + 1}"
        defaults: dict[str, Any] = {
            "model_id": model_id,
            "pred_df_path": forecast.get("forecast_csv_path"),
            "target_column": forecast.get("target_column"),
            "forecast_column": _default_forecast_column_for_mode(forecast.get("mode")),
        }
        override = overrides_by_id.get(str(model_id), {})
        # Per-entry override values trump auto-derived defaults.
        for k, v in override.items():
            if v is not None and v != "":
                defaults[k] = v
        derived.append(defaults)
    return derived


def _normalize_evaluations(payload: dict) -> list[dict]:
    # Upstream-bound `forecasts` list takes precedence — that's the path the
    # canonical workflow uses (Inference → Evaluate via a single edge).
    derived = _derive_evaluations_from_forecasts(payload)
    if derived:
        return derived

    evaluations = payload.get("evaluations")
    if isinstance(evaluations, list) and evaluations:
        return [dict(entry or {}) for entry in evaluations]
    return []


def _entry_payload_for(entry: dict, parent: dict) -> dict:
    """Materialize the effective payload for one evaluation entry."""
    materialized: dict[str, Any] = {}
    for key in _INHERITED_KEYS:
        if key in parent and parent[key] is not None:
            materialized[key] = parent[key]
    materialized.update({k: v for k, v in entry.items() if v is not None})
    return materialized


def _run_single_evaluation(entry_payload: dict) -> tuple[str, dict]:
    """Execute one evaluation request. Returns (evaluation_option, metrics)."""
    evaluation_option = (
        entry_payload.get("evaluation_option")
        or entry_payload.get("evaluation_type")
        or entry_payload.get("mode")
        or "normal"
    )
    evaluation_option = str(evaluation_option).lower()

    pred_df = (
        entry_payload.get("pred_df")
        or entry_payload.get("predictions_df")
        or entry_payload.get("predictions")
        or entry_payload.get("df")
    )

    if pred_df is None and entry_payload.get("pred_df_path"):
        import pandas as pd  # type: ignore

        pred_df = pd.read_csv(entry_payload["pred_df_path"])

    if isinstance(pred_df, list):
        import pandas as pd  # type: ignore

        pred_df = pd.DataFrame(pred_df)

    plot = bool(entry_payload.get("plot", False))
    baseline_id = int(entry_payload.get("baseline_id", 1))
    forecast_column = str(entry_payload.get("forecast_column") or "final_forecast")
    target_column = str(entry_payload.get("target_column") or "PVOUT")

    from .utils.error_evaluator import ErrorEvaluator

    evaluator = ErrorEvaluator()

    if pred_df is None:
        raise ValueError(
            "evaluation requires `payload['pred_df']` (or `predictions_df`/`predictions`/`df`)."
        )

    if evaluation_option == "normal":
        metrics = evaluator.evaluate(
            pred_df=pred_df,
            true_baseline_df=None,
            y_true=None,
            baseline_id=baseline_id,
            plot=plot,
            forecast_column=forecast_column,
            target_column=target_column,
        )
    elif evaluation_option in {"errorcorrection", "error_correction", "correction"}:
        y_true = entry_payload.get("y_true")
        true_baseline_df = entry_payload.get("true_baseline_df") or entry_payload.get(
            "baseline_df"
        )
        if true_baseline_df is None and entry_payload.get("true_baseline_df_path"):
            import pandas as pd  # type: ignore

            true_baseline_df = pd.read_csv(entry_payload["true_baseline_df_path"])
        if isinstance(true_baseline_df, list):
            import pandas as pd  # type: ignore

            true_baseline_df = pd.DataFrame(true_baseline_df)

        if y_true is not None:
            metrics = evaluator.evaluate(
                pred_df=pred_df,
                y_true=y_true,
                true_baseline_df=None,
                baseline_id=baseline_id,
                plot=plot,
            )
        elif true_baseline_df is not None:
            metrics = evaluator.evaluate(
                pred_df=pred_df,
                true_baseline_df=true_baseline_df,
                y_true=None,
                baseline_id=baseline_id,
                plot=plot,
            )
        else:
            raise ValueError(
                "errorcorrection evaluation requires either `payload['y_true']` "
                "or `payload['true_baseline_df']`."
            )
    else:
        raise ValueError(
            "evaluation_option must be one of: normal, errorcorrection."
        )

    return evaluation_option, metrics


class EvaluateMLModelPiece(BasePiece):
    def piece_function(self, input_data: InputModel):
        payload = input_data.payload_as_dict()
        self.logger.info("Running EvaluateMLModelPiece.")

        if not payload:
            return OutputModel(
                message="EvaluateMLModelPiece template executed (no-op).",
                artifacts={"input_payload": payload},
            )

        entries = _normalize_evaluations(payload)
        if not entries:
            return OutputModel(
                message="EvaluateMLModelPiece executed (no-op).",
                artifacts={"input_payload": payload},
            )

        results_dir = Path(self.results_path)
        results_dir.mkdir(parents=True, exist_ok=True)

        metrics_entries: list[MetricsEntry] = []
        per_model: dict[str, dict] = {}
        used_ids: set[str] = set()
        first_metrics_path: str | None = None

        for index, entry in enumerate(entries):
            entry_payload = _entry_payload_for(entry, payload)
            base_id = _model_id_for(entry, index)
            model_id = base_id
            suffix = 2
            while model_id in used_ids:
                model_id = f"{base_id}_{suffix}"
                suffix += 1
            used_ids.add(model_id)

            evaluation_option, metrics = _run_single_evaluation(entry_payload)

            metrics_name = (
                "metrics.json" if len(entries) == 1 else f"metrics_{model_id}.json"
            )
            metrics_path = str(results_dir / metrics_name)
            with open(metrics_path, "w", encoding="utf-8") as fh:
                json.dump(
                    {
                        "model_id": model_id,
                        "evaluation_option": evaluation_option,
                        "metrics": metrics,
                    },
                    fh,
                    indent=2,
                    default=str,
                )
            if first_metrics_path is None:
                first_metrics_path = metrics_path

            metrics_entries.append(
                MetricsEntry(
                    model_id=model_id,
                    evaluation_option=evaluation_option,
                    metrics_path=metrics_path,
                    metrics=metrics,
                )
            )
            per_model[model_id] = {
                "evaluation_option": evaluation_option,
                "metrics": metrics,
                "metrics_path": metrics_path,
            }

        if first_metrics_path:
            self.display_result = {"file_type": "txt", "file_path": first_metrics_path}

        head = metrics_entries[0]
        message = (
            "EvaluateMLModelPiece executed."
            if len(metrics_entries) == 1
            else f"EvaluateMLModelPiece executed for {len(metrics_entries)} evaluations."
        )

        return OutputModel(
            message=message,
            metrics=metrics_entries,
            artifacts={
                "input_payload": payload,
                "evaluation_option": head.evaluation_option,
                "metrics": head.metrics,
                "metrics_path": head.metrics_path,
                "per_model": per_model,
            },
        )
