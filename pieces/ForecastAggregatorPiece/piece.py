from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from domino.base_piece import BasePiece

from .models import InputModel, OutputModel


def _slugify_id(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()) or "model"
    return slug.strip("_") or "model"


def _model_id_for(entry: dict, index: int) -> str:
    candidate = entry.get("model_id")
    if candidate:
        return _slugify_id(str(candidate))
    csv_path = entry.get("forecast_csv_path")
    if csv_path:
        return _slugify_id(Path(str(csv_path)).stem)
    return f"model_{index + 1}"


def _load_forecast_frame(entry: dict):
    import pandas as pd  # type: ignore

    csv_path = entry.get("forecast_csv_path")
    if not csv_path:
        return None
    p = Path(csv_path)
    if not p.exists():
        raise FileNotFoundError(f"forecast CSV not found: {p}")
    return pd.read_csv(p)


class ForecastAggregatorPiece(BasePiece):
    def piece_function(self, input_data: InputModel):
        self.logger.info("Running ForecastAggregatorPiece.")
        payload = input_data.payload_as_dict()

        forecasts_input: list[dict[str, Any]] = list(payload.get("forecasts") or [])
        forecasts_input = [
            dict(entry) for entry in forecasts_input if isinstance(entry, dict)
        ]

        if not forecasts_input:
            return OutputModel(
                message="ForecastAggregatorPiece executed (no-op).",
                artifacts={"input_payload": payload},
            )

        import pandas as pd  # type: ignore

        datetime_column = str(payload.get("datetime_column") or "datetime")
        horizon_column = str(payload.get("horizon_column") or "pred_sequence_id")
        forecast_column = str(payload.get("forecast_column") or "final_forecast")
        include_diff = bool(payload.get("include_diff", False))
        output_csv_name = str(
            payload.get("output_csv_name") or "aggregated_forecast.csv"
        )

        target_column = None
        for entry in forecasts_input:
            if entry.get("target_column"):
                target_column = entry["target_column"]
                break

        merged: "pd.DataFrame | None" = None
        used_ids: set[str] = set()
        model_ids: list[str] = []
        join_keys: list[str] | None = None

        for index, entry in enumerate(forecasts_input):
            base_id = _model_id_for(entry, index)
            model_id = base_id
            suffix = 2
            while model_id in used_ids:
                model_id = f"{base_id}_{suffix}"
                suffix += 1
            used_ids.add(model_id)
            model_ids.append(model_id)

            df = _load_forecast_frame(entry)
            if df is None or forecast_column not in df.columns:
                # Skip entries that can't contribute a prediction column.
                continue

            current_join: list[str] = []
            if datetime_column in df.columns:
                current_join.append(datetime_column)
            if horizon_column in df.columns:
                current_join.append(horizon_column)

            if not current_join:
                raise ValueError(
                    f"forecast CSV for model `{model_id}` lacks both "
                    f"`{datetime_column}` and `{horizon_column}` columns."
                )

            slim = df[current_join + [forecast_column]].rename(
                columns={forecast_column: f"pred_{model_id}"}
            )

            if merged is None:
                merged = slim
                join_keys = current_join
            else:
                shared = [c for c in current_join if c in (join_keys or [])]
                if not shared:
                    raise ValueError(
                        f"Cannot align forecasts: model `{model_id}` shares no join keys "
                        f"with previously merged frame (have: {join_keys})."
                    )
                merged = merged.merge(slim, on=shared, how="outer")
                join_keys = shared

        if merged is None or merged.empty:
            return OutputModel(
                message="ForecastAggregatorPiece executed (no overlapping forecast rows).",
                artifacts={"input_payload": payload},
                model_ids=model_ids,
            )

        actual_csv_path = payload.get("actual_csv_path")
        if actual_csv_path and target_column:
            actual_path = Path(actual_csv_path)
            if actual_path.exists():
                actual_df = pd.read_csv(actual_path)
                if target_column in actual_df.columns and (
                    datetime_column in actual_df.columns
                    or horizon_column in actual_df.columns
                ):
                    keep_cols = [
                        c
                        for c in (datetime_column, horizon_column)
                        if c in actual_df.columns and c in (join_keys or [])
                    ]
                    if keep_cols:
                        actual_slim = (
                            actual_df[keep_cols + [target_column]]
                            .drop_duplicates(subset=keep_cols)
                            .rename(columns={target_column: f"actual_{target_column}"})
                        )
                        merged = merged.merge(actual_slim, on=keep_cols, how="left")

        if include_diff and target_column:
            actual_col = f"actual_{target_column}"
            if actual_col in merged.columns:
                for model_id in model_ids:
                    pred_col = f"pred_{model_id}"
                    if pred_col in merged.columns:
                        merged[f"diff_{model_id}"] = (
                            merged[pred_col] - merged[actual_col]
                        )

        if datetime_column in merged.columns:
            merged = merged.sort_values(datetime_column).reset_index(drop=True)

        results_path = Path(self.results_path)
        results_path.mkdir(parents=True, exist_ok=True)
        out_path = results_path / output_csv_name
        merged.to_csv(out_path, index=False)
        self.display_result = {"file_type": "txt", "file_path": str(out_path)}

        return OutputModel(
            message=(
                "ForecastAggregatorPiece executed."
                if len(model_ids) == 1
                else f"ForecastAggregatorPiece aggregated {len(model_ids)} models."
            ),
            aggregated_csv_path=str(out_path),
            model_ids=model_ids,
            n_rows=int(len(merged)),
            artifacts={
                "columns": list(merged.columns),
                "model_ids": model_ids,
                "n_rows": int(len(merged)),
                "target_column": target_column,
            },
        )
