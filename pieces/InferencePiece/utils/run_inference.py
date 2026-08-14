from __future__ import annotations

from typing import Any

from .loader import (
    load_input_dataframe,
    load_model_object,
    load_preprocessing_metadata,
)
from .output import (
    build_forecast_table,
    build_per_horizon_outputs,
    serialize_forecast_if_requested,
)
from .preprocess import (
    apply_horizon_filter,
    apply_optional_feature_derivations,
    build_price_baseline_from_profile,
    ensure_feature_schema,
    parse_datetime_column,
)
from .runners import run_price_ahead, run_price_level, run_pvout_correction


def run_inference(payload: dict) -> dict[str, Any]:
    mode = payload["mode"]
    datetime_column = payload.get("datetime_column", "datetime")
    horizon_column = payload.get("horizon_column", "pred_sequence_id")
    max_horizon = payload.get("max_horizon")
    per_horizon_outputs = bool(payload.get("per_horizon_outputs", False))
    strict_schema = bool(payload.get("strict_schema", True))
    missing_fill_value = float(payload.get("missing_fill_value", 0.0))
    return_debug = bool(payload.get("return_debug", False))

    df = load_input_dataframe(payload)
    model = load_model_object(payload)
    preprocessing_meta = load_preprocessing_metadata(payload)

    feature_columns = (
        payload.get("feature_columns")
        or preprocessing_meta.get("feature_columns_used")
        or preprocessing_meta.get("feature_columns")
        or []
    )
    if not feature_columns:
        raise ValueError("feature_columns must be provided in payload or metadata")

    base_forecast_column = payload.get("base_forecast_column", "PVOUT")

    df = parse_datetime_column(df, datetime_column=datetime_column)
    df = apply_optional_feature_derivations(df, datetime_column=datetime_column)
    df = apply_horizon_filter(df, horizon_column=horizon_column, max_horizon=max_horizon)

    if (
        mode == "price_ahead"
        and base_forecast_column not in df.columns
        and payload.get("build_baseline_if_missing")
    ):
        profile_path = payload.get("price_profile_path")
        if not profile_path:
            raise ValueError(
                "price_profile_path is required when build_baseline_if_missing=true"
            )
        df = build_price_baseline_from_profile(
            df,
            datetime_column=datetime_column,
            profile_path=profile_path,
            baseline_column=base_forecast_column,
        )

    X, missing_columns, added_columns = ensure_feature_schema(
        df,
        feature_columns=feature_columns,
        strict_schema=strict_schema,
        missing_fill_value=missing_fill_value,
    )

    if mode == "pvout_correction":
        pred_df = run_pvout_correction(
            model=model,
            source_df=df,
            X=X,
            base_forecast_column=base_forecast_column,
        )
    elif mode == "price_ahead":
        pred_df = run_price_ahead(
            model=model,
            source_df=df,
            X=X,
            baseline_column=base_forecast_column,
        )
    elif mode == "price_level":
        pred_df = run_price_level(
            model=model,
            source_df=df,
            X=X,
        )
    else:
        raise ValueError(f"Unsupported mode '{mode}'")

    # Ground-truth column in the saved forecast CSV: explicit `target_column`, or for
    # pvout_correction the baseline column (typically PVOUT — same as ground truth).
    # Price modes must set `target_column` separately from `base_forecast_column`.
    target_column = payload.get("target_column")
    if target_column is None and mode == "pvout_correction":
        target_column = base_forecast_column
    forecast = build_forecast_table(
        pred_df,
        datetime_column=datetime_column,
        horizon_column=horizon_column,
        target_column=target_column,
    )

    csv_path = serialize_forecast_if_requested(
        forecast_df=forecast,
        csv_path=payload.get("forecast_output_csv_path"),
    )

    out: dict[str, Any] = {
        "forecast": {
            "inline_records": forecast.to_dict(orient="records"),
            "csv_path": csv_path,
            "columns": list(forecast.columns),
        },
        "per_horizon": (
            build_per_horizon_outputs(forecast, horizon_column=horizon_column)
            if per_horizon_outputs
            else {}
        ),
        "metadata": {
            "mode": mode,
            "model_path": payload["model_path"],
            "rows_input": int(len(df)),
            "rows_output": int(len(forecast)),
            "max_horizon_applied": max_horizon,
            "schema_version": "v1",
        },
    }

    if return_debug:
        out["debug"] = {
            "missing_columns": missing_columns,
            "added_columns": added_columns,
            "feature_columns_used": feature_columns,
        }

    return out


def run_staged_inference(payload: dict) -> dict[str, Any]:
    """
    Apply multiple models in order on one working DataFrame loaded from payload.input.

    Each entry in payload["stages"] is a dict with at least `mode` and `model_path`, plus
    the same optional keys as single-step inference (`feature_columns`, `base_forecast_column`,
    `build_baseline_if_missing`, `price_profile_path`, ...).

    Optional `inject_forecast_as` on a stage writes that stage's `final_forecast` into a new
    column on the working frame for use in later stages.
    """
    datetime_column = payload.get("datetime_column", "datetime")
    horizon_column = payload.get("horizon_column", "pred_sequence_id")
    max_horizon = payload.get("max_horizon")
    per_horizon_outputs = bool(payload.get("per_horizon_outputs", False))
    stages = payload.get("stages")
    if not isinstance(stages, list) or not stages:
        raise ValueError("payload['stages'] must be a non-empty list")

    df = load_input_dataframe(payload)
    df = parse_datetime_column(df, datetime_column=datetime_column)
    df = apply_optional_feature_derivations(df, datetime_column=datetime_column)
    df = apply_horizon_filter(df, horizon_column=horizon_column, max_horizon=max_horizon)

    stage_summaries: list[dict[str, Any]] = []
    forecast: Any = None
    last_missing: list[str] = []
    last_added: list[str] = []
    last_features: list[str] = []

    for i, stage in enumerate(stages):
        if not isinstance(stage, dict):
            raise ValueError(f"stages[{i}] must be a dict")
        mode = stage.get("mode")
        if not mode:
            raise ValueError(f"stages[{i}] missing mode")
        if "model_path" not in stage:
            raise ValueError(f"stages[{i}] missing model_path")

        model = load_model_object(stage)
        preprocessing_meta = load_preprocessing_metadata(stage)
        feature_columns = (
            stage.get("feature_columns")
            or preprocessing_meta.get("feature_columns_used")
            or preprocessing_meta.get("feature_columns")
            or []
        )
        if not feature_columns:
            raise ValueError(
                f"stages[{i}]: provide feature_columns or colocated preprocessing_metadata.json"
            )

        strict_schema = bool(stage.get("strict_schema", payload.get("strict_schema", True)))
        missing_fill_value = float(
            stage.get("missing_fill_value", payload.get("missing_fill_value", 0.0))
        )
        base_forecast_column = stage.get("base_forecast_column", "PVOUT")
        stage_df = df.copy()
        if (
            mode == "price_ahead"
            and base_forecast_column not in stage_df.columns
            and stage.get("build_baseline_if_missing")
        ):
            profile_path = stage.get("price_profile_path")
            if not profile_path:
                raise ValueError(
                    f"stages[{i}]: price_profile_path required when build_baseline_if_missing=true"
                )
            stage_df = build_price_baseline_from_profile(
                stage_df,
                datetime_column=datetime_column,
                profile_path=profile_path,
                baseline_column=base_forecast_column,
            )

        X, missing_columns, added_columns = ensure_feature_schema(
            stage_df,
            feature_columns=feature_columns,
            strict_schema=strict_schema,
            missing_fill_value=missing_fill_value,
        )

        if mode == "pvout_correction":
            pred_df = run_pvout_correction(
                model=model,
                source_df=stage_df,
                X=X,
                base_forecast_column=base_forecast_column,
            )
        elif mode == "price_ahead":
            pred_df = run_price_ahead(
                model=model,
                source_df=stage_df,
                X=X,
                baseline_column=base_forecast_column,
            )
        elif mode == "price_level":
            pred_df = run_price_level(
                model=model,
                source_df=stage_df,
                X=X,
            )
        else:
            raise ValueError(f"Unsupported stage mode '{mode}'")

        inject = stage.get("inject_forecast_as")
        if inject:
            df = df.copy()
            df[inject] = pred_df["final_forecast"].values

        forecast = build_forecast_table(
            pred_df,
            datetime_column=datetime_column,
            horizon_column=horizon_column,
            target_column=base_forecast_column,
        )
        last_missing = missing_columns
        last_added = added_columns
        last_features = feature_columns
        stage_summaries.append(
            {
                "stage_index": i,
                "mode": mode,
                "model_path": stage["model_path"],
                "rows_output": int(len(forecast)),
            }
        )

    assert forecast is not None
    csv_path = serialize_forecast_if_requested(
        forecast_df=forecast,
        csv_path=payload.get("forecast_output_csv_path"),
    )

    out: dict[str, Any] = {
        "forecast": {
            "inline_records": forecast.to_dict(orient="records"),
            "csv_path": csv_path,
            "columns": list(forecast.columns),
        },
        "per_horizon": (
            build_per_horizon_outputs(forecast, horizon_column=horizon_column)
            if per_horizon_outputs
            else {}
        ),
        "metadata": {
            "pipeline_stages": len(stages),
            "schema_version": "v1-pipeline",
        },
        "stage_summaries": stage_summaries,
    }

    if payload.get("return_debug"):
        out["debug"] = {
            "missing_columns": last_missing,
            "added_columns": last_added,
            "feature_columns_used": last_features,
        }

    return out
