from __future__ import annotations

from typing import Any, Optional


def maybe_build_diagnostic_heatmaps(payload: dict) -> dict:
    """
    Build diagnostic-weighted error correction heatmaps.

    This implementation expects precomputed arrays in `payload["diagnostic"]`,
    because the underlying custom-loss helpers are not part of this repo.
    """

    diagnostic = payload.get("diagnostic") or {}
    x_train = payload.get("x_train") or diagnostic.get("x_train")

    if not diagnostic:
        return {
            "enabled": True,
            "status": "skipped",
            "reason": "payload['diagnostic'] is missing; cannot build heatmaps.",
        }

    import numpy as np
    import pandas as pd

    train_horizons = np.asarray(diagnostic["train_horizons"]).astype(int)
    train_regimes = np.asarray(diagnostic["train_regimes"]).astype(int)
    w_i = np.asarray(diagnostic["w_i"]).astype(float)
    grad_diag = np.asarray(diagnostic["grad_diag"]).astype(float)
    hess_diag = np.asarray(diagnostic["hess_diag"]).astype(float)
    y_tr_label = np.asarray(diagnostic["y_tr_label"]).astype(float)
    y_tr_pred = np.asarray(diagnostic["y_tr_pred"]).astype(float)

    difficulty_i = np.abs(y_tr_label - y_tr_pred)
    abs_grad = np.abs(grad_diag)
    w_times_hess = w_i * hess_diag

    grad_u = diagnostic.get("grad_u")
    hess_u = diagnostic.get("hess_u")
    abs_grad_u = (
        np.abs(np.asarray(grad_u)).astype(float).tolist()
        if grad_u is not None
        else None
    )

    regime_names = diagnostic.get("regime_names") or ["clear", "mixed", "cloudy"]
    hour_of_day = None
    if isinstance(x_train, pd.DataFrame) and "hour_of_day" in x_train.columns:
        hour_of_day = np.asarray(x_train["hour_of_day"]).astype(int)
    elif isinstance(x_train, pd.DataFrame) and "datetime" in x_train.columns:
        try:
            dt = pd.to_datetime(x_train["datetime"])
            hour_of_day = dt.dt.hour.values.astype(int)
        except Exception:
            hour_of_day = None

    if hour_of_day is None:
        hour_of_day = np.zeros_like(train_horizons, dtype=int)

    df_diag = pd.DataFrame(
        {
            "horizon": train_horizons,
            "regime": train_regimes,
            "regime_name": [regime_names[r] for r in train_regimes],
            "hour": hour_of_day,
            "w_i": w_i,
            "abs_grad": abs_grad,
            "w_hess": w_times_hess,
            "difficulty": difficulty_i,
        }
    )

    # Aggregate by (horizon, regime)
    agg_hr = (
        df_diag.groupby(["horizon", "regime_name"])
        .agg(
            mean_w=("w_i", "mean"),
            sum_abs_grad=("abs_grad", "sum"),
            mean_w_hess=("w_hess", "mean"),
            mean_difficulty=("difficulty", "mean"),
        )
        .reset_index()
    )

    pivot_w = agg_hr.pivot(index="horizon", columns="regime_name", values="mean_w")
    pivot_grad = agg_hr.pivot(
        index="horizon", columns="regime_name", values="sum_abs_grad"
    )
    pivot_diff = agg_hr.pivot(
        index="horizon", columns="regime_name", values="mean_difficulty"
    )

    # Aggregate by (horizon, hour)
    agg_hh = (
        df_diag.groupby(["horizon", "hour"])
        .agg(mean_w=("w_i", "mean"), sum_abs_grad=("abs_grad", "sum"))
        .reset_index()
    )

    pivot_hour = agg_hh.pivot(
        index="horizon", columns="hour", values="sum_abs_grad"
    ).fillna(0)

    # Optional unweighted pivots
    pivot_grad_u = None
    pivot_hess_u = None
    if grad_u is not None and hess_u is not None:
        df_diag_u = df_diag.copy()
        df_diag_u["abs_grad_u"] = np.abs(np.asarray(grad_u)).astype(float)
        df_diag_u["hess_u"] = np.asarray(hess_u).astype(float)
        agg_hr_u = (
            df_diag_u.groupby(["horizon", "regime_name"])
            .agg(
                sum_abs_grad_u=("abs_grad_u", "sum"),
                mean_hess_u=("hess_u", "mean"),
            )
            .reset_index()
        )
        pivot_grad_u = agg_hr_u.pivot(
            index="horizon", columns="regime_name", values="sum_abs_grad_u"
        )
        pivot_hess_u = agg_hr_u.pivot(
            index="horizon", columns="regime_name", values="mean_hess_u"
        )

    plots = {}
    try:
        import matplotlib.pyplot as plt  # type: ignore
        import base64
        from io import BytesIO

        def _heatmap_base64(pivot_df: "pd.DataFrame", cmap: str, title: str):
            fig, ax = plt.subplots(figsize=(10, 4))
            im = ax.imshow(pivot_df.values.T, aspect="auto", cmap=cmap)
            ax.set_xticks(range(len(pivot_df.index)))
            ax.set_xticklabels(pivot_df.index, rotation=0)
            ax.set_yticks(range(len(pivot_df.columns)))
            ax.set_yticklabels(pivot_df.columns)
            ax.set_xlabel("Horizon")
            ax.set_ylabel("Regime")
            ax.set_title(title)
            plt.colorbar(im, ax=ax)
            buf = BytesIO()
            fig.tight_layout()
            fig.savefig(buf, format="png", dpi=150)
            plt.close(fig)
            return base64.b64encode(buf.getvalue()).decode("ascii")

        plots["mean_w_horizon_regime"] = _heatmap_base64(
            pivot_w,
            cmap="viridis",
            title="Mean diagnostic weight w_i (horizon x regime)",
        )
        plots["sum_abs_grad_horizon_regime"] = _heatmap_base64(
            pivot_grad, cmap="plasma", title="Sum |gradient| (horizon x regime)"
        )
        plots["mean_difficulty_horizon_regime"] = _heatmap_base64(
            pivot_diff,
            cmap="cividis",
            title="Mean difficulty |residual| (horizon x regime)",
        )

        fig2, ax2 = plt.subplots(figsize=(10, 4))
        im2 = ax2.imshow(pivot_hour.values, aspect="auto", cmap="viridis")
        ax2.set_xlabel("Hour of day")
        ax2.set_ylabel("Horizon")
        ax2.set_title("Sum |gradient| (horizon x hour)")
        fig2.tight_layout()
        buf2 = BytesIO()
        fig2.savefig(buf2, format="png", dpi=150)
        plt.close(fig2)
        plots["sum_abs_grad_horizon_hour"] = base64.b64encode(buf2.getvalue()).decode(
            "ascii"
        )
    except Exception as e:
        plots = {"error": f"Plotting disabled/unavailable: {e}"}

    return {
        "enabled": True,
        "status": "ok",
        "pivot_w": {
            "index": pivot_w.index.tolist(),
            "columns": pivot_w.columns.tolist(),
            "values": pivot_w.values.tolist(),
        },
        "pivot_grad": {
            "index": pivot_grad.index.tolist(),
            "columns": pivot_grad.columns.tolist(),
            "values": pivot_grad.values.tolist(),
        },
        "pivot_diff": {
            "index": pivot_diff.index.tolist(),
            "columns": pivot_diff.columns.tolist(),
            "values": pivot_diff.values.tolist(),
        },
        "pivot_hour": {
            "index": pivot_hour.index.tolist(),
            "columns": pivot_hour.columns.tolist(),
            "values": pivot_hour.values.tolist(),
        },
        "plots": plots,
        "unweighted": {
            "pivot_grad_u": (
                None
                if pivot_grad_u is None
                else {
                    "index": pivot_grad_u.index.tolist(),
                    "columns": pivot_grad_u.columns.tolist(),
                    "values": pivot_grad_u.values.tolist(),
                }
            ),
            "pivot_hess_u": (
                None
                if pivot_hess_u is None
                else {
                    "index": pivot_hess_u.index.tolist(),
                    "columns": pivot_hess_u.columns.tolist(),
                    "values": pivot_hess_u.values.tolist(),
                }
            ),
        },
    }
