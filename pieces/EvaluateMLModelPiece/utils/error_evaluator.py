from __future__ import annotations

from typing import Optional


class ErrorEvaluator:
    def evaluate(
        self,
        pred_df,
        true_baseline_df=None,
        y_true=None,
        baseline_id: int = 1,
        plot: bool = False,
        forecast_column: str = "final_forecast",
        target_column: str = "PVOUT",
    ):
        """
        Evaluate prediction error vs true baseline.

        This is a light integration of the provided `ErrorEvaluator` code:
        - supports baseline embedded in `pred_df` (normal evaluation)
        - supports baseline provided separately or `y_true` provided (error correction evaluation)
        - heatmaps are generated only when `plot=True` (defaults to False for headless/smoke safety)
        """
        # Heavy deps are imported lazily for piece smoke tests.
        import numpy as np
        import pandas as pd

        # --- Resolve baseline: either inside pred_df, separate dataframe, or series ---
        df = pred_df.copy()
        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"])

        has_baseline_in_df = (
            "pred_sequence_id" in df.columns
            and (df["pred_sequence_id"] == baseline_id).any()
        )

        if y_true is not None:
            metrics = self._evaluate_test_split(pred_df, y_true)
            self._print_test_metrics(metrics)
            return metrics

        if true_baseline_df is not None:
            df = self._merge_baseline_into_pred(df, true_baseline_df, baseline_id)
            if df is None or df.empty:
                return {}
            diff_df = self._calculate_differences_vs_baseline(
                df, baseline_id=baseline_id, baseline_embedded=False
            )
        elif has_baseline_in_df:
            diff_df = self._calculate_differences_vs_baseline(
                df, baseline_id=baseline_id, baseline_embedded=True
            )
        elif forecast_column in df.columns and target_column in df.columns:
            return self._direct_forecast_metrics(
                df, forecast_column=forecast_column, target_column=target_column
            )
        else:
            # No baseline information available.
            return {}

        if diff_df is None or diff_df.empty:
            return {}

        # Calculate both raw and z-score normalized percentage differences only for plots.
        if plot:
            diff_df = self._calculate_raw_percentage_differences(diff_df)
            diff_df = self._calculate_percentage_differences(diff_df)

            summary_df_normalized = self._summary_differences_by_sequence(
                diff_df,
                original_df=df,
                baseline_id=baseline_id,
                use_percentage=True,
                use_raw_pct=False,
            )
            summary_df_raw = self._summary_differences_by_sequence(
                diff_df,
                original_df=df,
                baseline_id=baseline_id,
                use_percentage=True,
                use_raw_pct=True,
            )
            summary_df_absolute = self._summary_differences_by_sequence(
                diff_df, original_df=df, baseline_id=baseline_id, use_percentage=False
            )

            unique_seqs = diff_df["pred_sequence_id"].nunique()
            if unique_seqs >= 1:
                self._plot_difference_heatmap(
                    summary_df_normalized,
                    metric="abs_mean_diff_pct",
                    title_suffix="Z-Score Normalized Percentage",
                )
                self._plot_difference_heatmap(
                    summary_df_raw,
                    metric="abs_mean_diff_pct_raw",
                    title_suffix="Raw Percentage",
                )
                self._plot_difference_heatmap(
                    summary_df_absolute,
                    metric="abs_mean_diff",
                    title_suffix="Raw Absolute",
                )

        # Print test-set style metrics for PVOUT when we have diff_df with PVOUT
        pvout_diff = diff_df.get("PVOUT_diff")
        if pvout_diff is not None and len(pvout_diff.dropna()):
            err = pvout_diff.dropna()
            mape = np.nan
            if "PVOUT_baseline" in diff_df.columns:
                base = diff_df.loc[err.index, "PVOUT_baseline"].replace(0, np.nan)
                if base.notna().any():
                    mape = float((err.abs() / base).mean() * 100)
            metrics = {
                "mae": float(err.abs().mean()),
                "rmse": float(np.sqrt((err**2).mean())),
                "mape": mape,
            }
            self._print_test_metrics(metrics)
            return metrics

        return {}

    def get_summary_df_absolute(self, pred_df, baseline_id: int = 1):
        import pandas as pd

        df = pred_df.copy()
        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"])

        has_baseline = (
            "pred_sequence_id" in df.columns
            and (df["pred_sequence_id"] == baseline_id).any()
        )
        if not has_baseline:
            return pd.DataFrame()

        diff_df = self._calculate_differences_vs_baseline(
            df, baseline_id=baseline_id, baseline_embedded=True
        )
        if diff_df.empty:
            return pd.DataFrame()
        return self._summary_differences_by_sequence(
            diff_df, original_df=df, baseline_id=baseline_id, use_percentage=False
        )

    def _merge_baseline_into_pred(self, pred_df, true_baseline_df, baseline_id: int):
        import pandas as pd

        pred_df = pred_df.copy()
        true_baseline_df = true_baseline_df.copy()
        true_baseline_df["datetime"] = pd.to_datetime(true_baseline_df["datetime"])
        if "datetime" not in pred_df.columns:
            return None
        pred_df["datetime"] = pd.to_datetime(pred_df["datetime"])

        value_cols = ["PVOUT"]
        extra = [
            c
            for c in true_baseline_df.columns
            if c not in ("datetime", "PVOUT")
            and pd.api.types.is_numeric_dtype(true_baseline_df[c])
        ]
        value_cols = value_cols + extra
        value_cols = [c for c in value_cols if c in true_baseline_df.columns]

        baseline_sub = true_baseline_df[["datetime"] + value_cols].copy()
        baseline_sub = baseline_sub.rename(
            columns={c: f"{c}_baseline" for c in value_cols}
        )
        pred_df = pred_df.merge(baseline_sub, on="datetime", how="inner")
        if pred_df.empty:
            return None
        if "pred_sequence_id" not in pred_df.columns:
            pred_df["pred_sequence_id"] = baseline_id + 1
        return pred_df

    def _direct_forecast_metrics(self, df, forecast_column: str, target_column: str):
        """
        Compute MAE / RMSE / MAPE directly from a forecast column vs. truth column.

        Used when the pred_df contains both columns (typical Inference output) and no
        baseline/horizon information is available to drive the more elaborate analysis.
        """
        import numpy as np
        import pandas as pd

        y_pred = pd.to_numeric(df[forecast_column], errors="coerce")
        y_true = pd.to_numeric(df[target_column], errors="coerce")
        err = y_pred - y_true
        mask = err.notna() & y_true.notna()
        err, y_t = err[mask], y_true[mask]
        if len(err) == 0:
            return {}
        y_t_safe = y_t.replace(0, np.nan)
        mape = (
            float((err.abs() / y_t_safe).mean() * 100)
            if y_t_safe.notna().any()
            else float("nan")
        )
        metrics = {
            "mae": float(err.abs().mean()),
            "rmse": float(np.sqrt((err**2).mean())),
            "mape": mape,
            "forecast_column": forecast_column,
            "target_column": target_column,
            "n": int(len(err)),
        }
        self._print_test_metrics(metrics)
        return metrics

    def _evaluate_test_split(self, pred_df, y_true):
        import numpy as np
        import pandas as pd

        if "PVOUT" not in pred_df.columns:
            return {}
        y_pred = pred_df["PVOUT"]
        n = len(y_pred)
        y_true = pd.Series(np.asarray(y_true).ravel()[:n], index=y_pred.index)
        err = y_pred - y_true
        mask = pd.notna(err) & pd.notna(y_true)
        err, y_t = err[mask], y_true[mask]
        if len(err) == 0:
            return {}
        y_t_safe = y_t.replace(0, np.nan)
        mape = (err.abs() / y_t_safe).mean() * 100 if y_t_safe.notna().any() else np.nan
        return {
            "mae": float(err.abs().mean()),
            "rmse": float(np.sqrt((err**2).mean())),
            "mape": float(mape) if pd.notna(mape) else np.nan,
        }

    def _print_test_metrics(self, metrics: dict):
        if not metrics:
            return

    def _plot_difference_heatmap(self, summary_df, metric: str, title_suffix: str = ""):
        import matplotlib.pyplot as plt  # type: ignore
        import seaborn as sns  # type: ignore

        exclude_features = ["SA", "SE", "hour_of_day", "solar_elevation_sin"]

        if metric not in summary_df.columns:
            if "abs_mean_diff" in summary_df.columns:
                metric = "abs_mean_diff"
            else:
                return

        summary_df_filtered = summary_df[
            ~summary_df["variable"].isin(exclude_features)
        ].copy()
        pivot_data = summary_df_filtered.pivot(
            index="pred_sequence_id", columns="variable", values=metric
        )

        is_zscore = "pct" in metric.lower() and "raw" not in metric.lower()
        is_raw_pct = "pct_raw" in metric.lower()
        fmt_str = ".2f" if (is_zscore or is_raw_pct) else ".3f"

        if is_zscore:
            label_suffix = " (Z-Score)"
            cmap = "RdBu_r"
            center = 0
        elif is_raw_pct:
            label_suffix = " (%)"
            cmap = "YlOrRd"
            center = None
        else:
            label_suffix = ""
            cmap = "YlOrRd"
            center = None

        plt.figure(figsize=(16, max(8, len(pivot_data) * 0.5)))
        sns.heatmap(
            pivot_data,
            annot=True,
            fmt=fmt_str,
            cmap=cmap,
            cbar_kws={"label": f"{metric.replace('_', ' ').title()}{label_suffix}"},
            linewidths=0.5,
            linecolor="gray",
            center=center,
        )
        title = f"Heatmap of {title_suffix} Differences from Baseline (Seq 1)"
        plt.title(title, fontsize=14, fontweight="bold", pad=20)
        plt.xlabel("Variable", fontsize=12)
        plt.ylabel("Prediction Sequence ID", fontsize=12)
        plt.tight_layout()
        plt.show()

    def _summary_differences_by_sequence(
        self,
        diff_df,
        original_df=None,
        baseline_id: int = 1,
        use_percentage: bool = True,
        use_raw_pct: bool = False,
    ):
        import numpy as np
        import pandas as pd

        pct_cols = [
            col
            for col in diff_df.columns
            if col.endswith("_diff_pct") and not col.endswith("_diff_pct_raw")
        ]
        pct_raw_cols = [col for col in diff_df.columns if col.endswith("_diff_pct_raw")]
        diff_cols = [
            col
            for col in diff_df.columns
            if col.endswith("_diff")
            and not col.endswith("_diff_pct")
            and not col.endswith("_diff_pct_raw")
        ]

        if use_percentage:
            if use_raw_pct and pct_raw_cols:
                metric_cols = pct_raw_cols
                metric_suffix = "_pct_raw"
                metric_name = "abs_mean_diff_pct_raw"
            elif pct_cols:
                metric_cols = pct_cols
                metric_suffix = "_pct"
                metric_name = "abs_mean_diff_pct"
            else:
                metric_cols = diff_cols
                metric_suffix = ""
                metric_name = "abs_mean_diff"
        else:
            metric_cols = diff_cols
            metric_suffix = ""
            metric_name = "abs_mean_diff"

        summary_stats = []

        if original_df is not None:
            baseline_data = original_df[
                original_df["pred_sequence_id"] == baseline_id
            ].copy()
            if not baseline_data.empty:
                exclude_cols = ["datetime", "pred_sequence_id", "Date", "Time"]
                numeric_cols = [
                    col
                    for col in original_df.columns
                    if col not in exclude_cols
                    and pd.api.types.is_numeric_dtype(original_df[col])
                ]
                for var_name in numeric_cols:
                    values = baseline_data[var_name].dropna()
                    if len(values) > 0:
                        summary_stats.append(
                            {
                                "pred_sequence_id": baseline_id,
                                "variable": var_name,
                                "mean_diff_norm": 0.0,
                                "mean_diff": 0.0,
                                "std_diff": 0.0,
                                "min_diff": 0.0,
                                "max_diff": 0.0,
                                metric_name: 0.0,
                                f"abs_max_diff{metric_suffix}": 0.0,
                                "baseline_value": values.mean(),
                            }
                        )

        for seq_id in sorted(diff_df["pred_sequence_id"].unique()):
            seq_data = diff_df[diff_df["pred_sequence_id"] == seq_id]
            for col in metric_cols:
                if col.endswith("_diff_pct_raw"):
                    var_name = col.replace("_diff_pct_raw", "")
                elif col.endswith("_diff_pct"):
                    var_name = col.replace("_diff_pct", "")
                else:
                    var_name = col.replace("_diff", "")

                values = seq_data[col].dropna()
                if len(values) > 0:
                    summary_stats.append(
                        {
                            "pred_sequence_id": seq_id,
                            "variable": var_name,
                            "mean_diff_norm": (
                                values.mean() / values.abs().max()
                                if values.abs().max() > 0
                                else 0.0
                            ),
                            "mean_diff": values.mean(),
                            "std_diff": values.std(),
                            "min_diff": values.min(),
                            "max_diff": values.max(),
                            metric_name: values.abs().mean(),
                            f"abs_max_diff{metric_suffix}": values.abs().max(),
                            "baseline_value": np.nan,
                        }
                    )

        return pd.DataFrame(summary_stats)

    def _calculate_differences_vs_baseline(
        self, df, baseline_id: int = 1, baseline_embedded: bool = True
    ):
        import numpy as np
        import pandas as pd

        df = df.copy()
        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"])

        if not baseline_embedded:
            baseline_suffix = "_baseline"
            baseline_cols = [c for c in df.columns if c.endswith(baseline_suffix)]
            if not baseline_cols:
                return pd.DataFrame()

            value_cols = [c.replace(baseline_suffix, "") for c in baseline_cols]
            numeric_cols = [
                c
                for c in value_cols
                if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
            ]
            if not numeric_cols:
                return pd.DataFrame()

            out = (
                df[["datetime", "pred_sequence_id"]].copy()
                if "pred_sequence_id" in df.columns
                else df[["datetime"]].copy()
            )
            if "pred_sequence_id" not in out.columns:
                out["pred_sequence_id"] = baseline_id + 1
            out["baseline_id"] = baseline_id

            for col in numeric_cols:
                bcol = f"{col}_baseline"
                if bcol in df.columns:
                    out[f"{col}_diff"] = df[col] - df[bcol]
                    out[bcol] = df[bcol]
            return out

        baseline = df[df["pred_sequence_id"] == baseline_id].copy()
        if baseline.empty:
            return pd.DataFrame()

        exclude_cols = ["datetime", "pred_sequence_id", "Date", "Time"]
        numeric_cols = [
            col
            for col in df.columns
            if col not in exclude_cols
            and pd.api.types.is_numeric_dtype(df[col])
            and not col.endswith("_baseline")
        ]
        if not numeric_cols:
            return pd.DataFrame()

        differences = []
        for seq_id in sorted(df["pred_sequence_id"].unique()):
            if seq_id == baseline_id:
                continue
            current_data = df[df["pred_sequence_id"] == seq_id].copy()
            merged = current_data.merge(
                baseline[["datetime"] + numeric_cols],
                on="datetime",
                suffixes=("", "_baseline"),
                how="inner",
            )
            if merged.empty:
                continue
            for col in numeric_cols:
                merged[f"{col}_diff"] = merged[col] - merged[f"{col}_baseline"]
            merged["pred_sequence_id"] = seq_id
            merged["baseline_id"] = baseline_id
            keep_cols = (
                ["datetime", "pred_sequence_id", "baseline_id"]
                + [f"{col}_diff" for col in numeric_cols]
                + [f"{col}_baseline" for col in numeric_cols]
            )
            differences.append(merged[keep_cols])

        if differences:
            return pd.concat(differences, ignore_index=True)
        return pd.DataFrame()

    def _calculate_raw_percentage_differences(self, diff_df):
        import numpy as np

        if diff_df.empty:
            return diff_df
        diff_df_pct = diff_df.copy()
        diff_cols = [col for col in diff_df.columns if col.endswith("_diff")]
        for col in diff_cols:
            var_name = col.replace("_diff", "")
            baseline_col = f"{var_name}_baseline"
            if baseline_col in diff_df.columns:
                baseline_abs = diff_df[baseline_col].abs()
                baseline_safe = baseline_abs.replace(0, np.nan)
                pct_shift = (diff_df[col] / baseline_safe) * 100
                diff_df_pct[f"{col}_pct_raw"] = pct_shift
        return diff_df_pct

    def _calculate_percentage_differences(self, diff_df):
        import numpy as np
        import pandas as pd

        if diff_df.empty:
            return diff_df
        diff_df_pct = diff_df.copy()
        diff_cols = [col for col in diff_df.columns if col.endswith("_diff")]
        for col in diff_cols:
            var_name = col.replace("_diff", "")
            baseline_col = f"{var_name}_baseline"
            if baseline_col in diff_df.columns:
                baseline_abs = diff_df[baseline_col].abs()
                baseline_safe = baseline_abs.replace(0, np.nan)
                pct_shift = (diff_df[col] / baseline_safe) * 100

                pct_shift_clean = pct_shift.dropna()
                if len(pct_shift_clean) > 0:
                    mean_pct = pct_shift_clean.mean()
                    std_pct = pct_shift_clean.std()
                    if pd.notna(std_pct) and std_pct > 0:
                        z_score = (pct_shift - mean_pct) / std_pct
                        diff_df_pct[f"{col}_pct"] = z_score
                    else:
                        diff_df_pct[f"{col}_pct"] = 0.0
                else:
                    diff_df_pct[f"{col}_pct"] = np.nan
        return diff_df_pct
