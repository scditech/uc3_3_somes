"""
HorizonFeatureMetaLayer: horizon-aware feature weighting for error correction (and similar pipelines).

Integrates:
1. Days ahead (number of days until current time to predicted time) — horizon info so the model
   knows that farther horizons have more shifted sensor data.
2. Per-feature, per-horizon weights from differences/errors: bigger difference/error → less important
   feature (inverse-error weighting).
3. Optional fit from ErrorEvaluator-style summary (pred_sequence_id x variable → metric).

Usage:
  meta = HorizonFeatureMetaLayer(horizon_col="pred_sequence_id", add_days_ahead=True)
  meta.fit_from_diff_summary(summary_df, metric_col="abs_mean_diff")  # optional
  X_weighted = meta.transform(X)
  # Then train/predict with X_weighted (includes days_ahead and pred_sequence_id if requested).
"""

import numpy as np
import pandas as pd
from typing import Optional, List


class HorizonFeatureMetaLayer:
    def __init__(
        self,
        horizon_col: str = "pred_sequence_id",
        add_days_ahead: bool = True,
        days_ahead_col: str = "days_ahead",
        weight_formula: str = "inverse_error",
        inverse_scale: float = 1.0,
        # Extra knobs to make the meta-learner more expressive
        feature_importance_power: float = 1.5,
        horizon_importance_power: float = 1.0,
        local_weight_power: float = 1.5,
        # How to inject the meta-knowledge into X:
        # - scale_original_features: if True, multiply original features by weights
        # - add_weight_features: if True (default), add separate *_meta_weight columns
        scale_original_features: bool = False,
        add_weight_features: bool = True,
        # Hard feature pruning based on average error:
        # drop the worst frac of features globally (0.0 = keep all).
        drop_worst_feature_frac: float = 0.0,
        horizon_decay_rate: float = 0.1,
        skip_cols: Optional[List[str]] = None,
        normalize_metric: bool = True,
        min_weight: float = 0.0,
    ):
        """
        Args:
            horizon_col: Column name for horizon/sequence id (e.g. pred_sequence_id 1–15).
            add_days_ahead: If True, add days_ahead = horizon - 1 as a feature (0 for seq 1, 1 for seq 2, ...).
            days_ahead_col: Name of the added days_ahead column.
            weight_formula: "inverse_error" -> weight = 1/(1 + scale*metric), "exp_decay" -> weight = exp(-scale*metric).
            inverse_scale: Scale factor for metric in weight (higher = more aggressive downweighting).
            feature_importance_power: Exponent applied to global per-feature importance. >1 sharpens differences.
            horizon_importance_power: Exponent applied to global per-horizon importance.
            local_weight_power: Exponent applied to the per-(horizon, feature) weight from the metric itself.
            scale_original_features: If True, original features are multiplied by the per-row weights.
                If False (default), original features are left unchanged and meta information is only added as
                separate *_meta_weight columns.
            add_weight_features: If True (default), add one column per feature with the meta weight
                (e.g. \"temp__meta_weight\") plus a global horizon-importance column.
            drop_worst_feature_frac: If > 0, drop this fraction of features with the lowest
                global importance (highest average error) entirely from X during transform.
            horizon_decay_rate: Used when no diff summary is fitted: global decay by horizon, weight = exp(-rate*(h-1)).
            skip_cols: Columns to never weight (always 1.0); e.g. ["datetime", "pred_sequence_id", "days_ahead"].
            normalize_metric: If True (default), scale metric per variable by its max across horizons to [0, 1]
                before computing weights. Prevents raw abs_mean_diff (e.g. tens/hundreds) from over-downweighting.
            min_weight: Minimum weight per (horizon, feature). Use e.g. 0.2 to avoid fully zeroing features.
        """
        self.horizon_col = horizon_col
        self.add_days_ahead = add_days_ahead
        self.days_ahead_col = days_ahead_col
        self.weight_formula = weight_formula
        self.inverse_scale = inverse_scale
        self.feature_importance_power = feature_importance_power
        self.horizon_importance_power = horizon_importance_power
        self.local_weight_power = local_weight_power
        self.scale_original_features = scale_original_features
        self.add_weight_features = add_weight_features
        self.drop_worst_feature_frac = drop_worst_feature_frac
        self.horizon_decay_rate = horizon_decay_rate
        self.skip_cols = skip_cols or ["datetime"]
        self.normalize_metric = normalize_metric
        self.min_weight = min_weight

        # (horizon_id, feature_name) -> weight; filled by fit_from_diff_summary or fit
        self._weight_map: dict = {}
        self._feature_names: Optional[List[str]] = None
        self._horizons_seen: Optional[List[int]] = None
        self._fitted: bool = False
        # Diagnostics (not required by the rest of the code, but handy for debugging / inspection)
        self._feature_importance: Optional[pd.Series] = None
        self._horizon_importance: Optional[pd.Series] = None
        self._dropped_features: Optional[List[str]] = None

    def fit_from_diff_summary(
        self,
        summary_df: pd.DataFrame,
        metric_col: str = "abs_mean_diff",
        horizon_col_summary: Optional[str] = None,
        variable_col: Optional[str] = None,
    ) -> "HorizonFeatureMetaLayer":
        """
        Build per-horizon, per-feature weights from ErrorEvaluator-style summary.

        summary_df must have: horizon id column (default pred_sequence_id), variable name column (default "variable"),
        and metric_col (e.g. abs_mean_diff, abs_mean_diff_pct_raw). Bigger metric → smaller weight.

        If normalize_metric is True (default), the metric is scaled per variable by its max across horizons
        to [0, 1] before applying the weight formula, so raw abs_mean_diff (e.g. tens/hundreds) does not
        over-downweight features.

        Args:
            summary_df: DataFrame with columns [horizon_id, variable, metric_col].
            metric_col: Column name for difference/error metric (e.g. "abs_mean_diff", "abs_mean_diff_pct_raw").
            horizon_col_summary: Horizon column name in summary_df (default: self.horizon_col).
            variable_col: Variable/feature name column in summary_df (default: "variable").
        """
        hcol = horizon_col_summary or self.horizon_col
        vcol = variable_col or "variable"
        if metric_col not in summary_df.columns:
            raise ValueError(f"summary_df must have column '{metric_col}'")
        if hcol not in summary_df.columns or vcol not in summary_df.columns:
            raise ValueError(f"summary_df must have columns '{hcol}' and '{vcol}'")

        piv = summary_df.pivot(index=hcol, columns=vcol, values=metric_col)

        # Normalize metric per variable to [0, 1] so raw scale (e.g. abs_mean_diff in tens)
        # doesn't over-downweight. We keep this, but then build a *factorised* importance:
        #  - global per-feature importance (hard downweight consistently bad variables)
        #  - global per-horizon importance (downweight globally noisy horizons)
        #  - local (horizon, feature) importance.
        if self.normalize_metric:
            piv = piv.copy()
            for c in piv.columns:
                m = piv[c].max()
                if m is not None and pd.notna(m) and m > 0:
                    piv[c] = piv[c] / m
        self._horizons_seen = list(piv.index)
        self._feature_names = [c for c in piv.columns if c not in self.skip_cols]

        # Replace raw metric by something robust to NaNs for aggregation
        metric_values = piv.replace({np.inf: np.nan, -np.inf: np.nan})

        # Per-feature and per-horizon average metric (ignoring NaNs)
        feat_avg = metric_values.mean(axis=0, skipna=True)
        horiz_avg = metric_values.mean(axis=1, skipna=True)

        # Convert average metrics into importance terms in (0, 1].
        def _metric_to_weight(m: pd.Series) -> pd.Series:
            # Higher metric -> lower weight.
            if self.weight_formula == "inverse_error":
                w = 1.0 / (1.0 + self.inverse_scale * m.astype(float))
            else:  # exp_decay
                w = np.exp(-self.inverse_scale * m.astype(float))
            # NaN metrics fall back to 1.0 (no penalty).
            w = w.fillna(1.0)
            return w.clip(lower=0.0, upper=1.0)

        feat_importance = _metric_to_weight(feat_avg)
        horiz_importance = _metric_to_weight(horiz_avg)

        # Store for inspection / debugging.
        self._feature_importance = feat_importance
        self._horizon_importance = horiz_importance

        # Optional: decide a global feature subset to keep, based on average importance.
        # This makes the meta layer much more "opinionated" and can boost performance
        # by removing consistently bad signals.
        self._dropped_features = None
        if 0.0 < self.drop_worst_feature_frac < 1.0:
            # Only consider non-skip features that actually appear in the summary.
            feat_imp_non_skip = feat_importance.drop(
                index=[c for c in feat_importance.index if c in self.skip_cols],
                errors="ignore",
            )
            if len(feat_imp_non_skip) > 0:
                n_drop = int(
                    np.floor(len(feat_imp_non_skip) * self.drop_worst_feature_frac)
                )
                if n_drop >= 1:
                    # Sort ascending → smallest importance (worst) first.
                    worst = (
                        feat_imp_non_skip.sort_values(ascending=True)
                        .iloc[:n_drop]
                        .index.tolist()
                    )
                    self._dropped_features = worst

        # Build final per-(horizon, feature) weights as (local * feature * horizon),
        # each term optionally exponentiated for sharper contrast.
        for h in piv.index:
            base_h = float(horiz_importance.loc[h]) ** self.horizon_importance_power
            for f in piv.columns:
                if f in self.skip_cols:
                    continue
                val = piv.loc[h, f]
                if pd.isna(val):
                    local = 1.0
                else:
                    if self.weight_formula == "inverse_error":
                        local = 1.0 / (1.0 + self.inverse_scale * float(val))
                    else:  # exp_decay
                        local = float(np.exp(-self.inverse_scale * float(val)))
                base_f = float(feat_importance.loc[f]) ** self.feature_importance_power
                local = float(local) ** self.local_weight_power

                w = base_h * base_f * local

                # Ensure strictly within (0, 1] then apply min_weight if requested.
                w = max(min(w, 1.0), 0.0)
                if self.min_weight > 0:
                    w = max(w, self.min_weight)

                self._weight_map[(int(h), str(f))] = w
        self._fitted = True
        return self

    def fit(
        self,
        X: pd.DataFrame,
        horizon: Optional[pd.Series] = None,
    ) -> "HorizonFeatureMetaLayer":
        """
        Fit without a diff summary: use horizon-only decay (same weight for all features per horizon).
        Weight for horizon h = exp(-horizon_decay_rate * (h - 1)).
        """
        h = horizon if horizon is not None else X.get(self.horizon_col)
        if h is None:
            raise ValueError("horizon must be provided or X must contain horizon_col")
        self._horizons_seen = sorted(
            pd.Series(h).dropna().unique().astype(int).tolist()
        )
        feature_cols = [
            c
            for c in X.columns
            if c not in self.skip_cols
            and c != self.horizon_col
            and c != self.days_ahead_col
        ]
        self._feature_names = feature_cols
        for hi in self._horizons_seen:
            decay = np.exp(-self.horizon_decay_rate * (hi - 1))
            for f in self._feature_names:
                self._weight_map[(hi, f)] = decay
        self._fitted = True
        return self

    def _get_weight(self, horizon_id: int, feature_name: str) -> float:
        if feature_name in self.skip_cols:
            return 1.0
        key = (int(horizon_id), str(feature_name))
        return self._weight_map.get(key, 1.0)

    def get_sample_weights(
        self,
        X: pd.DataFrame,
        horizon: Optional[pd.Series] = None,
        aggregate: str = "mean",
    ) -> np.ndarray:
        """
        Return one weight per row for use as XGBoost sample_weight.
        Weight is aggregated from per-(horizon, feature) weights so rows from
        noisier horizons (or with noisier feature combos) get lower weight.

        Args:
            X: DataFrame with horizon_col and feature columns.
            horizon: Optional series; if None, taken from X[horizon_col].
            aggregate: "mean" or "min" over feature weights per row (default "mean").

        Returns:
            Array of shape (len(X),) with values in (0, 1] (or [min_weight, 1] if min_weight set).
        """
        h = horizon if horizon is not None else X.get(self.horizon_col)
        if h is None:
            return np.ones(len(X))
        h = pd.Series(h, index=X.index).astype(int)
        if not self._fitted:
            return np.ones(len(X))
        cols = [
            c
            for c in X.columns
            if c not in self.skip_cols
            and c != self.days_ahead_col
            and c != self.horizon_col
        ]
        if not cols:
            return np.ones(len(X))
        weights_per_col = np.column_stack(
            [h.map(lambda hi: self._get_weight(hi, c)).values for c in cols]
        )
        if aggregate == "min":
            out = np.min(weights_per_col, axis=1)
        else:
            out = np.mean(weights_per_col, axis=1)
        return np.asarray(out, dtype=float)

    def transform(
        self,
        X: pd.DataFrame,
        horizon: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """
        Scale features by per-horizon, per-feature weights. Optionally add days_ahead column.

        Non-feature columns (e.g. datetime) are left unchanged. Columns in skip_cols are not scaled.
        """
        X = X.copy()
        h = horizon if horizon is not None else X.get(self.horizon_col)
        if h is None:
            raise ValueError("horizon must be provided or X must contain horizon_col")
        h = pd.Series(h, index=X.index).astype(int)

        if self.add_days_ahead and self.days_ahead_col not in X.columns:
            X[self.days_ahead_col] = (h - 1).values

        if not self._fitted:
            return X

        # Drop globally bad features decided during fit_from_diff_summary.
        if self._dropped_features:
            X = X.drop(
                columns=[c for c in self._dropped_features if c in X.columns],
                errors="ignore",
            )

        # Optionally add a global horizon-importance column (same for all features in a row).
        if self.add_weight_features and self._horizon_importance is not None:
            X["meta_horizon_importance"] = h.map(
                lambda hi: float(self._horizon_importance.get(int(hi), 1.0))
            ).values

        for col in list(X.columns):
            # We never scale skip_cols, the synthetic days_ahead feature, or the horizon id itself.
            if (
                col in self.skip_cols
                or col == self.days_ahead_col
                or col == self.horizon_col
            ):
                continue

            # Only compute weights for "real" features (not already meta columns).
            if col.endswith("__meta_weight"):
                continue

            weights = h.map(lambda hi: self._get_weight(hi, col))

            # Optionally scale original features in-place.
            if self.scale_original_features:
                X[col] = X[col] * weights

            # Optionally expose weights as separate meta features so the downstream model
            # can decide how to use them (this tends to be safer than only scaling).
            if self.add_weight_features:
                meta_col = f"{col}__meta_weight"
                X[meta_col] = weights.values
        return X

    def fit_transform(
        self,
        X: pd.DataFrame,
        horizon: Optional[pd.Series] = None,
        summary_df: Optional[pd.DataFrame] = None,
        metric_col: str = "abs_mean_diff",
    ) -> pd.DataFrame:
        """Fit from summary_df if provided, else from horizon decay; then transform X."""
        if summary_df is not None:
            self.fit_from_diff_summary(summary_df, metric_col=metric_col)
        else:
            self.fit(X, horizon=horizon)
        return self.transform(X, horizon=horizon)
