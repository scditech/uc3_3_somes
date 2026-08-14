"""
Diagnostic-Weighted Error Correction Loss.

High-level form (per sample i):

    ℓ_i = w_i * ℓ_err(y_i, ŷ_i) + λ * ℓ_diag(i)

This module implements the gradient / Hessian of the core term
    w_i * ℓ_err(y_i, ŷ_i)
where the diagnostic weight decomposes as:

    w_i = w_diff(d_i) * w_horizon(h_i) * w_regime(z_i)

with:
- w_diff(d_i): difficulty-based weight from |residual|
- w_horizon(h_i): horizon-dependent importance
- w_regime(z_i): regime tag weight (e.g. clear / mixed / cloudy)

The piecewise error term ℓ_err is:
- Short horizon (1–3) & clear regime      → MSE
- Long horizon (>=8) & high uncertainty   → asymmetric (under-correction penalty)
- Cloudy regime                           → Huber-like
- Default                                 → MSE

The implementation is tailored for XGBoost custom objectives:
`diagnostic_weighted_grad_hess` returns (grad, hess),
and `diagnostic_xgb_objective` builds the `(preds, dtrain) -> (grad, hess)` closure.
"""

from __future__ import annotations

import numpy as np


def derive_regime(diffuse_fraction: np.ndarray) -> np.ndarray:
    """
    Simple regime tagging from diffuse_fraction:
      0 = clear   (< 0.35)
      1 = mixed   (0.35–0.65)
      2 = cloudy  (> 0.65)
    """
    df = np.asarray(diffuse_fraction, dtype=float).ravel()
    regime = np.zeros(len(df), dtype=np.int32)
    regime[(df >= 0.35) & (df < 0.65)] = 1
    regime[df >= 0.65] = 2
    return regime


def derive_uncertainty(
    horizon: np.ndarray,
    unc_column: np.ndarray | None = None,
) -> np.ndarray:
    """
    Uncertainty level: 0 = low, 1 = high.

    If unc_column is provided, split at its median. Otherwise, use
    a simple heuristic: horizons >= 8 are treated as high-uncertainty.
    """
    h = np.asarray(horizon).ravel()
    if unc_column is not None and len(unc_column) == len(h):
        u = np.asarray(unc_column).ravel()
        med = np.nanmedian(u)
        return (u > med).astype(np.int32)
    return (h >= 8).astype(np.int32)


def w_diff(
    difficulty: np.ndarray,
    scale: float = 1.0,
    alpha: float = 0.5,
    cap: float = 2.0,
) -> np.ndarray:
    """
    Difficulty component: upweight hard samples based on |residual|.

    w_diff(d) = 1 + alpha * min(d/scale, cap)
    """
    d = np.asarray(difficulty, dtype=float).ravel()
    s = scale if scale > 0 else 1.0
    raw = 1.0 + alpha * np.minimum(d / s, cap)
    return raw


def w_horizon(
    horizon: np.ndarray,
    beta: float = 0.3,
    h_max: float = 15.0,
) -> np.ndarray:
    """
    Horizon importance term (e.g. nearer horizons slightly higher weight).

    w_h = 1 / (1 + beta * (h - 1) / h_max)
    """
    h = np.asarray(horizon, dtype=float).ravel()
    return 1.0 / (1.0 + beta * (h - 1.0) / max(h_max, 1.0))


def w_regime(
    regime: np.ndarray,
    weights_per_regime: tuple[float, float, float] = (1.0, 1.1, 1.2),
) -> np.ndarray:
    """
    Regime tag weight: e.g. clear, mixed, cloudy.
    Default weights (1.0, 1.1, 1.2).
    """
    z = np.asarray(regime, dtype=int).ravel()
    w = np.ones(len(z), dtype=float)
    for r, v in enumerate(weights_per_regime):
        if r < len(weights_per_regime):
            w[z == r] = v
    return w


def _mse_grad_hess(residual: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """MSE per-sample: L = 0.5 * e^2, grad = -e, hess = 1."""
    e = np.asarray(residual).ravel()
    return -e, np.ones_like(e)


def _asymmetric_grad_hess(
    residual: np.ndarray,
    alpha_under: float = 1.5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Asymmetric squared loss:
      - under-correction (e >= 0) scaled by alpha_under,
      - over-correction (e < 0) scaled by 1.
    """
    e = np.asarray(residual).ravel()
    grad = np.where(e >= 0, -2.0 * alpha_under * e, -2.0 * e)
    hess = np.where(e >= 0, 2.0 * alpha_under, 2.0)
    return grad, hess


def _huber_grad_hess(
    residual: np.ndarray,
    delta: float = 1.5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Huber-like robust loss:
      L(e) = 0.5 * e^2       if |e| <= delta
             delta(|e| - .5δ) otherwise
    Smooth Hessian: 1 inside, delta/|e| (clipped) outside.
    """
    e = np.asarray(residual).ravel()
    abs_e = np.abs(e)
    grad = np.where(abs_e <= delta, -e, -delta * np.sign(e))
    hess = np.where(
        abs_e <= delta, np.ones_like(e), np.maximum(delta / (abs_e + 1e-12), 0.1)
    )
    return grad, hess


def diagnostic_weighted_grad_hess(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    horizon: np.ndarray,
    regime: np.ndarray,
    unc: np.ndarray,
    *,
    residual_scale: float = 1.0,
    w_diff_alpha: float = 0.5,
    w_horizon_beta: float = 0.3,
    alpha_under: float = 1.5,
    huber_delta: float = 1.5,
    normalize_weights: bool = True,
    return_weights: bool = False,
    apply_weights: bool = True,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Gradient and Hessian for the Diagnostic-Weighted Error Correction Loss.

    Residual e_i = y_true_i - y_pred_i.

    Weight:
        w_i = w_diff(d_i) * w_horizon(h_i) * w_regime(z_i)

    Piecewise ℓ_err:
      - Short (1–3) & clear (z=0)        → MSE
      - Long (>=8) & high-unc (unc>=.5) → asymmetric
      - Cloudy (z=2, and not above)     → Huber
      - Default                         → MSE

    If apply_weights=False, w_i is forced to 1 so grad/hess are constraint-only
    (no diagnostic scaling). This is useful for diagnostics / ablations.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    n = len(y_true)
    residual = y_true - y_pred
    h = np.asarray(horizon).ravel()
    z = np.asarray(regime).ravel()
    u = np.asarray(unc).ravel()
    if len(h) != n or len(z) != n or len(u) != n:
        raise ValueError("horizon, regime, unc must match length of y_true.")

    # Difficulty = |residual| (used for w_diff)
    difficulty = np.abs(residual)
    if residual_scale <= 0:
        residual_scale = (
            float(np.median(difficulty[difficulty > 0]))
            if np.any(difficulty > 0)
            else 1.0
        )
    wi_diff = w_diff(difficulty, scale=residual_scale, alpha=w_diff_alpha)
    wi_h = w_horizon(h, beta=w_horizon_beta)
    wi_r = w_regime(z)
    w_i = wi_diff * wi_h * wi_r
    if normalize_weights:
        w_i = w_i / (np.mean(w_i) + 1e-12)
    if not apply_weights:
        w_i = np.ones(n, dtype=float)

    # Branch masks: short h in [1,2,3], long h >= 8
    short_h = (h >= 1) & (h <= 3)
    long_h = h >= 8
    clear_r = z == 0
    cloudy_r = z == 2

    grad_err = np.empty(n)
    hess_err = np.empty(n)
    # Short + clear -> MSE
    mask_short_clear = short_h & clear_r
    g, he = _mse_grad_hess(residual[mask_short_clear])
    grad_err[mask_short_clear], hess_err[mask_short_clear] = g, he
    # Long + high unc -> asymmetric
    mask_long_unc = long_h & (u >= 0.5)
    g, he = _asymmetric_grad_hess(residual[mask_long_unc], alpha_under=alpha_under)
    grad_err[mask_long_unc], hess_err[mask_long_unc] = g, he
    # Cloudy -> Huber
    mask_cloudy = cloudy_r & ~mask_short_clear & ~mask_long_unc
    g, he = _huber_grad_hess(residual[mask_cloudy], delta=huber_delta)
    grad_err[mask_cloudy], hess_err[mask_cloudy] = g, he
    # Default -> MSE
    mask_default = ~mask_short_clear & ~mask_long_unc & ~mask_cloudy
    g, he = _mse_grad_hess(residual[mask_default])
    grad_err[mask_default], hess_err[mask_default] = g, he

    grad = w_i * grad_err
    hess = w_i * hess_err
    np.maximum(hess, 1e-6, out=hess)
    if return_weights:
        return grad, hess, w_i
    return grad, hess


def diagnostic_xgb_objective(
    train_horizons: np.ndarray,
    train_regimes: np.ndarray,
    train_unc: np.ndarray,
    residual_scale: float = 1.0,
    w_diff_alpha: float = 0.5,
    w_horizon_beta: float = 0.3,
    alpha_under: float = 1.5,
    huber_delta: float = 1.5,
):
    """
    Build an XGBoost custom objective for the Diagnostic-Weighted Error Correction Loss.

    XGBoost expects: obj(preds, dtrain) -> (grad, hess).
    Row order of dtrain must match train_horizons / train_regimes / train_unc.
    """

    def _obj(preds, dtrain):
        labels = dtrain.get_label()
        grad, hess = diagnostic_weighted_grad_hess(
            labels,
            preds,
            train_horizons,
            train_regimes,
            train_unc,
            residual_scale=residual_scale,
            w_diff_alpha=w_diff_alpha,
            w_horizon_beta=w_horizon_beta,
            alpha_under=alpha_under,
            huber_delta=huber_delta,
            normalize_weights=True,
            return_weights=False,
            apply_weights=True,
        )
        return grad, hess

    return _obj


__all__ = [
    "derive_regime",
    "derive_uncertainty",
    "diagnostic_weighted_grad_hess",
    "diagnostic_xgb_objective",
]
