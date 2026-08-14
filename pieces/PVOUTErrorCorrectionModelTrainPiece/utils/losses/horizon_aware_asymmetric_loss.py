"""
Horizon-Aware Asymmetric Loss (HAL) for multi-horizon forecast correction.

Reworked formulation to avoid underfitting high-σ (far) horizons and to align
better with RMSE/R² evaluation:

  L(e_i, h_i) = w(h_i) * phi(e_i),    phi(e) = alpha_under * e^2 if e >= 0 else alpha_over * e^2

Horizon weight w(h) can be:
  - "floor":  w(h) = max( 1 / (1 + beta * sigma_norm(h)), w_floor )  with sigma_norm = sigma(h)/median(sigma)
    So far horizons are never weighted below w_floor (default 0.4), avoiding underfitting.
  - "balanced": w(h) = clip( sigma(h)/median(sigma), 0.5, 2.0 )
    Emphasizes learning on noisier horizons so all horizons contribute to the loss.
  - "original": w(h) = 1 / (1 + beta * sigma(h))  [can underfit far horizons]

Asymmetry: alpha_under > alpha_over penalizes under-correction more (e >= 0).
           alpha_under = alpha_over = 1 gives symmetric MSE (recommended for best RMSE/R²).
"""

import numpy as np


def compute_sigma_per_horizon(
    y_correction: np.ndarray,
    horizons: np.ndarray,
) -> dict[int, float]:
    """
    Compute empirical standard deviation of correction targets per horizon.

    Args:
        y_correction: 1D array of correction targets (true_pvout - pred_pvout).
        horizons: 1D array of horizon ids, same length as y_correction.

    Returns:
        Dict mapping horizon_id -> std. Horizons with <2 samples get np.nan
        (caller should replace with fallback, e.g. global std or 1.0).
    """
    y_correction = np.asarray(y_correction).ravel()
    horizons = np.asarray(horizons, dtype=int).ravel()
    sigma_per_horizon = {}
    for h in np.unique(horizons):
        mask = horizons == h
        vals = y_correction[mask]
        if len(vals) >= 2:
            sigma_per_horizon[int(h)] = float(np.std(vals))
        else:
            sigma_per_horizon[int(h)] = np.nan
    return sigma_per_horizon


def _horizon_weights(
    horizons: np.ndarray,
    sigma_per_horizon: dict[int, float],
    beta: float,
    weight_mode: str,
    weight_floor: float,
    sigma_default: float,
) -> np.ndarray:
    """Compute per-sample horizon weights w(h). Returns 1D array same length as horizons."""
    sigma_h = np.array(
        [sigma_per_horizon.get(int(h), sigma_default) for h in horizons],
        dtype=float,
    )
    np.nan_to_num(sigma_h, copy=False, nan=sigma_default, posinf=sigma_default)
    np.maximum(sigma_h, 1e-10, out=sigma_h)

    valid_sigmas = [s for s in sigma_per_horizon.values() if np.isfinite(s) and s > 0]
    median_sigma = float(np.median(valid_sigmas)) if valid_sigmas else sigma_default
    if median_sigma <= 0:
        median_sigma = sigma_default
    sigma_norm = sigma_h / median_sigma

    if weight_mode == "original":
        # Use normalized sigma so beta is scale-invariant
        w_h = 1.0 / (1.0 + beta * sigma_norm)
    elif weight_mode == "floor":
        # Relative sigma so beta is scale-free; floor so far horizons still get meaningful gradient
        w_h = 1.0 / (1.0 + beta * sigma_norm)
        w_h = np.maximum(w_h, weight_floor)
    elif weight_mode == "balanced":
        # Upweight high-sigma horizons (so model learns them); clip to avoid extremes
        w_h = sigma_norm
        w_h = np.clip(w_h, 0.5, 2.0)
    else:
        w_h = np.ones_like(sigma_h, dtype=float)

    return w_h


def hal_grad_hess(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    horizons: np.ndarray,
    sigma_per_horizon: dict[int, float],
    alpha: float = 1.0,
    beta: float = 0.5,
    weight_mode: str = "floor",
    weight_floor: float = 0.4,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Gradient and Hessian of HAL loss w.r.t. predictions.

    Residual e_i = y_true_i - y_pred_i (under-correction: e >= 0, over-correction: e < 0).

    Loss: L_i = w(h_i) * ( alpha * e_i^2 if e_i >= 0 else e_i^2 )
    grad_i = dL/dy_pred_i = -2 * w(h_i) * ( alpha*e_i if e_i>=0 else e_i )
    hess_i = 2 * w(h_i) * ( alpha if e_i>=0 else 1 )

    Args:
        y_true: True correction targets (delta).
        y_pred: Predicted corrections.
        horizons: Horizon id per sample, same length as y_true/y_pred.
        sigma_per_horizon: Dict horizon_id -> std (from training set).
        alpha: Asymmetry (alpha for e>=0; 1 for e<0). Use 1.0 for symmetric (best for RMSE).
        beta: Horizon adaptation strength (used in "original" and "floor" modes).
        weight_mode: "original" | "floor" | "balanced". "floor" recommended.
        weight_floor: Minimum w(h) in "floor" mode (e.g. 0.4) so far horizons are not underfit.

    Returns:
        grad, hess: 1D arrays, same shape as y_pred.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    horizons = np.asarray(horizons, dtype=int).ravel()
    n = len(y_true)
    if len(y_pred) != n or len(horizons) != n:
        raise ValueError("y_true, y_pred, and horizons must have the same length.")

    residuals = y_true - y_pred  # e_i

    sigma_default = 1.0
    if sigma_per_horizon:
        valid_sigmas = [
            s for s in sigma_per_horizon.values() if np.isfinite(s) and s > 0
        ]
        if valid_sigmas:
            sigma_default = float(np.mean(valid_sigmas))

    w_h = _horizon_weights(
        horizons,
        sigma_per_horizon,
        beta=beta,
        weight_mode=weight_mode,
        weight_floor=weight_floor,
        sigma_default=sigma_default,
    )

    # Asymmetric: under-correction (e >= 0) scaled by alpha; over-correction by 1. alpha=1 => symmetric MSE
    grad = np.where(residuals >= 0, -2.0 * alpha * residuals, -2.0 * residuals)
    hess = np.where(residuals >= 0, 2.0 * alpha, 2.0)
    grad = grad * w_h
    hess = hess * w_h

    return grad, hess


def hal_xgb_objective(
    train_horizons: np.ndarray,
    sigma_per_horizon: dict[int, float],
    alpha: float = 1.0,
    beta: float = 0.5,
    weight_mode: str = "floor",
    weight_floor: float = 0.4,
):
    """
    Build an XGBoost custom objective for HAL.

    XGBoost expects obj(preds, dtrain) -> (grad, hess). Row order of dtrain
    must match train_horizons.

    Args:
        train_horizons: 1D array of horizon id per training sample (same order as dtrain).
        sigma_per_horizon: Dict from compute_sigma_per_horizon on training corrections.
        alpha: Asymmetry (1.0 = symmetric, recommended for RMSE/R²).
        beta: Horizon adaptation strength.
        weight_mode: "original" | "floor" | "balanced".
        weight_floor: Min weight in "floor" mode (default 0.4).

    Returns:
        A function (preds, dtrain) -> (grad, hess) for use in xgb.train(..., obj=...).
    """

    def _obj(preds, dtrain):
        labels = dtrain.get_label()
        grad, hess = hal_grad_hess(
            labels,
            preds,
            train_horizons,
            sigma_per_horizon,
            alpha=alpha,
            beta=beta,
            weight_mode=weight_mode,
            weight_floor=weight_floor,
        )
        return grad, hess

    return _obj
