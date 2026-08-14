"""Custom loss functions for error-correction and prediction models."""

from .horizon_aware_asymmetric_loss import (
    compute_sigma_per_horizon,
    hal_grad_hess,
    hal_xgb_objective,
)
from .diagnostic_weighted_loss import (
    derive_regime,
    derive_uncertainty,
    diagnostic_weighted_grad_hess,
    diagnostic_xgb_objective,
)

__all__ = [
    "compute_sigma_per_horizon",
    "hal_grad_hess",
    "hal_xgb_objective",
    "derive_regime",
    "derive_uncertainty",
    "diagnostic_weighted_grad_hess",
    "diagnostic_xgb_objective",
]
