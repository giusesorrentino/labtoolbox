"""Model fitting utilities for LabToolbox.

Exposes linear and generic model fitting routines along with bootstrap fitting.
"""

from .fit import bootstrap_fit
from ..stats import lin_fit, model_fit

__all__ = [
    "bootstrap_fit",
    "lin_fit",
    "model_fit"
]