"""Statistical tools for LabToolbox.

Provides histogram, residual analysis, outlier rejection, Bayesian inference,
and basic statistical estimators.
"""

from .stats import hist, residuals, posterior, propagate, bayes_factor, mean, lin_fit, model_fit

__all__ = [
    "hist", 
    "residuals", 
    "posterior", 
    "propagate", 
    "bayes_factor", 
    "mean", 
    "lin_fit", 
    "model_fit"
]