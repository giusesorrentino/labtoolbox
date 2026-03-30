"""Statistical tools for LabToolbox.

Provides histogram, residual analysis, outlier rejection, Bayesian inference,
and basic statistical estimators.
"""

from .stats import hist, residuals, remove_outliers, posterior, propagate, bayes_factor, mean, lin_fit, model_fit

__all__ = [
    "hist", 
    "residuals", 
    "remove_outliers", 
    "posterior", 
    "propagate", 
    "bayes_factor", 
    "mean", 
    "lin_fit", 
    "model_fit"
]