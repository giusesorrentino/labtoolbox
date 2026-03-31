import numpy as _np
from warnings import warn as _warn

# Removed functions kept commented for reference.
# def numerical(f, x_val, x_err, params=()):
#     """
#     Uncertainty propagation via numerical derivatives.
#     """
#     _warn(
#         "This function is part of a legacy module and may be removed in future versions. "
#         "Use labtoolbox.stats.propagate instead.",
#         category=DeprecationWarning,
#         stacklevel=2
#     )
#     N = x_val[0].shape[0]
#     n_vars = len(x_val)
#     f_val = f(*x_val, *params)
#     f_var = _np.zeros(N)
#     for i in range(n_vars):
#         x = x_val[i]
#         dx = _np.diff(x)
#         min_dx = _np.min(_np.abs(dx[dx != 0])) if _np.any(dx != 0) else 1.0
#         h = min_dx / 100
#         x_plus = [x.copy() for x in x_val]
#         x_minus = [x.copy() for x in x_val]
#         x_plus[i] += h
#         x_minus[i] -= h
#         f_plus = f(*x_plus, *params)
#         f_minus = f(*x_minus, *params)
#         df_dxi = (f_plus - f_minus) / (2 * h)
#         f_var += (df_dxi * x_err[i])**2
#     f_err = _np.sqrt(f_var)
#     return f_val, f_err
#
# def montecarlo(func, values, errs, N=10_000, seed=None):
#     """
#     Estimate the propagated uncertainty on a function of N variables using Monte Carlo simulation.
#     """
#     values = _np.array(values)
#     errs = _np.array(errs)
#     _warn(
#         "This function is part of a legacy module and may be removed in future versions. "
#         "Use labtoolbox.stats.propagate instead.",
#         category=DeprecationWarning,
#         stacklevel=2
#     )
#     if values.shape != errs.shape:
#         raise ValueError("values and uncertainties must have the same shape.")
#     rng = _np.random.default_rng(seed)
#     samples = [
#         rng.normal(loc=mu, scale=sigma, size=N)
#         for mu, sigma in zip(values, errs)
#     ]
#     samples = _np.array(samples)
#     outputs = func(*samples)
#     mean = _np.mean(outputs)
#     std = _np.std(outputs, ddof=1)
#     return mean, std
