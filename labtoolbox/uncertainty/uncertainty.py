import numpy as _np
from warnings import warn as _warn

def numerical(f, x_val, x_err, params=()):
    """
    Uncertainty propagation via numerical derivatives.

    Parameters
    ----------
        f : callable
            Function `f(x1, ..., xn; a1, ..., am)` that returns an array of shape (N,).
        x_val : list of np.ndarray
            List of input arrays `x1,..., xn`, each with shape (N,).
        x_err : list of np.ndarray
            List of uncertainty arrays corresponding to each `x_i`, shape (N,).
        params : tuple, optional
            Tuple of constant parameters `(a1, ..., am)` to be passed to the function.

    Returns
    ----------
        f_val : np.ndarray
            Central values of the function, shape (N,).
        f_err : np.ndarray
            Propagated uncertainty, shape (N,).
    """

    _warn(
        "This function is part of a legacy module and may be removed in future versions. "
        "Use labtoolbox.stats.propagate instead.",
        category=DeprecationWarning,
        stacklevel=2
    )

    N = x_val[0].shape[0]
    n_vars = len(x_val)

    # Valori centrali della funzione
    f_val = f(*x_val, *params)
    f_var = _np.zeros(N)

    for i in range(n_vars):
        x = x_val[i]

        # Calcolo h_i come distanza minima tra punti consecutivi diviso 100
        dx = _np.diff(x)
        min_dx = _np.min(_np.abs(dx[dx != 0])) if _np.any(dx != 0) else 1.0
        h = min_dx / 100

        # Copia degli array per ±h
        x_plus = [x.copy() for x in x_val]
        x_minus = [x.copy() for x in x_val]
        x_plus[i]  += h
        x_minus[i] -= h

        f_plus = f(*x_plus, *params)
        f_minus = f(*x_minus, *params)

        df_dxi = (f_plus - f_minus) / (2 * h)
        f_var += (df_dxi * x_err[i])**2

    f_err = _np.sqrt(f_var)

    return f_val, f_err

def montecarlo(func, values, errs, N=10_000, seed=None):
    """
    Estimate the propagated uncertainty on a function of N variables using Monte Carlo simulation.

    Parameters
    ----------
    func : callable
        The function to evaluate. Must accept the same number of arguments as
        the length of `values`.
    values : array-like
        Central values of the input variables. Must be of the same length as `errs`.
    errs : array-like
        Standard deviations (1-sigma uncertainties) of the input variables.
    N : int, optional
        Number of Monte Carlo samples to generate. Default is `1e4`.
    seed : int or None, optional
        Seed for the random number generator, for reproducibility. Default is None.

    Returns
    -------
    mean : float
        Mean value of the function evaluated over the sampled inputs.
    std : float
        Standard deviation (uncertainty) of the function output.

    Notes
    -----
    - The input variables are sampled as independent normal distributions with given means
      and standard deviations.
    - Correlations between input variables are not taken into account.

    Example
    -------
    >>> def f(x, y): return x * y
    >>> montecarlo(f, [2.0, 3.0], [0.1, 0.2])
    (6.00..., 0.42...)
    """
    values = _np.array(values)
    errs = _np.array(errs)

    _warn(
        "This function is part of a legacy module and may be removed in future versions. "
        "Use labtoolbox.stats.propagate instead.",
        category=DeprecationWarning,
        stacklevel=2
    )

    if values.shape != errs.shape:
        raise ValueError("values and uncertainties must have the same shape.")

    rng = _np.random.default_rng(seed)

    # Generate samples from normal distributions
    samples = [
        rng.normal(loc=mu, scale=sigma, size=N)
        for mu, sigma in zip(values, errs)
    ]

    # Evaluate the function over all sampled inputs
    samples = _np.array(samples)  # shape: (n_vars, N)
    outputs = func(*samples)

    mean = _np.mean(outputs)
    std = _np.std(outputs, ddof=1)

    return mean, std