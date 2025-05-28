import numpy as _np
from .stats import model_fit, lin_fit

def bootstrap_fit(func, xdata, ydata, y_err = None, p0 = None, punits = None, n_iter = 1000, bounds = (-_np.inf, _np.inf)):
    """
    Performs a bootstrap analysis of the fit to estimate the parameter distributions, optionally accounting for the uncertainties sigma_y.

    Parameters
    ----------
        func : callable
            Model function to be fitted, in the form `func(x, *params)`.
        xdata : array_like
            Independent data (x-values).
        ydata : array_like
            Dependent data (y-values).
        y_err : array_like, optional
            Uncertainties associated with `ydata`. If provided, a weighted fit will be performed.
        p0 : array_like, optional
            Initial guess for the fit parameters.
        punits : list of str, optional
            List of strings specifying the units of each parameter. Default is `None`.
        n_iter : int, optional
            Number of bootstrap iterations (default: `1000`).
        bounds : 2-tuple of arrays, optional
            Lower and upper bounds for the fit parameters.

    Returns
    ----------
        popt_mean : array
            Mean values of the parameters obtained from the bootstrap samples.
        popt_std : array
            Standard deviations of the parameters (as uncertainty estimates).
        all_popt : array
            Full array of all parameter estimates (shape: `[n_iter, n_params]`).

    Notes
    ----------
    If the i-th parameter is dimensionless (a pure number), simply use an empty string `""` as the corresponding element in the `punits` list.
    """

    import warnings
    warnings.warn("This function is deprecated and will be removed in a future release. Consider using scipy.stats.bootstrap", DeprecationWarning)

    xdata = _np.asarray(xdata)
    ydata = _np.asarray(ydata)
    if y_err is not None:
        y_err = _np.asarray(y_err)
    n_points = len(xdata)
    all_popt = []

    for _ in range(n_iter):
        indices = _np.random.choice(n_points, n_points, replace=True)
        x_sample = xdata[indices]
        y_sample = ydata[indices]
        if y_err is not None:
            sigma_sample = y_err[indices]
        else:
            sigma_sample = None

        try:
            from scipy.optimize import curve_fit
            popt, _ = curve_fit(func, x_sample, y_sample, p0=p0, bounds=bounds, sigma=sigma_sample, absolute_sigma=True)
            all_popt.append(popt)
        except Exception:
            continue  # Ignora i fit che non convergono

    all_popt = _np.array(all_popt)
    popt_mean = _np.mean(all_popt, axis=0)
    popt_std = _np.std(all_popt, axis=0)

    for i in range(len(all_popt)):
        value = popt_mean[i]
        error = popt_std[i]

        if punits is not None:
            unit = punits[i]
        else:
            unit = ""

        from .utils import PrintResult

        if value > 1e4 or abs(value) < 1e-3:
            # Scrittura in notazione scientifica
            exponent = int(_np.floor(_np.log10(abs(value)))) if value != 0 else 0
            scaled_value = value / 10**exponent
            scaled_error = error / 10**exponent
            name = f"Parameter {i+1} [1e{exponent}]"
            PrintResult(scaled_value, scaled_error, name=name, ux=unit)
        else:
            name = f"Parameter {i+1} "
            PrintResult(value, error, name=name, ux=unit)

    return popt_mean, popt_std, all_popt