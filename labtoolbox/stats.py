import math as _math
import numpy as _np
import matplotlib.pyplot as _plt
from warnings import warn

def hist(data, data_err, scale = 0, bins = "auto", label = "", unit = "", verbose = True):
    """
    Plots the histogram of a dataset and assess its Gaussianity using statistical indicators and a Shapiro-Wilk test.

    This function visualizes the empirical distribution of a dataset `data`, optionally accounting for individual measurement
    uncertainties `data_err`. It overlays a Gaussian curve parameterized by the estimated mean and standard deviation, and 
    performs a normality test, reporting skewness, kurtosis, and the p-value from the Shapiro-Wilk test.

    Parameters
    ----------
    data : array-like
        Numerical data representing the variable of interest.
    data_err : scalar or array-like or None
        Array of uncertainties associated with each element of `data`. If `None`, uncertainties are not included in the 
        computation of the effective standard deviation.
    scale : int, optional
        Scaling exponent for `data` and `data_err` (default is `0`). For example, `scale = -2` rescales the i_nputs by 1e2.
    bins : int or str, optional
        Number of bins or binning strategy passed to `matplotlib.pyplot.hist`. Defaults to `"auto"`.
    label : str, optional
        Label for the x-axis, typically the name of the variable.
    unit : str, optional
        Unit of measurement for the x-axis variable (e.g., "cm"). If provided, it will be displayed in the axis label and summary output.
    verbose : bool, optional
        If `True`, prints a formatted table of ... Default is `True`.

    Returns
    -------
    mean : float
        Arithmetic mean of the scaled data.
    sigma : float
        Effective standard deviation of the distribution, accounting for both the empirical spread and uncertainties (if provided).
    skewness : float
        Skewness of the distribution.
    kurtosis : float
        Kurtosis of the distribution.
    p_value : float
        p-value from the Shapiro-Wilk test for normality.

    Notes
    -----
    - The effective standard deviation is computed as `np.sqrt(data.std()**2 + np.sum(data_err**2)/len(data_err))` if `data_err` is provided.
    - The function rescales both `data` and `data_err` by `10**scale` for display purposes, but all statistics are computed on the scaled data.
    - The normal distribution is refferd to as `N(mu, sigma**2)`.

    Example
    -------
    >>> x = np.random.normal(loc=5, scale=0.2, size=100)
    >>> x_err = np.full_like(x, 0.05)
    >>> hist(x, x_err, scale=-2, label="Length", unit="cm")
    """

    from scipy.stats import norm, shapiro

    data = _np.asarray(data)

    if (not (_np.issubdtype(data.dtype, _np.floating) or _np.issubdtype(data.dtype, _np.integer))) or not _np.all(_np.isreal(data)):
            raise TypeError("'data' must contain only real numbers (int or float).")
    
    if not _np.all(_np.isfinite(data)):
            raise ValueError("'data' contains non-finite values (NaN or inf).")
    
    if not isinstance(scale, (int, float)):
        raise TypeError("'scale' must be a real number (int or float).")
    
    if not isinstance(bins, (int, float, str)):
        raise TypeError("'bins' must be a real number (int or float) or a string ('auto').")
    if isinstance(bins, str) and bins != "auto":
        raise ValueError("'bins' must be 'auto'.")
    
    if not isinstance(label, str):
        raise TypeError("'label' must be a string.")
    if not isinstance(unit, str):
        raise TypeError("'unit' must be a string.")

    data = data / 10**scale

    if data_err is not None:

        if _np.isscalar(data_err):
            if not isinstance(data_err, (int, float)):
                raise TypeError("'data_err' must be a real number (int or float).")
            data_err = _np.repeat(data_err, len(data))
        else:
            if (not (_np.issubdtype(data_err.dtype, _np.floating) or _np.issubdtype(data_err.dtype, _np.integer))) or not _np.all(_np.isreal(data_err)):
                raise TypeError("'data_err' must contain only real numbers (int or float).")
            if not _np.all(_np.isfinite(data)):
                    raise ValueError("'data_err' contains non-finite values (NaN or inf).")

        data_err = data_err / 10**scale
        sigma = _np.sqrt(data.std()**2 + _np.sum(data_err**2)/len(data))

    else:
        sigma = data.std()
    mean = data.mean()

    # Calcola l'esponente di sigma
    exponent = int(_math.floor(_math.log10(abs(sigma))))
    factor = 10**(exponent - 1)
    rounded_sigma = (round(sigma / factor) * factor)

    # Arrotonda la media
    rounded_mean = round(mean, -exponent + 1)

    # Converte in stringa mantenendo zeri finali
    fmt = f".{-exponent + 1}f" if exponent < 1 else "f"

    # ----------------------------

    # Calcola l'esponente di var
    exponent1 = int(_math.floor(_math.log10(abs(sigma**2))))
    factor = 10**(exponent1 - 1)
    rounded_var = (round(sigma**2 / factor) * factor)

    # Arrotonda la media
    rounded_mean1 = round(mean, -exponent1 + 1)

    # Converte in stringa mantenendo zeri finali
    fmt1 = f".{-exponent1 + 1}f" if exponent1 < 1 else "f"

    # ----------------------------

    # Prepara l'unità di misura, se presente
    ux_str = f" [$\\mathrm{{{unit}}}$]" if unit else ""

    label1 = f"$\\mathcal{{N}}({rounded_mean1:.{max(0, -exponent1 + 1)}f}, {rounded_var:.{max(0, -exponent1 + 1)}f})$"
    label2 = label+ux_str

    # histogram of the data
    _, bin_edges, _ = _plt.hist(data,bins=bins,color="blue",edgecolor='blue', histtype = "step", zorder=2, label='Data distribution')
    _plt.ylabel('Counts')

    lnspc = _np.linspace(data.min() - 2 * sigma, data.max() + 2 * sigma, 500) 
   
    bin_widths = _np.diff(bin_edges)
    mean_bin_width = _np.mean(bin_widths)
    f_gauss = data.size * mean_bin_width * norm.pdf(lnspc, mean, sigma)

    _plt.plot(lnspc, f_gauss, linewidth=1, color='r',linestyle='--', label = label1, zorder=1)
    _plt.xlabel(label2)
    _plt.xlim(mean - 3 * sigma, mean + 3 * sigma)
    _plt.legend()

    skewness = _np.sum((data - mean)**3) / (len(data) * sigma**3)
    kurtosis = _np.sum((data - mean)**4) / (len(data) * sigma**4) - 3 

    _, p_value = shapiro(data)

    if 0.10 <= p_value <= 1:
        pval_str = f"{p_value*100:.0f}%"
    elif 0.005 < p_value < 0.10:
        pval_str = f"{p_value * 100:.2f}%"
    elif 0.0005 < p_value <= 0.005:
        pval_str = f"{p_value * 100:.3f}%"
    else:
        pval_str = f"< 0.05%"
    
    if verbose:

        # Prepara l'unità di misura, se presente
        ux_str = f" {unit}" if unit else ""

        # Determina la larghezza massima delle etichette e dei valori
        entries = [
            ("Mean value", f"{rounded_mean:.{max(0, -exponent + 1)}f}{ux_str}"),
            ("Std dev", f"{rounded_sigma:.{max(0, -exponent + 1)}f}{ux_str}"),
            ("Skewness", f"{skewness:.2f}"),
            ("Kurtosis", f"{kurtosis:.2f}"),
            ("p-value", pval_str)
        ]
        label_width = max(len(label) for label, _ in entries) + 2
        value_width = max(len(str(value)) for _, value in entries) + 2

        # Costruisci il separatore dinamico
        title = " Normality Analysis "
        total_width = label_width + value_width + 3
        separator = "=" * total_width
        title_line = title.center(total_width, "*")

        # Crea il blocco di testo
        stamp = f"{separator}\n{title_line}\n{separator}\n"
        for label, value in entries:
            stamp += f"{label.ljust(label_width)}: {value.rjust(value_width)}\n"
        stamp += f"{separator}\n"

        # Aggiungi il risultato finale
        if p_value >= 0.05:
            result = "The data are consistent with a normal distribution."
        else:
            result = "The data deviate significantly from a normal distribution."
        stamp += f"\n{result}\n"

        # Stampa
        print(stamp)

    return mean, sigma, skewness, kurtosis, p_value

def analyze_residuals(*args, **kwargs):
    warn("This function is deprecated and will be removed in a future release. Use labtoolbox.stats.residuals instead.", DeprecationWarning)
    return residuals(args, kwargs)

def residuals(data, expected_data, data_err, scale = 0, unit = "", bins = "auto", confidence = 2, norm = False, verbose = True):
    """
    Analyzes and visualizes the residuals of the quantity of interes, including histogram, Gaussianity test, and autocorrelation test (Durbin-Watson statistic).

    Parameters
    ----------
    data : array-like
        Measured data points.
    expected_data : array-like
        Expected values to compare with `data` (e.g., from a model, theoretical prediction, or fit).
    data_err : array-like
        Uncertainties associated with each data point in `data`.
    scale : int, optional
        Scaling exponent applied to all quantities (e.g. `scale = -2` scales meters to centimeters). Default is `0`.
    unit : str, optional
        Unit of measurement of the data (e.g., `"cm"` or `"s"`). Used for labeling axes. Default is an empty string.
    bins : int or str, optional
        Number of bins or binning strategy passed to `matplotlib.pyplot.hist`. Default is `"auto"`.
    confidence : float, optional
        Confidence factor for visualizing bounds (e.g., `confidence = 2` draws ±2σ bounds). Default is `2`.
    norm : bool, optionale
        If `True`, residuals will be normalized. Default is `False`.
    verbose : bool, optional
        If `True`, prints a formatted table of ... Default is `True`.

    Returns
    -------
    mean : float
        Mean value of the residuals, after applying the specified scale.
    sigma : float
        Estimated standard deviation of the residuals, weighted by `data_err`.
    skewness : float
        Skewness (third standardized moment) of the residual distribution.
    kurtosis : float
        Excess kurtosis (fourth standardized moment minus 3) of the residual distribution.
    p_value : float
        p-value from the Shapiro–Wilk normality test.
    dw : float
        Durbin–Watson statistic for testing autocorrelation in the residuals.

    Notes
    -----
    - The residuals are computed as `resid = data - expected_data`, and scaled by `10**scale`.
    - The standard deviation is computed as`np.sqrt(resid.std()**2 + np.sum(data_err**2)/len(data_err))`.
    - The normal distribution is refferd to as `N(mu, sigma**2)`.
    - The Shapiro–Wilk test is used to test for normality of the residuals:
        - If `p_value >= 0.05`, residuals are considered consistent with a normal distribution.
    - The Durbin–Watson statistic is used to detect first-order autocorrelation:
        - Values ≈ 2 suggest no autocorrelation.
        - Values < 1.5 suggest positive autocorrelation.
        - Values > 2.5 suggest negative autocorrelation.
    """

    from matplotlib.ticker import MaxNLocator
    from scipy.stats import norm, shapiro

    try:
        from statsmodels.stats.stattools import durbin_watson
    except ImportError:
        raise ImportError(
            "The 'statsmodels' package is not installed. "
            "Please install it by running 'pip install statsmodels'."
        )
    
    data = _np.asarray(data)
    expected_data = _np.asarray(expected_data)
    data_err = _np.asarray(data_err)

    if (not (_np.issubdtype(data.dtype, _np.floating) or _np.issubdtype(data.dtype, _np.integer))) or not _np.all(_np.isreal(data)):
            raise TypeError("'data' must contain only real numbers (int or float).")
    
    if not _np.all(_np.isfinite(data)):
            raise ValueError("'data' contains non-finite values (NaN or inf).")
    
    if (not (_np.issubdtype(data_err.dtype, _np.floating) or _np.issubdtype(data_err.dtype, _np.integer))) or not _np.all(_np.isreal(data_err)):
            raise TypeError("'data_err' must contain only real numbers (int or float).")
    
    if not _np.all(_np.isfinite(data_err)):
            raise ValueError("'data_err' contains non-finite values (NaN or inf).")
    
    if (not (_np.issubdtype(expected_data.dtype, _np.floating) or _np.issubdtype(expected_data.dtype, _np.integer))) or not _np.all(_np.isreal(expected_data)):
            raise TypeError("'expected_data' must contain only real numbers (int or float).")
    
    if not _np.all(_np.isfinite(expected_data)):
            raise ValueError("'expected_data' contains non-finite values (NaN or inf).")
    
    if not isinstance(scale, (int, float)):
        raise TypeError("'scale' must be a real number (int or float).")
    
    if not isinstance(confidence, (int, float)):
            raise TypeError("'confidence' must be a real number (int or float).")
    
    if not isinstance(bins, (int, float, str)):
        raise TypeError("'bins' must be a real number (int or float) or a string ('auto').")
    if isinstance(bins, str) and bins != "auto":
        raise ValueError("'bins' must be 'auto'.")
    
    if not isinstance(unit, str):
        raise TypeError("'unit' must be a string.")

    if not (len(data) == len(expected_data) == len(data_err)):
        raise ValueError("'data', 'expected_data' and 'data_err' must have the same length.")

    x_data = _np.linspace(1, len(data), len(data))
    data = data / 10**scale
    expected_data = expected_data / 10**scale
    data_err = data_err / 10**scale

    resid = data - expected_data

    if norm == True:
        resid = resid / data_err
        data_err /= data_err

    mean = resid.mean()

    sigma = _np.sqrt(resid.std()**2 + _np.sum(data_err**2)/len(data_err))

    # Calcola l'esponente di sigma
    exponent = int(_math.floor(_math.log10(abs(sigma))))
    factor = 10**(exponent - 1)
    rounded_sigma = (round(sigma / factor) * factor)

    # Arrotonda la media
    rounded_mean = round(mean, -exponent + 1)

    # Converte in stringa mantenendo zeri finali
    fmt = f".{-exponent + 1}f" if exponent < 1 else "f"

    # ----------------------------

    # Calcola l'esponente di sigma
    exponent1 = int(_math.floor(_math.log10(abs(sigma**2))))
    factor = 10**(exponent1 - 1)
    rounded_var = (round(sigma**2 / factor) * factor)

    # Arrotonda la media
    rounded_mean1 = round(mean, -exponent1 + 1)

    # Converte in stringa mantenendo zeri finali
    fmt1 = f".{-exponent1 + 1}f" if exponent1 < 1 else "f"

    # ----------------------------

    if norm == True:
        bar1 = _np.repeat(1, len(x_data))
        bar2 = resid / data_err
        dash = _np.repeat(confidence, len(x_data))
    else:
        bar1 = data_err
        bar2 = resid
        dash = confidence * data_err

    # The following code is adapted from the VoigtFit library,
    # originally developed by Jens-Kristian Krogager under the MIT License.
    # https://github.com/jkrogager/VoigtFit

    fig = _plt.figure(figsize=(6.4, 4.8))
    gs = fig.add_gridspec(2, hspace=0, height_ratios=[0.1, 0.9])
    axs = gs.subplots()
    # Aggiungi linee di riferimento
    axs[0].axhline(0., ls='--', color='0.7', lw=0.8)
    axs[0].errorbar(x_data, bar2, bar1, ls='', color='gray', lw=1.)
    axs[0].plot(x_data, bar2, color='k', drawstyle='steps-mid', lw=1.15)
    axs[0].plot(x_data, dash, ls='dotted', color='crimson', lw=1.6)
    axs[0].plot(x_data, -dash, ls='dotted', color='crimson', lw=1.6)
    axs[0].set_ylim(-_np.nanmean(3 * dash / 2), _np.nanmean(3 * dash / 2))

    # Configurazioni estetiche per il pannello dei residui
    axs[0].tick_params(labelbottom=False)
    axs[0].set_yticklabels('')
    axs[0].set_xlim(x_data.min(), x_data.max())

    if norm == False:
    # Prepara l'unità di misura, se presente
        uy_str = f" [$\\mathrm{{{unit}}}$]" if unit else ""
        label1 = f"Residuals value{uy_str}"
    else:
        label1 = "Normalized residuals value"

    label = f"$\\mathcal{{N}}({rounded_mean1:.{max(0, -exponent1 + 1)}f}, {rounded_var:.{max(0, -exponent1 + 1)}f})$"
    # label1 = f"$\\text{{Residual}} = y_\\text{{data}} - y_\\text{{expected}}${uy_str}"

    # histogram of the data
    _, bin_edges, _ = axs[1].hist(resid, color="blue",edgecolor='blue', histtype = "step", bins=bins, label ="Residuals distribution", zorder=2)
    axs[1].set_ylabel('Counts')

    lnspc = _np.linspace(resid.min() - 2 * sigma, resid.max() + 2 * sigma, 500) 

    bin_widths = _np.diff(bin_edges)
    mean_bin_width = _np.mean(bin_widths)
    f_gauss = data.size * mean_bin_width * norm.pdf(lnspc, mean, sigma)

    axs[1].plot(lnspc, f_gauss, linewidth=1, color='r',linestyle='--', label = label, zorder=1)
    axs[1].set_xlabel(label1)
    axs[1].yaxis.set_major_locator(MaxNLocator(integer=True))
    axs[1].set_xlim(mean - 3 * sigma, mean + 3 * sigma)

    axs[1].legend()

    skewness = _np.sum((resid - mean)**3) / (len(resid) * sigma**3)
    kurtosis = _np.sum((resid - mean)**4) / (len(resid) * sigma**4) - 3

    _, p_value = shapiro(resid)
    dw = durbin_watson(resid)

    if verbose:

        if 0.10 <= p_value <= 1:
            pval_str = f"{p_value*100:.0f}%"
        elif 0.005 < p_value < 0.10:
            pval_str = f"{p_value * 100:.2f}%"
        elif 0.0005 < p_value <= 0.005:
            pval_str = f"{p_value * 100:.3f}%"
        else:
            pval_str = f"< 0.05%"

        # Prepara l'unità di misura, se presente
        uy_str = f" {unit}" if unit else ""

        # Prepara le voci da stampare
        entries = [
            ("Mean value", f"{rounded_mean:.{max(0, -exponent + 1)}f}{uy_str}"),
            ("Standard deviation", f"{rounded_sigma:.{max(0, -exponent + 1)}f}{uy_str}"),
            ("Skewness", f"{skewness:.2f}"),
            ("Kurtosis", f"{kurtosis:.2f}"),
            ("p-value", pval_str),
            ("Durbin-Watson", f"{dw:.3f}")
        ]

        # Calcola le larghezze dinamiche
        label_width = max(len(label) for label, _ in entries) + 2
        value_width = max(len(str(value)) for _, value in entries) + 2
        total_width = label_width + value_width + 3

        # Costruisci il separatore dinamico e il titolo centrato
        title = " Residuals Analysis "
        separator = "=" * total_width
        title_line = title.center(total_width, "*")

        # Costruisci il blocco di testo
        stamp = f"{separator}\n{title_line}\n{separator}\n"
        for label, value in entries:
            stamp += f"{label.ljust(label_width)}: {value.rjust(value_width)}\n"
        stamp += f"{separator}\n"

        # Aggiungi le interpretazioni finali
        if p_value >= 0.05:
            result_norm = "Residuals are consistent with a normal distribution."
        else:
            result_norm = "Residuals deviate significantly from a normal distribution."

        if dw < 1.5:
            result_dw = "Residuals show evidence of positive autocorrelation."
        elif dw > 2.5:
            result_dw = "Residuals show evidence of negative autocorrelation."
        else:
            result_dw = "Residuals do not show significant autocorrelation."

        # Aggiungi le interpretazioni al testo finale
        stamp += f"\n{result_norm}\n{result_dw}\n"

        # Stampa
        print(stamp)

    return mean, sigma, skewness, kurtosis, p_value, dw

def samples(n, distribution='normal', **params):
    """
    Generates synthetic data from common probability distributions.

    Parameters
    ----------
    n : int
        Number of data points to generate.
    distribution : {'normal', 'uniform', 'exponential', 'poisson', 'binomial', 'gamma', 'beta', 'lognormal', 'weibull', 'chi2', 't'}, optional
        Type of distribution to sample from. Default is 'normal'.
    **params : dict
        Distribution-specific parameters:
        - normal:      mu (mean), sigma (stddev)
        - uniform:     low, high
        - exponential: scale (1/lambda)
        - poisson:     lam (expected rate)
        - binomial:    n (number of trials), p (success probability)
        - gamma:       shape, scale
        - beta:        alpha, beta
        - lognormal:   mean, sigma
        - weibull:     shape
        - chi2:        df (degrees of freedom)
        - t:           df (degrees of freedom)

    Returns
    -------
    data : ndarray
        Array of length `n` with samples drawn from the specified distribution.

    Examples
    --------
    >>> samples(1000, 'normal', mu=0, sigma=1)
    array([...])
    >>> samples(500, 'uniform', low=0, high=10)
    array([...])
    >>> samples(200, 'poisson', lam=3)
    array([...])
    """
    warn("This function is deprecated and will be removed in a future release. Consider using scipy.stats", DeprecationWarning)

    dist = distribution.lower()
    rng = _np.random.default_rng()
    
    if dist == 'normal':
        mu = params.get('mu')
        sigma = params.get('sigma')
        if mu is None or sigma is None:
            raise ValueError("For 'normal' distribution, provide 'mu' and 'sigma'.")
        return rng.normal(loc=mu, scale=sigma, size=n)
    
    elif dist == 'uniform':
        low = params.get('low')
        high = params.get('high')
        if low is None or high is None:
            raise ValueError("For 'uniform' distribution, provide 'low' and 'high'.")
        return rng.uniform(low=low, high=high, size=n)
    
    elif dist == 'exponential':
        scale = params.get('scale')
        if scale is None:
            raise ValueError("For 'exponential' distribution, provide 'scale'.")
        return rng.exponential(scale=scale, size=n)
    
    elif dist == 'poisson':
        lam = params.get('lam')
        if lam is None:
            raise ValueError("For 'poisson' distribution, provide 'lam'.")
        return rng.poisson(lam=lam, size=n)
    
    elif dist == 'binomial':
        trials = params.get('n')
        p = params.get('p')
        if trials is None or p is None:
            raise ValueError("For 'binomial' distribution, provide 'n' (trials) and 'p' (probability).")
        return rng.binomial(n=trials, p=p, size=n)
    
    elif dist == 'gamma':
        shape = params.get('shape')
        scale = params.get('scale', 1)
        if shape is None:
            raise ValueError("For 'gamma' distribution, provide 'shape'.")
        return rng.gamma(shape=shape, scale=scale, size=n)
    
    elif dist == 'beta':
        alpha = params.get('alpha')
        beta = params.get('beta')
        if alpha is None or beta is None:
            raise ValueError("For 'beta' distribution, provide 'alpha' and 'beta'.")
        return rng.beta(alpha=alpha, beta=beta, size=n)
    
    elif dist == 'lognormal':
        mean = params.get('mean')
        sigma = params.get('sigma')
        if mean is None or sigma is None:
            raise ValueError("For 'lognormal' distribution, provide 'mean' and 'sigma'.")
        return rng.lognormal(mean=mean, sigma=sigma, size=n)
    
    elif dist == 'weibull':
        shape = params.get('shape')
        if shape is None:
            raise ValueError("For 'weibull' distribution, provide 'shape'.")
        return rng.weibull(a=shape, size=n)
    
    elif dist == 'chi2':
        df = params.get('df')
        if df is None:
            raise ValueError("For 'chi2' distribution, provide 'df'.")
        return rng.chisquare(df=df, size=n)
    
    elif dist == 't':
        df = params.get('df')
        if df is None:
            raise ValueError("For 't' distribution, provide 'df'.")
        return rng.standard_t(df=df, size=n)
    
    else:
        raise ValueError(f"Unsupported distribution '{distribution}'. "
                         f"Choose from 'normal', 'uniform', 'exponential', 'poisson', 'binomial', "
                         f"'gamma', 'beta', 'lognormal', 'weibull', 'chi2', 't'.")
    
def remove_outliers(data, data_err=None, expected=None, method="zscore", threshold=3.0):
    """
    Removes outliers from a data array according to the specified method.

    Parameters
    ----------
    data : array-like
        Observed data.
    data_err : array-like, optional
        Uncertainties on the data. Necessary if comparing with `'expected'`.
    expected : array-like, optional
        Expected values for the data. If provided, the `'zscore'` method is automatically used.
    method : str, optional
        Method to use (`"zscore"`, `"mad"`, or `"iqr"`). Default: `"zscore"`.
    threshold : float, optional
        Threshold value to identify outliers. Default: `3.0`.

    Returns
    ----------
    data_clean : ndarray
        Data without outliers.
    """
    data = _np.asarray(data)

    # Caso 1: confronto con expected → forza 'zscore'
    if expected is not None:
        if data_err is None:
            raise ValueError("If you provide 'expected', you must also provide 'data_err'.")
        
        expected = _np.asarray(expected)
        data_err = _np.asarray(data_err)

        if len(data) != len(expected) or len(data) != len(data_err):
            raise ValueError("'data', 'expected', and 'data_err' must have the same length.")

        # Metodo unico valido
        z_scores = _np.abs((data - expected) / data_err)
        mask = z_scores < threshold

    else:
        # Caso 2: solo dati osservati → puoi scegliere il metodo
        if method == "zscore":
            mean = _np.mean(data)
            std = _np.std(data)
            z_scores = _np.abs((data - mean) / std)
            mask = z_scores < threshold

        elif method == "mad":
            median = _np.median(data)
            mad = _np.median(_np.abs(data - median))
            modified_z_scores = 0.6745 * (data - median) / mad
            mask = _np.abs(modified_z_scores) < threshold

        elif method == "iqr":
            q1 = _np.percentile(data, 25)
            q3 = _np.percentile(data, 75)
            iqr = q3 - q1
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
            mask = (data >= lower_bound) & (data <= upper_bound)

        else:
            raise ValueError("Unrecognized method. Use 'zscore', 'mad', or 'iqr'.")

    return data[mask]

def posterior(x, y, y_err, f, p0, burn=1000, steps=5000, thin=10, maxfev=5000, verbose = True,
              names=None, prior_bounds=None, plot_dataset = False, plot_density = True, color = 'k', **kwargs):
    """
    Performs a Bayesian parameter estimation using MCMC for a user-defined model function.

    This function fits a given model `f` to the experimental data `(x, y)` with associated uncertainties `y_err` 
    by first performing a frequentist optimization (`curve_fit`) to obtain initial estimates, and then 
    running a Markov Chain Monte Carlo (MCMC) sampling using the `emcee` package to derive the full 
    posterior distribution of the model parameters. It returns the flattened MCMC sample chain representing the posterior samples, 
    the maximum likelihood estimate (MLE) of the parameters, and it visualizes the posterior distribution via a corner plot.

    Parameters
    ----------
    x : array-like,
        Independent variable data points.
    y : array-like,
        Dependent variable data points to be fitted.
    y_err : array-like
        Uncertainties of the dependent data `y`. Used for computing chi-squared
        and log-likelihood in the MCMC sampling. Can be `None`.
    f : callable
        The model function to fit. Must be of the form `f(x, *params)`, where `x` is the independent variable
        and `params` are the free parameters of the model.
    p0 : list or array-like
        Initial guess for the M free parameters of the model. Used both for `curve_fit` and to initialize 
        the MCMC walkers. Can be `None`.
    burn : int, optional
        Number of burn-in steps to discard from the beginning of each walker chain before flattening.
        These initial steps are typically biased by the starting conditions. Default is 1000.
    steps : int, optional
        Total number of MCMC steps for each walker. Default is 5000
    thin : int, optional
        Subsampling factor to reduce autocorrelation between samples. Only every `thin`-th sample is retained. Default is 10.
    maxfev : int, optional
        Maximum number of function evaluations for the `curve_fit` routine. If exceeded, the fit will fail. Default is 5000.
    verbose : bool, optional
        If `True`, prints a formatted table of ... Default is `True`.
    names : list of str, optional
        Parameter names to be used in the `corner` plot and output. If `None`, defaults to ['p0', 'p1', ..., 'pN'].
    prior_bounds : list of tuple, optional
        Prior bounds on the parameters, as a list of `(min, max)` tuples for each parameter. 
        If `None`, assumes uninformative priors that only reject non-positive values.
    plot_dataset : bool, optional
        Draw the individual data points. Default is `False`.
    plot_density : bool, optional
        Draw the density colormap. Default is `False`.
    color : str, optional
        A `matplotlib` style color for all histograms. Default is `k`
    **kwargs
        Any remaining keyword arguments are passed to `corner.corner`.

    Returns
    -------
    mle_params : ndarray, shape (M,)
        The parameter vector corresponding to the sample with the highest log-probability (maximum
        likelihood estimate) in the posterior chain

    flat_samples : ndarray, shape (n_samples, M)
        Flattened MCMC sample chain (after burn-in and thinning). Each row is a sample of the M parameters.
        This chain can be used for uncertainty analysis, plotting posteriors, or further statistical inference.

    Notes
    -----
    - This implementation assumes a **uniform prior** over all parameters, constrained to be strictly positive. 
      Parameters less than or equal to zero are automatically rejected (log-prob = -inf).
    - The `params` object returned is for reference only; it does not contain MCMC results. 
      To use posterior values in further analysis, refer to `flat_samples`.
    - The performance and correctness of the posterior heavily depend on the choice of priors, model structure, 
      and convergence of the sampler. Always check convergence diagnostics in real analyses.

    Example
    --------
    >>> def model(x, a, b):
    ...     return a * x + b
    >>> x = _np.linspace(0, 10, 50)
    >>> y = model(x, 2.5, 1.0) + _np.random.normal(0, 0.5, size=x.size)
    >>> y_err = 0.5 * _np.ones_like(y)
    >>> posterior(x, y, y_err, model, [1, 1])
    """

    missing = []
    try:
        import emcee
    except ImportError:
        missing.append("emcee")

    try:
        import corner
    except ImportError:
        missing.append("corner")
    
    if missing:
        raise ImportError(
            f"The following packages are missing: {', '.join(missing)}. "
            "Please install them using pip."
        )

    if not callable(f):
        raise TypeError("'f' must be a callable function.")
    
    x = _np.asarray(x)
    y = _np.asarray(y)
    y_err = _np.asarray(y_err)

    if not (len(x) == len(y) == len(y_err)):
        raise ValueError("'x', 'y' and 'y_err' must have the same length.")

    if (not (_np.issubdtype(x.dtype, _np.floating) or _np.issubdtype(x.dtype, _np.integer))) or not _np.all(_np.isreal(x)):
            raise TypeError("'x' must contain only real numbers (int or float).")
    if not _np.all(_np.isfinite(x)):
            raise ValueError("'x' contains non-finite values (NaN or inf).")

    if (not (_np.issubdtype(y.dtype, _np.floating) or _np.issubdtype(y.dtype, _np.integer))) or not _np.all(_np.isreal(y)):
            raise TypeError("'y' must contain only real numbers (int or float).")
    if not _np.all(_np.isfinite(y)):
            raise ValueError("'y' contains non-finite values (NaN or inf).")

    if (not (_np.issubdtype(y_err.dtype, _np.floating) or _np.issubdtype(y_err.dtype, _np.integer))) or not _np.all(_np.isreal(y_err)):
            raise TypeError("'y_err' must contain only real numbers (int or float).")
    if not _np.all(_np.isfinite(y_err)):
            raise ValueError("'y_err' contains non-finite values (NaN or inf).")
    
    if not isinstance(burn, (int)):
        raise TypeError("'burn' must be an integer.")
    if burn <= 0:
        raise ValueError("'burn' must be equal or greater than 1.")
    
    if not isinstance(steps, (int)):
        raise TypeError("'steps' must be an integer.")
    if steps <= 0:
        raise ValueError("'steps' must be equal or greater than 1.")
    
    if not isinstance(thin, (int)):
        raise TypeError("'thin' must be an integer.")
    if thin <= 0:
        raise ValueError("'thin' must be equal or greater than 1.")
    
    if not isinstance(maxfev, (int)):
        raise TypeError("'maxfev' must be an integer.")
    if maxfev <= 0:
        raise ValueError("'maxfev' must be equal or greater than 1.")

    p0 = _np.array(p0)

    if (not (_np.issubdtype(p0.dtype, _np.floating) or _np.issubdtype(p0.dtype, _np.integer))) or not _np.all(_np.isreal(p0)):
            raise TypeError("'p0' must contain only real numbers (int or float).")
    if not _np.all(_np.isfinite(p0)):
            raise ValueError("'p0' contains non-finite values (NaN or inf).")

    ndim = len(p0)
    nwalkers = 2 * ndim

    if names is None:
        names = [f"p{i}" for i in range(ndim)]

    if prior_bounds is None:
        # Wide uninformative priors
        prior_bounds = [(-_np.inf, _np.inf)] * ndim

    from scipy.optimize import curve_fit
    # Fit iniziale per inizializzare bene i walkers
    popt, _ = curve_fit(f, x, y, p0=p0, sigma=y_err, absolute_sigma=True, maxfev=maxfev)

    # Log-likelihood
    def log_likelihood(theta):
        model = f(x, *theta)
        return -0.5 * _np.sum(((y - model) / y_err) ** 2)

    # Log-prior uniforme (con bound)
    def log_prior(theta):
        for i, (p, (low, high)) in enumerate(zip(theta, prior_bounds)):
            if not (low < p < high):
                return -_np.inf
        return 0.0

    # Log-posterior
    def log_posterior(theta):
        lp = log_prior(theta)
        if not _np.isfinite(lp):
            return -_np.inf
        return lp + log_likelihood(theta)

    # Inizializzazione walker intorno a popt
    p0_walkers = popt + 1e-4 * _np.random.randn(nwalkers, ndim)

    # Sampling
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior)
    sampler.run_mcmc(p0_walkers, steps, progress=True)

    # Estrai i samples
    flat_samples = sampler.get_chain(discard=burn, thin=thin, flat=True)

    # ho tolto truths = popt

    # Calcola posteriori con arrotondamenti consistenti
    posterior_results = []
    log_prob = sampler.get_log_prob(discard=burn, thin=thin, flat=True)
    mle_index = _np.argmax(log_prob)
    mle_params = flat_samples[mle_index]

    for i in range(len(names)):
        median = _np.median(flat_samples[:, i])
        lower = _np.percentile(flat_samples[:, i], 16)
        upper = _np.percentile(flat_samples[:, i], 84)
        plus = upper - median
        minus = median - lower

        exponent1 = int(_math.floor(_math.log10(abs(plus)))) if plus != 0 else 0
        exponent2 = int(_math.floor(_math.log10(abs(minus)))) if minus != 0 else 0
        common_exponent = min(exponent1, exponent2)
        factor = 10 ** (common_exponent - 1)

        rounded_plus = round(plus / factor) * factor
        rounded_minus = round(minus / factor) * factor
        decimal_places = max(-common_exponent + 1, 0)
        rounded_median = round(median, decimal_places)
        rounded_mle = round(mle_params[i], decimal_places)

        param_name = f"Parameter {i+1}"
        posterior_results.append((param_name, rounded_median, rounded_plus, rounded_minus, rounded_mle, decimal_places))

    if verbose: 
        # Determina le larghezze massime per ciascuna colonna
        col_headers = ["Parameter", "Median", "+1σ", "-1σ", "MLE"]
        columns = [ [r[0] for r in posterior_results],      # Parametri
                    [f"{r[1]:.{r[5]}f}" for r in posterior_results],  # Mediana
                    [f"{r[2]:.{r[5]}f}" for r in posterior_results],  # +1σ
                    [f"{r[3]:.{r[5]}f}" for r in posterior_results],  # -1σ
                    [f"{r[4]:.{r[5]}f}" for r in posterior_results] ] # MLE

        # Calcola le larghezze dinamiche (massimo tra intestazione e valori)
        col_widths = []
        for header, col_data in zip(col_headers, columns):
            max_width = max(len(header), max(len(str(val)) for val in col_data))
            col_widths.append(max_width)

        # Calcola larghezza totale della tabella
        total_width = sum(col_widths) + 3 * len(col_widths) + 1  # +1 per '|', +3 per spazi e separatori

        # Costruisci il titolo centrato
        title = "Summary Report"
        padding = (total_width - len(title)) // 2
        header_line = "=" * total_width
        title_line = " " * padding + title + " " * padding
        if len(title_line) < total_width:
            title_line += " "

        # Stampa la tabella
        print(header_line)
        print(title_line)
        print(header_line)

        # Intestazione
        header = "|"
        for header_text, width in zip(col_headers, col_widths):
            header += f" {header_text.center(width)} |"
        print(header)
        print("-" * total_width)

        # Righe con allineamento centrato
        for row in posterior_results:
            param, median, plus, minus, mle, dec = row
            row_str = "|"
            row_str += f" {param.center(col_widths[0])} |"
            row_str += f" {f'{median:.{dec}f}'.center(col_widths[1])} |"
            row_str += f" {f'{plus:.{dec}f}'.center(col_widths[2])} |"
            row_str += f" {f'{minus:.{dec}f}'.center(col_widths[3])} |"
            row_str += f" {f'{mle:.{dec}f}'.center(col_widths[4])} |"
            print(row_str)

        print(header_line)

    # Plot corner
    # Rimuovi eventuali duplicati nel kwargs
    kwargs.setdefault("plot_density", plot_density)
    kwargs.setdefault("plot_datapoints", plot_dataset)
    # corner.corner(flat_samples, labels=names, no_fill_contours=True, color="royalblue", hist_kwargs={"color": "royalblue", "alpha": 0.8, "linewidth": 1.5}, 
    #               contour_kwargs={"colors": ["darkblue"], "levels": [0.864, 0.675,]}, **kwargs)
    fig = corner.corner(flat_samples, 
                        labels=names, 
                        fill_contours=True,      # riempie le regioni dei contorni con colori sfumati
                        #no_fill_contours=True, 
                        color = color,
                        hist_kwargs={"linewidth": 0.8}, 
                        #contourf_kwargs={"cmap": "viridis", "colors": None},
                        **kwargs)

    # Ora scorro tutti gli assi (subplots) e modifico la linewidth delle linee dei contorni
    # Per ogni asse (subplot) cerco se ci sono contorni 2D (QuadContourSet)
    for ax in fig.axes:
        # ax.collections contiene le collezioni di artisti, i contorni sono lì dentro
        for collection in ax.collections:
            # Modifico la linewidth della collezione
            collection.set_linewidth(0.8)
                # Modifica linewidth per gli istogrammi 1D (linee)
        for line in ax.lines:
            line.set_linewidth(0.8)

    _plt.show()

    return mle_params, flat_samples

def propagate(func, x_val, x_err, params = None, method='Monte_Carlo', MC_sample_size = 10000):
    """
    Propagates uncertainty from i_nput arrays to a generic function using the `uncertainty_class` library.

    Parameters
    ----------
    func : callable
        The base function, in the form `f(x, a)` where:
        - `x` is a vector of variables.
        - `a` is a vector of parameters (optional).
        
    x_val : list of numpy.array
        List containing the i_nput variable arrays `[x1, x2, ..., xn]`. Each entry can be:
        - a scalar,
        - a numpy array of values (same length across all xi).
        
    x_err : list or numpy.array
        List of uncertainties for each variable (scalars or arrays), 
        or a full covariance matrix.
        
    params : list or numpy.array, optional
        List or array of constant parameters `[a1, a2, ..., am]`.
        
    method : str, optional
        The uncertainty propagation method ('Delta' or 'Monte_Carlo').
        
    MC_sample_size : int, optional
        Sample size for the Monte Carlo method.
        
    Returns
    --------
        f_values : numpy.ndarray
            Values of the function calculated at each point `j`.
        f_err : numpy.ndarray
            Propagated uncertainties on the output function for each point `j`.
        confidence_bands : tuple of numpy.ndarray
            Lower and upper confidence bands for each point `j`.
    """

    try:
        import seaborn as sns
    except ImportError:
        raise ImportError(
            "The 'seaborn' package is not installed. "
            "Please install it by running 'pip install seaborn'."
        )

    from ._helper import uncert_prop

    # --- Controllo func ---
    if not callable(func):
        raise TypeError("'func' must be a callable function.")

    # --- Controllo x_val ---
    if not isinstance(x_val, list):
        raise TypeError("'x_val' must be a list of numpy.array or real numbers (int or float).")
    if len(x_val) == 0:
        raise ValueError("'x_val' cannot be empty.")

    array_lengths = []
    for i, xi in enumerate(x_val):
        if _np.isscalar(xi):
            if not isinstance(xi, (int, float)):
                raise TypeError(f"'x_val[{i}]' scalar must be a real number (int or float).")
        elif isinstance(xi, _np.ndarray):
            if not (_np.issubdtype(xi.dtype, _np.floating) or _np.issubdtype(xi.dtype, _np.integer)):
                raise TypeError(f"'x_val[{i}]' must contain real numbers (int or float).")
            if not _np.all(_np.isfinite(xi)):
                raise ValueError(f"'x_val[{i}]' contains non-finite values (NaN or inf)..")
            array_lengths.append(len(xi))
        else:
            raise TypeError(f"'x_val[{i}]' must be a scalar or numpy.ndarray.")

    if array_lengths:
        if len(set(array_lengths)) != 1:
            raise ValueError("All 'x_val[i]' arrays must all have the same length.")

    # --- Controllo x_err ---
    if isinstance(x_err, list):
        if len(x_err) != len(x_val):
            raise ValueError("'x_err' list must have the same length as 'x_val'.")
        for i, err_i in enumerate(x_err):
            if not (_np.isscalar(err_i) or isinstance(err_i, _np.ndarray)):
                raise TypeError(f"'x_err[{i}]' must be a scalar or numpy.ndarray.")
            if _np.isscalar(err_i):
                if not isinstance(err_i, (int, float)):
                    raise TypeError(f"'x_err[{i}]' scalar must be a real number (int or float).")
            if isinstance(err_i, _np.ndarray):
                if not (_np.issubdtype(err_i.dtype, _np.floating) or _np.issubdtype(err_i.dtype, _np.integer)):
                    raise TypeError(f"'x_err[{i}]' must contain real numbers (int or float).")
                if not _np.all(_np.isfinite(err_i)):
                    raise ValueError(f"'x_err[{i}]' contains non-finite values (NaN or inf).")
                if array_lengths and len(err_i) != array_lengths[0]:
                    raise ValueError(f"'x_err[{i}]' must have the same length as 'x_val' arrays.")
    elif isinstance(x_err, _np.ndarray):
        if x_err.ndim != 2 or x_err.shape[0] != x_err.shape[1]:
            raise ValueError("'x_err' numpy.ndarray covariance matrix must be 2D square.")
        expected_size = len(x_val)
        if x_err.shape[0] != expected_size:
            raise ValueError(f"'x_err' covariance matrix must be of size {expected_size}x{expected_size}.")
        if not (_np.issubdtype(x_err.dtype, _np.floating) or _np.issubdtype(x_err.dtype, _np.integer)):
            raise TypeError("'x_err' covariance matrix must contain real numbers (int or float).")
        if not _np.all(_np.isfinite(x_err)):
            raise ValueError("'x_err' covariance matrix contains non-finite values (NaN or inf).")
    else:
        raise TypeError("'x_err' must be a list or numpy.ndarray.")

    # --- Controllo params ---
    if params is not None:
        if not (isinstance(params, list) or isinstance(params, _np.ndarray)):
            raise TypeError("'params' must be a list or numpy.ndarray.")
        else:
            # Se vuoi, puoi aggiungere controlli sul contenuto di params qui
            pass

    # --- Controllo MC_sample_size ---
    if not isinstance(MC_sample_size, int):
        raise TypeError("'MC_sample_size' must be an integer.")
    if MC_sample_size <= 0:
        raise ValueError("'MC_sample_size' must be equal or greater than 1.")

    # Normalizza x_val in lista di array
    x_val = [_np.atleast_1d(xi) for xi in x_val]

    # Determina la lunghezza degli array
    lengths = [len(xi) for xi in x_val]
    unique_lengths = set(lengths)

    if len(unique_lengths) == 1:
        n_points = lengths[0]
    else:
        raise ValueError("All input arrays (or scalars) must have the same length.")
    
    # Inizializza gli array di output
    f_values = _np.zeros(n_points)
    f_err = _np.zeros(n_points)
    confidence_bands_lower = _np.zeros(n_points)
    confidence_bands_upper = _np.zeros(n_points)
    
    # Prepara la funzione wrapper che accetta un vettore di variabili
    def wrapped_func(x_vector):
        if params is not None:
            return func(*x_vector, *params)
        else:
            return func(*x_vector)
    
    # Per ogni punto j, calcola f[j] e la sua incertezza
    for j in range(n_points):
        # Estrai i valori per il punto j
        x_point = _np.array([x[j] for x in x_val])
        
        # Prepara la matrice di covarianza
        if isinstance(x_err, list):
            # Se uncertainties è una lista di incertezze per ogni variabile
            if all(isinstance(u, (int, float)) for u in x_err):
                # Se sono scalari, crea una matrice diagonale
                cov_matrix = _np.diag([u**2 for u in x_err])
            else:
                # Se sono array, prendi il valore per il punto j
                cov_matrix = _np.diag([u[j]**2 for u in x_err])
        else:
            # Assume che uncertainties sia già una matrice di covarianza
            cov_matrix = x_err
            
        # Crea l'oggetto uncert_prop
        uncertainty_propagator = uncert_prop(
            func = wrapped_func,
            x = x_point,
            cov_matrix = cov_matrix,
            method = method,
            MC_sample_size = MC_sample_size
        )
        
        # Calcola il valore della funzione
        f_values[j] = wrapped_func(x_point)
        
        # Calcola l'incertezza propagata
        f_err[j] = uncertainty_propagator.SEM()
        
        # Calcola le bande di confidenza
        lcb, ucb = uncertainty_propagator.confband()
        confidence_bands_lower[j] = lcb
        confidence_bands_upper[j] = ucb
    
    return f_values, f_err, (confidence_bands_lower, confidence_bands_upper)

def bayes_factor(x, y, y_err, f1, p0_1, f2, p0_2, burn=1000, steps=5000, thin=10, maxfev=5000, prior_bounds1=None, prior_bounds2=None, verbose = True):
    """
    Estimate the Bayes factor between two models using the Bayesian Information Criterion (BIC).

    Parameters
    ----------
    x : array-like
        Independent variable values of the dataset.
    y : array-like
        Dependent variable (observed data).
    y_err : array-like
        Uncertainties (standard deviations) on the observed data.
    f1 : callable
        First model function to compare.
    p0_1 : list or array-like
        Initial guess for the parameters of the first model.
    f2 : callable
        Second model function to compare.
    p0_2 : list or array-like
        Initial guess for the parameters of the second model.
    burn : int, optional
        Number of initial MCMC steps to discard (default is 1000).
    steps : int, optional
        Total number of MCMC steps per walker (default is 5000).
    thin : int, optional
        Thinning factor applied when flattening the MCMC chains (default is 10).
    maxfev : int, optional
        Maximum number of function evaluations for the curve fitting (default is 5000).
    prior_bounds1 : list of tuples, optional
        Bounds for the uniform priors of the first model. Each element must be a (min, max) tuple.
        If None, unbounded uniform priors are assumed.
    prior_bounds2 : list of tuples, optional
        Bounds for the uniform priors of the second model.
    verbose : bool, optional
        If `True`, prints a formatted table of ... Default is `True`.

    Returns
    -------
    lnB12 : float
        Estimated natural logarithm of the Bayes factor ln(B12).
        A positive value favors model M1; a negative value favors model M1.
    BIC1 : float
        Bayesian Information Criterion for the first model.
    BIC2 : float
        Bayesian Information Criterion for the second model.

    Notes
    -----
    Interpretation of ln(B12):

        - ln B12 > 5            : Strong evidence for model 1
        - ln B12 ∈ [2.5, 5)     : Moderate evidence for model 1
        - ln B12 ∈ [1, 2.5)     : Weak evidence for model 1
        - ln B12 ∈ [-1, 1)      : Inconclusive
        - ln B12 ∈ [-2.5, -1)   : Weak evidence for model 2
        - ln B12 ∈ [-5, -2.5)   : Moderate evidence for model 2
        - ln B12 < -5           : Strong evidence for model 2

    The approximation assumes that the prior volume is not too informative and that the maximum a posteriori estimate is close to the maximum likelihood.
    """

    from ._helper import format_smart
    from scipy.optimize import curve_fit

    try:
        import emcee
    except ImportError:
        raise ImportError(
            "The 'emcee' package is not installed. "
            "Please install it by running 'pip install emcee'."
        )

    # --- Controllo x ---
    if not hasattr(x, '__iter__'):
        raise TypeError("'x' must be array-like.")
    x_arr = _np.asarray(x)
    if not (_np.issubdtype(x_arr.dtype, _np.floating) or _np.issubdtype(x_arr.dtype, _np.integer)):
        raise TypeError("'x' must contain real numbers (int or float).")
    if not _np.all(_np.isfinite(x_arr)):
        raise ValueError("'x' contains non-finite values (NaN or inf).")

    # --- Controllo y ---
    if not hasattr(y, '__iter__'):
        raise TypeError("'y' must be array-like.")
    y_arr = _np.asarray(y)
    if not (_np.issubdtype(y_arr.dtype, _np.floating) or _np.issubdtype(y_arr.dtype, _np.integer)):
        raise TypeError("'y' must contain only real numbers (int or float).")
    if not _np.all(_np.isfinite(y_arr)):
        raise ValueError("'y' contains non-finite values (NaN or inf).")

    # --- Controllo y_err ---
    if not hasattr(y_err, '__iter__'):
        raise TypeError("'y_err' must be array-like.")
    y_err_arr = _np.asarray(y_err)
    if not (_np.issubdtype(y_err_arr.dtype, _np.floating) or _np.issubdtype(y_err_arr.dtype, _np.integer)):
        raise TypeError("'y_err' must contain only real numbers (int or float).")
    if not _np.all(_np.isfinite(y_err_arr)):
        raise ValueError("'y_err' contains non-finite values (NaN or inf).")

    # --- Controllo lunghezze coerenti ---
    if not (len(x_arr) == len(y_arr) == len(y_err_arr)):
        raise ValueError("'x', 'y' and 'y_err' must have the same length.")

    # --- Controllo f1 ---
    if not callable(f1):
        raise TypeError("'f1' must be callable.")

    # --- Controllo p0_1 ---
    p0_1 = _np.array(p0_1)

    if (not (_np.issubdtype(p0_1.dtype, _np.floating) or _np.issubdtype(p0_1.dtype, _np.integer))) or not _np.all(_np.isreal(p0_1)):
            raise TypeError("'p0_1' must contain only real numbers (int or float).")
    if not _np.all(_np.isfinite(p0_1)):
            raise ValueError("'p0_1' contains non-finite values (NaN or inf).")

    # --- Controllo f2 ---
    if not callable(f2):
        raise TypeError("'f2' must be callable.")

    # --- Controllo p0_2 ---
    p0_2 = _np.array(p0_2)

    if (not (_np.issubdtype(p0_2.dtype, _np.floating) or _np.issubdtype(p0_2.dtype, _np.integer))) or not _np.all(_np.isreal(p0_2)):
            raise TypeError("'p0_2' must contain only real numbers (int or float).")
    if not _np.all(_np.isfinite(p0_2)):
            raise ValueError("'p0_2' contains non-finite values (NaN or inf).")

    # --- Controllo burn ---
    if not (isinstance(burn, int) and burn >= 0):
        raise ValueError("'burn' must be a non-negative integer.")

    # --- Controllo steps ---
    if not (isinstance(steps, int) and steps > 0):
        raise ValueError("'steps' must be a positive integer.")

    # --- Controllo thin ---
    if not (isinstance(thin, int) and thin > 0):
        raise ValueError("'thin' must be a positive integer.")

    # --- Controllo maxfev ---
    if not (isinstance(maxfev, int) and maxfev > 0):
        raise ValueError("'maxfev' must be a positive integer.")

    # --- Controllo prior_bounds1 ---
    if prior_bounds1 is not None:
        if not isinstance(prior_bounds1, list):
            raise TypeError("'prior_bounds1' must be a list of (min, max) tuples or None.")
        for i, bound in enumerate(prior_bounds1):
            if not (isinstance(bound, tuple) and len(bound) == 2):
                raise TypeError(f"'prior_bounds1[{i}]' must be a tuple of length 2.")
            if not all(isinstance(v, (int, float)) for v in bound):
                raise TypeError(f"Elements of 'prior_bounds1[{i}]' must be a real numbers (int or float).")

    # --- Controllo prior_bounds2 ---
    if prior_bounds2 is not None:
        if not isinstance(prior_bounds2, list):
            raise TypeError("'prior_bounds2' must be a list of (min, max) tuples or None.")
        for i, bound in enumerate(prior_bounds2):
            if not (isinstance(bound, tuple) and len(bound) == 2):
                raise TypeError(f"'prior_bounds2[{i}]' must be a tuple of length 2.")
            if not all(isinstance(v, (int, float)) for v in bound):
                raise TypeError(f"Elements of 'prior_bounds2[{i}]' must be a real numbers (int or float).")

    if len(x) <= 10 * len(p0_1) or len(x) <= 10 * len(p0_2):
        warn("The BIC approximation is only valid for sample size much larger than the number of parameters in the model. Results may be inaccurate.", Warning)

    if not (len(x) == len(y) == len(y_err)):
        raise ValueError("'x', 'y' and 'y_err' must have the same length.")

    def fit_model(f, p0, prior_bounds):
        ndim = len(p0)
        nwalkers = 2 * ndim
        popt, _ = curve_fit(f, x, y, p0=p0, sigma=y_err, absolute_sigma=True, maxfev=maxfev)

        def log_likelihood(theta):
            model = f(x, *theta)
            return -0.5 * _np.sum(((y - model) / y_err) ** 2)

        def log_prior(theta):
            if prior_bounds is None:
                return 0.0
            for p, (low, high) in zip(theta, prior_bounds):
                if not (low < p < high):
                    return -_np.inf
            return 0.0

        def log_posterior(theta):
            lp = log_prior(theta)
            if not _np.isfinite(lp):
                return -_np.inf
            return lp + log_likelihood(theta)

        # Sampling
        p0_walkers = popt + 1e-4 * _np.random.randn(nwalkers, ndim)
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior)
        sampler.run_mcmc(p0_walkers, steps, progress=True)

        flat_log_prob = sampler.get_log_prob(discard=burn, thin=thin, flat=True)
        max_log_like = _np.max(flat_log_prob)

        bic = ndim * _np.log(len(x)) - 2 * max_log_like
        return bic, max_log_like

    BIC1, logL1 = fit_model(f1, p0_1, prior_bounds1)
    BIC2, logL2 = fit_model(f2, p0_2, prior_bounds2)

    lnB12 = -0.5 * (BIC1 - BIC2)

    if verbose:

        label_width = 10

        # Definisci le stringhe e il risultato
        if lnB12 >= 5:
            result = "Strong evidence for model 1"
        elif 2.5 <= lnB12 < 5:
            result = "Moderate evidence for model 1"
        elif 1 <= lnB12 < 2.5:
            result = "Weak evidence for model 1"
        elif -1 <= lnB12 < 1:
            result = 'Inconclusive'
        elif -2.5 <= lnB12 < -1:
            result = 'Weak evidence for model 2'
        elif -5 <= lnB12 < -2.5:
            result = 'Moderate evidence for model 2'
        else:
            result = 'Strong evidence for model 2'

        # Prepariamo le intestazioni e i dati
        header = f"{'Model':<{label_width}} | {'BIC':>12} | {'logL':>12}"
        model1_line = f"{'Model 1':<{label_width}} | {format_smart(BIC1, equalsign=False):>12} | {format_smart(logL1, equalsign=False):>12}"
        model2_line = f"{'Model 2':<{label_width}} | {format_smart(BIC2, equalsign=False):>12} | {format_smart(logL2, equalsign=False):>12}"
        divider = "-" * len(header)

        # Stampa con layout elegante e dinamico
        print()
        print("=" * len(header))
        print(header)
        print(divider)
        print(model1_line)
        print(model2_line)
        print(divider)
        print(f"{'Result':<{label_width}} | {'log(BF)':>12} | {format_smart(lnB12, equalsign=False):>12}")
        print("=" * len(header))
        # Evidenzia il risultato con un messaggio elegante
        result_box = f"*** {result} ***"
        # Calcola la lunghezza dinamica per centrare il messaggio
        box_width = max(len(header), len(result_box))
        print(result_box.center(box_width))
        print("=" * box_width)

    return lnB12, BIC1, BIC2

def mean(x, kind='arithmetic'):
    """
    Compute the mean of an i_nput scalar or numpy array, with the specified type of mean.

    Parameters
    ----------
    x : scalar or array-like
        Input data. Can be a scalar or a numpy array. If a scalar is provided, it is returned as the mean.
    
    kind : str or float, optional
        Type of mean to compute. Supported options are:
            - 'max'          : maximum value of x
            - 'min'          : minimum value of x
            - 'arith'        : arithmetic mean (default)
            - 'geom'         : geometric mean
            - 'harmonic'     : harmonic mean
            - 'rms'          : root mean square (quadratic mean)
            - 'cubic'        : cubic mean (third-order)
            - 'agm'          : arithmetic-geometric mean
            - float (number) : generalized mean of order p

    Returns
    -------
    result : float
        The computed mean of x, according to the specified type.

    Examples
    --------
    >>> from labtoolbox.stats import mean
    >>> mean([1, 2, 3, 4], 'arithmetic')
    2.5
    >>> mean([1, 2, 3, 4], 'rms')
    2.7386127875258306
    >>> mean([1, 2, 3, 4], 2)
    2.7386127875258306
    """
    x = _np.asarray(x)
    
    # Handle scalar case
    if _np.isscalar(x):
        if not isinstance(x, (int, float)):
            raise TypeError("'x' must be a real number (int or float).")
        return x
    else:
        if (not (_np.issubdtype(x.dtype, _np.floating) or _np.issubdtype(x.dtype, _np.integer))) or not _np.all(_np.isreal(x)):
            raise TypeError("'x' must contain only real numbers (int or float).")
        if not _np.all(_np.isfinite(x)):
                raise ValueError("'x' contains non-finite values (NaN or inf).")
    
    if kind == 'arithmetic':
        return _np.mean(x)
    elif kind == 'geometric':
        if _np.any(x <= 0):
            raise ValueError("Geometric mean requires all elements to be positive.")
        return _np.exp(_np.mean(_np.log(x)))
    elif kind == 'harmonic':
        if _np.any(x == 0):
            raise ValueError("Harmonic mean requires non-zero elements.")
        return len(x) / _np.sum(1.0 / x)
    elif kind == 'maximum':
        return _np.max(x)
    elif kind == 'minimum':
        return _np.min(x)
    elif kind == 'rms':
        return _np.sqrt(_np.mean(x**2))
    elif kind == 'cubic':
        return _np.cbrt(_np.mean(x**3))
    elif kind == 'agm':
        if _np.any(x <= 0):
            raise ValueError("Arithmetic-Geometric Mean requires all elements to be positive.")
        a = _np.mean(x)
        g = _np.exp(_np.mean(_np.log(x)))
        while not _np.isclose(a, g, rtol=1e-10):
            a_new = 0.5 * (a + g)
            g = _np.sqrt(a * g)
            a = a_new
        return a
    elif isinstance(kind, (float, int)):
        p = kind
        if p == 0:
            if _np.any(x <= 0):
                raise ValueError("Generalized mean of order 0 (geometric) requires positive values.")
            return _np.exp(_np.mean(_np.log(x)))
        return (_np.mean(x**p))**(1/p)
    else:
        raise ValueError(f"Unsupported mean type '{kind}'.")
    
def lin_fit(x, y, y_err, x_err = None, fitmodel = "wls", xlabel="x [ux]", ylabel="y [uy]", xlim = [], ylim = [], showlegend = True, legendloc = None, log = None,
            xscale = 0, yscale = 0, mscale = 0, cscale = 0, m_units = "", c_units = "", confidence = 2, confidencerange = True, residuals=True, norm = True, verbose = False, summary = True):
    """
    Performs a linear fit (Weighted Least Squares or Ordinary Least Squares) and displays experimental data along with the regression line and uncertainty band.

    Parameters
    ----------
        x : array-like
            Values of the independent variable.
        y : array-like
            Values of the dependent variable.
        y_err : array-like
            Uncertainties associated with y values.
        x_err : array-like, optional
            Uncertainties associated with x values.
        fitmodel : str, optional
            Fitting model, either "wls" or "ols". Default is "wls".
        xlabel : str, optional
            Label for the x-axis, including units in square brackets (e.g., "x [m]").
        ylabel : str, optional
            Label for the y-axis, including units in square brackets (e.g., "y [s]").
        showlegend : bool, optional
            If `True`, displays a legend with the values of m and c on the plot. 
        legendloc : str, optional
            Location of the legend on the plot ('upper right', 'lower left', 'upper center', etc.). Default is `None`.
        log : str, optional
            If set to `'x'` or `'y'`, the corresponding axis is plotted on a logarithmic scale; if `'xy'`, both axes. Defaul is `None`.
        xscale : int, optional
            Scaling factor for the x-axis (e.g., `xscale = -2` corresponds to 1e-2, to convert meters to centimeters).
        yscale : int, optional
            Scaling factor for the y-axis.
        xlim : tuple, optional
            Limits for the x-axis, in the form (xmin, xmax). The values should
            already be scaled with respect to `xscale`. If None or an empty tuple,
            the default limits will be automatically determined from the data.
        ylim : tuple, optional
            Limits for the y-axis, in the form (ymin, ymax). The values should
            already be scaled with respect to `yscale`. If None or an empty tuple,
            the default limits will be automatically determined from the data.
        mscale : int, optional
            Scaling factor for the slope `m`.
        cscale : int, optional
            Scaling factor for the intercept `c`.
        m_units : str, optional
            Unit of measurement for `m` (note the consistency with x, y, and scale factors). Default is `""`.
        c_units : str, optional
            Unit of measurement for `c` (note the consistency with x, y, and scale factors). Default is `""`.
        confidence : int, optional
            Residual confidence interval to display, i.e., `[-confidence, +confidence]`.
        confidencerange : bool, optional
            If `True`, shows the 1σ uncertainty band around the fit line.
        residuals : bool, optional
            If `True`, adds an upper panel showing fit residuals.
        norm : bool, optional
            If `True`, residuals in the upper panel will be normalized.
        verbose : bool, optional
            If `True`, prints the output of `wls_fit` (or `ols_fit`) to the screen. Default is `False`.
        summary : bool, optional  
            If `True`, prints a formatted summary report of the fit, including  
            reduced chi-square, p-value, and percentage of residuals within ±2σ.

    Returns
    ----------
        m : float
            Slope of the regression line.
        c : float
            Intercept of the regression line.
        sigma_m : float
            Uncertainty on the slope.
        sigma_c : float
            Uncertainty on the intercept.
        chi2_red : float
            Reduced chi-square value (χ²/dof).
        p_value : float
            Fit p-value (probability that the observed χ² is compatible with the model).

    Notes
    ----------
    - The values of `xscale` and `yscale` affect only the axis scaling in the plot and have no impact on the fitting computation itself. All model parameters are estimated using the original i_nput data as provided.
    - LaTeX formatting is already embedded in the strings used to display the units of `m` and `c`. You do not need to use "$...$".
    - If `c_scale = 0` (recommended when using `c_units`), then `c_units` will represent the suffix corresponding to 10^yscale (+ `y_units`).
    - If `m_scale = 0` (recommended when using `m_units`), then `m_units` will represent the suffix corresponding to 10^(yscale - xscale) [+ `y_units/x_units`].
    """

    from scipy.stats import chi2
    from ._helper import my_cov, my_mean, my_var, my_line, y_estrapolato

    try:
        import statsmodels.api as sm
    except ImportError:
        raise ImportError(
            "The 'statsmodels' package is not installed. "
            "Please install it by running 'pip install statsmodels'."
        )
    
    x = _np.asarray(x)
    y = _np.asarray(y)
    y_err = _np.asarray(y_err)

    if x_err is not None:
        x_err = _np.asarray(x_err)
        if not (len(x) == len(y) == len(y_err) == len(x_err)):
                raise ValueError("'x', 'y', 'x_err' and 'y_err' must have the same length.")
        if (not (_np.issubdtype(x_err.dtype, _np.floating) or _np.issubdtype(x_err.dtype, _np.integer))) or not _np.all(_np.isreal(x_err)):
            raise TypeError("'y_err' must contain only real numbers (int or float).")
        if not _np.all(_np.isfinite(x_err)):
            raise ValueError("'x_err' contains non-finite values (NaN or inf).")
    else:
        if not (len(x) == len(y) == len(y_err)):
                raise ValueError("'x', 'y' and 'y_err' must have the same length.")
    
    if (not (_np.issubdtype(x.dtype, _np.floating) or _np.issubdtype(x.dtype, _np.integer))) or not _np.all(_np.isreal(x)):
        raise TypeError("'x' must contain only real numbers (int or float).")
    
    if (not (_np.issubdtype(y.dtype, _np.floating) or _np.issubdtype(y.dtype, _np.integer))) or not _np.all(_np.isreal(y)):
        raise TypeError("'y' must contain only real numbers (int or float).")
    
    if (not (_np.issubdtype(y_err.dtype, _np.floating) or _np.issubdtype(y_err.dtype, _np.integer))) or not _np.all(_np.isreal(y_err)):
        raise TypeError("'y_err' must contain only real numbers (int or float).")
    
    if not _np.all(_np.isfinite(x)):
            raise ValueError("'x' contains non-finite values (NaN or inf).")
    if not _np.all(_np.isfinite(y)):
            raise ValueError("'y' contains non-finite values (NaN or inf).")
    if not _np.all(_np.isfinite(y_err)):
            raise ValueError("'y_err' contains non-finite values (NaN or inf).")
    
    if not isinstance(xscale, (int, float)):
        raise TypeError("'xscale' must be a real number (int or float).")
    if not isinstance(yscale, (int, float)):
        raise TypeError("'yscale' must be a real number (int or float).")

    if not isinstance(xlim, list):
        raise TypeError("'xlim' must be a list (either empty or containing two real numbers).")
    if len(xlim) != 0:
        if len(xlim) != 2:
            raise TypeError("'xlim' must be empty or a list of exactly two real numbers.")
        if not all(isinstance(u, (int, float)) and _np.isfinite(u) for u in xlim):
            raise TypeError("Both elements in 'xlim' must be finite real numbers (int or float).")
        
    if not isinstance(ylim, list):
        raise TypeError("'ylim' must be a list (either empty or containing two real numbers).")
    if len(ylim) != 0:
        if len(ylim) != 2:
            raise TypeError("'ylim' must be empty or a list of exactly two real numbers.")
        if not all(isinstance(u, (int, float)) and _np.isfinite(u) for u in ylim):
            raise TypeError("Both elements in 'ylim' must be finite real numbers (int or float).")
        
    if not isinstance(xlabel, (str)):
        raise TypeError("'xlabel' must be a string.")
    if not isinstance(ylabel, (str)):
        raise TypeError("'ylabel' must be a string.")

    if log is not None:
        if log not in ("x", "y", "xy"):
            raise ValueError("The value of 'log' must be 'x', 'y' or 'xy'.")

    xscale = 10**xscale
    yscale = 10**yscale
    
    # Aggiunta dell'intercetta (colonna di 1s per il termine costante)
    X = sm.add_constant(x)  # Aggiunge una colonna di 1s per il termine costante

    # Calcolo dei pesi come inverso delle varianze
    weights = 1 / y_err**2

    # Modello di regressione pesata
    if fitmodel == "wls":
        model = sm.WLS(y, X, weights=weights)  # Weighted Least Squares (OLS con pesi)
    elif fitmodel == "ols":
        model = sm.OLS(y, X)
    else:
        raise ValueError('Invalid model. Only "wls" or "ols" allowed.')
    results = model.fit()

    if verbose:
        print(results.summary())
        print("\n")

    # Parametri stimati
    m = float(results.params[1])
    c = float(results.params[0])

    # Errori standard dei parametri stimati
    sigma_m = float(results.bse[1])  # Incertezza sul coefficiente angolare (m)
    sigma_c = float(results.bse[0])  # Incertezza sull'intercetta (c)

    chi2_value = _np.sum(((y - (m * x + c)) / y_err) ** 2)

    # Gradi di libertà (DOF)
    dof = len(x) - 2

    # Chi-quadrato ridotto
    chi2_red = chi2_value / dof

    # p-value
    p_value = chi2.sf(chi2_value, dof)
        
    m2 = my_cov(x, y, weights) / my_var(x, weights)
    var_m2 = 1 / ( my_var(x, weights) * _np.sum(weights) )
        
    c2 = my_mean(y, weights) - my_mean(x, weights) * m
    var_c2 = my_mean(x*x, weights)  / ( my_var(x, weights) * _np.sum(weights) )

    sigma_m2 = var_m2 ** 0.5
    sigma_c2 = var_c2 ** 0.5
        
    cov_mc = - my_mean(x, weights) / ( my_var(x, weights) * _np.sum(weights) )

    # ------------------------ 

    # Applica lo scaling
    mean_scaled = m / (10**mscale)
    sigma_scaled = sigma_m / (10**mscale)

    exponent = int(_math.floor(_math.log10(abs(sigma_scaled))))
    factor = 10**(exponent - 1)
    rounded_sigma = round(sigma_scaled / factor) * factor

    # 2. Arrotonda mean allo stesso ordine di grandezza di sigma
    rounded_mean = round(mean_scaled, -exponent + 1)

    # 3. Converte in stringa mantenendo zeri finali
    fmt = f".{-exponent + 1}f" if exponent < 1 else "f"
    mean_str = f"{rounded_mean:.{max(0, -exponent + 1)}f}"
    sigma_str = f"{rounded_sigma:.{max(0, -exponent + 1)}f}"

    # Costruzione stringa LaTeX ottimizzata
    unit_str = f" \\, \\mathrm{{{m_units}}}" if m_units else ""
    scale_str = f" \\times 10^{{{mscale}}}" if mscale != 0 else ""

    if unit_str or scale_str:
        result1 = f"$m = ({mean_str} \pm {sigma_str}){scale_str}{unit_str}$"
    else:
        result1 = f"$m = {mean_str} \pm {sigma_str}$"
    
    # ------------------------ 

    # Applica lo scaling
    mean_scaled = c / (10**cscale)
    sigma_scaled = sigma_c / (10**cscale)

    exponent = int(_math.floor(_math.log10(abs(sigma_scaled))))
    factor = 10**(exponent - 1)
    rounded_sigma = round(sigma_scaled / factor) * factor

    # 2. Arrotonda mean allo stesso ordine di grandezza di sigma
    rounded_mean = round(mean_scaled, -exponent + 1)

    # 3. Converte in stringa mantenendo zeri finali
    fmt = f".{-exponent + 1}f" if exponent < 1 else "f"
    mean_str = f"{rounded_mean:.{max(0, -exponent + 1)}f}"
    sigma_str = f"{rounded_sigma:.{max(0, -exponent + 1)}f}"

    # Costruzione stringa LaTeX ottimizzata
    unit_str = f" \\, \\mathrm{{{c_units}}}" if c_units else ""
    scale_str = f" \\times 10^{{{cscale}}}" if cscale != 0 else ""

    if unit_str or scale_str:
        result2 = f"$c = ({mean_str} \pm {sigma_str}){scale_str}{unit_str}$"
    else:
        result2 = f"$c = {mean_str} \pm {sigma_str}$"
    
    # ------------------------ 

    # Calcolo dei residui normalizzati
    resid = y - (m * x + c)
    resid_norm = resid / y_err

    k = _np.sum((-1 <= resid_norm) & (resid_norm <= 1))

    n = k / len(resid_norm)

    labels = ["Reduced χ² (χ²/dof)", "p-value", "Residuals within ±2σ"]
    values = []

    # χ²/dof dinamico
    if chi2_red > 100:
        values.append("> 100")
    elif 10 <= chi2_red <= 100:
        values.append(f"{chi2_red:.0f}")
    elif 0.01 < chi2_red < 10:
        values.append(f"{chi2_red:.2f}")
    else:
        values.append("< 0.01")

    if p_value >= 0.10:
        values.append(f"{p_value * 100:.0f}%")
    elif 0.005 < p_value < 0.10:
        values.append(f"{p_value * 100:.2f}%")
    elif 0.0005 < p_value <= 0.005:
        values.append(f"{p_value * 100:.3f}%")
    else:
        values.append("< 0.05%")

    # Residui dinamici
    if n >= 0.10:
        values.append(f"{n*100:.0f}%")
    elif 0.005 < n < 0.10:
        values.append(f"{n*100:.2f}%")
    else:
        values.append("Virtually no one")

    if summary:
        max_label_len = max(len(label) for label in labels)
        max_value_len = max(len(value) for value in values)
        total_width = max_label_len + max_value_len + 5

        # Costruzione righe
        report_lines = []
        title = "Summary Report"
        title_line = title.center(total_width)
        separator = "=" * total_width

        report_lines.append(separator)
        report_lines.append(title_line)
        report_lines.append(separator)

        for label, value in zip(labels, values):
            line = f"{label:<{max_label_len}}  :  {value:>{max_value_len}}"
            report_lines.append(line)

        report_lines.append(separator)

        report = "\n".join(report_lines)
        print(report)
        
    xmin_plot = x.min() - .12 * (x.max() - x.min())
    xmax_plot = x.max() + .12 * (x.max() - x.min())
    x1 = _np.linspace(xmin_plot, xmax_plot, 500)
    y1 = my_line(x1, m, c) / yscale

    y1_plus_1sigma = y1 + y_estrapolato(x1, m2, c2, sigma_m2, sigma_c2, cov_mc)[1] / yscale
    y1_minus_1sigma = y1 - y_estrapolato(x1, m2, c2, sigma_m2, sigma_c2, cov_mc)[1] / yscale

    y = y / yscale
    x = x / xscale
    x1 = x1 / xscale
    y_err = y_err / yscale
    
    if x_err is not None:
        x_err = x_err / xscale

    if showlegend:
        label = (
            "Best fit\n"
            + result1 + "\n"
            + result2
        )
    else :
        label = "Best fit"

    if norm == True:
        bar1 = _np.repeat(1, len(x))
        bar2 = resid_norm
        dash = _np.repeat(confidence, len(x1))
    else :
        bar1 = y_err
        bar2 = resid / yscale
        dash = confidence * y_err

    fig = _plt.figure(figsize=(6.4, 4.8))

    # The following code is adapted from the VoigtFit library,
    # originally developed by Jens-Kristian Krogager under the MIT License.
    # https://github.com/jkrogager/VoigtFit

    # dashed, lw = 1.
    # steps-mid, lw = 1.

    if residuals:
        gs = fig.add_gridspec(2, hspace=0, height_ratios=[0.1, 0.9])
        axs = gs.subplots(sharex=True)
        #axs = gs.subplots()
        # Aggiungi linee di riferimento
        axs[0].axhline(0., ls='--', color='0.7', lw=0.8)
        axs[0].errorbar(x, bar2, bar1, ls='', color='gray', lw=1.15)
        axs[0].plot(x, bar2, color='k', drawstyle='steps-mid', lw=1.15)
        if norm == True:
            axs[0].plot(x1, dash, ls='dotted', color='crimson', lw=1.6)
            axs[0].plot(x1, -dash, ls='dotted', color='crimson', lw=1.6)
        else:
            axs[0].plot(x, dash, ls='dotted', color='crimson', lw=1.6)
            axs[0].plot(x, -dash, ls='dotted', color='crimson', lw=1.6)
        axs[0].set_ylim(-_np.nanmean(3 * dash / 2), _np.nanmean(3 * dash / 2))

        # Configurazioni estetiche per il pannello dei residui
        axs[0].tick_params(labelbottom=False)
        axs[0].set_yticklabels('')
        #axs[0].set_xlim(x.min(), x.max())
    else:
        gs = fig.add_gridspec(2, hspace=0, height_ratios=[0, 1])
        axs = gs.subplots(sharex=True)
        #axs = gs.subplots()
        axs[0].remove()  # Rimuovi axs[0], axs[1] rimane valido

    axs[1].plot(x1, y1, color="blue", ls="-", linewidth=0.8, label = label)

    if confidencerange == True:
        axs[1].fill_between(x1, y1_plus_1sigma, y1_minus_1sigma,  
                            where=(y1_plus_1sigma > y1_minus_1sigma), color='blue', alpha=0.3, edgecolor='none', label="Confidence interval")

    if x_err is None:
        axs[1].errorbar(x, y, yerr=y_err, ls='', marker='.', 
                        color="black", label='Experimental data', capsize=2)       
    else:
        axs[1].errorbar(x, y, yerr=y_err, xerr=x_err, ls='', marker='.', 
                        color="black", label='Experimental data', capsize=2)
    
    axs[1].set_xlabel(xlabel)
    axs[1].set_ylabel(ylabel)

    if xlim and len(xlim) == 2:
        if residuals:
            axs[0].set_xlim(xlim)
        axs[1].set_xlim(xlim)
    else:
        if residuals:
            axs[0].set_xlim(xmin_plot/xscale, xmax_plot/xscale)
        axs[1].set_xlim(xmin_plot/xscale, xmax_plot/xscale)

    if ylim and len(ylim) == 2:
        axs[1].set_ylim(ylim)

    if legendloc == None:
        axs[1].legend()
    else:
        axs[1].legend(loc = legendloc)

    if log == "x":
        axs[1].set_xscale("log")
        if residuals:
            axs[0].set_xscale("log")
    elif log == "y":
        axs[1].set_yscale("log")
    elif log == "xy":
        axs[1].set_xscale("log")
        axs[1].set_yscale("log")
        if residuals:
            axs[0].set_xscale("log")

    return m, c, sigma_m, sigma_c, chi2_red, p_value

def model_fit(x, y, f, x_err = None, y_err = None, p0 = None, xlabel="x [ux]", ylabel="y [uy]", xlim = [], ylim = [], showlegend = True, legendloc = None, 
              bounds = None, confidencerange = True, log=None, maxfev=5000, xscale=0, yscale=0, confidence = 2, residuals=True, norm = True, verbose = True, print_parameters = True):
    """
    General-purpose fit of multi-parameter functions, with an option to display residuals.

    Parameters
    ----------
        x : array-like
            Measured values of the independent variable.
        y : array-like
            Measured values of the dependent variable.
        f : function
            Function of one variable (first argument of `f`) with `N` free parameters.
        x_err : array-like, optional
            Uncertainties associated with the independent variable. Default is `None`.
        y_err : array-like, optional
            Uncertainties associated with the dependent variable. Default is `None`.
        p0 : list, optional
            Initial guess for the model parameters, in the form `[a, ..., z]`. Default is `None`.
        xlabel : str, optional
            Label (and units) for the independent variable.
        ylabel : str, optional
            Label (and units) for the dependent variable.
        xlim : tuple, optional
            Limits for the x-axis, in the form (xmin, xmax). The values should
            already be scaled with respect to `xscale`. If None or an empty tuple,
            the default limits will be automatically determined from the data.
        ylim : tuple, optional
            Limits for the y-axis, in the form (ymin, ymax). The values should
            already be scaled with respect to `yscale`. If None or an empty tuple,
            the default limits will be automatically determined from the data.
        showlegend : bool, optional
            If `True`, displays a legend with the reduced chi-square and p-value in the plot.
        legendloc : str, optional
            Location of the legend in the plot ('upper right', 'lower left', 'upper center', etc.). Default is `None`.
        bounds : 2-tuple of array-like, optional
            Tuple `([lower_bounds], [upper_bounds])` specifying bounds for the parameters. Default is `None`.
        confidencerange : bool, optional
            If `True`, displays the 1σ uncertainty band around the best-fit curve.
        log : str, optional
            If set to `'x'` or `'y'`, the corresponding axis is plotted on a logarithmic scale; if `'xy'`, both axes. Defaul is `None`.
        maxfev : int, optional
            Maximum number of iterations allowed by `curve_fit`.
        xscale : int, optional
            Scaling factor for the x-axis (e.g., `xscale = -2` corresponds to 1e-2, to convert meters to centimeters).
        yscale : int, optional
            Scaling factor for the y-axis.
        confidence : int, optional
            Residual confidence interval to display, i.e., `[-confidence, +confidence]`.
        residuals : bool, optional
            If `True`, adds an upper panel showing fit residuals.
        norm : bool, optional
            If `True`, residuals in the upper panel will be normalized.
        verbose : bool, optional  
            If `True`, prints a formatted summary report of the fit, including  
            reduced chi-square, p-value, and percentage of residuals within ±2σ.
        print_parameters : bool, optional  
            If `True`, prints the best-fit parameters along with their standard uncertainties.


    Returns
    ----------
        popt : array-like
            Array of optimal parameters estimated from the fit.
        perr : array-like
            Uncertainties on the optimal parameters. Only if `y_err` is provided.
        chi2_red : float
            Reduced chi-square value (χ²/dof).
        p_value : float
            Fit p-value (probability that the observed χ² is compatible with the model).

    Notes
    ----------
    The values of `xscale` and `yscale` affect only the axis scaling in the plot and have no impact on the fitting computation itself. 
    All model parameters are estimated using the original i_nput data as provided.
    """

    from scipy.optimize import curve_fit

    x = _np.asarray(x)
    y = _np.asarray(y)

    if y_err is not None:
        y_err = _np.asarray(y_err)
        if (not (_np.issubdtype(y_err.dtype, _np.floating) or _np.issubdtype(y_err.dtype, _np.integer))) or not _np.all(_np.isreal(y_err)):
            raise TypeError("'y_err' must contain only real numbers (int or float).")
        if not _np.all(_np.isfinite(y_err)):
            raise ValueError("'y_err' contains non-finite values (NaN or inf).")
    if x_err is not None:
        x_err = _np.asarray(x_err)
        if (not (_np.issubdtype(x_err.dtype, _np.floating) or _np.issubdtype(x_err.dtype, _np.integer))) or not _np.all(_np.isreal(x_err)):
            raise TypeError("'x_err' must contain only real numbers (int or float).")
        if not _np.all(_np.isfinite(x_err)):
            raise ValueError("'x_err' contains non-finite values (NaN or inf).")
        
    if x_err is not None and y_err is not None:
        if not (len(x) == len(y) == len(y_err) == len(x_err)):
                raise ValueError("'x', 'y', 'x_err' and 'y_err' must have the same length.")
    elif x_err is not None and y_err is None:
        if not (len(x) == len(y) == len(x_err)):
                raise ValueError("'x', 'y' and 'x_err' must have the same length.")
    elif x_err is None and y_err is not None:
        if not (len(x) == len(y) == len(y_err)):
                raise ValueError("'x', 'y' and 'y_err' must have the same length.")
    else:
        if not (len(x) == len(y)):
                raise ValueError("'x' and 'y' must have the same length.")

    if (not (_np.issubdtype(x.dtype, _np.floating) or _np.issubdtype(x.dtype, _np.integer))) or not _np.all(_np.isreal(x)):
        raise TypeError("'x' must contain only real numbers (int or float).")
    
    if (not (_np.issubdtype(y.dtype, _np.floating) or _np.issubdtype(y.dtype, _np.integer))) or not _np.all(_np.isreal(y)):
        raise TypeError("'y' must contain only real numbers (int or float).")
    
    if not _np.all(_np.isfinite(x)):
            raise ValueError("'x' contains non-finite values (NaN or inf).")
    if not _np.all(_np.isfinite(y)):
            raise ValueError("'y' contains non-finite values (NaN or inf).")
    
    if not isinstance(xscale, (int, float)):
        raise TypeError("'xscale' must be a real number (int or float).")
    if not isinstance(yscale, (int, float)):
        raise TypeError("'yscale' must be a real number (int or float).")

    if not isinstance(xlim, list):
        raise TypeError("'xlim' must be a list (either empty or containing two real numbers).")
    if len(xlim) != 0:
        if len(xlim) != 2:
            raise TypeError("'xlim' must be empty or a list of exactly two real numbers.")
        if not all(isinstance(u, (int, float)) and _np.isfinite(u) for u in xlim):
            raise TypeError("Both elements in 'xlim' must be finite real numbers (int or float).")
        
    if not isinstance(ylim, list):
        raise TypeError("'ylim' must be a list (either empty or containing two real numbers).")
    if len(ylim) != 0:
        if len(ylim) != 2:
            raise TypeError("'ylim' must be empty or a list of exactly two real numbers.")
        if not all(isinstance(u, (int, float)) and _np.isfinite(u) for u in ylim):
            raise TypeError("Both elements in 'ylim' must be finite real numbers (int or float).")
        
    if not isinstance(xlabel, (str)):
        raise TypeError("'xlabel' must be a string.")
    if not isinstance(ylabel, (str)):
        raise TypeError("'ylabel' must be a string.")
    
    p0 = _np.array(p0)

    if (not (_np.issubdtype(p0.dtype, _np.floating) or _np.issubdtype(p0.dtype, _np.integer))) or not _np.all(_np.isreal(p0)):
            raise TypeError("'p0' must contain only real numbers (int or float).")
    if not _np.all(_np.isfinite(p0)):
            raise ValueError("'p0' contains non-finite values (NaN or inf).")

    if log is not None:
        if log not in ("x", "y", "xy"):
            raise ValueError("The value of 'log' must be 'x', 'y' or 'xy'.")

    xscale = 10**xscale
    yscale = 10**yscale

    if bounds is None and p0 is not None:
        bounds = (_np.repeat(-_np.inf, len(p0)), _np.repeat(_np.inf, len(p0)))
    elif bounds is None and p0 is None:
        bounds = (-_np.inf, _np.inf)

    popt, pcov = curve_fit(
        f,
        x,
        y,
        p0=p0,
        sigma=y_err,
        absolute_sigma=True,
        maxfev=maxfev,
        bounds=bounds
    )

    perr = _np.sqrt(_np.diag(pcov))

    # Calcolo del chi-quadrato
    y_fit = f(x, *popt)

    if print_parameters:
        if y_err is not None:
            # Stampa dei parametri con incertezze
            for i in range(len(popt)):

                # Calcola l'esponente di sigma
                exponent = int(_math.floor(_math.log10(abs(perr[i]))))
                factor = 10**(exponent - 1)
                rounded_sigma = (round(perr[i] / factor) * factor)

                # Arrotonda la media
                rounded_mean = round(popt[i], -exponent + 1) 

                # Converte in stringa mantenendo zeri finali
                fmt = f".{-exponent + 1}f" if exponent < 1 else "f"

                if popt[i] != 0:
                    nu = perr[i] / popt[i]
                    print(
                        f"Parameter {i + 1} = ({rounded_mean:.{max(0, -exponent + 1)}f} ± {rounded_sigma:.{max(0, -exponent + 1)}f}) [{_np.abs(nu) * 100:.2f}%]"
                    )
                else:
                    print(f"Parameter {i + 1} = ({rounded_mean:.{max(0, -exponent + 1)}f} ± {rounded_sigma:.{max(0, -exponent + 1)}f})")
        else:
            print(f"Parameter {i + 1} = {popt[i]:.2g}")

        print()

    # k = _np.sum((-1 <= resid_norm) & (resid_norm <= 1))

    if y_err is not None:

        from scipy.stats import chi2

        resid = y - y_fit
        resid_norm = resid / y_err

        chi2_value = _np.sum((resid_norm) ** 2)

        dof = len(x) - len(popt)

        chi2_red = chi2_value / dof

        p_value = chi2.sf(chi2_value, dof)
    
        n = _np.sum((-1 <= resid_norm) & (resid_norm <= 1)) / len(resid_norm)

        labels = ["Reduced χ² (χ²/dof)", "p-value", "Residuals within ±2σ"]
        values = []

        if chi2_red > 100:
            values.append("> 100")
            _chi2_str = f"> 100"
        elif 10 <= chi2_red <= 100:
            values.append(f"{chi2_red:.0f}")
            _chi2_str = f"= {chi2_red:.0f}"
        elif 0.01 < chi2_red < 10:
            values.append(f"{chi2_red:.2f}")
            _chi2_str = f"= {chi2_red:.2f}"
        else:
            values.append("< 0.01")
            _chi2_str = f"< 0.01"

        if p_value >= 0.10:
            values.append(f"{p_value * 100:.0f}%")
            pval_str = f"p-value = {p_value*100:.0f}%"
        elif 0.005 < p_value < 0.10:
            values.append(f"{p_value * 100:.2f}%")
            pval_str = f"p-value = {p_value * 100:.2f}%"
        elif 0.0005 < p_value <= 0.005:
            values.append(f"{p_value * 100:.3f}%")
            pval_str = f"p-value = {p_value * 100:.3f}%"
        else:
            values.append("< 0.05%")
            pval_str = f"p-value < 0.05%"

        if n >= 0.10:
            values.append(f"{n*100:.0f}%")
        elif 0.005 < n < 0.10:
            values.append(f"{n*100:.2f}%")
        else:
            values.append("Virtually no one")

        if verbose:
            max_label_len = max(len(label) for label in labels)
            max_value_len = max(len(value) for value in values)
            total_width = max_label_len + max_value_len + 5 

            # Costruzione righe
            report_lines = []
            title = "Summary Report"
            title_line = title.center(total_width)
            separator = "=" * total_width

            report_lines.append(separator)
            report_lines.append(title_line)
            report_lines.append(separator)

            for label, value in zip(labels, values):
                line = f"{label:<{max_label_len}}  :  {value:>{max_value_len}}"
                report_lines.append(line)

            report_lines.append(separator)

            # Unione e stampa
            report = "\n".join(report_lines)
            print(report)
             
    xmin_plot = x.min() - .12 * (x.max() - x.min())
    xmax_plot = x.max() + .12 * (x.max() - x.min())

    x1 = _np.linspace(xmin_plot, xmax_plot, 500)
    y_fit_cont = f(x1, *popt)

    if y_err is not None:

        if norm is True:
            bar1 = _np.repeat(1, len(x))
            bar2 = resid_norm
            dash = _np.repeat(confidence, len(x1))
        else :
            bar1 = y_err
            bar2 = resid / yscale
            dash = confidence * y_err
        # Ripeti ciascun parametro per len(x1) volte
        parametri_ripetuti = [_np.repeat(p, len(x1)) for p in popt]
        errori_ripetuti = [_np.repeat(e, len(x1)) for e in perr]

        # Costruisci lista dei valori e delle incertezze
        lista = [x1] + parametri_ripetuti
        lista_err = [_np.repeat(0, len(x1))] + errori_ripetuti

        from .stats import propagate as _propagate
        # Ora puoi usarli nella propagazione
        _, _ , confid = _propagate(f, lista, lista_err)

        y1_plus_1sigma = confid[1] / yscale
        y1_minus_1sigma = confid[0] / yscale

        y_err = y_err / yscale

    x1 = x1 / xscale
    xmax_plot = xmax_plot / xscale
    xmin_plot = xmin_plot / xscale
    x = x / xscale
    y = y / yscale
    y_fit_cont = y_fit_cont / yscale
    y_fit = y_fit / yscale

    if x_err is not None:
        x_err = x_err / xscale

    fig = _plt.figure(figsize=(6.4, 4.8))

    # The following code is adapted from the VoigtFit library,
    # originally developed by Jens-Kristian Krogager under the MIT License.
    # https://github.com/jkrogager/VoigtFit

    if residuals is True and y_err is not None:
        gs = fig.add_gridspec(2, hspace=0, height_ratios=[0.1, 0.9])
        axs = gs.subplots(sharex=True)
        axs[0].axhline(0., ls='--', color='0.7', lw=0.8)
        axs[0].errorbar(x, bar2, bar1, ls='', color='gray', lw=1.15)
        axs[0].plot(x, bar2, color='k', drawstyle='steps-mid', lw=1.15)
        if norm == True:
            axs[0].plot(x1, dash, ls='dotted', color='crimson', lw=1.6)
            axs[0].plot(x1, -dash, ls='dotted', color='crimson', lw=1.6)
        else:
            axs[0].plot(x, dash, ls='dotted', color='crimson', lw=1.6)
            axs[0].plot(x, -dash, ls='dotted', color='crimson', lw=1.6)
        axs[0].set_ylim(-_np.nanmean(3 * dash / 2), _np.nanmean(3 * dash / 2))

        axs[0].tick_params(labelbottom=False)
        axs[0].set_yticklabels('')
    else: 
        gs = fig.add_gridspec(2, hspace=0, height_ratios=[0, 1])
        axs = gs.subplots(sharex=True)
        axs[0].remove()

    if showlegend and y_err is not None:
        label = f"Best fit\n$\\chi^2/\\text{{dof}}{_chi2_str}$\n{pval_str}"
    else :
        label = "Best fit"

    axs[1].plot(x1, y_fit_cont, color="blue", ls="-", linewidth=0.8, label = label)

    if confidencerange is True and y_err is not None:
        axs[1].fill_between(x1, y1_plus_1sigma, y1_minus_1sigma,  
                            where=(y1_plus_1sigma > y1_minus_1sigma), color='blue', alpha=0.3, edgecolor='none', label="Confidence interval")

    if y_err is not None:
        if x_err is None:
            axs[1].errorbar(x, y, yerr=y_err, ls='', marker='.', 
                            color="black", label='Experimental data', capsize=2)       
        else:
            axs[1].errorbar(x, y, yerr=y_err, xerr=x_err, ls='', marker='.', 
                            color="black", label='Experimental data', capsize=2)
    else:
        axs[1].plot(x, y, ls='', marker='.', 
                    color="black", label='Experimental data', capsize=2)
    
    axs[1].set_xlabel(xlabel)
    axs[1].set_ylabel(ylabel)

    if xlim and len(xlim) == 2:
        if residuals:
            axs[0].set_xlim(xlim)
        axs[1].set_xlim(xlim)
    else:
        if residuals:
            axs[0].set_xlim(xmin_plot, xmax_plot)
        axs[1].set_xlim(xmin_plot, xmax_plot)

    if ylim and len(ylim) == 2:
        axs[1].set_ylim(ylim)

    if legendloc is None:
        axs[1].legend()
    else:
        axs[1].legend(loc = legendloc)
    
    if log == "x":
        axs[1].set_xscale("log")
        if residuals:
            axs[0].set_xscale("log")
    elif log == "y":
        axs[1].set_yscale("log")
    elif log == "xy":
        axs[1].set_xscale("log")
        axs[1].set_yscale("log")
        if residuals:
            axs[0].set_xscale("log")

    if y_err is not None:
        return popt, perr, chi2_red, p_value
    else:
        return popt, chi2_red, p_value