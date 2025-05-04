from LabToolbox import np, plt, math, spstats, curve_fit
from matplotlib.ticker import MaxNLocator
from statsmodels.stats.stattools import durbin_watson
import emcee
import corner
from lmfit import Model, Parameters

def histogram(data, sigma_data, scale = 0, label = "", unit = ""):
    """
    Plot the histogram of a dataset and assess its Gaussianity using statistical indicators and a Shapiro-Wilk test.

    This function visualizes the empirical distribution of a dataset `data`, optionally accounting for individual measurement
    uncertainties `sigma_data`. It overlays a Gaussian curve parameterized by the estimated mean and standard deviation, and 
    performs a normality test, reporting skewness, kurtosis, and the p-value from the Shapiro-Wilk test.

    Parameters
    ----------
    data : array-like
        Numerical data representing the variable of interest.
    sigma_data : array-like or None
        Array of uncertainties associated with each element of `data`. If `None`, uncertainties are not included in the 
        computation of the effective standard deviation.
    scale : int, optional
        Scaling exponent for `data` and `sigma_data` (default is `0`). For example, `scale = -2` rescales the inputs by 1e2.
    label : str, optional
        Label for the x-axis, typically the name of the variable.
    unit : str, optional
        Unit of measurement for the x-axis variable (e.g., "cm"). If provided, it will be displayed in the axis label and summary output.

    Returns
    -------
    mean : float
        Arithmetic mean of the scaled data.
    sigma : float
        Effective standard deviation of the distribution, accounting for both the empirical spread and uncertainties (if provided).
    skewness : float
        Skewness of the distribution.
    kurtosis : float
        Kurtosis (excess) of the distribution.
    p_value : float
        p-value from the Shapiro-Wilk test for normality.

    Notes
    -----
    - The effective standard deviation is computed as `np.sqrt(data.std()**2 + np.sum(sigma_data**2)/len(sigma_data))` if `sigma_data` is provided.
    - The function rescales both `data` and `sigma_data` by `10**scale` for display purposes, but all statistics are computed on the scaled data.
    - The histogram is plotted using adaptive binning (`bins="auto"`) and normalized using the mean bin width to allow direct comparison with the overlaid Gaussian PDF.
    - The normal distribution is refferd to as `N(mu, sigma**2)`.
    - The Shapiro-Wilk test is used to assess normality.
    - The printed output includes the mean, standard deviation, skewness, kurtosis, and a qualitative assessment of Gaussianity.

    Example
    -------
    >>> x = np.random.normal(loc=5, scale=0.2, size=100)
    >>> sigmax = np.full_like(x, 0.05)
    >>> histogram(x, sigmax, scale=-2, label="Length", unit="cm")
    """

    data = data / 10**scale

    if sigma_data is not None:
        sigma_data = sigma_data / 10**scale
        sigma = np.sqrt(data.std()**2 + np.sum(sigma_data**2)/len(data))
    else:
        sigma = data.std()
    mean = data.mean()

    # Calcola l'esponente di sigma
    exponent = int(math.floor(math.log10(abs(sigma))))
    factor = 10**(exponent - 1)
    rounded_sigma = (round(sigma / factor) * factor)

    # Arrotonda la media
    rounded_mean = round(mean, -exponent + 1)

    # Converte in stringa mantenendo zeri finali
    fmt = f".{-exponent + 1}f" if exponent < 1 else "f"

    # ----------------------------

    # Calcola l'esponente di sigma
    exponent1 = int(math.floor(math.log10(abs(sigma**2))))
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
    _, bin_edges, _ = plt.hist(data,bins="auto",color="blue",edgecolor='blue', histtype = "step", zorder=2, label='Data distribution')
    plt.ylabel('Counts')

    lnspc = np.linspace(data.min() - 2 * sigma, data.max() + 2 * sigma, 500) 
   
    bin_widths = np.diff(bin_edges)
    mean_bin_width = np.mean(bin_widths)
    f_gauss = data.size * mean_bin_width * spstats.norm.pdf(lnspc, mean, sigma)

    plt.plot(lnspc, f_gauss, linewidth=1, color='r',linestyle='--', label = label1, zorder=1)
    plt.xlabel(label2)
    plt.xlim(mean - 3 * sigma, mean + 3 * sigma)
    plt.legend()

    skewness = np.sum((data - mean)**3) / (len(data) * sigma**3)
    kurtosis = np.sum((data - mean)**4) / (len(data) * sigma**4) - 3 

    _, p_value = spstats.shapiro(data)

    if p_value >= 0.10:
        pval_str = f"{p_value*100:.0f}%"
    elif 0.005 < p_value < 0.10:
        pval_str = f"{p_value * 100:.2f}%"
    elif 0.0005 < p_value <= 0.005:
        pval_str = f"{p_value * 1000:.2f}‰"
    elif 1e-6 < p_value <= 0.0005:
        pval_str = f"{p_value:.2e}"
    else:
        pval_str = f"< 1e-6"

    # Prepara l'unità di misura, se presente
    ux_str = f" {unit}" if unit else ""

    # Crea la stringa stampabile
    stamp = (
        f"Mean value: {rounded_mean:.{max(0, -exponent + 1)}f}{ux_str}\n"
        f"Standard deviation: {rounded_sigma:.{max(0, -exponent + 1)}f}{ux_str}\n"
        f"Skewness: {skewness:.2f}\n"
        f"Kurtosis: {kurtosis:.2f}\n"
        f"p-value: {pval_str}\n"
    )
    print(stamp)

    if p_value >= 0.05:
        print("The data are consistent with a normal distribution.")
    else:
        print("The data deviate significantly from a normal distribution.")

    return mean, sigma, skewness, kurtosis, p_value

def analyze_residuals(data, expected_data, sigma_data, scale = 0, unit = "", bin = None, confidence = 2, norm = False):
    """
    Analyze and visualize the residuals of a fit, including histogram, Gaussianity test, and autocorrelation test (Durbin-Watson statistic).

    Parameters
    ----------
    data : array-like
        Measured data points.
    expected_data : array-like
        Expected values to compare with `data` (e.g., from a model, theoretical prediction, or fit).
    sigma_data : array-like
        Uncertainties (standard deviations) associated with each data point in `data`.
    scale : int, optional
        Scaling exponent applied to all quantities (e.g. `scale = -2` scales meters to centimeters). Default is `0`.
    unit : str, optional
        Unit of measurement of the data (e.g., `"cm"` or `"s"`). Used for labeling axes. Default is an empty string.
    bin : int or str, optional
        Number of bins or binning strategy passed to `matplotlib.pyplot.hist`. If `None`, uses `"auto"`.
    confidence : float, optional
        Confidence factor for visualizing bounds (e.g., `confidence = 2` draws ±2σ bounds). Default is `2`.
    norm : bool, optionale
        If `True`, residuals in the upper panel will be normalized. Default is `False`.

    Returns
    -------
    mean : float
        Mean value of the residuals, after applying the specified scale.
    sigma : float
        Estimated standard deviation of the residuals, weighted by `sigma_data`.
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
    - The standard deviation is computed as `np.sqrt(resid.std()**2 + np.sum(sigma_data**2)/len(sigma_data))`.
    - A histogram of residuals is plotted, along with a best-fit Gaussian PDF scaled to match the histogram area.
    - The normal distribution is refferd to as `N(mu, sigma**2)`.
    - The Shapiro–Wilk test is used to test for normality of the residuals:
        - If `p_value >= 0.05`, residuals are considered consistent with a normal distribution.
    - The Durbin–Watson statistic is used to detect first-order autocorrelation:
        - Values ≈ 2 suggest no autocorrelation.
        - Values < 1.5 suggest positive autocorrelation.
        - Values > 2.5 suggest negative autocorrelation.
    - The `confidence` parameter sets the width of the dashed bounds plotted above and below the residuals.
    """

    x_data = np.linspace(1, len(data), len(data))
    data = data / 10**scale
    expected_data = expected_data / 10**scale
    sigma_data = sigma_data / 10**scale

    resid = data - expected_data

    mean = resid.mean()

    sigma = np.sqrt(resid.std()**2 + np.sum(sigma_data**2)/len(sigma_data))

    # Calcola l'esponente di sigma
    exponent = int(math.floor(math.log10(abs(sigma))))
    factor = 10**(exponent - 1)
    rounded_sigma = (round(sigma / factor) * factor)

    # Arrotonda la media
    rounded_mean = round(mean, -exponent + 1)

    # Converte in stringa mantenendo zeri finali
    fmt = f".{-exponent + 1}f" if exponent < 1 else "f"

    # ----------------------------

    # Calcola l'esponente di sigma
    exponent1 = int(math.floor(math.log10(abs(sigma**2))))
    factor = 10**(exponent1 - 1)
    rounded_var = (round(sigma**2 / factor) * factor)

    # Arrotonda la media
    rounded_mean1 = round(mean, -exponent1 + 1)

    # Converte in stringa mantenendo zeri finali
    fmt1 = f".{-exponent1 + 1}f" if exponent1 < 1 else "f"

    # ----------------------------

    if norm == True:
        bar1 = np.repeat(1, len(x_data))
        bar2 = resid / sigma_data
        dash = np.repeat(confidence, len(x_data))
    else :
        bar1 = sigma_data
        bar2 = resid
        dash = confidence * sigma_data

    # The following code (lines 239-247) is adapted from the VoigtFit library,
    # originally developed by Jens-Kristian Krogager under the MIT License.
    # https://github.com/jkrogager/VoigtFit

    fig = plt.figure(figsize=(6.4, 4.8))
    gs = fig.add_gridspec(2, hspace=0, height_ratios=[0.1, 0.9])
    axs = gs.subplots()
    # Aggiungi linee di riferimento
    axs[0].axhline(0., ls='--', color='0.7', lw=0.8)
    axs[0].errorbar(x_data, bar2, bar1, ls='', color='gray', lw=1.)
    axs[0].plot(x_data, bar2, color='k', drawstyle='steps-mid', lw=1.)
    axs[0].plot(x_data, dash, ls='dashed', color='crimson', lw=1.)
    axs[0].plot(x_data, -dash, ls='dashed', color='crimson', lw=1.)
    axs[0].set_ylim(-np.nanmean(3 * dash / 2), np.nanmean(3 * dash / 2))

    # Configurazioni estetiche per il pannello dei residui
    axs[0].tick_params(labelbottom=False)
    axs[0].set_yticklabels('')
    axs[0].set_xlim(x_data.min(), x_data.max())

    # Prepara l'unità di misura, se presente
    uy_str = f" [$\\mathrm{{{unit}}}$]" if unit else ""

    label = f"$\\mathcal{{N}}({rounded_mean1:.{max(0, -exponent1 + 1)}f}, {rounded_var:.{max(0, -exponent1 + 1)}f})$"
    # label1 = f"$\\text{{Residual}} = y_\\text{{data}} - y_\\text{{expected}}${uy_str}"
    label1 = f"Residual values{uy_str}"

    temp = bin if bin is not None else 'auto'

    # histogram of the data
    _, bin_edges, _ = axs[1].hist(resid, color="blue",edgecolor='blue', histtype = "step", bins=temp, label ="Residual distribution", zorder=2)
    axs[1].set_ylabel('Counts')

    lnspc = np.linspace(resid.min() - 2 * sigma, resid.max() + 2 * sigma, 500) 

    bin_widths = np.diff(bin_edges)
    mean_bin_width = np.mean(bin_widths)
    f_gauss = data.size * mean_bin_width * spstats.norm.pdf(lnspc, mean, sigma)

    axs[1].plot(lnspc, f_gauss, linewidth=1, color='r',linestyle='--', label = label, zorder=1)
    axs[1].set_xlabel(label1)
    axs[1].yaxis.set_major_locator(MaxNLocator(integer=True))
    axs[1].set_xlim(mean - 3 * sigma, mean + 3 * sigma)

    axs[1].legend()

    skewness = np.sum((resid - mean)**3) / (len(resid) * sigma**3)
    kurtosis = np.sum((resid - mean)**4) / (len(resid) * sigma**4) - 3

    _, p_value = spstats.shapiro(resid)

    if p_value >= 0.10:
        pval_str = f"{p_value*100:.0f}%"
    elif 0.005 < p_value < 0.10:
        pval_str = f"{p_value * 100:.2f}%"
    elif 0.0005 < p_value <= 0.005:
        pval_str = f"{p_value * 1000:.2f}‰"
    elif 1e-6 < p_value <= 0.0005:
        pval_str = f"{p_value:.2e}"
    else:
        pval_str = f"< 1e-6"

    dw = durbin_watson(resid)

    # Prepara l'unità di misura, se presente
    uy_str = f" {unit}" if unit else ""

    # Crea la stringa stampabile
    stamp = (
        f"Mean value: {rounded_mean:.{max(0, -exponent + 1)}f}{uy_str}\n"
        f"Standard deviation: {rounded_sigma:.{max(0, -exponent + 1)}f}{uy_str}\n"
        f"Skewness: {skewness:.2f}\n"
        f"Kurtosis: {kurtosis:.2f}\n"
        f"p-value: {pval_str}\n"
        f"Durbin-Watson statistic: {dw:.3f}"
    )

    print(stamp)

    if p_value >= 0.05:
        print("\nResiduals are consistent with a normal distribution.")
    else:
        print("\nResiduals deviate significantly from a normal distribution.")
    
    if dw < 1.5:
        print("Residuals show evidence of positive autocorrelation.")
    elif dw > 2.5:
        print("Residuals show evidence of negative autocorrelation.")
    else:
        print("Residuals do not show significant autocorrelation.")

    return mean, sigma, skewness, kurtosis, p_value, dw

def samples(n, distribution='normal', **params):
    """
    Generate synthetic data from common probability distributions.

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

    Raises
    ------
    ValueError
        If an unsupported `distribution` is specified or required parameters are missing.

    Examples
    --------
    >>> samples(1000, 'normal', mu=0, sigma=1)
    array([...])
    >>> samples(500, 'uniform', low=0, high=10)
    array([...])
    >>> samples(200, 'poisson', lam=3)
    array([...])
    """
    dist = distribution.lower()
    rng = np.random.default_rng()
    
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
    data = np.asarray(data)

    # Caso 1: confronto con expected → forza 'zscore'
    if expected is not None:
        if data_err is None:
            raise ValueError("If you provide 'expected', you must also provide 'data_err'.")
        
        expected = np.asarray(expected)
        data_err = np.asarray(data_err)

        if len(data) != len(expected) or len(data) != len(data_err):
            raise ValueError("'data', 'expected', and 'data_err' must have the same length.")

        # Metodo unico valido
        z_scores = np.abs((data - expected) / data_err)
        mask = z_scores < threshold

    else:
        # Caso 2: solo dati osservati → puoi scegliere il metodo
        if method == "zscore":
            mean = np.mean(data)
            std = np.std(data)
            z_scores = np.abs((data - mean) / std)
            mask = z_scores < threshold

        elif method == "mad":
            median = np.median(data)
            mad = np.median(np.abs(data - median))
            modified_z_scores = 0.6745 * (data - median) / mad
            mask = np.abs(modified_z_scores) < threshold

        elif method == "iqr":
            q1 = np.percentile(data, 25)
            q3 = np.percentile(data, 75)
            iqr = q3 - q1
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
            mask = (data >= lower_bound) & (data <= upper_bound)

        else:
            raise ValueError("Unrecognized method. Use 'zscore', 'mad', or 'iqr'.")

    return data[mask]

def posterior(x, y, sy, f, p0, burn=1000, steps=5000, thin=10, maxfev=5000):
    """
    Bayesian analysis with emcee for fitting a function with many parameters. 
    
    This function performs a Markov Chain Monte Carlo (MCMC) analysis to obtain a posterior distribution of the parameters, 
    then calculates the Maximum Likelihood Estimation (MLE) parameters and visualizes the corner plot of the results.

    Parameters
    ----------
        x : array-like
            Measured values for the independent variable.
        y : array-like
            Measured values for the dependent variable (to be fitted to the model).
        sy : array-like
            Uncertainties on the measurements of the dependent variable.
        f : function
            Model function to be fitted to the data. The function should accept an independent variable 
            `x` as the first argument and the free parameters as subsequent arguments.
        p0 : list
            List of initial values for the free parameters of the model. 
            Example: [a0, b0, c0], where each element corresponds to the initial value of a parameter.
        burn : int, optional
            Number of "burn-in steps" to exclude the first samples from the Markov chain 
            that might be correlated (default is 1000).
        steps : int, optional
            Total number of steps for the Markov chain (default is 5000).
        thin : int, optional
            Subsampling factor (default is 10), to reduce correlation between samples.
        maxfev : int
            Maximum number of iterations for the `curve_fit` function.

    Returns
    ----------
        res.params : Params
            Object containing the optimized parameters and uncertainties on the obtained parameters.
        res.flatchain : array-like
            Flattened chain of MCMC samples, useful for statistical analysis.

    The function visualizes a corner plot of the posterior parameters and prints the median values and uncertainties 
    on the parameters, along with the results of the Maximum Likelihood Estimation (MLE).

    Notes
    ----------
    https://lmfit.github.io/lmfit-py/fitting.html#lmfit.minimizer.MinimizerResult
    """
    
    # Creazione del modello lmfit con la funzione f
    mod = Model(f)
    params = Parameters()

    # Inizializzazione dei parametri con p0
    for i, name in enumerate(mod.param_names):
        params.add(name, value=p0[i])

    # --- Fitting con curve_fit per ottenere una stima iniziale dei parametri ---
    popt, pcov = curve_fit(f, x, y, p0=p0, sigma=sy, absolute_sigma=True, maxfev=maxfev)
    
    # Residui normalizzati
    residual = (y - f(x, *popt)) / sy

    # --- Esegui l'analisi bayesiana con emcee ---
    def lnprob(p, x, y, sy):
        # Calcola la log-likelihood e aggiungi una prior (come una prior uniforme)
        model = f(x, *p)
        chi_squared = np.sum(((y - model) / sy) ** 2)
        log_likelihood = -0.5 * chi_squared
        
        # Prior (in questo caso una prior uniforme)
        log_prior = 0
        for param in p:
            if param <= 0:
                return -np.inf  # Prior che rifiuta parametri non positivi
        
        return log_likelihood + log_prior

    # Configurazione di emcee
    ndim = len(p0)  # Numero di parametri da stimare
    nwalkers = 2 * ndim  # Numero di walkers
    p0_emcee = [popt + 1e-4 * np.random.randn(ndim) for i in range(nwalkers)]  # Inizializzazione dei walkers
    
    # Esecuzione del campionamento MCMC
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(x, y, sy))
    sampler.run_mcmc(p0_emcee, steps)
    
    # Appiattire la catena MCMC
    flat_samples = sampler.get_chain(discard=burn, thin=thin, flat=True)
    
    # Visualizzazione della corner plot
    corner.corner(flat_samples, labels=mod.param_names, truths=popt)
    plt.show()

    # Stampa la mediana e le incertezze della distribuzione posteriore
    print("Median of posterior probability distribution:")
    print("-------------------------------------------")
    for i, name in enumerate(mod.param_names):
        median = np.median(flat_samples[:, i])
        lower = np.percentile(flat_samples[:, i], 16)
        upper = np.percentile(flat_samples[:, i], 84)
        print(f"{name}: {median:.5f} (+{upper - median:.5f}, -{median - lower:.5f})")

    # Massima verosimiglianza (MLE) - Otteniamo l'indice corretto
    # Qui vogliamo l'indice della log-probabilità massima
    log_prob = sampler.get_log_prob(discard=burn, thin=thin, flat=True)
    highest_prob_index = np.argmax(log_prob)
    
    # Estraiamo i parametri corrispondenti
    mle_soln = flat_samples[highest_prob_index]

    print("\nMaximum Likelihood Estimation (MLE):")
    print("-------------------------------------")
    for i, name in enumerate(mod.param_names):
        print(f"{name}: {mle_soln[i]:.5f}")

    return params, flat_samples