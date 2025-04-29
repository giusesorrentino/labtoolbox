from LabToolbox import math, np, plt, stats, chi2

def PrintResult(mean, sigma, name = "", ux = ""):
    """
    Returns a formatted string in the "mean ± sigma" format, with sigma to two significant figures,
    and the mean rounded consistently.

    Parameters
    ----------
    mean : float
        Value of the variable.
    sigma : float
        Uncertainty of the variable considered.
    name : str, optional
        Name of the variable to display before the value (default is an empty string).
    ux : str, optional
        Unit of measurement to display after the value in parentheses (default is an empty string).

    Returns
    -------
    None
        Prints the formatted string directly.
    """

    # 1. Arrotonda sigma a due cifre significative
    if sigma == 0:
        raise ValueError("The uncertainty cannot be zero.")
        
    exponent = int(math.floor(math.log10(abs(sigma))))
    factor = 10**(exponent - 1)
    rounded_sigma = round(sigma / factor) * factor

    # 2. Arrotonda mean allo stesso ordine di grandezza di sigma
    rounded_mean = round(mean, -exponent + 1)

    # 3. Converte in stringa mantenendo zeri finali
    fmt = f".{-exponent + 1}f" if exponent < 1 else "f"
    mean_str = f"{rounded_mean:.{max(0, -exponent + 1)}f}"
    sigma_str = f"{rounded_sigma:.{max(0, -exponent + 1)}f}"

    # 4. Crea la stringa risultante
    if ux != "":
        if rounded_mean != 0:
            nu = rounded_sigma / rounded_mean
            result = f"{name} = ({mean_str} ± {sigma_str}) {ux} [{np.abs(nu)*100:.2f}%]"
        else:
            result = f"{name} = ({mean_str} ± {sigma_str}) {ux}"
    else:
        if rounded_mean != 0:
            nu = rounded_sigma / rounded_mean
            result = f"{name} = ({mean_str} ± {sigma_str}) [{np.abs(nu)*100:.2f}%]"
        else:
            result = f"{name} = ({mean_str} ± {sigma_str})"

    print(result)

def histogram(x, sigmax, xscale = 0, xlabel = "", ux = ""):
    """
    Plots the histogram of occurrences of a variable x, checking for its Gaussianity.

    Parameters
    ----------
    x : array-like
        Array of the parameter of interest.
    sigmax : array-like
        Array of the uncertainties for each element of x. Can be `None`.
    xscale : int
        Scaling factor for the x-axis (e.g., `xscale = -2` corresponds to 1e-2, to convert meters to centimeters).
    xlabel : str
        Name of the variable x.
    ux : str
        Unit of measurement of the variable x. Default is `""`.
    """

    x = x / 10**xscale

    if sigmax is not None:
        sigmax = sigmax / 10**xscale
        sigma = np.sqrt(x.std()**2 + np.sum(sigmax**2)/len(x))
    else:
        sigma = x.std()
    mean = x.mean()

    # err_exp = int(np.floor(np.log10(abs(sigma))))
    # err_coeff = sigma / 10**err_exp

    # if err_coeff < 1.5:
    #     err_exp -= 1
    #     err_coeff = sigma / 10**err_exp

    # sigma1 = round(sigma, -err_exp + 1)
    # mean1 = round(mean, -err_exp + 1)

    # Calcola l'esponente di sigma
    exponent = int(math.floor(math.log10(abs(sigma))))
    factor = 10**(exponent - 1)
    rounded_sigma = (round(sigma / factor) * factor)

    # Arrotonda la media
    rounded_mean = round(mean, -exponent + 1)

    # Converte in stringa mantenendo zeri finali
    fmt = f".{-exponent + 1}f" if exponent < 1 else "f"

    # label_ist = (f"Istogramma delle occorrenze")

    N = len(x)
    
    # Calcolare il numero di bin in base ai metodi
    sturges_bins = int(np.ceil(np.log2(N) + 1))  # Metodo di Sturges
    sqrt_bins = int(np.ceil(np.sqrt(N)))  # Metodo della radice quadrata
    freedman_binsize = 2 * np.percentile(x, 75) - np.percentile(x, 25) / np.cbrt(N)
    freedman_bins = int(np.ceil((np.max(x) - np.min(x)) / freedman_binsize))  # Metodo Freedman-Diaconis
    
    # Se i dati sono approssimativamente gaussiani, usare la regola di Sturges o la radice quadrata
    # Il metodo di Freedman-Diaconis è più robusto se i dati non sono gaussiani
    if (sigma / mean) < 0.5:  # Condizione approssimativa per la normalità
        bins = sturges_bins
    else:
        bins = freedman_bins
        
    # Calcolo il bin size basato sul numero di bin scelto
    bins = np.linspace(np.min(x), np.max(x), bins + 1)
    binsize = bins[1] - bins[0]

    # histogram of the data
    plt.hist(x,bins=bins,color="blue",edgecolor='blue',alpha=0.75, histtype = "step")
    plt.ylabel('Counts')
    # plt.title(label=label_ist)

    # ==> draw a gaussian function
    # create an array with 500 equally separated values in the x axis interval
    lnspc = np.linspace(x.min()- sigma, x.max() + sigma, 500) 
    # create an array with f(x) values, one for each of the above points
    # normalize properly the function such that integral from -inf to +inf is the total number of events
    norm_factor = x.size * binsize
    f_gaus = norm_factor*stats.norm.pdf(lnspc,mean,sigma)  
    # draw the function
    if ux != "":
        plt.plot(lnspc, f_gaus, linewidth=1, color='r',linestyle='--', label = f"Gaussian\n$\mu = {rounded_mean:.{max(0, -exponent + 1)}f}$ "+ux+f"\n$\sigma = {rounded_sigma:.{max(0, -exponent + 1)}f}$ "+ux)
        plt.xlabel(xlabel+" ["+ux+"]")
    else:
        plt.plot(lnspc, f_gaus, linewidth=1, color='r',linestyle='--', label = f"$\mu = {rounded_mean:.{max(0, -exponent + 1)}f}$\n$\sigma = {rounded_sigma:.{max(0, -exponent + 1)}f}$")
        plt.xlabel(xlabel)

    plt.legend()

    skewness = np.sum((x - x.mean())**3) / (len(x) * sigmax**3)

    print(f"This histogram has a skewness of {skewness:.2f}") #gamma 1

    curtosi = np.sum((x - x.mean())**4) / (len(x) * sigmax**4) - 3  # momento terzo - 3, vedi wikipedia

    print(f"This histogram has a kurtosis of {curtosi:.2f}")

def residuals(x_data, y_data, y_att, sy, N, xlabel, ux = "", uy = "", marker = "d", xscale = 0, yscale = 0, confidence = 2, norm = True, legendloc = None, newstyle = True, log = None):
    """
    Plots the normalized residuals.

    Parameters
    ----------
    x_data : array-like
        Measured values for the independent variable.
    y_data : array-like
        Measured values for the dependent variable.
    y_att : array-like
        Predicted values from the model for the dependent variable.
    sy : array-like
        Uncertainties for the measured dependent variable.
    N : int, optional     
        Number of free parameters in the model. Can be `None`.
    xlabel : str          
        Name of the independent variable.
    ux : str 
        Unit of measurement of the independent variable. Default is `""`.
    uy : str
        Unit of measurement of the dependent variable. Default is `""`.
    marker : str
        Marker for the residual. Only applicable if `newstyle = False`.
    xscale : int
        Scaling factor (10^xscale) for the x-axis (e.g., xscale = -2 if converting from meters to centimeters). 
    yscale : int
        Scaling factor (10^yscale) for the y-axis (e.g., yscale = -2 if converting from meters to centimeters). 
    confidence : int
        Defines the confidence interval `[-confidence, +confidence]`. Must be a positive number. Default is `2`.
    norm : bool
        If `True`, the residuals will be normalized. Default is `True`. 
    legendloc : str
        Positioning of the legend in the plot ('upper right', 'lower left', 'upper center', etc.). Default is `None`.
    newstyle : bool
        Alternative style for the plot.
    log : bool
        If `x`, the x-axis will be in logarithmic scale. Default is `None`.

    Returns
    ----------
    None

    Notes
    ----------
    If `N = None`, the values of χ²/dof and p-value will not be displayed.
    """

    if confidence <= 0:
        raise ValueError("Il parametro 'confidence' deve essere maggiore di zero.")

    xscale = 10**xscale
    yscale = 10**yscale

    resid = y_data - y_att
    resid_norm = resid/sy

    if N is not None:
        chi2_value = np.sum(resid_norm ** 2)

        # Gradi di libertà (DOF)
        dof = len(x_data) - N

        # Chi-quadrato ridotto
        chi2_red = chi2_value / dof

        # p-value
        p_value = chi2.sf(chi2_value, dof)

        if p_value > 0.005:
            pval_str = f"$\\text{{p–value}} = {p_value * 100:.2f}$%"
        elif 0.0005 < p_value <= 0.005:
            pval_str = f"$\\text{{p–value}} ={p_value * 1000:.2f}$‰"
        elif 1e-6 < p_value <= 0.0005:
            pval_str = f"$\\text{{p–value}} = {p_value:.2e}$"
        else:
            pval_str = f"$\\text{{p–value}} < 10^{{-6}}$"

        label2 = f"\n{pval_str}\n$\chi^2/\\text{{dof}} = {chi2_red:.2f}$"
    
    else:
        label2 = f"\n$\chi^2 = {chi2_value:.2f}$"

    if norm == True:
        label1 = "Normalized residuals"
        bar1 = np.repeat(1, len(x_data))
        bar2 = resid_norm
        dash = np.repeat(confidence, len(x1))
    else :
        label1 = "Residuals"
        bar1 = sy / yscale
        bar2 = resid / yscale
        dash = confidence * sy/yscale

    label = label2+label1

    amp = np.abs(x_data.max()-x_data.min())/20

    x_data = x_data / xscale

    xmin_plot = x_data.min()-amp
    xmax_plot = x_data.max()+amp
    x1 = np.linspace(xmin_plot, xmax_plot, 500)

    if newstyle:
        plt.axhline(0., ls='--', color='0.7', lw=0.8)
        plt.errorbar(x_data, bar2, bar1, ls='', color='gray', lw=1., label = f"Confidence interval $[-{confidence},\,{confidence}]$")
        plt.plot(x_data, bar2, color='k', drawstyle='steps-mid', lw=1., label = label)
        plt.plot(x1, dash, ls='dashed', color='crimson', lw=1.)
        plt.plot(x1, -dash, ls='dashed', color='crimson', lw=1.)
        plt.ylim(-np.nanmean(3 * bar1 * confidence/2), np.nanmean(3 * bar1 * confidence/2))
        if norm == False:
            if uy != "":
                plt.ylabel("Residuals"+" ["+uy+"]")
            else:
                plt.xlabel("Residuals")
    else:
        plt.plot(xmin_plot, xmax_plot, [0, 0], 'r--')
        plt.errorbar(x_data, bar2, 1, marker=marker, linestyle="", capsize = 2, color='black', label = label)

    plt.xlim(xmin_plot, xmax_plot)

    if legendloc == None:
        plt.legend()
    else:
        plt.legend(legendloc)

    if ux != "":
        plt.xlabel(xlabel+" ["+ux+"]")
    else:
        plt.xlabel(xlabel)

    if log == "x":
        plt.xscale("log")

    k = np.sum((-1 <= resid_norm) & (resid_norm <= 1))

    n = k / len(resid_norm)

    if n >= 0.10:
        print(f"{n*100:.0f}% of the residuals lie within ±2σ of zero.")
    elif 0.005 < n < 0.10:
        print(f"{n*100:.2f}% of the residuals lie within ±2σ of zero.")
    elif 0.0005 < p_value <= 0.005:
        print(f"{n*1000:.2f}‰ of the residuals lie within ±2σ of zero.")
    else:
        print(f"{n:.2e} of the residuals lie within ±2σ of zero.")

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