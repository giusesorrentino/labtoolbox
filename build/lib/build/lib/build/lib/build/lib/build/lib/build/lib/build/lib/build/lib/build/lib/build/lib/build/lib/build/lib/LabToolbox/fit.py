import math
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.stats import chi2
from scipy.optimize import curve_fit
from .utils import PrintResult
from .uncertainty import propagate
from ._helper import my_cov, my_mean, my_var, my_line, y_estrapolato

def lin_fit(x, y, y_err, x_err = None, fitmodel = "wls", xlabel="x [ux]", ylabel="y [uy]", showlegend = True, legendloc = None, log = None,
            xscale = 0, yscale = 0, mscale = 0, cscale = 0, m_units = "", c_units = "", confidence = 2, confidencerange = True, residuals=True, norm = True, result = False):
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
        result : bool, optional
            If `True`, prints the output of `wls_fit` (or `ols_fit`) to the screen. Default is `False`.

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
    - The values of `xscale` and `yscale` affect only the axis scaling in the plot and have no impact on the fitting computation itself. All model parameters are estimated using the original input data as provided.
    - LaTeX formatting is already embedded in the strings used to display the units of `m` and `c`. You do not need to use "$...$".
    - If `c_scale = 0` (recommended when using `c_units`), then `c_units` will represent the suffix corresponding to 10^yscale (+ `y_units`).
    - If `m_scale = 0` (recommended when using `m_units`), then `m_units` will represent the suffix corresponding to 10^(yscale - xscale) [+ `y_units/x_units`].
    """

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

    if result:
        print(results.summary())

    # Parametri stimati
    m = float(results.params[1])
    c = float(results.params[0])

    # Errori standard dei parametri stimati
    sigma_m = float(results.bse[1])  # Incertezza sul coefficiente angolare (m)
    sigma_c = float(results.bse[0])  # Incertezza sull'intercetta (c)

    chi2_value = np.sum(((y - (m * x + c)) / y_err) ** 2)

    # Gradi di libertà (DOF)
    dof = len(x) - 2

    # Chi-quadrato ridotto
    chi2_red = chi2_value / dof

    # p-value
    p_value = chi2.sf(chi2_value, dof)

    print(f"χ²/dof = {chi2_red:.2f}") # ≈ 1 se il fit è buono

    if p_value >= 0.10:
        print(f"p-value = {p_value*100:.0f}%")
    elif 0.005 < p_value < 0.10:
        print(f"p-value = {p_value*100:.2f}%")
    elif 0.0005 < p_value <= 0.005:
        print(f"p-value = {p_value*1000:.2f}‰")
    elif 1e-6 < p_value <= 0.0005:
        print(f"p-value = {p_value:.2e}")
    else:
        print(f"p-value < 1e-6")
        
    m2 = my_cov(x, y, weights) / my_var(x, weights)
    var_m2 = 1 / ( my_var(x, weights) * np.sum(weights) )
        
    c2 = my_mean(y, weights) - my_mean(x, weights) * m
    var_c2 = my_mean(x*x, weights)  / ( my_var(x, weights) * np.sum(weights) )

    sigma_m2 = var_m2 ** 0.5
    sigma_c2 = var_c2 ** 0.5
        
    cov_mc = - my_mean(x, weights) / ( my_var(x, weights) * np.sum(weights) )

    # ------------------------ 

    # Applica lo scaling
    mean_scaled = m / (10**mscale)
    sigma_scaled = sigma_m / (10**mscale)

    exponent = int(math.floor(math.log10(abs(sigma_scaled))))
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

    exponent = int(math.floor(math.log10(abs(sigma_scaled))))
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

    k = np.sum((-1 <= resid_norm) & (resid_norm <= 1))

    n = k / len(resid_norm)

    # if n >= 0.10:
    #     print(f"{n*100:.0f}% of the residuals lie within ±2σ of zero.")
    # elif 0.005 < n < 0.10:
    #     print(f"{n*100:.2f}% of the residuals lie within ±2σ of zero.")
    # elif 0.0001 < p_value <= 0.005:
    #     print(f"{n*1000:.2f}‰ of the residuals lie within ±2σ of zero.")
    # else:
    #     print(f"{n:.2e} of the residuals lie within ±2σ of zero.")

    if n >= 0.10:
        print(f"{n*100:.0f}% of the residuals lie within ±2σ of zero.")
    elif 0.005 < n < 0.10:
        print(f"{n*100:.2f}% of the residuals lie within ±2σ of zero.")
    else:
        print("Virtually no residuals lie within ±2σ of zero.")

    # costruisco dei punti x su cui valutare la retta del fit              
    xmin = float(np.min(x)) 
    xmax = float(np.max(x))
    xmin_plot = xmin-.12*(xmax-xmin)
    xmax_plot = xmax+.12*(xmax-xmin)
    x1 = np.linspace(xmin_plot, xmax_plot, 500)
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
        bar1 = np.repeat(1, len(x))
        bar2 = resid_norm
        dash = np.repeat(confidence, len(x1))
    else :
        bar1 = y_err
        bar2 = resid / yscale
        dash = confidence * y_err

    fig = plt.figure(figsize=(6.4, 4.8))

    # The following code (lines 257-274) is adapted from the VoigtFit library,
    # originally developed by Jens-Kristian Krogager under the MIT License.
    # https://github.com/jkrogager/VoigtFit

    if residuals:
        gs = fig.add_gridspec(2, hspace=0, height_ratios=[0.1, 0.9])
        axs = gs.subplots(sharex=True)
        #axs = gs.subplots()
        # Aggiungi linee di riferimento
        axs[0].axhline(0., ls='--', color='0.7', lw=0.8)
        axs[0].errorbar(x, bar2, bar1, ls='', color='gray', lw=1.)
        axs[0].plot(x, bar2, color='k', drawstyle='steps-mid', lw=1.)
        if norm == True:
            axs[0].plot(x1, dash, ls='dashed', color='crimson', lw=1.)
            axs[0].plot(x1, -dash, ls='dashed', color='crimson', lw=1.)
        else:
            axs[0].plot(x, dash, ls='dashed', color='crimson', lw=1.)
            axs[0].plot(x, -dash, ls='dashed', color='crimson', lw=1.)
        axs[0].set_ylim(-np.nanmean(3 * dash / 2), np.nanmean(3 * dash / 2))

        # Configurazioni estetiche per il pannello dei residui
        axs[0].tick_params(labelbottom=False)
        axs[0].set_yticklabels('')
        axs[0].set_xlim(xmin_plot, xmax_plot)
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
    axs[1].set_xlim(xmin_plot/xscale, xmax_plot/xscale)

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

def model_fit(x, y, y_err, f, p0, x_err = None, xlabel="x [ux]", ylabel="y [uy]", showlegend = True, legendloc = None, 
              bounds = None, confidencerange = True, log=None, maxfev=5000, xscale=0, yscale=0, confidence = 2, residuals=True, norm = True):
    """
    General-purpose fit of multi-parameter functions, with an option to display residuals.

    Parameters
    ----------
        x : array-like
            Measured values of the independent variable.
        y : array-like
            Measured values of the dependent variable.
        y_err : array-like
            Uncertainties associated with the dependent variable.
        f : function
            Function of one variable (first argument of `f`) with `N` free parameters.
        p0 : list
            Initial guess for the model parameters, in the form `[a, ..., z]`.
        x_err : array-like, optional
            Uncertainties associated with the independent variable. Default is `None`.
        xlabel : str, optional
            Label (and units) for the independent variable.
        ylabel : str, optional
            Label (and units) for the dependent variable.
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

    Returns
    ----------
        popt : array-like
            Array of optimal parameters estimated from the fit.
        errors : array-like
            Uncertainties on the optimal parameters.
        chi2_red : float
            Reduced chi-square value (χ²/dof).
        p_value : float
            Fit p-value (probability that the observed χ² is compatible with the model).

    Notes
    ----------
    The values of `xscale` and `yscale` affect only the axis scaling in the plot and have no impact on the fitting computation itself. 
    All model parameters are estimated using the original input data as provided.
    """

    if log is not None:
        if log not in ("x", "y", "xy"):
            raise ValueError("The value of 'log' must be 'x', 'y' or 'xy'.")

    xscale = 10**xscale
    yscale = 10**yscale

    if bounds is None:
        bounds = (np.repeat(-np.inf, len(p0)), np.repeat(np.inf, len(p0)))

    popt, pcov = curve_fit(
        f,
        x,
        y,
        p0=p0,
        sigma=y_err,
        absolute_sigma=True,
        maxfev=maxfev
    )

    perr = np.sqrt(np.diag(pcov))

    # Calcolo del chi-quadrato
    y_fit = f(x, *popt)

    resid = y - y_fit
    resid_norm = resid / y_err

    chi2_value = np.sum((resid_norm) ** 2)

    # Gradi di libertà (DOF)
    dof = len(x) - len(popt)

    # Chi-quadrato ridotto
    chi2_red = chi2_value / dof

    # p-value
    p_value = chi2.sf(chi2_value, dof)

    # Stampa dei parametri con incertezze
    for i in range(len(popt)):

        # Calcola l'esponente di sigma
        exponent = int(math.floor(math.log10(abs(perr[i]))))
        factor = 10**(exponent - 1)
        rounded_sigma = (round(perr[i] / factor) * factor)

        # Arrotonda la media
        rounded_mean = round(popt[i], -exponent + 1) 

        # Converte in stringa mantenendo zeri finali
        fmt = f".{-exponent + 1}f" if exponent < 1 else "f"

        if popt[i] != 0:
            nu = perr[i] / popt[i]
            print(
                f"Parameter {i + 1} = ({rounded_mean:.{max(0, -exponent + 1)}f} +/- {rounded_sigma:.{max(0, -exponent + 1)}f}) [{np.abs(nu) * 100:.2f}%]"
            )
        else:
            print(f"Parameter {i + 1} = ({rounded_mean:.{max(0, -exponent + 1)}f} +/- {rounded_sigma:.{max(0, -exponent + 1)}f})")

    
    print(f"\nχ²/dof = {chi2_red:.2f}")  # ≈ 1 se il fit è buono

    if p_value >= 0.10:
        print(f"p-value = {p_value*100:.0f}%")
        pval_str = f"$\\text{{p–value}} = {p_value*100:.0f}$%"
    elif 0.005 < p_value < 0.10:
        print(f"p-value = {p_value*100:.2f}%\n")
        pval_str = f"$\\text{{p–value}} = {p_value * 100:.2f}$%"
    elif 0.0001 < p_value <= 0.005:
        print(f"p-value = {p_value*1000:.2f}‰")
        pval_str = f"$\\text{{p–value}} = {p_value * 1000:.2f}$‰"
    else:
        print(f"p-value < 1e-6")
        pval_str = f"$\\text{{p–value}} < 10^{{-4}}$"

    k = np.sum((-1 <= resid_norm) & (resid_norm <= 1))

    n = k / len(resid_norm)

    # if n >= 0.10:
    #     print(f"{n*100:.0f}% of the residuals lie within ±2σ of zero.")
    # elif 0.005 < n < 0.10:
    #     print(f"{n*100:.2f}% of the residuals lie within ±2σ of zero.")
    # elif 0.0005 < p_value <= 0.005:
    #     print(f"{n*1000:.2f}‰ of the residuals lie within ±2σ of zero.")
    # else:
    #     print(f"{n:.2e} of the residuals lie within ±2σ of zero.")

    if n >= 0.10:
        print(f"{n*100:.0f}% of the residuals lie within ±2σ of zero.")
    elif 0.005 < n < 0.10:
        print(f"{n*100:.2f}% of the residuals lie within ±2σ of zero.")
    else:
        print("Virtually no residuals lie within ±2σ of zero.")

    # costruisco dei punti x su cui valutare la retta del fit              
    xmin = float(np.min(x)) 
    xmax = float(np.max(x))
    xmin_plot = xmin-.12*(xmax-xmin)
    xmax_plot = xmax+.12*(xmax-xmin)

    x1 = np.linspace(xmin_plot, xmax_plot, 500)
    y_fit_cont = f(x1, *popt)

    # Ripeti ciascun parametro per len(x1) volte
    parametri_ripetuti = [np.repeat(p, len(x1)) for p in popt]
    errori_ripetuti = [np.repeat(e, len(x1)) for e in perr]

    # Costruisci lista dei valori e delle incertezze
    lista = [x1] + parametri_ripetuti
    lista_err = [np.repeat(0, len(x1))] + errori_ripetuti

    # Ora puoi usarli nella propagazione
    _, _ , confid = propagate(f, lista, lista_err)

    y1_plus_1sigma = confid[1] / yscale
    y1_minus_1sigma = confid[0] / yscale

    x1 = x1 / xscale
    xmax_plot = xmax_plot / xscale
    xmin_plot = xmin_plot / xscale
    x = x / xscale
    y = y / yscale
    y_err = y_err / yscale
    y_fit_cont = y_fit_cont / yscale
    y_fit = y_fit / yscale

    if x_err is not None:
        x_err = x_err / xscale

    if norm == True:
        bar1 = np.repeat(1, len(x))
        bar2 = resid_norm
        dash = np.repeat(confidence, len(x1))
    else :
        bar1 = y_err
        bar2 = resid / yscale
        dash = confidence * y_err

    fig = plt.figure(figsize=(6.4, 4.8))

    # The following code (lines 518-535) is adapted from the VoigtFit library,
    # originally developed by Jens-Kristian Krogager under the MIT License.
    # https://github.com/jkrogager/VoigtFit

    if residuals:
        gs = fig.add_gridspec(2, hspace=0, height_ratios=[0.1, 0.9])
        axs = gs.subplots(sharex=True)
        # axs = gs.subplots()
        # Aggiungi linee di riferimento
        axs[0].axhline(0., ls='--', color='0.7', lw=0.8)
        axs[0].errorbar(x, bar2, bar1, ls='', color='gray', lw=1.)
        axs[0].plot(x, bar2, color='k', drawstyle='steps-mid', lw=1.)
        if norm == True:
            axs[0].plot(x1, dash, ls='dashed', color='crimson', lw=1.)
            axs[0].plot(x1, -dash, ls='dashed', color='crimson', lw=1.)
        else:
            axs[0].plot(x, dash, ls='dashed', color='crimson', lw=1.)
            axs[0].plot(x, -dash, ls='dashed', color='crimson', lw=1.)
        axs[0].set_ylim(-np.nanmean(3 * dash / 2), np.nanmean(3 * dash / 2))

        # Configurazioni estetiche per il pannello dei residui
        axs[0].tick_params(labelbottom=False)
        axs[0].set_yticklabels('')
        axs[0].set_xlim(xmin_plot, xmax_plot)
        #axs[0].set_xlim(x.min(), x.max())
    else: 
        gs = fig.add_gridspec(2, hspace=0, height_ratios=[0, 1])
        axs = gs.subplots(sharex=True)
        #axs = gs.subplots()
        axs[0].remove()  # Rimuovi axs[0], axs[1] rimane valido

    if showlegend:
        label = f"Best fit\n$\\chi^2/\\text{{dof}} = {chi2_red:.2f}$\n{pval_str}"
    else :
        label = "Best fit"

    axs[1].plot(x1, y_fit_cont, color="blue", ls="-", linewidth=0.8, label = label)

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
    axs[1].set_xlim(xmin_plot, xmax_plot)

    if legendloc == None:
        axs[1].legend()
    else:
        axs[1].legend(loc = legendloc)
    
    # Gestione delle scale logaritmiche
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

    return popt, perr, chi2_red, p_value

def bootstrap_fit(func, xdata, ydata, y_err = None, p0 = None, punits = None, n_iter = 1000, bounds = (-np.inf, np.inf)):
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


    xdata = np.asarray(xdata)
    ydata = np.asarray(ydata)
    if y_err is not None:
        y_err = np.asarray(y_err)
    n_points = len(xdata)
    all_popt = []

    for _ in range(n_iter):
        indices = np.random.choice(n_points, n_points, replace=True)
        x_sample = xdata[indices]
        y_sample = ydata[indices]
        if y_err is not None:
            sigma_sample = y_err[indices]
        else:
            sigma_sample = None

        try:
            popt, _ = curve_fit(func, x_sample, y_sample, p0=p0, bounds=bounds, sigma=sigma_sample, absolute_sigma=True)
            all_popt.append(popt)
        except Exception:
            continue  # Ignora i fit che non convergono

    all_popt = np.array(all_popt)
    popt_mean = np.mean(all_popt, axis=0)
    popt_std = np.std(all_popt, axis=0)

    for i in range(len(all_popt)):
        value = popt_mean[i]
        error = popt_std[i]

        if punits is not None:
            unit = punits[i]
        else:
            unit = ""

        if value > 1e4 or abs(value) < 1e-3:
            # Scrittura in notazione scientifica
            exponent = int(np.floor(np.log10(abs(value)))) if value != 0 else 0
            scaled_value = value / 10**exponent
            scaled_error = error / 10**exponent
            name = f"Parameter {i+1} [1e{exponent}]"
            PrintResult(scaled_value, scaled_error, name=name, ux=unit)
        else:
            name = f"Parameter {i+1} "
            PrintResult(value, error, name=name, ux=unit)

    return popt_mean, popt_std, all_popt