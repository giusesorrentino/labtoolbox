from LabToolbox import curve_fit, plt, np, sm, chi2, math
from .basics import my_cov, my_mean, my_var, my_line, y_estrapolato
from .misc import PrintResult
from .uncertainty import propagate_uncertainty

def lin_fit(x, y, sy, sx = None, fitmodel = "wls", xlabel="x [ux]", ylabel="y [uy]", showlegend = True, legendloc = None, 
            xscale = 0, yscale = 0, mscale = 0, cscale = 0, m_units = "", c_units = "", confidence = 2, confidencerange = True, residuals=True, norm = True, result = False):
    """
    Esegue un fit lineare (Weighted Least Squares o Ordinary Least Squares) e visualizza i dati sperimentali con retta di regressione e incertezza.

    Parameters
    ----------
        x : array-like
            Valori della variabile indipendente.
        y : array-like
            Valori della variabile dipendente.
        sy : array-like
            Incertezze associate ai valori di y.
        sx : array-like
            Incertezze associate ai valori di x.
        fitmodel : str
            Modello del fit, "wls" o "ols". Default è "wls".
        xlabel : str
            Etichetta dell'asse x, con unità tra parentesi quadre (es. "x [m]").
        ylabel : str
            Etichetta dell'asse y, con unità tra parentesi quadre (es. "y [s]").
        showlegend : bool
            Se `True`, mostra l'etichetta con i valori di m e c nel plot. 
        legendloc : str
            Posizionamento della legenda nel grafico ('upper right', 'lower left', 'upper center' etc.). Default è `None`.
        xscale : float
            Fattore di scala dell'asse x (es. `xscale = -2`, cioè 10e-2, per passare da m a cm).
        yscale : float
            Fattore di scala dell'asse y.
        mscale : float
            Fattore di scala di `m`.
        cscale : float
            Fattore di scala di `c`.
        m_units : str
            Unità di misura di m (attenzione alla scala di m, x ed y). Default è `""`.
        c_units : str
            Unità di misura di c (attenzione alla scala di c, x ed y). Default è `""`.
        confidence : int
            Intervallo di confidenza dei residui, cioè `[-confidenze, +confidence]`.
        confidencerange : bool
            Se `True`, mostra la fascia di incertezza del fit (1σ) come area evidenziata attorno alla retta del fit.
        residuals : bool
            Se `True`, aggiunge un pannello superiore con i residui del fit.
        norm : bool
            Se `True`, i residui nel pannello superiore saranno normalizzati.
        result : bool
            Se `True`, stampa su schermo il risultato di `wls_fit`. Default è `False`.

    Returns
    ----------
        m : float
            Coefficiente angolare della retta di regressione.
        c : float
            Intercetta della retta di regressione.
        sigma_m : float
            Incertezza sul coefficiente angolare.
        sigma_c : float
            Incertezza sull'intercetta.
        chi2_red : float
            Valore del chi-quadro ridotto (χ²/dof).
        p_value : float
            p-value del fit (probabilità che il χ² osservato sia compatibile con il modello).

    Notes
    ----------
    Il formato latex è già preimpostato all'interno delle stringhe che permettono la visualizzazione delle unità di misura di m e c. Non vi è bisogno di scrivere "$...$".
    Se `c_scale = 0` (scelta consigliata se si utilizza l'opzione di unità di misura per `c`), allora `c_units` è il suffisso corrispondente a 10^yscale (+ `y_units`).
    Se `m_scale = 0` (scelta consigliata se si utilizza l'opzione di unità di misura per `m`), allora `m_units` è il suffisso corrispondente a 10^(yscale - xscale) [+ `y_units/x_units`].
    """

    xscale = 10**xscale
    yscale = 10**yscale
    
    # Aggiunta dell'intercetta (colonna di 1s per il termine costante)
    X = sm.add_constant(x)  # Aggiunge una colonna di 1s per il termine costante

    # Calcolo dei pesi come inverso delle varianze
    weights = 1 / sy**2

    # Modello di regressione pesata
    if fitmodel == "wls":
        model = sm.WLS(y, X, weights=weights)  # Weighted Least Squares (OLS con pesi)
    elif fitmodel == "ols":
        model = sm.OLS(y, X)
    else:
        raise ValueError('Errore! Modello non valido. Solo "wls" o "ols"')
    results = model.fit()

    if result:
        print(results.summary())

    # Parametri stimati
    m = float(results.params[1])
    c = float(results.params[0])

    # Errori standard dei parametri stimati
    sigma_m = float(results.bse[1])  # Incertezza sul coefficiente angolare (m)
    sigma_c = float(results.bse[0])  # Incertezza sull'intercetta (c)

    chi2_value = np.sum(((y - (m * x + c)) / sy) ** 2)

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

    # Calcola l'esponente di sigma
    exponent = int(math.floor(math.log10(abs(sigma_m))))
    factor = 10**(exponent - 1)
    rounded_sigma = (round(sigma_m / factor) * factor) / (10**mscale)

    # Arrotonda la media
    rounded_mean = round(m, -exponent + 1) / (10**mscale)

    # Converte in stringa mantenendo zeri finali
    fmt = f".{-exponent + 1}f" if exponent < 1 else "f"
    mean_str = f"{rounded_mean:.{max(0, -exponent + 1)}f}"
    sigma_str = f"{rounded_sigma:.{max(0, -exponent + 1)}f}"

    # Crea la stringa risultante
    if m_units != "":
        if mscale != 0:
            result = rf"$m = ({mean_str} \pm {sigma_str}) \\times 10^{{{mscale}}} \, \mathrm{{{m_units}}}$"
        else:
            result = rf"$m = ({mean_str} \pm {sigma_str}) \, {m_units}$"
    else:
        if mscale != 0:
            result = f"$m = ({mean_str} \pm {sigma_str}) \\times 10^{{{mscale}}}$"
        else:
            result = f"$m = {mean_str} \pm {sigma_str}$"
    
    # ------------------------ 

    # Calcola l'esponente di sigma
    exponent = int(math.floor(math.log10(abs(sigma_c))))
    factor = 10**(exponent - 1)
    rounded_sigma = (round(sigma_c / factor) * factor) / (10**cscale)

    # Arrotonda la media
    rounded_mean = round(c, -exponent + 1) / (10**cscale)

    # Converte in stringa mantenendo zeri finali
    fmt = f".{-exponent + 1}f" if exponent < 1 else "f"
    mean_str = f"{rounded_mean:.{max(0, -exponent + 1)}f}"
    sigma_str = f"{rounded_sigma:.{max(0, -exponent + 1)}f}"

        # Crea la stringa risultante
    if c_units != "":
        if cscale != 0:
            result1 = rf"$c = ({mean_str} \pm {sigma_str}) \\times 10^{{{cscale}}} \, \mathrm{{{c_units}}}$"
        else:
            result1 = rf"$c = ({mean_str} \pm {sigma_str}) \, {c_units}$"
    else:
        if cscale != 0:
            result1 = f"$c = ({mean_str} \pm {sigma_str}) \\times 10^{{{cscale}}}$"
        else:
            result1 = f"$c = {mean_str} \pm {sigma_str}$"
    
    # ------------------------ 

    # Calcolo dei residui normalizzati
    resid = y - (m * x + c)
    resid_norm = resid / sy

    k = np.sum((-1 <= resid_norm) & (resid_norm <= 1))

    n = k / len(resid_norm)

    print(f"Percentuale di residui compatibili con zero: {n*100:.1f}%")

    # costruisco dei punti x su cui valutare la retta del fit              
    xmin = float(np.min(x)) 
    xmax = float(np.max(x))
    xmin_plot = xmin-.2*(xmax-xmin) / xscale
    xmax_plot = xmax+.2*(xmax-xmin) / xscale
    x1 = np.linspace(xmin_plot, xmax_plot, 500)
    y1 = my_line(x1, m, c) / yscale

    y1_plus_1sigma = y1 + y_estrapolato(x1, m2, c2, sigma_m2, sigma_c2, cov_mc)[1] / yscale
    y1_minus_1sigma = y1 - y_estrapolato(x1, m2, c2, sigma_m2, sigma_c2, cov_mc)[1] / yscale

    y = y / yscale
    x = x / xscale
    sy = sy / yscale
    if sx is not None:
        sx = sx / xscale

    if showlegend:
        label = (
            "Best fit\n"
            + result + "\n"
            + result1
        )
    else :
        label = "Best fit"

    if norm == True:
        bar1 = np.repeat(1, len(x))
        bar2 = resid_norm
        dash = np.repeat(confidence, len(x1))
    else :
        bar1 = sy
        bar2 = resid / yscale
        dash = confidence * sy

    fig = plt.figure(figsize=(6.4, 4.8))

    if residuals:
        gs = fig.add_gridspec(2, hspace=0, height_ratios=[0.1, 0.9])
        axs = gs.subplots(sharex=True)
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
    else:
        gs = fig.add_gridspec(2, hspace=0, height_ratios=[0, 1])
        axs = gs.subplots(sharex=True)
        axs[0].remove()  # Rimuovi axs[0], axs[1] rimane valido

    axs[1].plot(x1, y1, color="blue", ls="-", linewidth=0.8, label = label)

    if confidencerange == True:
        axs[1].fill_between(x1, y1_plus_1sigma, y1_minus_1sigma,  
                            where=(y1_plus_1sigma > y1_minus_1sigma), color='blue', alpha=0.3, edgecolor='none', label="Intervallo di confidenza")

    if sx == None:
        axs[1].errorbar(x, y, yerr=sy, ls='', marker='.', 
                        color="black", label='Dati sperimentali', capsize=2)       
    else:
        axs[1].errorbar(x, y, yerr=sy, xerr=sx, ls='', marker='.', 
                        color="black", label='Dati sperimentali', capsize=2)
    
    axs[1].set_xlabel(xlabel)
    axs[1].set_ylabel(ylabel)
    axs[1].set_xlim(xmin_plot, xmax_plot)

    if legendloc == None:
        axs[1].legend()
    else:
        axs[1].legend(loc = legendloc)

    return m, c, sigma_m, sigma_c, chi2_red, p_value

def model_fit(x, y, sy, f, p0, sx = None, xlabel="x [ux]", ylabel="y [uy]", showlegend = True, legendloc = None, 
              bounds = None, confidencerange = True, log=None, maxfev=5000, xscale=0, yscale=0, confidence = 2, residuals=True, norm = True):
    """
    Fit universale di funzioni a molti parametri, con opzione per visualizzare i residui.

    Parameters
    ----------
        x : array-like
            Valori misurati per la variabile indipendente.
        y : array-like
            Valori misurati per la variabile dipendente.
        sy : array-like
            Incertezze della variabile dipendente misurata.
        f : function
            Funzione ad una variabile (primo argomento di `f`) con `N` parametri liberi.
        p0 : list
            Lista dei valori iniziali dei parametri liberi del modello, nella forma `[a, ..., z]`.
        sx : array-like
            Incertezze della variabile indipendente misurata. Default è `None`.
        xlabel : str
            Nome (e unità) della variabile indipendente.
        ylabel : str
            Nome (e unità) della variabile dipendente.
        showlegend : bool
            Se `True`, mostra l'etichetta del chi-quadro ridotto e p-value nel plot. 
        legendloc : str
            Posizionamento della legenda nel grafico ('upper right', 'lower left', 'upper center' etc.). Default è `None`.
        bounds : 2-tuple of array-like
            Lista `([lower_bound],[upper_bound])` dei limiti dei parametri. Default è `None`.
        confidencerange : bool
            Se `True`, mostra la fascia di incertezza del fit (1σ) come area evidenziata attorno alla curva del best fit.
        log : str
            Se `x` o `y`, l'asse x o y sarà in scala logaritmica; se `xy`, entrambi gli assi.
        maxfev : int
            Numero massimo di iterazioni della funzione `curve_fit`.
        xscale : int
            Fattore di scala dell'asse x (es. `xscale = -2`, cioè 10e-2, per passare da m a cm).
        yscale : int
            Fattore di scala dell'asse y.
        confidence : int
            Intervallo di confidenza dei residui, cioè `[-confidenze, +confidence]`.
        residuals : bool
            Se `True`, aggiunge un pannello superiore con i residui del fit.
        norm : bool
            Se `True`, i residui nel pannello superiore saranno normalizzati.

    Returns
    ----------
        popt : array-like
            Array dei parametri ottimali ottenuti dal fit.
        errors : array-like
            Incertezze sui parametri ottimali.
        chi2_red : float
            Valore del chi-quadro ridotto (χ²/dof).
        p_value : float
            p-value del fit (probabilità che il χ² osservato sia compatibile con il modello).
    """

    xscale = 10**xscale
    yscale = 10**yscale

    # Fit con curve_fit
    if bounds is not None:
        popt, pcov = curve_fit(
            f,
            x,
            y,
            p0=p0,
            sigma=sy,
            bounds=bounds,
            absolute_sigma=True,
            maxfev=maxfev
        )
    else:
        popt, pcov = curve_fit(
            f,
            x,
            y,
            p0=p0,
            sigma=sy,
            absolute_sigma=True,
            maxfev=maxfev
        )

    errors = np.sqrt(np.diag(pcov))

    # Calcolo del chi-quadrato
    y_fit = f(x, *popt)

    resid = y - y_fit
    resid_norm = resid / sy

    chi2_value = np.sum((resid_norm) ** 2)

    # Gradi di libertà (DOF)
    dof = len(x) - len(popt)

    # Chi-quadrato ridotto
    chi2_red = chi2_value / dof

    # p-value
    p_value = chi2.sf(chi2_value, dof)

    # Stampa dei parametri con incertezze
    for i in range(len(popt)):
        err_exp = int(np.floor(np.log10(abs(errors[i]))))
        err_coeff = errors[i] / 10**err_exp

        if err_coeff < 1.5:
            err_exp -= 1
            err_coeff = errors[i] / 10**err_exp

        sigma1 = round(errors[i], -err_exp + 1)
        mean1 = round(popt[i], -err_exp + 1)

        if mean1 != 0:
            nu = sigma1 / mean1
            print(
                f"Parametro {i + 1} = ({mean1} +/- {sigma1}) [{np.abs(nu) * 100:.2f}%]"
            )
        else:
            print(f"Parametro {i + 1} = ({mean1} +/- {sigma1})")

    
    print(f"χ²/dof = {chi2_red:.2f}")  # ≈ 1 se il fit è buono

    if p_value >= 0.10:
        print(f"p-value = {p_value*100:.0f}%")
        pval_str = f"$\\text{{p–value}} = {p_value*100:.0f}$%"
    elif 0.005 < p_value < 0.10:
        print(f"p-value = {p_value*100:.2f}%")
        pval_str = f"$\\text{{p–value}} = {p_value * 100:.2f}$%"
    elif 0.0005 < p_value <= 0.005:
        print(f"p-value = {p_value*1000:.2f}‰")
        pval_str = f"$\\text{{p–value}} ={p_value * 1000:.2f}$‰"
    elif 1e-6 < p_value <= 0.0005:
        print(f"p-value = {p_value:.2e}")
        pval_str = f"$\\text{{p–value}} = {p_value:.2e}$"
    else:
        print(f"p-value < 1e-6")
        pval_str = f"$\\text{{p–value}} < 10^{{-6}}$"

    k = np.sum((-1 <= resid_norm) & (resid_norm <= 1))

    n = k / len(resid_norm)

    print(f"Percentuale di residui compatibili con zero: {n*100:.1f}%")

    amp = np.abs(x.max() - x.min()) / 20

    x1 = np.linspace(min(x) - amp, max(x) + amp, 1000)
    y_fit_cont = f(x1, *popt)

    # Ripeti ciascun parametro per len(x1) volte
    parametri_ripetuti = [np.repeat(p, len(x1)) for p in popt]
    errori_ripetuti = [np.repeat(e, len(x1)) for e in errors]

    # Costruisci lista dei valori e delle incertezze
    lista = [x1] + parametri_ripetuti
    lista_err = [np.repeat(0, len(x1))] + errori_ripetuti

    # Ora puoi usarli nella propagazione
    _, _ , confid = propagate_uncertainty(f, lista, lista_err)

    y1_plus_1sigma = confid[1] / yscale
    y1_minus_1sigma = confid[0] / yscale

    x1 = x1 / xscale
    x = x / xscale
    y = y / yscale
    sy = sy / yscale
    y_fit_cont = y_fit_cont / yscale
    y_fit = y_fit / yscale

    if sx is not None:
        sx = sx / xscale

    if norm == True:
        bar1 = np.repeat(1, len(x))
        bar2 = resid_norm
        dash = np.repeat(confidence, len(x1))
    else :
        bar1 = sy
        bar2 = resid / yscale
        dash = confidence * sy

    fig = plt.figure(figsize=(6.4, 4.8))

    if residuals:
        gs = fig.add_gridspec(2, hspace=0, height_ratios=[0.1, 0.9])
        axs = gs.subplots(sharex=True)
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
        axs[0].set_xlim((x.min() - amp), (x.max() + amp))
    else: 
        gs = fig.add_gridspec(2, hspace=0, height_ratios=[0, 1])
        axs = gs.subplots(sharex=True)
        axs[0].remove()  # Rimuovi axs[0], axs[1] rimane valido

    if showlegend:
        label = f"Best fit\n$\\chi^2/\\text{{dof}} = {chi2_red:.2f}$\n{pval_str}"
    else :
        label = "Best fit"

    axs[1].plot(x1, y_fit_cont, color="blue", ls="-", linewidth=0.8, label = label)

    if confidencerange == True:
        axs[1].fill_between(x1, y1_plus_1sigma, y1_minus_1sigma,  
                            where=(y1_plus_1sigma > y1_minus_1sigma), color='blue', alpha=0.3, edgecolor='none', label="Intervallo di confidenza")

    if sx == None:
        axs[1].errorbar(x, y, yerr=sy, ls='', marker='.', 
                        color="black", label='Dati sperimentali', capsize=2)       
    else:
        axs[1].errorbar(x, y, yerr=sy, xerr=sx, ls='', marker='.', 
                        color="black", label='Dati sperimentali', capsize=2)
    
    axs[1].set_xlabel(xlabel)
    axs[1].set_ylabel(ylabel)
    axs[1].set_xlim((x.min() - amp), (x.max() + amp))

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

    return popt, errors, chi2_red, p_value

def bootstrap_fit(func, xdata, ydata, sigma_y = None, p0 = None, punits = None, n_iter = 1000, bounds = (-np.inf, np.inf)):
    """
    Esegue un bootstrap del fit per stimare la distribuzione dei parametri, considerando opzionalmente le incertezze sigma_y.

    Parameters
    ----------
        func : callable
            Funzione modello da fittare, della forma `func(x, *params)`.
        xdata : array_like
            Dati indipendenti (ascisse).
        ydata : array_like
            Dati dipendenti (ordinate).
        sigma_y : array_like, opzionale
            Incertezze associate a `ydata`. Se fornite, il fit sarà pesato.
        p0 : array_like, opzionale
            Parametri iniziali per il fit.
        punits : list of str, opzionale
            Lista di stringhe (unità di misura dei parametri). Defaul è None.
        n_iter : int, opzionale
            Numero di iterazioni di bootstrap (default: `1000`).
        bounds : 2-tuple di array, opzionale
            Limiti inferiori e superiori sui parametri per il fit.

    Returns
    ----------
        popt_mean : array
            Parametri medi ottenuti dal bootstrap.
        popt_std : array
            Deviazioni standard dei parametri (stima delle incertezze).
        all_popt : array
            Array completo di tutte le stime dei parametri (forma: `[n_iter, n_params]`).

    Notes
    ----------
    Se il parametro i-esimo è un numero puro, è sufficiente inserire `""` al corrispondente elemento della lista.
    """

    xdata = np.asarray(xdata)
    ydata = np.asarray(ydata)
    if sigma_y is not None:
        sigma_y = np.asarray(sigma_y)
    n_points = len(xdata)
    all_popt = []

    for _ in range(n_iter):
        indices = np.random.choice(n_points, n_points, replace=True)
        x_sample = xdata[indices]
        y_sample = ydata[indices]
        if sigma_y is not None:
            sigma_sample = sigma_y[indices]
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
            name = f"Parametro {i+1}-esimo [1e{exponent}]"
            PrintResult(scaled_value, scaled_error, name=name, ux=unit)
        else:
            name = f"Parametro {i+1}-esimo"
            PrintResult(value, error, name=name, ux=unit)

    return popt_mean, popt_std, all_popt