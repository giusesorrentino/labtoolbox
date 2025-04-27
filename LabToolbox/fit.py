from LabToolbox import curve_fit, plt, np, sm, chi2
from .basics import my_cov, my_mean, my_var, my_line, y_estrapolato, format_value_auto
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

    if p_value > 0.005:
        print(f"p-value = {p_value*100:.2f}%")
    elif 0.0005 < p_value <= 0.005:
        print(f"p-value = {p_value*1000:.2f}‰")
    elif 1e-6 < p_value <= 0.0005:
        print(f"p-value = {p_value:.2e}")
    else:
        print(f"p-value < 1e-6")

    #pesi
    w_y = np.power(sy.astype(float), -2)
        
    m2 = my_cov(x, y, w_y) / my_var(x, w_y)
    var_m2 = 1 / ( my_var(x, w_y) * np.sum(w_y) )
        
    c2 = my_mean(y, w_y) - my_mean(x, w_y) * m
    var_c2 = my_mean(x*x, w_y)  / ( my_var(x, w_y) * np.sum(w_y) )

    sigma_m2 = var_m2 ** 0.5
    sigma_c2 = var_c2 ** 0.5
        
    cov_mc = - my_mean(x, weights) / ( my_var(x, weights) * np.sum(weights) ) 

    # Arrotonda m1 e sm1
    # exponent = int(math.floor(math.log10(abs(sigma_m))))
    # factor = 10**(exponent - 1)
    # rounded_sigma = round(sigma_m / factor) * factor

    # # 2. Arrotonda mean allo stesso ordine di grandezza di sigma
    # rounded_mean = round(m, -exponent + 1)

    # # 3. Converte in stringa mantenendo zeri finali
    # fmt = f".{-exponent + 1}f" if exponent < 1 else "f"
    # m1_str = f"{(rounded_mean/mscale2):.{max(0, -exponent + 1)}f}"
    # sm1_str = f"{(rounded_sigma/mscale2):.{max(0, -exponent + 1)}f}"

    # # Arrotondac1 e sc1
    # exponent = int(math.floor(math.log10(abs(sigma_c))))
    # factor = 10**(exponent - 1)
    # rounded_sigma = round(sigma_c / factor) * factor

    # # 2. Arrotonda mean allo stesso ordine di grandezza di sigma
    # rounded_mean = round(c, -exponent + 1)

    # # 3. Converte in stringa mantenendo zeri finali
    # fmt = f".{-exponent + 1}f" if exponent < 1 else "f"
    # c1_str = f"{(rounded_mean / cscale2):.{max(0, -exponent + 1)}f}"
    # sc1_str = f"{(rounded_sigma / cscale2):.{max(0, -exponent + 1)}f}"

    # digits = np.abs(n_cifre(sigma_m)) + 1
    # if(n_cifre(sigma_m) >= 0):
    #     sm1 = round(sigma_m, -int(digits) + 2)
    # else:
    #     sm1 = round(sigma_m, digits)
    # m1 = riscrivi_con_cifre_decimali(sm1, m)

    # digits = np.abs(n_cifre(sigma_c)) + 1
    # if(n_cifre(sigma_c) >= 0):
    #     sc1 = round(sigma_c, -int(digits) + 2)
    # else:
    #     sc1 = round(sigma_c, digits)
    # c1 = riscrivi_con_cifre_decimali(sc1, c)
    
    err_exp = int(np.floor(np.log10(abs(sigma_m))))
    err_coeff = sigma_m / 10**err_exp

    if err_coeff < 1.5:
        err_exp -= 1
        err_coeff = sigma_m / 10**err_exp

    sm1 = round(sigma_m, -err_exp + 1)
    m1 = round(m, -err_exp + 1)

    err_exp = int(np.floor(np.log10(abs(sigma_c))))
    err_coeff = sigma_c / 10**err_exp

    if err_coeff < 1.5:
        err_exp -= 1
        err_coeff = sigma_c / 10**err_exp

    sc1 = round(sigma_c, -err_exp + 1)
    c1 = round(c, -err_exp + 1)

    # Calcolo dei residui normalizzati
    resid = y - (m * x + c)
    resid_norm = resid / sy

    k = np.sum((-1 <= resid_norm) & (resid_norm <= 1))

    n = k / len(resid_norm)

    print(f"Percentuale di residui compatibili con zero: {n*100:.1f}%")

    # costruisco dei punti x su cui valutare la retta del fit              
    xmin = float(np.min(x)) 
    xmax = float(np.max(x))
    xmin_plot = xmin-.2*(xmax-xmin)
    xmax_plot = xmax+.2*(xmax-xmin)
    x1 = np.linspace(xmin_plot, xmax_plot, 500)
    y1 = my_line(x1, m, c)

    y1_plus_1sigma = y1 + y_estrapolato(x1, m2, c2, sigma_m2, sigma_c2, cov_mc)[1]
    y1_minus_1sigma = y1 - y_estrapolato(x1, m2, c2, sigma_m2, sigma_c2, cov_mc)[1] 

    label = (
        "Best Fit\n"
        + f"$m={format_value_auto(m1, sm1, unit = m_units, scale = mscale)}$\n"
        + f"$c={format_value_auto(c1, sc1, unit = c_units, scale = cscale)}$"
    )

    if residuals:
        # Ottieni le dimensioni standard di una figura matplotlib
        #default_figsize = figure.figaspect(1.0)  # Dimensioni standard
        
        # Crea una figura con le dimensioni standard più spazio per il pannello residui
        fig = plt.figure(figsize=(6.4, 4.8))
        #fig = plt.figure()
        
        # Crea due pannelli con GridSpec, mantenendo la dimensione originale per il pannello principale
        # gs = GridSpec(2, 1, height_ratios=[0.1, 0.9],hspace=0)  # 1/10 per residui, 9/10 per il grafico principale
        gs = fig.add_gridspec(2, hspace=0, height_ratios=[0.1, 0.9])
        axs = gs.subplots(sharex=True)
        
        # Pannello residui (1/10 dell'altezza totale)
        # ax_residuals = fig.add_subplot(gs[0])
        
        # Pannello principale (occupa 9/10 dell'altezza totale)
        # ax_main = fig.add_subplot(gs[1:])

        # Plot dei residui
        
        # Aggiungi linee di riferimento
        axs[0].axhline(0., ls='--', color='0.7', lw=0.8)
        #axs[0].axhline(2, ls='dashed', color='crimson', lw=0.6)
        #axs[0].axhline(-2, ls='dashed', color='crimson', lw=0.6)
        if norm == False:
            axs[0].errorbar(x/xscale, resid/yscale, sy, ls='', color='gray', lw=1.)
            axs[0].plot(x/xscale, resid/yscale, color='k', drawstyle='steps-mid', lw=1.)
            # axs[0].errorbar(x/xscale, resid/yscale, sy, ls='', color='gray', lw=1.) # <-- prima era qui
            axs[0].plot(x/ xscale, confidence*sy/yscale, ls='dashed', color='crimson', lw=1.)
            axs[0].plot(x/ xscale, -confidence*sy/yscale, ls='dashed', color='crimson', lw=1.)
            res_min = -np.nanmean(3*sy*confidence/2)
            res_max = np.nanmean(3*sy*confidence/2)
            axs[0].set_ylim(res_min,res_max)
        else:
            axs[0].errorbar(x/xscale, resid_norm, 1, ls='', color='gray', lw=1.)
            axs[0].plot(x/xscale, resid_norm, color='k', drawstyle='steps-mid', lw=1.)
            # axs[0].errorbar(x/xscale, resid_norm, 1, ls='', color='gray', lw=1.) # <-- prima era qui
            axs[0].plot(x1/ xscale, np.repeat(confidence, len(x1)), ls='dashed', color='crimson', lw=1.)
            axs[0].plot(x1/ xscale, np.repeat(-confidence, len(x1)), ls='dashed', color='crimson', lw=1.)
            axs[0].set_ylim(-3*confidence/2, 3*confidence/2)
        
        # Configurazioni estetiche per il pannello dei residui
        axs[0].tick_params(labelbottom=False)
        axs[0].set_yticklabels('')
        # axs[0].set_ylim(-3*confidence/2, 3*confidence/2)
        axs[1].set_xlim(xmin_plot/xscale,xmax_plot/xscale)

        if showlegend:

            axs[1].plot(x1/xscale, y1/yscale, linestyle="-", color="blue", linewidth=0.8, label=label)
        
        else:

            axs[1].plot(x1/xscale, y1/yscale, linestyle = "-", color = "blue", linewidth = 0.8, label = f"Best Fit")

        # Plot principale con dati e fit
        if sx == None:
            axs[1].errorbar(x / xscale, y / yscale, yerr=sy / yscale, ls='', marker='.', 
                            color="black", label='Dati sperimentali', capsize=2)
        else:
            axs[1].errorbar(x / xscale, y / yscale, yerr=sy / yscale, xerr=sx/xscale, ls='', marker='.', 
                            color="black", label='Dati sperimentali', capsize=2)

        if confidencerange == True:
            axs[1].fill_between(x1/xscale, y1_plus_1sigma/yscale, y1_minus_1sigma/yscale, where=(y1_plus_1sigma/yscale > y1_minus_1sigma/yscale), color='blue', alpha=0.3, edgecolor='none', label="Intervallo di confidenza")
        
        axs[1].set_xlabel(xlabel)
        axs[1].set_ylabel(ylabel)
        axs[1].set_xlim(xmin_plot/xscale,xmax_plot/xscale)
        axs[1].legend()
        if legendloc == None:
            axs[1].legend()
        else:
            axs[1].legend(loc = legendloc)

    else:

        if showlegend:

            plt.plot(x1/xscale, y1/yscale, linestyle="-", color="blue", linewidth=0.8, label=label)


        else:

            plt.plot(x1/xscale, y1/yscale, linestyle = "-", color = "blue", linewidth = 0.8, label = f"Best Fit")


        if sx == None:
            plt.errorbar(x/xscale, y/yscale, yerr=sy/yscale, capsize=2, label = "Dati sperimentali", linestyle='', marker = ".", color="black")
        else:
            plt.errorbar(x/xscale, y/yscale, yerr=sy/yscale, xerr = sx/xscale, capsize=2, label = "Dati sperimentali", linestyle='', marker = ".", color="black")

        if confidencerange == True:
            plt.fill_between(x1/xscale, y1_plus_1sigma/yscale, y1_minus_1sigma/yscale, where=(y1_plus_1sigma/yscale > y1_minus_1sigma/yscale), color='blue', alpha=0.3, edgecolor='none', label="Intervallo di confidenza")
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.xlim(xmin_plot/xscale,xmax_plot/xscale)

        if legendloc == None:
            plt.legend()
        else:
            plt.legend(loc = legendloc)

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

    Parameters
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

    if p_value > 0.005:
        print(f"p-value = {p_value*100:.2f}%")
    elif 0.0005 < p_value <= 0.005:
        print(f"p-value = {p_value*1000:.2f}‰")
    elif 1e-6 < p_value <= 0.0005:
        print(f"p-value = {p_value:.2e}")
    else:
        print(f"p-value < 1e-6")

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

    y1_plus_1sigma = confid[1]
    y1_minus_1sigma = confid[0]

    # Costruzione della stringa p-value
    if p_value > 0.005:
        pval_str = f"$\\text{{p–value}} = {p_value * 100:.2f}$%"
    elif 0.0005 < p_value <= 0.005:
        pval_str = f"$\\text{{p–value}} ={p_value * 1000:.2f}$‰"
    elif 1e-6 < p_value <= 0.0005:
        pval_str = f"$\\text{{p–value}} = {p_value:.2e}$"
    else:
        pval_str = f"$\\text{{p–value}} < 10^{{-6}}$"

    if residuals:
        # Ottieni le dimensioni standard di una figura matplotlib
        #default_figsize = figure.figaspect(1.0)  # Dimensioni standard
        
        # Crea una figura con le dimensioni standard più spazio per il pannello residui
        fig = plt.figure(figsize=(6.4, 4.8))
        #fig = plt.figure()
        
        # Crea due pannelli con GridSpec, mantenendo la dimensione originale per il pannello principale
        # gs = GridSpec(2, 1, height_ratios=[0.1, 0.9],hspace=0)  # 1/10 per residui, 9/10 per il grafico principale
        gs = fig.add_gridspec(2, hspace=0, height_ratios=[0.1, 0.9])
        axs = gs.subplots(sharex=True)
        
        # Pannello residui (1/10 dell'altezza totale)
        # ax_residuals = fig.add_subplot(gs[0])
        
        # Pannello principale (occupa 9/10 dell'altezza totale)
        # ax_main = fig.add_subplot(gs[1:])
        
        # Aggiungi linee di riferimento
        axs[0].axhline(0., ls='--', color='0.7', lw=0.8)
        #axs[0].axhline(2, ls='dashed', color='crimson', lw=0.6)
        #axs[0].axhline(-2, ls='dashed', color='crimson', lw=0.6)
        if norm == False:
            axs[0].errorbar(x/xscale, resid/yscale, sy, ls='', color='gray', lw=1.)
            axs[0].plot(x/xscale, resid/yscale, color='k', drawstyle='steps-mid', lw=1.)
            # axs[0].errorbar(x/xscale, resid/yscale, sy, ls='', color='gray', lw=1.) # <-- prima era qui
            axs[0].plot(x/xscale, confidence*sy/yscale, ls='dashed', color='crimson', lw=1.)
            axs[0].plot(x/xscale, -confidence*sy/yscale, ls='dashed', color='crimson', lw=1.)
            res_min = -np.nanmean(3*sy*confidence/2)
            res_max = np.nanmean(3*sy*confidence/2)
            axs[0].set_ylim(res_min,res_max)
        else:
            axs[0].errorbar(x/xscale, resid_norm, 1, ls='', color='gray', lw=1.)
            axs[0].plot(x/xscale, resid_norm, color='k', drawstyle='steps-mid', lw=1.)
            # axs[0].errorbar(x/xscale, resid_norm, 1, ls='', color='gray', lw=1.) # <-- prima era qui
            axs[0].plot(x1/xscale, np.repeat(confidence, len(x1)), ls='dashed', color='crimson', lw=1.)
            axs[0].plot(x1/xscale, np.repeat(-confidence, len(x1)), ls='dashed', color='crimson', lw=1.)
            axs[0].set_ylim(-3*confidence/2, 3*confidence/2)
        
        # Configurazioni estetiche per il pannello dei residui
        axs[0].tick_params(labelbottom=False)
        axs[0].set_yticklabels('')
        # axs[0].set_ylim(-3*confidence/2, 3*confidence/2)
        axs[0].set_xlim((x.min() - amp)/xscale, (x.max() + amp)/xscale)

        if showlegend:

            # Uso in label
            axs[1].plot(
                x1 / xscale,
                y_fit_cont / yscale,
                color="blue",
                ls="-",
                linewidth=0.8,
                label=(
                    f"Best fit\n$\\chi^2/\\text{{dof}} = {chi2_red:.2f}$\n{pval_str}"
                )
            )

            if confidencerange == True:
                axs[1].fill_between(x1/xscale, y1_plus_1sigma/yscale, y1_minus_1sigma/yscale, where=(y1_plus_1sigma/yscale > y1_minus_1sigma/yscale), color='blue', alpha=0.3, edgecolor='none', label="Intervallo di confidenza")

            if sx == None:
                axs[1].errorbar(x / xscale, y/yscale, yerr=sy/yscale, ls='', marker='.', 
                                color="black", label='Dati sperimentali', capsize=2)       
            else:
                axs[1].errorbar(x / xscale, y/yscale, yerr=sy/yscale, xerr=sx/xscale, ls='', marker='.', 
                                color="black", label='Dati sperimentali', capsize=2)
        
        else:

            axs[1].plot(x1 / xscale, y_fit_cont / yscale, color="blue", ls="-", 
                            linewidth=0.8, label=f"Best fit")
            
            if confidencerange == True:
                axs[1].fill_between(x1/xscale, y1_plus_1sigma/yscale, y1_minus_1sigma/yscale, where=(y1_plus_1sigma/yscale > y1_minus_1sigma/yscale), color='blue', alpha=0.3, edgecolor='none', label="Intervallo di confidenza")

            if sx == None:
                axs[1].errorbar(x / xscale, y/yscale, yerr=sy/yscale, ls='', marker='.', 
                                color="black", label='Dati sperimentali', capsize=2)       
            else:
                axs[1].errorbar(x / xscale, y/yscale, yerr=sy/yscale, xerr=sx/xscale, ls='', marker='.', 
                                color="black", label='Dati sperimentali', capsize=2) 
        
        axs[1].set_xlabel(xlabel)
        axs[1].set_ylabel(ylabel)
        axs[1].set_xlim((x.min() - amp)/xscale, (x.max() + amp)/xscale)

        if legendloc == None:
            axs[1].legend()
        else:
            axs[1].legend(loc = legendloc)
        
        # Gestione delle scale logaritmiche
        if log == "x":
            axs[1].set_xscale("log")
            axs[0].set_xscale("log")
        elif log == "y":
            axs[1].set_yscale("log")
        elif log == "xy":
            axs[1].set_xscale("log")
            axs[1].set_yscale("log")
            axs[0].set_xscale("log")
        
        #plt.tight_layout()
        
    else:
        # Plot originale senza residui
        plt.figure()

        if showlegend:
        
            # Uso in label
            plt.plot(
                x1 / xscale,
                y_fit_cont / yscale,
                color="blue",
                ls="-",
                linewidth=0.8,
                label=(
                    f"Best fit\n$\\chi^2/\\text{{dof}} = {chi2_red:.2f}$\n{pval_str}"
                )
            )

            if confidencerange == True:
                plt.fill_between(x1/xscale, y1_plus_1sigma/yscale, y1_minus_1sigma/yscale, where=(y1_plus_1sigma/yscale > y1_minus_1sigma/yscale), color='blue', alpha=0.3, edgecolor='none', label="Intervallo di confidenza")

            if sx == None:
                plt.errorbar(x / xscale, y/yscale, yerr=sy/yscale, ls='', marker='.', 
                                color="black", label='Dati sperimentali', capsize=2)       
            else:
                plt.errorbar(x / xscale, y/yscale, yerr=sy/yscale, xerr=sx/xscale, ls='', marker='.', 
                                color="black", label='Dati sperimentali', capsize=2)
        
        else:

            plt.plot(x1 / xscale, y_fit_cont / yscale, color="blue", ls="-", 
                            linewidth=0.8, label=f"Best fit")
            
            if confidencerange == True:
                plt.fill_between(x1/xscale, y1_plus_1sigma/yscale, y1_minus_1sigma/yscale, where=(y1_plus_1sigma/yscale > y1_minus_1sigma/yscale), color='blue', alpha=0.3, edgecolor='none', label="Intervallo di confidenza")

            if sx == None:
                plt.errorbar(x / xscale, y/yscale, yerr=sy/yscale, ls='', marker='.', 
                                color="black", label='Dati sperimentali', capsize=2)       
            else:
                plt.errorbar(x / xscale, y/yscale, yerr=sy/yscale, xerr=sx/xscale, ls='', marker='.', 
                                color="black", label='Dati sperimentali', capsize=2) 

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.xlim((x.min() - amp)/xscale, (x.max() + amp)/xscale)
        if legendloc == None:
            plt.legend()
        else:
            plt.legend(loc = legendloc)

        if log == "x":
            plt.xscale("log")
        elif log == "y":
            plt.yscale("log")
        elif log == "xy":
            plt.xscale("log")
            plt.yscale("log")

    plt.show()

    return popt, errors, chi2_red, p_value