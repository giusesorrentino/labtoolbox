from LabToolbox import math, np, plt, stats, chi2

def PrintResult(mean, sigma, name = "", ux = ""):
    """
    Restituisce una stringa formattata nel formato "mean ± sigma", con sigma a due cifre significative,
    e mean arrotondato in modo coerente.

    Parameters
    ----------
    mean : float
        Valore della variabile.
    sigma : float
        Incertezza della variabile considerata.
    name : str, optional
        Nome della variabile da visualizzare prima del valore (default è stringa vuota).
    ux : str, optional
        Unità di misura da mostrare dopo il valore tra parentesi (default è stringa vuota).

    Returns
    -------
    None
        Stampa direttamente la stringa formattata.
    """

    # 1. Arrotonda sigma a due cifre significative
    if sigma == 0:
        raise ValueError("Sigma non può essere zero.")
        
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

def histogram(x, sigmax, xlabel = "", ux = ""):
    """
    Grafica l'istogramma delle occorrenze di una variabile x, verificandone la gaussianità.

    Parameters
    ----------
        x : array-like
            Array del parametro d'interesse.
        sigmax : array-like
            Array delle incertezze dei singoli elementi di x.
        xlabel : str
            Nome della variabile x.
        ux : str
            Unità di misura della variabile x. Defaul è `""`.
    """

    sigma = np.sqrt(x.std()**2 + np.sum(sigmax**2)/len(x))
    mean = x.mean()

    err_exp = int(np.floor(np.log10(abs(sigma))))
    err_coeff = sigma / 10**err_exp

    if err_coeff < 1.5:
        err_exp -= 1
        err_coeff = sigma / 10**err_exp

    sigma1 = round(sigma, -err_exp + 1)
    mean1 = round(mean, -err_exp + 1)

    label_ist = (f"Istogramma delle occorrenze")

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
    plt.ylabel('Conteggi')
    plt.title(label=label_ist)

    # ==> draw a gaussian function
    # create an array with 500 equally separated values in the x axis interval
    lnspc = np.linspace(x.min()- sigma1, x.max() + sigma1, 500) 
    # create an array with f(x) values, one for each of the above points
    # normalize properly the function such that integral from -inf to +inf is the total number of events
    norm_factor = x.size * binsize
    f_gaus = norm_factor*stats.norm.pdf(lnspc,mean1,sigma1)  
    # draw the function
    if ux != "":
        plt.plot(lnspc, f_gaus, linewidth=1, color='r',linestyle='--', label = f"Gaussiana\n$\mu = {mean1}$ "+ux+f"\n$\sigma = {sigma1}$ "+ux)
        plt.xlabel(xlabel+" ["+ux+"]")
    else:
        plt.plot(lnspc, f_gaus, linewidth=1, color='r',linestyle='--', label = f"$\mu = {mean1}$\n$\sigma = {sigma1}$")
        plt.xlabel(xlabel)

    plt.legend()

    tot = x

    skewness = np.sum((tot - tot.mean())**3) / (len(tot) * sigmax**3)

    print(f"La skewness di questo istogramma è: {skewness:.2f}") #gamma 1

    curtosi = np.sum((tot - tot.mean())**4) / (len(tot) * sigmax**4) - 3  # momento terzo - 3, vedi wikipedia

    print(f"La curtosi di questo istogramma è: {curtosi:.2f}")

def residuals(x_data, y_data, y_att, sy, N, xlabel, ux = "", uy = "", marker = "d", xscale = 0, yscale = 0, confidence = 2, norm = True, legendloc = None, newstyle = True, log = None):
    """
    Grafica i residui normalizzati.

    Parameters
    ----------
    x_data : array-like
        Valori misurati per la variabile indipendente.
    y_data : array-like
        Valori misurati per la variabile dipendente.
    y_att : array-like
        Valori previsti dal modello per la variabile dipendente.
    sy : array-like
        Incertezze della variabile dipendente misurata.
    N : int, opzionale     
        Numero di parametri liberi del modello. Può essere `None`
    xlabel : str          
        Nome della variabile indipendente.
    ux : str 
        Unità di misura della variabile indipendente. Default è `""`.
    uy : str
        Unità di misura della variabile indipendente. Default è `""`.
    marker : str
        Marker del residuo. Applicabile solo se `newstyle = False`.
    xscale : int
        Fattore di scala (10^xscale) dell'asse x (es. xscale = -2 se si vuole passare da m a cm). 
    yscale : int
        Fattore di scala (10^yscale) dell'asse y (es. yscale = -2 se si vuole passare da m a cm). 
    confidence : int
        Definisce l'intervallo di confidenza `[-confidence, +confidence]`. Deve essere un numero positivo. Default è `2`.
    norm : bool
        Se `True`, i residui saranno normalizzarti. Default è `True`. 
    legendloc : str
        Posizionamento della legenda nel grafico ('upper right', 'lower left', 'upper center' etc.). Default è `None`.
    newstyle : bool
        Stile alternativo per il plot.
    log : bool
        Se `x` l'asse x sarà in scala logaritmica. Default è `None`.

    Returns
    ----------
    None

    Notes
    ----------
    Se `N = None`, allora non verranno visualizzati i valori di χ²/dof e p-value.
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
        label1 = "Residui normalizzati"
        bar1 = np.repeat(1, len(x_data))
        bar2 = resid_norm
        dash = np.repeat(confidence, len(x1))
    else :
        label1 = "Residui"
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
        plt.errorbar(x_data, bar2, bar1, ls='', color='gray', lw=1., label = f"Intervallo di confidenza $[-{confidence},\,{confidence}]$")
        plt.plot(x_data, bar2, color='k', drawstyle='steps-mid', lw=1., label = label)
        plt.plot(x1, dash, ls='dashed', color='crimson', lw=1.)
        plt.plot(x1, -dash, ls='dashed', color='crimson', lw=1.)
        plt.ylim(-np.nanmean(3 * bar1 * confidence/2), np.nanmean(3 * bar1 * confidence/2))
        if norm == False:
            if uy != "":
                plt.ylabel("Residui"+" ["+uy+"]")
            else:
                plt.xlabel("Residui")
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

    print(f"Percentuale di residui compatibili con zero: {n*100:.1f}%")

def remove_outliers(data, data_err=None, expected=None, method="zscore", threshold=3.0):
    """
    Rimuove outlier da un array di dati secondo il metodo specificato.

    Parameters
    ----------
        data : array-like
            Dati osservati.
        data_err : array-like, opzionale
            Incertezze sui dati. Necessario se si vuole confrontare con `'expected'`.
        expected : array-like, opzionale
            Valori attesi per i dati. Se forniti, viene usato automaticamente il metodo `'zscore'`.
        method : str, opzionale
            Metodo da usare (`"zscore"`, `"mad"` o `"iqr"`). Default: `"zscore"`.
        threshold : float, opzionale
            Valore soglia per identificare gli outlier. Default: `3.0`.

    Returns
    ----------
        data_clean : ndarray
            Dati senza outlier.
    """
    data = np.asarray(data)

    # Caso 1: confronto con expected → forza 'zscore'
    if expected is not None:
        if data_err is None:
            raise ValueError("Se fornisci 'expected', devi fornire anche 'data_err'.")
        
        expected = np.asarray(expected)
        data_err = np.asarray(data_err)

        if len(data) != len(expected) or len(data) != len(data_err):
            raise ValueError("'data', 'expected' e 'data_err' devono avere la stessa lunghezza.")

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
            raise ValueError("Metodo non riconosciuto. Usa 'zscore', 'mad' o 'iqr'.")

    return data[mask]