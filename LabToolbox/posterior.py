import emcee
import corner
from lmfit import Model, Parameters
from LabToolbox import np, plt, curve_fit

def posterior(x, y, sy, f, p0, burn=1000, steps=5000, thin=10, maxfev=5000):
    """
    Analisi bayesiana con emcee per il fitting di una funzione a molti parametri. 
    La funzione esegue un'analisi MCMC (Markov Chain Monte Carlo) per ottenere una distribuzione posteriore dei parametri, 
    quindi calcola i parametri di massima verosimiglianza (MLE) e visualizza la corner plot dei risultati.

    Parameters
    ----------
        x : array-like
            Valori misurati per la variabile indipendente.
        y : array-like
            Valori misurati per la variabile dipendente (da adattare al modello).
        sy : array-like
            Incertezze sulle misure della variabile dipendente.
        f : function
            Funzione modello da adattare ai dati. La funzione deve accettare una variabile indipendente 
            `x` come primo argomento e i parametri liberi come argomenti successivi.
        p0 : list
            Lista dei valori iniziali dei parametri liberi del modello. 
            Esempio: [a0, b0, c0], dove ogni elemento corrisponde al valore iniziale del parametro.
        burn : int, opzionale
            Numero di "passi di burn-in" per escludere i primi campioni della catena di Markov 
            che potrebbero essere correlati (default è 1000).
        steps : int, opzionale
            Numero totale di passi per la catena di Markov (default è 5000).
        thin : int, opzionale
            Fattore di sottocampionamento (default è 10), per ridurre la correlazione tra i campioni.
        maxfev : int
            Numero massimo di iterazioni della funzione `curve_fit`.

    Returns
    ----------
        res.params : Params
            Oggetto contenente i parametri ottimizzati e le incertezze sui parametri ottenuti.
        res.flatchain : array-like
            Catena appiattita dei campioni MCMC, utile per l'analisi statistica.

    La funzione visualizza una corner plot dei parametri posteriore e stampa i valori della mediana e delle incertezze 
    sui parametri, insieme ai risultati della massima verosimiglianza (MLE).

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