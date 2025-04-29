import emcee
import corner
from lmfit import Model, Parameters
from LabToolbox import np, plt, curve_fit

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