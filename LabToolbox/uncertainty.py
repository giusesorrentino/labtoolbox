from LabToolbox import np
from uncertainty_class import uncert_prop

def uncertainty_diff(f, x_vars, sigma_x, params=()):
    """
    Propagazione dell'incertezza tramite derivate numeriche.

    Parameters
    ----------
        f : callable
            Funzione `f(x1, ..., xn; a1, ..., am)`, restituisce array shape (N,).
        x_vars : list of np.ndarray
            Lista degli array `x1,..., xn` ciascuno di shape (N,).
        sigma_x : list of np.ndarray
            Lista delle incertezze corrispondenti per ogni `x_i`, shape (N,).
        params : tuple
            Parametri `(a1, ..., am)` passati alla funzione.

    Returns
    ----------
        f_vals : np.ndarray
            Valori centrali della funzione, shape (N,).
        f_std : np.ndarray
            Incertezza propagata, shape (N,).
    """
    N = x_vars[0].shape[0]
    n_vars = len(x_vars)

    # Valori centrali della funzione
    f_central = f(*x_vars, *params)
    f_var = np.zeros(N)

    for i in range(n_vars):
        x = x_vars[i]

        # Calcolo h_i come distanza minima tra punti consecutivi diviso 100
        dx = np.diff(x)
        min_dx = np.min(np.abs(dx[dx != 0])) if np.any(dx != 0) else 1.0
        h = min_dx / 100

        # Copia degli array per ±h
        x_plus = [x.copy() for x in x_vars]
        x_minus = [x.copy() for x in x_vars]
        x_plus[i]  += h
        x_minus[i] -= h

        f_plus = f(*x_plus, *params)
        f_minus = f(*x_minus, *params)

        df_dxi = (f_plus - f_minus) / (2 * h)
        f_var += (df_dxi * sigma_x[i])**2

    f_std = np.sqrt(f_var)

    return f_central, f_std

def propagate_uncertainty(func, x_arrays, uncertainties, params = None, method='Delta', MC_sample_size = 10000):
    """
    Propaga l'incertezza dagli array di input ad una generica funzione utilizzando la liberia `uncertainty_class`.
    
    Parameters
    ----------
    func : callable
        La funzione base, nella forma `f(x,a)` dove:
        - `x` è un vettore di variabili.
        - `a` è un vettore di parametri (opzionale).
        
    x_arrays : list of numpy.ndarray
        Lista contenente gli array delle variabili di input `[x1, x2, ..., xn]`.
        Ogni `xi` deve avere la stessa lunghezza.
        
    uncertainties : list or numpy.ndarray
        Lista delle incertezze per ciascuna variabile, o matrice di covarianza completa.
        Se è una lista di incertezze sigma, verrà costruita una matrice di covarianza diagonale.
        
    params : list or numpy.ndarray, optional
        Lista o array dei parametri costanti `[a1, a2, ..., am]`.
        
    method : str, optional
        Metodo di propagazione dell'incertezza ('Delta' o 'Monte_Carlo').
        
    MC_sample_size : int, optional
        Dimensione del campione per il metodo Monte Carlo.
        
    Returns
    --------
        f_values : numpy.ndarray
            Valori della funzione calcolati in ogni punto `j`.
        f_uncertainties : numpy.ndarray
            Incertezze propagate sulla funzione di output per ogni punto `j`.
        confidence_bands : tuple of numpy.ndarray
            Bande di confidenza inferiore e superiore per ogni punto `j`.

    Notes
    --------
    https://github.com/yiorgoskost/Uncertainty-Propagation/tree/master
    """

    # Verifica che tutti gli array di input abbiano la stessa lunghezza
    n_points = len(x_arrays[0])
    for i, x in enumerate(x_arrays[1:], 1):
        if len(x) != n_points:
            raise ValueError(f"L'array di input x{i+1} ha una lunghezza diversa dagli altri")
    
    # Inizializza gli array di output
    f_values = np.zeros(n_points)
    f_uncertainties = np.zeros(n_points)
    confidence_bands_lower = np.zeros(n_points)
    confidence_bands_upper = np.zeros(n_points)
    
    # Prepara la funzione wrapper che accetta un vettore di variabili
    def wrapped_func(x_vector):
        if params is not None:
            return func(*[x_vector[i] for i in range(len(x_vector))], *params)
        else:
            return func(*[x_vector[i] for i in range(len(x_vector))])
    
    # Per ogni punto j, calcola f[j] e la sua incertezza
    for j in range(n_points):
        # Estrai i valori per il punto j
        x_point = np.array([x[j] for x in x_arrays])
        
        # Prepara la matrice di covarianza
        if isinstance(uncertainties, list):
            # Se uncertainties è una lista di incertezze per ogni variabile
            if all(isinstance(u, (int, float)) for u in uncertainties):
                # Se sono scalari, crea una matrice diagonale
                cov_matrix = np.diag([u**2 for u in uncertainties])
            else:
                # Se sono array, prendi il valore per il punto j
                cov_matrix = np.diag([u[j]**2 for u in uncertainties])
        else:
            # Assume che uncertainties sia già una matrice di covarianza
            cov_matrix = uncertainties
            
        # Crea l'oggetto uncert_prop
        uncertainty_propagator = uncert_prop(
            func=wrapped_func,
            x=x_point,
            cov_matrix=cov_matrix,
            method=method,
            MC_sample_size=MC_sample_size
        )
        
        # Calcola il valore della funzione
        f_values[j] = wrapped_func(x_point)
        
        # Calcola l'incertezza propagata
        f_uncertainties[j] = uncertainty_propagator.SEM()
        
        # Calcola le bande di confidenza
        lcb, ucb = uncertainty_propagator.confband()
        confidence_bands_lower[j] = lcb
        confidence_bands_upper[j] = ucb
    
    return f_values, f_uncertainties, (confidence_bands_lower, confidence_bands_upper)