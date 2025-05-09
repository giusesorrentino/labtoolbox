import numpy as np
import matplotlib.pyplot as plt
import LabToolbox as lab
from LabToolbox.misc import PrintResult

# >-------------------------

x = 1.264382
sigmax = 0.357

PrintResult(x, sigmax, name = "Corrente", ux = "A")

# >-------------------------

np.random.seed(0)
x = np.random.normal(100, 7, size = (50))

lab.misc.histogram(x, sigmax = x.std(), ux = "g", xlabel = "Massa di una mela")

# >-------------------------

# Generazione dei dati di esempio
np.random.seed(0)  # Per rendere l'esempio riproducibile
x = np.linspace(0, 10, 20)  # 20 punti equidistanti tra 0 e 10
y = 2.5 * x + 1.5 + np.random.normal(0, 2, size=x.shape)  # dati di esempio

np.random.seed(1)
sy = np.random.uniform(1.5, 2.5, size=len(y))  # Incertezze tra ±1.5 e ±2.5

m, c, sigma_m, sigma_c, chi2_red, p_value = lab.fit.lin_fit(x = x, y = y, sy = sy, sx = None, 
                                                            fitmodel = "wls", residuals = True)

# >-------------------------

# Generazione dei dati di esempio
np.random.seed(0)  # Per rendere l'esempio riproducibile
x = np.linspace(0, 10, 20)  # 20 punti equidistanti tra 0 e 10
y = np.sin(2*x) + 1.5 + np.random.normal(0, 2, size=x.shape)  # dati di esempio

np.random.seed(1)
sy = np.random.uniform(1.5, 2.5, size=len(y))  # Incertezze tra ±1.5 e ±2.5

def funzione(x, a, b):
    return np.sin(a * x) + b

# ho definito la funzione secondo la quale dovrebbero distribuirsi i dati.
# 'a' e 'b' sono parametri liberi del fit, vanno messi dopo la variabile indipendente nell'argomento della funzione.
# la funzine `model_fit` permette una stima dei parametri 'a' e 'b'.

popt, perr, chi2_red, p_value = lab.fit.model_fit(x, y, sy, f = funzione, p0 = [1, 2])


# >------------------------- DA TESTARE

# lab.misc.remove_outliers()

# >-------------------------

np.random.seed(0)
h = np.random.normal(5, 0.25, size = (10))
r = np.random.normal(7, 0.20, size = (10))

np.random.seed(1)
sh = np.random.uniform(0.15, 0.25, size=len(h))  # Incertezze tra ±1.5 e ±2.5
sr = np.random.uniform(0.1, 0.3, size=len(h))  # Incertezze tra 1 e 3

V = np.pi * r**2 * h 

# ho misurato 10 volte due determinate variabile (es: altezza di un cilindro e raggio in cm), ognuna con incertezza sh e sr.
# la variabile V è il volume di un cilindro (in cm^3).
# se ora voglio propagare le incertezze di h ed r su V, posso usare la funzione `propagate_uncertainty`.

def volume(altezza, raggio):
    return np.pi * raggio**2 * altezza**2

# ho definito la funzione che permette di calcolare il volume a partire dal raggio e dall'altezza.

V_value, sigma_V, _ = lab.uncertainty.propagate_uncertainty(func = volume, x_arrays=[h, r], uncertainties = [sh, sr], params = None)

# ulteriori descrizioni delle variabili di `propagate_uncertainty` sono disponibili trascinando il cursore sopra la funzione stessa.

for i in range(len(V_value)):
    PrintResult(V_value[i], sigma_V[i], name = f"Volume (misura {i+1}-esima)", ux = "cm^3")

# >------------------------- DA TESTARE

# lab.fit.bootstrap_fit()