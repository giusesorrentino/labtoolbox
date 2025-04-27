# Import principali
import numpy as np
import math
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Moduli specifici di scipy
from scipy import stats
from scipy.stats import norm, chi2, multivariate_normal
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline

# Importazioni esplicite da moduli interni
from .basics import *  # Importa tutte le funzioni dal modulo basics
from .fit import *     # Importa tutte le funzioni dal modulo fit
from .posterior import *  # Importa tutte le funzioni dal modulo posterior
from .uncertainty import *  # Importa tutte le funzioni dal modulo uncertainty
from .uncertainty_class import *  # Importa tutte le funzioni dal modulo uncertainty_class

# Esportazione automatica dei membri globali
__all__ = [
    'np', 'math', 'plt', 'sm', 'stats', 'norm', 'chi2', 'multivariate_normal',
    'curve_fit', 'UnivariateSpline'
]