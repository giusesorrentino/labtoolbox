# Import principali
import numpy as np
import math
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Moduli specifici di scipy
import scipy.stats as spstats
from scipy.stats import norm, chi2, multivariate_normal
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline

# Importazioni esplicite da moduli interni
# from .io import *
# from .signals import *
from .utils import *
from .fit import *     # Importa tutte le funzioni dal modulo fit
from .stats import *
from .uncertainty import *  # Importa tutte le funzioni dal modulo uncertainty

# Esportazione automatica dei membri globali
__all__ = [
    'np', 'math', 'plt', 'sm', 'spstats', 'norm', 'chi2', 'multivariate_normal',
    'curve_fit', 'UnivariateSpline'
]