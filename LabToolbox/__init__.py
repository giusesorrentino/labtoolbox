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

# Importazioni dinamiche dei moduli interni
import os
import importlib

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

# Percorso della cartella del pacchetto
package_dir = os.path.dirname(__file__)

# Importazione automatica dei moduli nel pacchetto
modules = ['misc', 'fit', 'uncertainty', 'posterior']

for module_name in modules:
    # Importa ogni modulo dinamicamente
    module = importlib.import_module(f'.{module_name}', package='LabToolbox')
    
    # Ottieni tutti i membri del modulo
    module_members = dir(module)
    
    # Aggiungi tutte le funzioni e classi al namespace globale, evitando membri speciali
    for member in module_members:
        if not member.startswith('__'):  # Ignora i membri speciali
            globals()[member] = getattr(module, member)