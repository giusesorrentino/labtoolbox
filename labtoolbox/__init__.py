"""
LabToolbox: A scientific data analysis package
==============================================

Documentation is available in the docstrings and on GitHub:
https://github.com/giusesorrentino/labtoolbox

Modules
----------
::

 numerical     --- Numerical analysis routines
 signals       --- Signal processing routines
 special       --- Special mathematical functions
 stats         --- Statistical and probabilistic analysis
 utils         --- Utility functions

Public API in the LabToolbox namespace
--------------------------------------
::
 __version__   --- 
 PrintResult   --- Pretty-print numerical results with uncertainties
 convert       --- Units converter
 mean          --- Generalised means
 genspace      --- Generalized linspace generator
"""

# labtoolbox - A Python library for scientific data analysis
# Copyright (c) 2025, Giuseppe Sorrentino
# Licensed under the BSD 3-Clause License. See LICENSE file for details.

__version__ = "3.1.0"

# Public API
from .utils import PrintResult, convert, genspace
from .stats import mean

# Available submodules
from . import signals
from . import utils
from . import fit
from . import stats
from . import uncertainty
from . import numerical
from . import special

# Public symbols
__all__ = [
    'signals',
    'utils',
    'fit',
    'stats',
    'uncertainty',
    'numerical',
    'special',
    'PrintResult',
    'convert',
    'mean',
    'genspace',
]

# --- Version checker ---

import json
import warnings
import importlib.metadata
from urllib.request import urlopen

def _check_latest_version(package_name='labtoolbox'):
    try:
        # Ottieni la versione installata
        local_version = importlib.metadata.version(package_name)

        # Ottieni la versione più recente da PyPI
        with urlopen(f"https://pypi.org/pypi/{package_name}/json") as response:
            data = json.load(response)
            latest_version = data["info"]["version"]

        if local_version != latest_version:
            warnings.warn(
                f"A new version of {package_name} is available: {latest_version} (installed: {local_version})",
                UserWarning
            )
    except Exception:
        pass

_check_latest_version()