"""
LabToolbox: A scientific data analysis package
==============================================

Documentation is available in the docstrings and on GitHub:
https://github.com/giusesorrentino/labtoolbox

Submodules
----------
::

 numerical     --- 
 signals       --- Signal processing routines
 stats         --- Statistical and probabilistic analysis
 utils         --- Utility functions

Public API in the LabToolbox namespace
--------------------------------------
::

 PrintResult   --- Nicely formats numbers
 convert       --- Units converter
 average       --- 
"""

# Public API
from .utils import PrintResult, convert
from .stats import average

# Available submodules
from . import signals
from . import utils
from . import fit
from . import stats
from . import uncertainty
from . import linalg
from . import optics
from . import numerical
from . import special

# Public symbols
__all__ = [
    'signals',
    'utils',
    'fit',
    'stats',
    'uncertainty',
    'linalg',
    'optics',
    'numerical',
    'special',
    'PrintResult',
    'convert',
    'average',
]

# --- Version checker ---

import warnings
import importlib.metadata
import json
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