"""
LabToolbox: A scientific data analysis package
==============================================

Documentation is available in the docstrings and on GitHub:
https://github.com/giusesorrentino/LabToolbox

Submodules
----------
::

 utils         --- Utility functions
 fit           --- Curve fitting tools
 stats         --- Statistical and probabilistic analysis
 signals       --- Signal processing routines
 uncertainty   --- Propagation of uncertainty

Public API in the LabToolbox namespace
--------------------------------------
::

 PrintResult   --- Nicely formats numbers
 convert       --- Units converter
"""

# Public API
from .utils import PrintResult, convert

__all__ = ['PrintResult', 'convert']

# Available submodules
from . import signals
from . import utils
from . import fit
from . import stats
from . import uncertainty

# Public symbols
__all__ = [
    'signals',
    'utils',
    'fit',
    'stats',
    'uncertainty',
]