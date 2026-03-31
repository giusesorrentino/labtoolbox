"""Signal processing functions for LabToolbox.

This submodule provides FFT/IFFT, Fourier series, and envelope extraction
utilities for time series data.
"""

from .signals import fft, ifft, fourier_series, envelope

# Removed functions kept commented for reference.
# from .signals import dfs, harmonic, decompose

__all__ = [
    "fft", 
    "ifft", 
    "fourier_series", 
    "envelope"
]

# Removed functions kept commented for reference.
# __all__.extend([
#     "dfs",
#     "harmonic",
#     "decompose",
# ])
