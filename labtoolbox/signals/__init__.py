"""Signal processing functions for LabToolbox.

This submodule provides FFT/IFFT, Fourier series, harmonic detection,
component decomposition, and envelope extraction utilities for time series data.
"""

from .signals import fft, ifft, dfs, fourier_series, harmonic, decompose, envelope

__all__ = [
    "fft", 
    "ifft", 
    "dfs", 
    "fourier_series", 
    "harmonic", 
    "decompose", 
    "envelope"
]