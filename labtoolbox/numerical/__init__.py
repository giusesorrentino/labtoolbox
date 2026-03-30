"""Numerical methods of LabToolbox.

Includes integration, root finding, and interpolation utilities.
"""

from .numerical import boole, romberg, newton, lagrange

__all__ = [
    "boole", 
    "romberg", 
    "newton", 
    "lagrange"
]