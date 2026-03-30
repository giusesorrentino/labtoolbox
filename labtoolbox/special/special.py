import numpy as _np
from typing import Union, Tuple, List, Callable
from numpy.typing import ArrayLike

def sgn(x: Union[float, ArrayLike]) -> Union[float, ArrayLike]:
    """
    Compute the sign function of the input.

    Parameters
    ----------
    x : float or numpy.ndarray
        Input value(s), scalar or 1D array of real numbers.

    Returns
    -------
    float or numpy.ndarray
        The sign of `x`. Returns a scalar if `x` is scalar, otherwise a 1D array.

    Raises
    ------
    TypeError
        If `x` contains non-real numbers.
    ValueError
        If `x` is not a scalar or 1D array, or contains non-finite values.

    Examples
    --------
    >>> from labtoolbox.special import sgn
    >>> sgn(3.0)
    1.0
    >>> sgn([-2, 0, 1])
    array([-1.,  0.,  1.])
    """
    x = _np.asarray(x, dtype=float)
    if not _np.all(_np.isreal(x)):
        raise TypeError("'x' must contain only real numbers.")
    if x.ndim > 1:
        raise ValueError("'x' must be a scalar or 1D array.")
    if not _np.all(_np.isfinite(x)):
        raise ValueError("'x' contains non-finite values (NaN or inf).")
    
    result = _np.where(x < 0, -1.0, _np.where(x > 0, 1.0, 0.0))
    return result.item() if result.size == 1 else result

def rect(x: Union[float, ArrayLike]) -> Union[float, ArrayLike]:
    """
    Compute the rectangular function (boxcar function).

    Parameters
    ----------
    x : float or numpy.ndarray
        Input value(s), scalar or 1D array of real numbers.

    Returns
    -------
    float or numpy.ndarray
        The rectangular function evaluated at `x`. Returns a scalar if `x` is scalar,
        otherwise a 1D array.

    Raises
    ------
    TypeError
        If `x` contains non-real numbers.
    ValueError
        If `x` is not a scalar or 1D array, or contains non-finite values.

    Examples
    --------
    >>> from labtoolbox.special import rect
    >>> rect(0.4)
    1.0
    >>> rect([-0.6, 0.5, 0.0])
    array([0. , 0.5, 1. ])
    """
    x = _np.asarray(x, dtype=float)
    if not _np.all(_np.isreal(x)):
        raise TypeError("'x' must contain only real numbers.")
    if x.ndim > 1:
        raise ValueError("'x' must be a scalar or 1D array.")
    if not _np.all(_np.isfinite(x)):
        raise ValueError("'x' contains non-finite values (NaN or inf).")
    
    abs_x = _np.abs(x)
    result = _np.where(abs_x < 0.5, 1.0, _np.where(abs_x == 0.5, 0.5, 0.0))
    return result.item() if result.size == 1 else result

def tri(x: Union[float, ArrayLike]) -> Union[float, ArrayLike]:
    """
    Compute a triangle pulse. Not to be confused with `labtoolbox.triangle`.

    Parameters
    ----------
    x : float or numpy.ndarray
        Input value(s), scalar or 1D array of real numbers.

    Returns
    -------
    float or numpy.ndarray
        The triangle wave evaluated at x. Returns a scalar if `x` is scalar,
        otherwise a 1D array.

    Raises
    ------
    TypeError
        If `x` contains non-real numbers.
    ValueError
        If `x` is not a scalar or 1D array, or contains non-finite values.

    Examples
    --------
    >>> from labtoolbox.special import tri
    >>> tri(0.25)
    0.75
    >>> tri([0, 0.25, 0.5])
    array([1, 0.75, 0.5])
    """
    x = _np.asarray(x, dtype=float)
    if not _np.all(_np.isreal(x)):
        raise TypeError("'x' must contain only real numbers.")
    if x.ndim > 1:
        raise ValueError("'x' must be a scalar or 1D array.")
    if not _np.all(_np.isfinite(x)):
        raise ValueError("'x' contains non-finite values (NaN or inf).")
    
    result = rect(x/2) * (1 - _np.abs(x))
    return result.item() if result.size == 1 else result

def saw(x: Union[float, ArrayLike], T: float = 1.0) -> Union[float, ArrayLike]:
    """
    Compute the sawtooth wave function.

    Parameters
    ----------
    x : float or numpy.ndarray
        Input value(s), scalar or 1D array of real numbers.
    T : float, optional
        Period of the sawtooth wave. Must be positive. Defaults to `1.0`.

    Returns
    -------
    float or numpy.ndarray
        The sawtooth wave evaluated at `x`. Returns a scalar if `x` is scalar,
        otherwise a 1D array.

    Raises
    ------
    TypeError
        If `x` contains non-real numbers.
    ValueError
        If `x` is not a scalar or 1D array, contains non-finite values, or `T` is not positive.

    Examples
    --------
    >>> from labtoolbox.special import saw
    >>> saw(0.25, T=1.0)
    0.5
    >>> saw([0, 0.5, 1.0], T=1.0)
    array([ 0.,  1., -1.])
    """
    x = _np.asarray(x, dtype=float)
    if not _np.all(_np.isreal(x)):
        raise TypeError("'x' must contain only real numbers.")
    if x.ndim > 1:
        raise ValueError("'x' must be a scalar or 1D array.")
    if not _np.all(_np.isfinite(x)):
        raise ValueError("'x' contains non-finite values (NaN or inf).")
    
    if not isinstance(T, (int, float)) or not _np.isfinite(T):
        raise TypeError("'T' must be a finite real number.")
    if T <= 0:
        raise ValueError("'T' must be positive.")
    
    result = 2 * (x / T - _np.floor(0.5 + x / T))
    return result.item() if result.size == 1 else result

def triangle(x: Union[float, ArrayLike], T: float = 1.0) -> Union[float, ArrayLike]:
    """
    Compute the triangle wave function.

    Parameters
    ----------
    x : float or numpy.ndarray
        Input value(s), scalar or 1D array of real numbers.
    T : float, optional
        Period of the triangle wave. Must be positive. Defaults to `1.0`.

    Returns
    -------
    float or numpy.ndarray
        The triangle wave evaluated at `x`. Returns a scalar if `x` is scalar,
        otherwise a 1D array.

    Raises
    ------
    TypeError
        If `x` contains non-real numbers or `T` is not a finite real number.
    ValueError
        If `x` is not a scalar or 1D array, contains non-finite values, or `T` is not positive.

    Examples
    --------
    >>> from labtoolbox.special import triangle
    >>> triangle(0.25)
    1.0
    >>> triangle([0, 0.25, 0.5])
    array([0., 1., 0.])
    """
    x = _np.asarray(x, dtype=float)
    if not _np.all(_np.isreal(x)):
        raise TypeError("'x' must contain only real numbers.")
    if x.ndim > 1:
        raise ValueError("'x' must be a scalar or 1D array.")
    if not _np.all(_np.isfinite(x)):
        raise ValueError("'x' contains non-finite values (NaN or inf).")
    
    if not isinstance(T, (int, float)) or not _np.isfinite(T):
        raise TypeError("'T' must be a finite real number.")
    if T <= 0:
        raise ValueError("'T' must be positive.")
    
    result = 2 * _np.abs(x / T - _np.floor(0.5 + x / T))
    return result.item() if result.size == 1 else result

def square(x: Union[float, ArrayLike], T: float = 1.0) -> Union[float, ArrayLike]:
    """
    Compute the square wave function.

    The function is periodic with period `T`, alternating between -1 and 1 based
    on the sign of sin(2 * pi * x / T).

    Parameters
    ----------
    x : float or numpy.ndarray
        Input value(s), scalar or 1D array of real numbers.
    T : float, optional
        Period of the square wave. Must be positive. Defaults to `1.0`.

    Returns
    -------
    float or numpy.ndarray
        The square wave evaluated at `x`. Returns a scalar if `x` is scalar,
        otherwise a 1D array.

    Raises
    ------
    TypeError
        If `x` contains non-real numbers or `T` is not a finite real number.
    ValueError
        If `x` is not a scalar or 1D array, contains non-finite values, or `T` is not positive.

    Examples
    --------
    >>> from labtoolbox.special import square
    >>> square(0.25)
    1.0
    >>> square([0, 0.5, 1.0])
    array([ 1., -1.,  1.])
    """
    x = _np.asarray(x, dtype=float)
    if not _np.all(_np.isreal(x)):
        raise TypeError("'x' must contain only real numbers.")
    if x.ndim > 1:
        raise ValueError("'x' must be a scalar or 1D array.")
    if not _np.all(_np.isfinite(x)):
        raise ValueError("'x' contains non-finite values (NaN or inf).")
    
    if not isinstance(T, (int, float)) or not _np.isfinite(T):
        raise TypeError("'T' must be a finite real number.")
    if T <= 0:
        raise ValueError("'T' must be positive.")
    
    result = sgn(_np.sin(2 * _np.pi * x / T))
    return result.item() if result.size == 1 else result

def step(x: Union[float, ArrayLike]) -> Union[float, ArrayLike]:
    """
    Compute the Heaviside step function.

    Returns `0` for `x < 0`, 1 for `x >= 0`.

    Parameters
    ----------
    x : float or numpy.ndarray
        Input value(s), scalar or 1D array of real numbers.

    Returns
    -------
    float or numpy.ndarray
        The step function evaluated at `x`. Returns a scalar if `x` is scalar,
        otherwise a 1D array.

    Raises
    ------
    TypeError
        If `x` contains non-real numbers.
    ValueError
        If `x` is not a scalar or 1D array, or contains non-finite values.

    Examples
    --------
    >>> from labtoolbox.special import step
    >>> step(-1.0)
    0.0
    >>> step([-1, 0, 1])
    array([0., 1., 1.])
    """
    x = _np.asarray(x, dtype=float)
    if not _np.all(_np.isreal(x)):
        raise TypeError("'x' must contain only real numbers.")
    if x.ndim > 1:
        raise ValueError("'x' must be a scalar or 1D array.")
    if not _np.all(_np.isfinite(x)):
        raise ValueError("'x' contains non-finite values (NaN or inf).")
    
    result = _np.where(x < 0, 0.0, 1.0)
    return result.item() if result.size == 1 else result

def lorentz(x: Union[float, ArrayLike], x0: float = 0.0, g: float = 1.0) -> Union[float, ArrayLike]:
    """
    Compute the Lorentzian (Cauchy) distribution function.

    Parameters
    ----------
    x : float or numpy.ndarray
        Input value(s), scalar or 1D array of real numbers.
    x0 : float, optional
        Center of the Lorentzian distribution. Defaults to `0.0`.
    g : float, optional
        Width parameter. Must be positive. Defaults to `1.0`.

    Returns
    -------
    float or numpy.ndarray
        The Lorentzian function evaluated at `x`. Returns a scalar if `x` is scalar,
        otherwise a 1D array.

    Raises
    ------
    TypeError
        If `x` contains non-real numbers, or `x0` or `g` are not finite real numbers.
    ValueError
        If `x` is not a scalar or 1D array, contains non-finite values, or `g` is not positive.

    Examples
    --------
    >>> from labtoolbox.special import lorentz
    >>> lorentz(0.0, x0=0.0, g=1.0)
    0.3183098861837907
    >>> lorentz([0, 1, 2], x0=0.0, g=1.0)
    array([0.31830989, 0.15915494, 0.06366198])
    """
    x = _np.asarray(x, dtype=float)
    if not _np.all(_np.isreal(x)):
        raise TypeError("'x' must contain only real numbers.")
    if x.ndim > 1:
        raise ValueError("'x' must be a scalar or 1D array.")
    if not _np.all(_np.isfinite(x)):
        raise ValueError("'x' contains non-finite values (NaN or inf).")
    
    if not isinstance(x0, (int, float)) or not _np.isfinite(x0):
        raise TypeError("'x0' must be a finite real number.")
    if not isinstance(g, (int, float)) or not _np.isfinite(g):
        raise TypeError("'g' must be a finite real number.")
    if g <= 0:
        raise ValueError("'g' must be positive.")
    
    result = (g / ((x - x0)**2 + g**2)) / _np.pi
    return result.item() if result.size == 1 else result

# def indicator(point: Union[float, Tuple[float, float], List[float]], 
#               domain_type: str, 
#               domain_params: Union[Tuple, List, Callable], 
#               return_type: str = "bool") -> Union[bool, int]:
#     """
#     Determine if a point belongs to a specified domain, returning a boolean or integer indicator.

#     Parameters
#     ----------
#         point : float, tuple, or list
#             The point to check for domain membership. 
#             - For 1D: a float (e.g., `x`).
#             - For 2D: a tuple of two floats (e.g., `(x, y)`).
#             - For nD: a list of floats (e.g., `[x1, x2, ..., xn]`).
#         domain_type : str
#             The type of domain. Supported values:
#             - 'interval': 1D interval [a, b].
#             - 'rectangle': 2D rectangle [a, b] x [c, d].
#             - 'ball': nD ball defined by center and radius.
#             - 'custom': user-defined domain via a callable function.
#         domain_params : tuple, list, or callable
#             Parameters defining the domain, depending on domain_type:
#             - For 'interval': tuple (a, b) where a < b.
#             - For 'rectangle': tuple ((a, b), (c, d)) where a < b, c < d.
#             - For 'ball': tuple (center, radius) where center is a list of n floats, radius is a positive float.
#             - For 'custom': a callable that takes a point (float, tuple, or list) and returns `True` or `False`.
#         return_type : str, optional
#             The type of return value. Options:
#             - 'bool': returns `True` or `False` (default).
#             - 'int': returns `1` or `0`.

#     Returns
#     -------
#         bool or int
#             Indicator of domain membership:
#             - `True` or `1` if the point is in the domain.
#             - `False` or `0` if the point is not in the domain.
#     """
#     # Validate return_type
#     if return_type not in ["bool", "int"]:
#         raise TypeError("'return_type' must be 'bool' or 'int'.")

#     # Validate domain_type
#     valid_domain_types = ["interval", "rectangle", "ball", "custom"]
#     if domain_type not in valid_domain_types:
#         raise ValueError(f"'domain_type' must be one of {valid_domain_types}.")

#     # Validate point type and get dimension
#     if isinstance(point, (int, float)):
#         dim = 1
#         point = float(point)
#     elif isinstance(point, tuple) and len(point) == 2 and all(isinstance(x, (int, float)) for x in point):
#         dim = 2
#         point = tuple(float(x) for x in point)
#     elif isinstance(point, list) and all(isinstance(x, (int, float)) for x in point):
#         dim = len(point)
#         point = [float(x) for x in point]
#     else:
#         raise ValueError("'point' must be a float (1D), tuple of two floats (2D), or list of floats (nD).")

#     try:
#         if domain_type == "interval":
#             # Validate dimension
#             if dim != 1:
#                 raise ValueError("interval domain requires a 1D point (float).")
            
#             # Validate domain_params
#             if not isinstance(domain_params, (tuple, list)) or len(domain_params) != 2:
#                 raise TypeError("'domain_params' for interval must be a tuple (a, b).")
#             a, b = domain_params
#             if not all(isinstance(x, (int, float)) for x in [a, b]):
#                 raise ValueError("'domain_params' for interval must contain numeric bounds.")
#             if a >= b:
#                 raise ValueError("interval must satisfy a < b.")
            
#             # Check membership
#             result = a <= point <= b

#         elif domain_type == "rectangle":
#             # Validate dimension
#             if dim != 2:
#                 raise ValueError("rectangle domain requires a 2D point (tuple of two floats).")
            
#             # Validate domain_params
#             if not isinstance(domain_params, (tuple, list)) or len(domain_params) != 2:
#                 raise TypeError("'domain_params' for rectangle must be a tuple ((a, b), (c, d)).")
#             if not all(isinstance(interval, (tuple, list)) and len(interval) == 2 for interval in domain_params):
#                 raise ValueError("'domain_params' for rectangle must contain two intervals.")
#             (a, b), (c, d) = domain_params
#             if not all(isinstance(x, (int, float)) for x in [a, b, c, d]):
#                 raise ValueError("'domain_params' for rectangle must contain numeric bounds.")
#             if a >= b or c >= d:
#                 raise ValueError("rectangle intervals must satisfy a < b and c < d.")
            
#             # Check membership
#             x, y = point
#             result = a <= x <= b and c <= y <= d

#         elif domain_type == "ball":
#             # Validate domain_params
#             if not isinstance(domain_params, (tuple, list)) or len(domain_params) != 2:
#                 raise ValueError("'domain_params' for ball must be a tuple (center, radius).")
#             center, radius = domain_params
#             if not isinstance(center, list) or not all(isinstance(x, (int, float)) for x in center):
#                 raise TypeError("center for ball must be a list of floats.")
#             if not isinstance(radius, (int, float)) or radius <= 0:
#                 raise TypeError("radius for ball must be a positive float.")
#             if len(center) != dim:
#                 raise TypeError(f"center dimension ({len(center)}) must match point dimension ({dim}).")
            
#             # Compute Euclidean distance
#             center = _np.array(center, dtype=float)
#             point_array = _np.array(point, dtype=float) if dim > 1 else _np.array([point], dtype=float)
#             distance = _np.sqrt(_np.sum((point_array - center) ** 2))
            
#             # Check membership
#             result = distance <= radius

#         elif domain_type == "custom":
#             # Validate domain_params
#             if not callable(domain_params):
#                 raise TypeError("'domain_params' for custom domain must be a callable function.")
            
#             # Evaluate custom indicator function
#             try:
#                 result = domain_params(point)
#                 if not isinstance(result, bool):
#                     raise ValueError("Custom indicator function must return a boolean.")
#             except Exception as e:
#                 raise ValueError(f"Error evaluating custom indicator function: {str(e)}")

#         # Convert result to desired return_type
#         if return_type == "int":
#             return 1 if result else 0
#         return result

#     except Exception as e:
#         raise ValueError(f"Error processing indicator: {str(e)}")