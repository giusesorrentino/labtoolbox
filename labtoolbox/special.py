import numpy as _np
from typing import Union, Tuple, List, Callable

def dirichlet(f, g, n):
    """
    Compute the Dirichlet convolution (f * g)(n) for one or more positive integers n.

    The Dirichlet convolution of arithmetic functions f and g at each integer n is defined as:
        (f * g)(n) = sum_{d | n} f(d) * g(n // d)

    Parameters
    ----------
    f : callable
        An arithmetic function f: N -> R, defined for positive integers.
    g : callable
        An arithmetic function g: N -> R, defined for positive integers.
    n : int or array_like of int
        Positive integer(s) at which to evaluate the Dirichlet convolution.

    Returns
    -------
    float or np.ndarray
        The value(s) of the Dirichlet convolution (f * g)(n). The output is a float for scalar input,
        or an array of floats for array input.

    Raises
    ------
    TypeError
        If f or g is not callable.
    ValueError
        If any n is not a positive integer.
    """
    if not callable(f):
        raise TypeError("'f' must be a callable function.")
    if not callable(g):
        raise TypeError("'g' must be a callable function.")

    n = _np.atleast_1d(n)

    if not _np.issubdtype(n.dtype, _np.integer) or not _np.all(n > 0):
        raise ValueError("All elements in 'n' must be positive integers.")

    results = _np.zeros_like(n, dtype=float)
    for i, ni in enumerate(n):
        total = 0.0
        for d in range(1, ni + 1):
            if ni % d == 0:
                total += f(d) * g(ni // d)
        results[i] = total

    return results[0] if results.size == 1 else results

def sgn(x):
    x = _np.asarray(x)
    result = _np.where(x < 0, -1, _np.where(x > 0, 1, 0))
    return result.item() if result.size == 1 else result

def rect(x):
    x = _np.asarray(x)
    abs_x = _np.abs(x)
    result = _np.where(abs_x < 0.5, 1.0,
                      _np.where(abs_x == 0.5, 0.5, 0.0))
    return result.item() if result.size == 1 else result

def saw(x, T = 1):
    return 2 * (x / T - _np.floor(0.5 + x / T))

def triangle(x, T = 1):
     return 2 * _np.abs((x / T - _np.floor(0.5 + x / T)))

def square(x, T = 1):
    return sgn(_np.sin(2 * _np.pi * x / T))

def step(x):
    x = _np.asarray(x)
    result = _np.where(x < 0, 0, 1)
    return result.item() if result.size == 1 else result

class IndicatorError(Exception):
    """Custom exception for errors in the indicator function."""
    pass

def indicator(point: Union[float, Tuple[float, float], List[float]], 
              domain_type: str, 
              domain_params: Union[Tuple, List, Callable], 
              return_type: str = "bool") -> Union[bool, int]:
    """
    Determine if a point belongs to a specified domain, returning a boolean or integer indicator.

    Parameters
    ----------
        point : float, tuple, or list
            The point to check for domain membership. 
            - For 1D: a float (e.g., x).
            - For 2D: a tuple of two floats (e.g., (x, y)).
            - For nD: a list of floats (e.g., [x1, x2, ..., xn]).
        domain_type : str
            The type of domain. Supported values:
            - 'interval': 1D interval [a, b].
            - 'rectangle': 2D rectangle [a, b] x [c, d].
            - 'ball': nD ball defined by center and radius.
            - 'custom': user-defined domain via a callable function.
        domain_params : tuple, list, or callable
            Parameters defining the domain, depending on domain_type:
            - For 'interval': tuple (a, b) where a < b.
            - For 'rectangle': tuple ((a, b), (c, d)) where a < b, c < d.
            - For 'ball': tuple (center, radius) where center is a list of n floats, radius is a positive float.
            - For 'custom': a callable that takes a point (float, tuple, or list) and returns `True` or `False`.
        return_type : str, optional
            The type of return value. Options:
            - 'bool': returns `True` or `False` (default).
            - 'int': returns `1` or `0`.

    Returns
    -------
        bool or int
            Indicator of domain membership:
            - True or 1 if the point is in the domain.
            - False or 0 if the point is not in the domain.
    """
    # Validate return_type
    if return_type not in ["bool", "int"]:
        raise IndicatorError("return_type must be 'bool' or 'int'.")

    # Validate domain_type
    valid_domain_types = ["interval", "rectangle", "ball", "custom"]
    if domain_type not in valid_domain_types:
        raise IndicatorError(f"domain_type must be one of {valid_domain_types}.")

    # Validate point type and get dimension
    if isinstance(point, (int, float)):
        dim = 1
        point = float(point)
    elif isinstance(point, tuple) and len(point) == 2 and all(isinstance(x, (int, float)) for x in point):
        dim = 2
        point = tuple(float(x) for x in point)
    elif isinstance(point, list) and all(isinstance(x, (int, float)) for x in point):
        dim = len(point)
        point = [float(x) for x in point]
    else:
        raise IndicatorError("point must be a float (1D), tuple of two floats (2D), or list of floats (nD).")

    try:
        if domain_type == "interval":
            # Validate dimension
            if dim != 1:
                raise IndicatorError("interval domain requires a 1D point (float).")
            
            # Validate domain_params
            if not isinstance(domain_params, (tuple, list)) or len(domain_params) != 2:
                raise IndicatorError("domain_params for interval must be a tuple (a, b).")
            a, b = domain_params
            if not all(isinstance(x, (int, float)) for x in [a, b]):
                raise IndicatorError("domain_params for interval must contain numeric bounds.")
            if a >= b:
                raise IndicatorError("interval must satisfy a < b.")
            
            # Check membership
            result = a <= point <= b

        elif domain_type == "rectangle":
            # Validate dimension
            if dim != 2:
                raise IndicatorError("rectangle domain requires a 2D point (tuple of two floats).")
            
            # Validate domain_params
            if not isinstance(domain_params, (tuple, list)) or len(domain_params) != 2:
                raise IndicatorError("domain_params for rectangle must be a tuple ((a, b), (c, d)).")
            if not all(isinstance(interval, (tuple, list)) and len(interval) == 2 for interval in domain_params):
                raise IndicatorError("domain_params for rectangle must contain two intervals.")
            (a, b), (c, d) = domain_params
            if not all(isinstance(x, (int, float)) for x in [a, b, c, d]):
                raise IndicatorError("domain_params for rectangle must contain numeric bounds.")
            if a >= b or c >= d:
                raise IndicatorError("rectangle intervals must satisfy a < b and c < d.")
            
            # Check membership
            x, y = point
            result = a <= x <= b and c <= y <= d

        elif domain_type == "ball":
            # Validate domain_params
            if not isinstance(domain_params, (tuple, list)) or len(domain_params) != 2:
                raise IndicatorError("domain_params for ball must be a tuple (center, radius).")
            center, radius = domain_params
            if not isinstance(center, list) or not all(isinstance(x, (int, float)) for x in center):
                raise IndicatorError("center for ball must be a list of floats.")
            if not isinstance(radius, (int, float)) or radius <= 0:
                raise IndicatorError("radius for ball must be a positive float.")
            if len(center) != dim:
                raise IndicatorError(f"center dimension ({len(center)}) must match point dimension ({dim}).")
            
            # Compute Euclidean distance
            center = _np.array(center, dtype=float)
            point_array = _np.array(point, dtype=float) if dim > 1 else _np.array([point], dtype=float)
            distance = _np.sqrt(_np.sum((point_array - center) ** 2))
            
            # Check membership
            result = distance <= radius

        elif domain_type == "custom":
            # Validate domain_params
            if not callable(domain_params):
                raise IndicatorError("domain_params for custom domain must be a callable function.")
            
            # Evaluate custom indicator function
            try:
                result = domain_params(point)
                if not isinstance(result, bool):
                    raise IndicatorError("custom indicator function must return a boolean.")
            except Exception as e:
                raise IndicatorError(f"Error evaluating custom indicator function: {str(e)}")

        # Convert result to desired return_type
        if return_type == "int":
            return 1 if result else 0
        return result

    except Exception as e:
        raise IndicatorError(f"Error processing indicator: {str(e)}")