import numpy as _np
import warnings as _warnings
from inspect import signature as _signature
from typing import Callable, Tuple, List, Union, Optional

def boole(f, a, b, n = None, varname = None, max_step = 0.1, **kwargs):
    """
    Approximate the definite integral of a function using Boole's Rule.

    Parameters
    ----------
    f : callable
        The function to integrate. Must accept the integration variable as a named argument.
    a : float
        Lower limit of integration.
    b : float
        Upper limit of integration.
    n : int, optional
        Number of Boole segments. Must be equal or greater than 1. If not provided, an optimal value is estimated
        to ensure segment width ≤ max_step.
    varname : str, optional
        Name of the integration variable as expected by `f`. If not provided and `f` is a lambda
        or function with one positional argument, it is inferred automatically.
    max_step : float, optional
        Maximum width of a Boole segment. Only used if `n` is not provided. Default is 0.1.
    **kwargs
        Additional parameters passed to `f`.

    Returns
    -------
    float
        Approximation of the definite integral using Boole's Rule.
    """

    if not callable(f):
        raise TypeError("'f' must be a callable function.")

    if not isinstance(a, (int, float)):
        raise TypeError("'a' must be a real number (int or float).")
    if not isinstance(b, (int, float)):
        raise TypeError("'b' must be a real number (int or float).")
    
    if b < a:
        a, b = b, a
        _warnings.warn("Integration limits 'a' and 'b' have been swapped.", UserWarning)

    if a == b:
        _warnings.warn("Warning: Integration limits 'a' and 'b' are equal; integral is zero.")
        return 0.0

    if n is not None and n < 1:
        raise ValueError("'n' must be a positive integer.")

    # Infer variable name if not given
    if varname is None:
        try:
            sig = _signature(f)
            pos_params = [p.name for p in sig.parameters.values() if p.kind in (p.POSITIONAL_OR_KEYWORD, p.POSITIONAL_ONLY)]
            if len(pos_params) == 1:
                varname = pos_params[0]
            else:
                raise ValueError("Unable to infer integration variable name. Pass it explicitly via varname.")
        except Exception:
            raise ValueError("Function signature inspection failed. Pass varname explicitly.")

    if max_step <= 0:
        raise ValueError("'max_step' must be positive.")

    # Determine number of segments
    if n is None:
        total_length = abs(b - a)
        n = max(1, int(_np.ceil(total_length / (4 * max_step))))
    else:
        if not isinstance(n, int) or n < 1:
            raise ValueError("'n' must be a positive integer.")

    n = int(n)

    # Compute integration points
    h = (b - a) / (4 * n)
    x = _np.linspace(a, b, 4 * n + 1)

    if len(x) < 5:
        raise ValueError("The number of points must be at least 5 for Boole's rule.")

    vector_f = _np.vectorize(lambda xi: f(**{varname: xi}, **kwargs))
    y = vector_f(x)

    if not _np.all(_np.isfinite(y)):
        raise ValueError("Function evaluations contain non-finite values (NaN or inf).")

    # Apply Boole's weights in blocks of 5 points
    total = 0.0
    for i in range(n):
        j = 4 * i
        weights = _np.array([7, 32, 12, 32, 7])
        segment = y[j:j+5]
        total += _np.dot(weights, segment)

    return (2 * h / 45) * total

def BooleDiscrete(x, y):
    """
    Approximate the definite integral of discrete data using Boole's Rule.

    Parameters
    ----------
    x : array-like
        Array of x-coordinates (must be equispaced and sorted in ascending order).
    y : array-like
        Array of y-coordinates (function values at x points).

    Returns
    -------
    float
        Approximation of the definite integral using Boole's Rule.
    """
    x = _np.asarray(x)
    y = _np.asarray(y)

    if len(x) != len(y):
        raise ValueError("'x' and 'y' must have the same length.")
    if len(x) < 5:
        raise ValueError("At least 5 points are required for Boole's Rule.")
    
    if (not (_np.issubdtype(x.dtype, _np.floating) or _np.issubdtype(x.dtype, _np.integer))) or not _np.all(_np.isreal(x)):
            raise TypeError("'x' must contain only real numbers (int or float).")
    if (not (_np.issubdtype(y.dtype, _np.floating) or _np.issubdtype(y.dtype, _np.integer))) or not _np.all(_np.isreal(y)):
            raise TypeError("'y' must contain only real numbers (int or float).")
    
    if not _np.all(_np.isfinite(x)):
                raise ValueError("'x' contains non-finite values (NaN or inf).")
    if not _np.all(_np.isfinite(y)):
                raise ValueError("'y' contains non-finite values (NaN or inf).")

    if not _np.all(x[:-1] < x[1:]):
        _warnings.warn("'x' was not sorted in ascending order; sorting automatically.", UserWarning)
        sorted_indices = _np.argsort(x)
        x = x[sorted_indices]
        y = y[sorted_indices]
    
    h = x[1] - x[0]
    if not _np.allclose(_np.diff(x), h):
        raise ValueError("'x' must be equispaced.")

    n = (len(x) - 1) // 4  # Number of Boole segments
    # remainder = (len(x)-1) % 4

    # if remainder != 0:
    #     _warnings.warn(
    #         f"The number of points ({len(x)}) does not satisfy the Boole's rule condition (4n + 1). "
    #         f"Only the first {4 * n + 1} points will be used; {remainder} points will be ignored. "
    #         "This may introduce a bias if the omitted data is significant.",
    #         UserWarning
    #     )

    if n < 1 or (len(x) - 1) % 4 != 0:
        _warnings.warn("Number of points does not satisfy 4n + 1; extending arrays with zeros.", UserWarning)
        points_needed = 4 * (n + 1) + 1  # Prossimo numero valido di punti (4n + 1)
        if n < 1:
            points_needed = 5  # Minimo numero di punti per Boole
        h = x[1] - x[0]  # Passo equispaziato
        x_new = _np.linspace(x[0], x[0] + (points_needed - 1) * h, points_needed)
        y_new = _np.zeros(points_needed)
        y_new[:len(y)] = y  # Copia i valori originali, il resto rimane zero
        x, y = x_new, y_new
        n = (len(x) - 1) // 4  # Aggiorna n

    # Apply Boole's Rule
    total = 0.0
    for i in range(n):
        j = 4 * i
        weights = _np.array([7, 32, 12, 32, 7])
        segment = y[j:j+5]
        total += _np.dot(weights, segment)

    return (2 * h / 45) * total

def romberg(f, a, b, varname=None, tol=1e-8, max_iter=10, **kwargs):
    """
    Perform numerical integration using Romberg's method.

    Parameters
    ----------
    f : callable
        The function to integrate. Must accept the integration variable as a named argument.
    a : float
        Lower limit of integration.
    b : float
        Upper limit of integration.
    varname : str, optional
        Name of the integration variable. If None, it's inferred automatically (only if `f` has one arg).
    tol : float, optional
        Desired absolute tolerance. Default is 1e-8.
    max_iter : int, optional
        Maximum number of Romberg iterations. Default is 10.
    **kwargs
        Additional keyword arguments passed to `f`.

    Returns
    -------
    float
        Approximation of the integral.
    """

    if not callable(f):
        raise TypeError("'f' must be a callable function.")

    if not isinstance(a, (int, float)):
        raise TypeError("'a' must be a real number (int or float).")
    if not isinstance(b, (int, float)):
        raise TypeError("'b' must be a real number (int or float).")
    
    if b < a:
        a, b = b, a
        _warnings.warn("Integration limits 'a' and 'b' have been swapped.", UserWarning)

    if a == b:
        _warnings.warn("Warning: Integration limits 'a' and 'b' are equal; integral is zero.")
        return 0.0
    
    if not isinstance(tol, (int, float)) or tol <= 0:
         raise ValueError("'tol' must be a positive real number (int or float).")
    
    if not isinstance(max_iter, int) or max_iter < 1:
        raise ValueError("'max_iter' must be a real number (int or float) greater than 1.")

    # Infer varname if not provided
    if varname is None:
        try:
            sig = _signature(f)
            pos_args = [p.name for p in sig.parameters.values()
                        if p.kind in (p.POSITIONAL_OR_KEYWORD, p.POSITIONAL_ONLY)]
            if len(pos_args) == 1:
                varname = pos_args[0]
            else:
                raise ValueError("Unable to infer variable of integration; provide 'varname' explicitly.")
        except Exception:
            raise ValueError("Could not inspect function signature. Pass 'varname' explicitly.")

    def eval_f(x):
        result = f(**{varname: x}, **kwargs)
        if not _np.isfinite(result):
            raise ValueError(f"Function evaluation at x = {x} returned non-finite value (NaN or inf).")
        return result

    # Romberg integration table
    R = _np.zeros((max_iter, max_iter))
    h = b - a

    # First row: trapezoid rule
    R[0, 0] = 0.5 * h * (eval_f(a) + eval_f(b))

    for k in range(1, max_iter):
        h /= 2
        if h < _np.finfo(float).eps:
            _warnings.warn(f"Step size h = {h} is below machine precision; stopping early.", UserWarning)
            return R[k-1, k-1]
        # Composite Trapezoid Rule refinement
        subtotal = sum(eval_f(a + (2 * i - 1) * h) for i in range(1, 2**(k-1)+1))
        R[k, 0] = 0.5 * R[k - 1, 0] + h * subtotal

        # Romberg extrapolation
        for j in range(1, k + 1):
            R[k, j] = (4**j * R[k, j - 1] - R[k - 1, j - 1]) / (4**j - 1)

        # Convergence check
        if abs(R[k, k] - R[k - 1, k - 1]) < tol:
            return R[k, k]

    # If it gets here, it didn't converge
    _warnings.warn(f"Romberg integration did not converge after {max_iter} iterations.", UserWarning)
    return R[max_iter - 1, max_iter - 1]

def RombergDiscrete(x, y, tol=1e-8, max_iter=10):
    """
    Perform numerical integration on discrete data using Romberg's method.

    Parameters
    ----------
    x : array-like
        Array of x-coordinates (must be equispaced and sorted in ascending order).
    y : array-like
        Array of y-coordinates (function values at x points).
    tol : float, optional
        Desired absolute tolerance. Default is 1e-8.
    max_iter : int, optional
        Maximum number of Romberg iterations. Default is 10.

    Returns
    -------
    float
        Approximation of the definite integral.
    """
    x = _np.asarray(x)
    y = _np.asarray(y)

    if len(x) != len(y):
        raise ValueError("'x' and 'y' must have the same length.")
    
    if (not (_np.issubdtype(x.dtype, _np.floating) or _np.issubdtype(x.dtype, _np.integer))) or not _np.all(_np.isreal(x)):
            raise TypeError("'x' must contain only real numbers (int or float).")
    if (not (_np.issubdtype(y.dtype, _np.floating) or _np.issubdtype(y.dtype, _np.integer))) or not _np.all(_np.isreal(y)):
            raise TypeError("'y' must contain only real numbers (int or float).")
    
    if not _np.all(_np.isfinite(x)):
                raise ValueError("'x' contains non-finite values (NaN or inf).")
    if not _np.all(_np.isfinite(y)):
                raise ValueError("'y' contains non-finite values (NaN or inf).")

    if len(x) < 2:
        raise ValueError("At least 2 points are required for Romberg integration.")
    if tol <= 0:
        raise ValueError("'tol' must be positive.")
    if not isinstance(max_iter, int) or max_iter < 1:
        raise ValueError("'max_iter' must be a positive integer.")

    if not _np.all(x[:-1] < x[1:]):
        _warnings.warn("'x' was not sorted in ascending order; sorting automatically.", UserWarning)
        sorted_indices = _np.argsort(x)
        x = x[sorted_indices]
        y = y[sorted_indices]

    h = x[1] - x[0]
    if not _np.allclose(_np.diff(x), h):
        raise ValueError("'x' must be equispaced.")

    R = _np.zeros((max_iter, max_iter))

    n_points = len(x)
    max_k = int(_np.floor(_np.log2(n_points - 1)))
    expected_points = 2**max_k + 1

    if n_points != expected_points:
        _warnings.warn(
            f"The number of points ({n_points}) does not satisfy the Romberg condition (2^k + 1). "
            f"Only the first {expected_points} points will be used; {n_points - expected_points} points will be ignored. "
            "This may introduce a bias if the omitted data is significant.",
            UserWarning
        )
        x = x[:expected_points]
        y = y[:expected_points]
        n_points = expected_points

    a, b = x[0], x[-1]
    h_initial = b - a

    # First row: trapezoid rule
    R[0, 0] = 0.5 * h_initial * (y[0] + y[-1])

    for k in range(1, min(max_iter, max_k + 1)):
        h = h_initial / (2**k)
        if h < _np.finfo(float).eps:
            _warnings.warn(f"Step size h = {h} is below machine precision; stopping early.", UserWarning)
            return R[k-1, k-1]

        # Composite Trapezoid Rule refinement
        step = 2**(k-1)
        indices = _np.arange(1, n_points, step)
        if indices[-1] >= n_points:
            indices = indices[:-1]
        subtotal = sum(y[i] for i in indices)
        R[k, 0] = 0.5 * R[k-1, 0] + h * subtotal

        # Romberg extrapolation
        for j in range(1, k + 1):
            R[k, j] = (4**j * R[k, j-1] - R[k-1, j-1]) / (4**j - 1)

        # Convergence check
        if abs(R[k, k] - R[k-1, k-1]) < tol:
            return R[k, k]

    _warnings.warn(f"Romberg integration did not converge after {max_iter} iterations.", UserWarning)
    return R[k-1, k-1]

def newton(f, x0, fprime=None, varname=None, tol=1e-10, maxiter=50, dx=1e-6, **kwargs):
    """
    Find the root of a scalar function using the Newton-Raphson method.

    Parameters
    ----------
    f : callable
        Function whose root is to be found. Must accept the variable of interest as a named argument.
    x0 : float
        Initial guess.
    fprime : callable, optional
        Derivative function. If None, numerical differentiation is used.
    varname : str, optional
        Name of the variable with respect to which we take the root. Required if `f` has multiple arguments.
    tol : float, optional
        Absolute tolerance for convergence. Default is 1e-10.
    maxiter : int, optional
        Maximum number of iterations. Default is 50.
    dx : float, optional
        Step size for numerical differentiation. Default is 1e-6.
    **kwargs
        Additional keyword arguments passed to `f` (and fprime if provided).

    Returns
    -------
    float
        Approximated root of the function.

    Raises
    ------
    RuntimeError
        If the method fails to converge within `maxiter` iterations.
    """

    if not callable(f):
        raise TypeError("'f' must be a callable function.")
    
    if fprime is not None and not callable(fprime):
        raise TypeError("'fprime' must be a callable function.")

    if not isinstance(x0, (int, float)):
        raise TypeError("'x0' must be a real number (int or float).")
    
    if tol <= 0:
        raise ValueError("'tol' must be positive.")
    if not isinstance(maxiter, int) or maxiter < 1:
        raise ValueError("'maxiter' must be a positive integer.")
    if dx <= 0:
        raise ValueError("'dx' must be positive for numerical differentiation.")

    # Infer variable name if needed
    if varname is None:
        try:
            sig = _signature(f)
            pos_args = [p.name for p in sig.parameters.values()
                        if p.kind in (p.POSITIONAL_OR_KEYWORD, p.POSITIONAL_ONLY)]
            if len(pos_args) == 1:
                varname = pos_args[0]
            else:
                raise ValueError("Unable to infer variable of integration; provide 'varname' explicitly.")
        except Exception:
            raise ValueError("Could not inspect function signature. Pass 'varname' manually.")

    def eval_f(x):
        result = f(**{varname: x}, **kwargs)
        if not _np.isfinite(result):
            raise ValueError(f"Function evaluation at x={x} returned non-finite value (NaN or inf).")
        return result

    def eval_df(x):
        if fprime:
            result = fprime(**{varname: x}, **kwargs)
            if not _np.isfinite(result):
                raise ValueError(f"Derivative evaluation at x={x} returned non-finite value (NaN or inf).")
            return result
        else:
            # Central finite difference
            return (eval_f(x + dx) - eval_f(x - dx)) / (2 * dx)

    x = x0

    for i in range(maxiter):
        fx = eval_f(x)
        dfx = eval_df(x)

        if dfx == 0:
            raise ZeroDivisionError(f"Derivative is zero at iteration {i} (x = {x}).")
        
        if abs(dfx) < _np.finfo(float).eps:
            print(f"Warning: Derivative is near zero at iteration {i} (x = {x}); stopping early.")
            return x
        dx_newton = fx / dfx
        x_new = x - dx_newton

        if abs(dx_newton) < tol:
            return x_new

        x = x_new

    raise RuntimeError(f"Newton-Raphson did not converge after {maxiter} iterations.")

from scipy.stats import qmc as _qmc

class LebesgueIntegrationError(Exception):
    """Custom exception for Lebesgue integration errors."""
    pass

def lebesgue(func: Callable[[float], float], 
             interval: Tuple[float, float], 
             num_samples: int = 10000, 
             use_quasi_mc: bool = True,
             indicator_func: Optional[Callable[[float], bool]] = None,
             return_error: bool = False) -> Union[float, Tuple[float, float]]:
    """
    Compute the 1D Lebesgue integral of a function over a specified interval using Monte Carlo or Quasi-Monte Carlo.

    Parameters
    ----------
        func : callable
            The integrand function, which must be Lebesgue measurable.
        interval : tuple
            The integration interval [a, b], where a < b.
        num_samples : int, optional
            Number of Monte Carlo samples. Defaults to `10000`.
        use_quasi_mc : bool, optional
            If `True`, uses Quasi-Monte Carlo (Sobol sequence) for sampling. Defaults to `True`.
        indicator_func : callable, optional
            A function that returns `True` if a point is in the integration domain, `False` otherwise.
            If `None`, the entire interval [a, b] is used. Defaults to `None`.
        return_error : bool, optional
            If `True`, returns a tuple (integral, error_estimate). Defaults to `False`.

    Returns
    -------
        float or tuple
            The approximate value of the Lebesgue integral. If `return_error` is `True`, returns `(integral, error_estimate)`.
    """
    # Input validation
    if not callable(func):
        raise LebesgueIntegrationError("The function 'func' must be callable.")
    if not isinstance(interval, (tuple, list)) or len(interval) != 2:
        raise LebesgueIntegrationError("Interval must be a tuple of two floats (a, b).")
    a, b = interval
    if not (isinstance(a, (int, float)) and isinstance(b, (int, float))):
        raise LebesgueIntegrationError("Interval bounds must be numeric.")
    if a >= b:
        raise LebesgueIntegrationError("Interval must satisfy a < b.")
    if not isinstance(num_samples, int) or num_samples <= 0:
        raise LebesgueIntegrationError("Number of samples must be a positive integer.")
    if indicator_func is not None and not callable(indicator_func):
        raise LebesgueIntegrationError("The indicator function must be callable.")

    # Compute the Lebesgue measure of the interval
    measure = b - a

    # Generate samples
    if use_quasi_mc:
        sampler = _qmc.Sobol(d=1, scramble=True)
        samples = sampler.random(n=num_samples)
        points = a + (b - a) * samples[:, 0]
    else:
        points = _np.random.uniform(a, b, num_samples)

    try:
        # Evaluate function at sample points
        values = _np.array([func(x) for x in points], dtype=float)

        # Apply indicator function if provided
        if indicator_func is not None:
            indicator_values = _np.array([indicator_func(x) for x in points], dtype=_np.bool_)
            # Estimate measure of the domain
            measure = measure * _np.mean(indicator_values)
            values = values * indicator_values

        # Check for non-finite values
        if not _np.all(_np.isfinite(values)):
            raise LebesgueIntegrationError("Function evaluations contain non-finite values.")

        # Compute Monte Carlo estimate
        integral = measure * _np.mean(values)
        
        # Compute error estimate if requested
        if return_error:
            variance = _np.var(values)
            error_estimate = _np.sqrt(variance / num_samples) * measure
            return integral, error_estimate
        
        return integral
    except Exception as e:
        raise LebesgueIntegrationError(f"Error during integration: {str(e)}")

def lebesgue2(func: Callable[[float, float], float], 
              domain: Tuple[Tuple[float, float], Tuple[float, float]], 
              num_samples: int = 10000, 
              use_quasi_mc: bool = True,
              indicator_func: Optional[Callable[[Tuple[float, float]], bool]] = None,
              return_error: bool = False) -> Union[float, Tuple[float, float]]:
    """
    Compute the 2D Lebesgue integral of a function over a rectangular domain using Monte Carlo or Quasi-Monte Carlo.

    Parameters
    ----------
        func : callable
            The integrand function f(x, y), which must be Lebesgue measurable.
        domain : tuple
            The integration domain ([a, b], [c, d]).
        num_samples : int, optional
            Number of Monte Carlo samples. Defaults to `10000`.
        use_quasi_mc : bool, optional
            If `True`, uses Quasi-Monte Carlo (Sobol sequence) for sampling. Defaults to `True`.
        indicator_func : callable, optional
            A function that returns True if a point (x, y) is in the integration domain, False otherwise.
            If `None`, the entire domain [a, b] x [c, d] is used. Defaults to `None`.
        return_error : bool, optional
            If `True`, returns a tuple `(integral, error_estimate)`. Defaults to `False`.

    Returns:
    ----------
        float or tuple
            The approximate value of the Lebesgue integral. If `return_error` is `True`, returns `(integral, error_estimate)`.
    """
    # Input validation
    if not callable(func):
        raise LebesgueIntegrationError("The function 'func' must be callable.")
    if not isinstance(domain, (tuple, list)) or len(domain) != 2:
        raise LebesgueIntegrationError("Domain must be a tuple of two intervals ([a, b], [c, d]).")
    (a, b), (c, d) = domain
    if not all(isinstance(x, (int, float)) for x in [a, b, c, d]):
        raise LebesgueIntegrationError("Domain bounds must be numeric.")
    if a >= b or c >= d:
        raise LebesgueIntegrationError("Domain intervals must satisfy a < b and c < d.")
    if not isinstance(num_samples, int) or num_samples <= 0:
        raise LebesgueIntegrationError("Number of samples must be a positive integer.")
    if indicator_func is not None and not callable(indicator_func):
        raise LebesgueIntegrationError("The indicator function must be callable.")

    # Compute the Lebesgue measure of the domain
    measure = (b - a) * (d - c)

    # Generate samples
    if use_quasi_mc:
        sampler = _qmc.Sobol(d=2, scramble=True)
        samples = sampler.random(n=num_samples)
        x_points = a + (b - a) * samples[:, 0]
        y_points = c + (d - c) * samples[:, 1]
    else:
        x_points = _np.random.uniform(a, b, num_samples)
        y_points = _np.random.uniform(c, d, num_samples)

    try:
        # Evaluate function at sample points
        values = _np.array([func(x, y) for x, y in zip(x_points, y_points)], dtype=float)

        # Apply indicator function if provided
        if indicator_func is not None:
            indicator_values = _np.array([indicator_func((x, y)) for x, y in zip(x_points, y_points)], dtype=_np.bool_)
            # Estimate measure of the domain
            measure = measure * _np.mean(indicator_values)
            values = values * indicator_values

        # Check for non-finite values
        if not _np.all(_np.isfinite(values)):
            raise LebesgueIntegrationError("Function evaluations contain non-finite values.")

        # Compute Monte Carlo estimate
        integral = measure * _np.mean(values)
        
        # Compute error estimate if requested
        if return_error:
            variance = _np.var(values)
            error_estimate = _np.sqrt(variance / num_samples) * measure
            return integral, error_estimate
        
        return integral
    except Exception as e:
        raise LebesgueIntegrationError(f"Error during integration: {str(e)}")

def nlebesgue(func: Callable[[List[float]], float], 
              domain: List[Tuple[float, float]], 
              num_samples: int = 10000, 
              use_quasi_mc: bool = True,
              indicator_func: Optional[Callable[[List[float]], bool]] = None,
              return_error: bool = False) -> Union[float, Tuple[float, float]]:
    """
    Compute the n-dimensional Lebesgue integral of a function over a rectangular domain using Monte Carlo or Quasi-Monte Carlo.

    Parameters
    ----------
        func : callable
            The integrand function f(x_1, ..., x_n), which must be Lebesgue measurable.
        domain : list
            List of intervals [(a_1, b_1), ..., (a_n, b_n)] defining the domain.
        num_samples : int, optional 
            Number of Monte Carlo samples. Defaults to `10000`.
        use_quasi_mc : bool, optional
            If `True`, uses Quasi-Monte Carlo (Sobol sequence) for sampling. Defaults to `True`.
        indicator_func : callable, optional
            A function that returns `True` if a point (x_1, ..., x_n) is in the integration domain, `False` otherwise.
            If `None`, the entire domain is used. Defaults to `None`.
        return_error : bool, optional
            If `True`, returns a tuple `(integral, error_estimate)`. Defaults to `False`.

    Returns:
    ----------
        float or tuple
            The approximate value of the Lebesgue integral. If `return_error` is `True`, returns `(integral, error_estimate)`.
    """
    # Input validation
    if not callable(func):
        raise LebesgueIntegrationError("The function 'func' must be callable.")
    if not isinstance(domain, (list, tuple)) or not domain:
        raise LebesgueIntegrationError("Domain must be a non-empty list of intervals.")
    if not all(isinstance(interval, (tuple, list)) and len(interval) == 2 for interval in domain):
        raise LebesgueIntegrationError("Each domain entry must be a tuple of two floats.")
    if not all(isinstance(x, (int, float)) for interval in domain for x in interval):
        raise LebesgueIntegrationError("Domain bounds must be numeric.")
    if not all(a < b for a, b in domain):
        raise LebesgueIntegrationError("Each interval must satisfy a < b.")
    if not isinstance(num_samples, int) or num_samples <= 0:
        raise LebesgueIntegrationError("Number of samples must be a positive integer.")
    if indicator_func is not None and not callable(indicator_func):
        raise LebesgueIntegrationError("The indicator function must be callable.")

    # Compute the Lebesgue measure of the domain
    measure = _np.prod([b - a for a, b in domain])

    # Generate samples
    n_dims = len(domain)
    if use_quasi_mc:
        sampler = _qmc.Sobol(d=n_dims, scramble=True)
        samples = sampler.random(n=num_samples)
        points = _np.array([domain[i][0] + (domain[i][1] - domain[i][0]) * samples[:, i] for i in range(n_dims)]).T
    else:
        points = _np.array([_np.random.uniform(a, b, num_samples) for a, b in domain]).T

    try:
        # Evaluate function at sample points
        values = _np.array([func(point) for point in points], dtype=float)

        # Apply indicator function if provided
        if indicator_func is not None:
            indicator_values = _np.array([indicator_func(point) for point in points], dtype=_np.bool_)
            # Estimate measure of the domain
            measure = measure * _np.mean(indicator_values)
            values = values * indicator_values

        # Check for non-finite values
        if not _np.all(_np.isfinite(values)):
            raise LebesgueIntegrationError("Function evaluations contain non-finite values.")

        # Compute Monte Carlo estimate
        integral = measure * _np.mean(values)
        
        # Compute error estimate if requested
        if return_error:
            variance = _np.var(values)
            error_estimate = _np.sqrt(variance / num_samples) * measure
            return integral, error_estimate
        
        return integral
    except Exception as e:
        raise LebesgueIntegrationError(f"Error during integration: {str(e)}")