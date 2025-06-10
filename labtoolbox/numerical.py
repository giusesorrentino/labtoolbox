import numpy as _np
from numpy.typing import ArrayLike
import warnings
from inspect import signature as _signature
from typing import Callable, Tuple, List, Union, Optional
from ._helper import GenericError

def boole(f: Callable[[Union[float, ArrayLike]], Union[float, ArrayLike]], 
          a: float, b: float, n: Optional[int] = None, 
          varname: Optional[str] = None, max_step: float = 0.1, **kwargs) -> float:
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
        Maximum width of a Boole segment. Only used if `n` is not provided. Default is `0.1`.
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
        warnings.warn("Integration limits 'a' and 'b' have been swapped.", UserWarning)

    if a == b:
        warnings.warn("Warning: Integration limits 'a' and 'b' are equal; integral is zero.")
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

def romberg(f: Callable[[Union[float, ArrayLike]], Union[float, ArrayLike]], a: float, b: float, 
            varname: Optional[str] = None, tol: float = 1e-8, max_iter: int = 10, **kwargs) -> float:
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
        Name of the integration variable. If `None`, it's inferred automatically (only if `f` has one arg).
    tol : float, optional
        Desired absolute tolerance. Default is `1e-8`.
    max_iter : int, optional
        Maximum number of Romberg iterations. Default is `10`.
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
        warnings.warn("Integration limits 'a' and 'b' have been swapped.", UserWarning)

    if a == b:
        warnings.warn("Warning: Integration limits 'a' and 'b' are equal; integral is zero.")
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
            warnings.warn(f"Step size h = {h} is below machine precision; stopping early.", UserWarning)
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
    warnings.warn(f"Romberg integration did not converge after {max_iter} iterations.", UserWarning)
    return R[max_iter - 1, max_iter - 1]

def newton(f: Callable[[Union[float, ArrayLike]], Union[float, ArrayLike]], 
           x0: float, fprime: Optional[Callable[[Union[float, ArrayLike]], Union[float, ArrayLike]]] = None, 
           varname: Optional[str] = None, tol: float = 1e-10, maxiter: int = 50, dx: float = 1e-6, **kwargs) -> float:
    """
    Find the root of a scalar function using the Newton-Raphson method.

    Parameters
    ----------
    f : callable
        Function whose root is to be found. Must accept the variable of interest as a named argument.
    x0 : float
        Initial guess.
    fprime : callable, optional
        Derivative function. If `None`, numerical differentiation is used.
    varname : str, optional
        Name of the variable with respect to which we take the root. Required if `f` has multiple arguments.
    tol : float, optional
        Absolute tolerance for convergence. Default is `1e-10`.
    maxiter : int, optional
        Maximum number of iterations. Default is `50`.
    dx : float, optional
        Step size for numerical differentiation. Default is `1e-6`.
    **kwargs
        Additional keyword arguments passed to `f` (and `fprime` if provided).

    Returns
    -------
    float
        Approximated root of the function.
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

def lebesgue(func: Callable[[Union[float, ArrayLike]], Union[float, ArrayLike]], 
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
            The integration interval `[a, b]`, where a < b.
        num_samples : int, optional
            Number of Monte Carlo samples. Defaults to `10000`.
        use_quasi_mc : bool, optional
            If `True`, uses Quasi-Monte Carlo (Sobol sequence) for sampling. Defaults to `True`.
        indicator_func : callable, optional
            A function that returns `True` if a point is in the integration domain, `False` otherwise.
            If `None`, the entire interval `[a, b]` is used. Defaults to `None`.
        return_error : bool, optional
            If `True`, returns a tuple (integral, error_estimate). Defaults to `False`.

    Returns
    -------
        float or tuple
            The approximate value of the Lebesgue integral. If `return_error` is `True`, returns `(integral, error_estimate)`.
    """

    from scipy.stats import qmc

    # Input validation
    if not callable(func):
        raise TypeError("'func' must be callable.")
    if not isinstance(interval, (tuple, list)) or len(interval) != 2:
        raise TypeError("'interval' must be a tuple of two floats (a, b).")
    a, b = interval
    if not (isinstance(a, (int, float)) and isinstance(b, (int, float))):
        raise ValueError("'interval' bounds must be numeric.")
    if a >= b:
        raise ValueError("'interval' must satisfy a < b.")
    if not isinstance(num_samples, int) or num_samples <= 0:
        raise ValueError("'num_samples' must be a positive integer.")
    if indicator_func is not None and not callable(indicator_func):
        raise TypeError("'indicator_func' must be callable.")

    # Compute the Lebesgue measure of the interval
    measure = b - a

    # Generate samples
    if use_quasi_mc:
        sampler = qmc.Sobol(d=1, scramble=True)
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
            raise ValueError("Function evaluations contain non-finite values.")

        # Compute Monte Carlo estimate
        integral = measure * _np.mean(values)
        
        # Compute error estimate if requested
        if return_error:
            variance = _np.var(values)
            error_estimate = _np.sqrt(variance / num_samples) * measure
            return integral, error_estimate
        
        return integral
    except Exception as e:
        raise GenericError(f"Error during integration: {str(e)}")

def dblebesgue(func: Callable[[Union[float, ArrayLike]], Union[float, ArrayLike]], 
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
            The integrand function `f(x, y)`, which must be Lebesgue measurable.
        domain : tuple
            The integration domain `([a, b], [c, d])`.
        num_samples : int, optional
            Number of Monte Carlo samples. Defaults to `10000`.
        use_quasi_mc : bool, optional
            If `True`, uses Quasi-Monte Carlo (Sobol sequence) for sampling. Defaults to `True`.
        indicator_func : callable, optional
            A function that returns True if a point `(x, y)` is in the integration domain, False otherwise.
            If `None`, the entire domain `([a, b], [c, d])` is used. Defaults to `None`.
        return_error : bool, optional
            If `True`, returns a tuple `(integral, error_estimate)`. Defaults to `False`.

    Returns:
    ----------
        float or tuple
            The approximate value of the Lebesgue integral. If `return_error` is `True`, returns `(integral, error_estimate)`.
    """
    from scipy.stats import qmc
    # Input validation
    if not callable(func):
        raise TypeError("'func' must be callable.")
    if not isinstance(domain, (tuple, list)) or len(domain) != 2:
        raise TypeError("'domain' must be a tuple of two intervals ([a, b], [c, d]).")
    (a, b), (c, d) = domain
    if not all(isinstance(x, (int, float)) for x in [a, b, c, d]):
        raise ValueError("'domain' bounds must be numeric.")
    if a >= b or c >= d:
        raise ValueError("'domain' intervals must satisfy a < b and c < d.")
    if not isinstance(num_samples, int) or num_samples <= 0:
        raise ValueError("'num_samples' must be a positive integer.")
    if indicator_func is not None and not callable(indicator_func):
        raise TypeError("'indicator_func' must be callable.")

    # Compute the Lebesgue measure of the domain
    measure = (b - a) * (d - c)

    # Generate samples
    if use_quasi_mc:
        sampler = qmc.Sobol(d=2, scramble=True)
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
            raise ValueError("Function evaluations contain non-finite values.")

        # Compute Monte Carlo estimate
        integral = measure * _np.mean(values)
        
        # Compute error estimate if requested
        if return_error:
            variance = _np.var(values)
            error_estimate = _np.sqrt(variance / num_samples) * measure
            return integral, error_estimate
        
        return integral
    except Exception as e:
        raise GenericError(f"Error during integration: {str(e)}")

def nlebesgue(func: Callable[[Union[float, ArrayLike]], Union[float, ArrayLike]], 
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
            The integrand function `f(x_1, ..., x_n)`, which must be Lebesgue measurable.
        domain : list
            List of intervals `[(a_1, b_1), ..., (a_n, b_n)]` defining the domain.
        num_samples : int, optional 
            Number of Monte Carlo samples. Defaults to `10000`.
        use_quasi_mc : bool, optional
            If `True`, uses Quasi-Monte Carlo (Sobol sequence) for sampling. Defaults to `True`.
        indicator_func : callable, optional
            A function that returns `True` if a point `(x_1, ..., x_n)` is in the integration domain, `False` otherwise.
            If `None`, the entire domain is used. Defaults to `None`.
        return_error : bool, optional
            If `True`, returns a tuple `(integral, error_estimate)`. Defaults to `False`.

    Returns:
    ----------
        float or tuple
            The approximate value of the Lebesgue integral. If `return_error` is `True`, returns `(integral, error_estimate)`.  
    """
    from scipy.stats import qmc
    # Input validation
    if not callable(func):
        raise TypeError("'func' must be callable.")
    if not isinstance(domain, (list, tuple)) or not domain:
        raise TypeError("'domain' must be a non-empty list of intervals.")
    if not all(isinstance(interval, (tuple, list)) and len(interval) == 2 for interval in domain):
        raise ValueError("Each domain entry must be a tuple of two floats.")
    if not all(isinstance(x, (int, float)) for interval in domain for x in interval):
        raise ValueError("'domain' bounds must be numeric.")
    if not all(a < b for a, b in domain):
        raise ValueError("Each interval must satisfy a < b.")
    if not isinstance(num_samples, int) or num_samples <= 0:
        raise ValueError("'num_samples' must be a positive integer.")
    if indicator_func is not None and not callable(indicator_func):
        raise TypeError("'indicator_func' must be callable.")

    # Compute the Lebesgue measure of the domain
    measure = _np.prod([b - a for a, b in domain])

    # Generate samples
    n_dims = len(domain)
    if use_quasi_mc:
        sampler = qmc.Sobol(d=n_dims, scramble=True)
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
            raise ValueError("Function evaluations contain non-finite values.")

        # Compute Monte Carlo estimate
        integral = measure * _np.mean(values)
        
        # Compute error estimate if requested
        if return_error:
            variance = _np.var(values)
            error_estimate = _np.sqrt(variance / num_samples) * measure
            return integral, error_estimate
        
        return integral
    except Exception as e:
        raise GenericError(f"Error during integration: {str(e)}")

def lagrange(f: Callable[[Union[float, ArrayLike]], Union[float, ArrayLike]], 
             constraints: List[Callable[[ArrayLike], float]], 
             x0: ArrayLike, 
             tol: float = 1e-6) -> Tuple[ArrayLike, ArrayLike]:
    """
    Solve a constrained optimization problem using Lagrange multipliers.

    Parameters
    ----------
        f : callable
            The objective function `f`. Must return a scalar (float).
        constraints : list of callables
            List of constraint functions `[g_1(x), ..., g_m(x)]`, where each `g_i(x)`
            takes a 1D numpy array and returns a scalar (float).
            Each g_i(x) = 0 defines a constraint.
        x0 : array_like
            Initial guess for the solution, a 1D array of length `n` (number of variables).
        tol : float, optional
            Tolerance for the solver (used in `scipy.optimize.fsolve`).
            Defaults to `1e-6`.

    Returns
    -------
        tuple of (ndarray, ndarray)
            - x_opt: The optimal point (1D array of length `n`).
            - lambda_opt: The Lagrange multipliers (1D array of length `m`, where `m` is the number of constraints).

    Notes
    -----
    The solver may not converge for poorly conditioned problems or bad initial guesses.

    Examples
    --------
    >>> from labtoolbox.numerical import lagrange
    >>> import numpy as np
    >>> # Minimize f(x, y) = x^2 + y^2 subject to x + y = 1
    >>> f = lambda x: x[0]**2 + x[1]**2
    >>> constraints = [lambda x: x[0] + x[1] - 1]
    >>> x0 = np.array([0.5, 0.5])
    >>> x_opt, lambda_opt = lagrange(f, constraints, x0)
    >>> print(x_opt, lambda_opt)  # Expected: x_opt ≈ [0.5, 0.5], lambda_opt ≈ [1.0]
    [0.5 0.5] [1.00000002]
    """

    from scipy.optimize import fsolve, approx_fprime
    
    # Validate f
    if not callable(f):
        raise TypeError("'f' must be a callable function.")

    # Validate constraints
    if not isinstance(constraints, list) or not all(callable(g) for g in constraints):
        raise TypeError("'constraints' must be a list of callable functions.")
    if not constraints:
        raise ValueError("'constraints' list cannot be empty.")

    # Validate x0
    x0 = _np.asarray(x0, dtype=float)
    if x0.ndim != 1 or not _np.all(_np.isfinite(x0)):
        raise ValueError("'x0' must be a 1D array of finite values.")

    # Validate tol
    if not isinstance(tol, (int, float)) or tol <= 0 or not _np.isfinite(tol):
        raise ValueError("'tol' must be a positive finite float.")

    n = len(x0)  # Number of variables
    m = len(constraints)  # Number of constraints

    def system(vars: ArrayLike) -> ArrayLike:
        """System of equations: ∇f = Σ λ_i ∇g_i, g_i = 0."""
        x = vars[:n]  # Variables
        lam = vars[n:]  # Lagrange multipliers
        # Compute ∇f
        grad_f = approx_fprime(x, f, epsilon=1e-8)
        # Compute Σ λ_i ∇g_i
        grad_g_sum = _np.zeros(n)
        for i, g in enumerate(constraints):
            grad_g = approx_fprime(x, g, epsilon=1e-8)
            grad_g_sum += lam[i] * grad_g
        # Equations: ∇f - Σ λ_i ∇g_i = 0
        eq1 = grad_f - grad_g_sum
        # Equations: g_i(x) = 0
        eq2 = _np.array([g(x) for g in constraints])
        return _np.concatenate([eq1, eq2])

    # Initial guess: x0 for variables, zeros for multipliers
    initial_guess = _np.concatenate([x0, _np.zeros(m)])

    try:
        # Solve the system
        solution = fsolve(system, initial_guess, xtol=tol)
        if not _np.all(_np.isfinite(solution)):
            raise ValueError("'scipy.optimize.fsolve' returned non-finite values.")
        x_opt = solution[:n]
        lambda_opt = solution[n:]
        # Verify constraints
        constraint_vals = _np.array([g(x_opt) for g in constraints])
        if not _np.all(_np.abs(constraint_vals) < tol * 10):
            raise ValueError("Solution does not satisfy constraints within tolerance.")
        return x_opt, lambda_opt
    
    except Exception as e:
        raise GenericError(
            message=f"Error solving Lagrange multipliers: {str(e)}",
            context="executing lagrange",
            original_error=e,
            details={"x0_shape": x0.shape, "n_constraints": m}
        )