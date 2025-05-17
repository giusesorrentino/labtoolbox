import numpy as np
from inspect import signature

def boole(f, a, b, n = None, varname = None, *, max_step = 0.1, **kwargs):
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
        Number of Boole segments. Must be >=1. If not provided, an optimal value is estimated
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

    # Infer variable name if not given
    if varname is None:
        try:
            sig = signature(f)
            pos_params = [p.name for p in sig.parameters.values() if p.kind in (p.POSITIONAL_OR_KEYWORD, p.POSITIONAL_ONLY)]
            if len(pos_params) == 1:
                varname = pos_params[0]
            else:
                raise ValueError("Unable to infer integration variable name. Pass it explicitly via varname.")
        except Exception:
            raise ValueError("Function signature inspection failed. Pass varname explicitly.")

    # Determine number of segments
    if n is None:
        total_length = abs(b - a)
        n = max(1, int(np.ceil(total_length / (4 * max_step))))
    if n < 1 or n != int(n):
        raise ValueError("n must be a positive integer.")
    n = int(n)

    # Compute integration points
    h = (b - a) / (4 * n)
    x = np.linspace(a, b, 4 * n + 1)

    # Evaluate f at the integration points
    y = np.array([f(**{varname: xi}, **kwargs) for xi in x])

    # Apply Boole's weights in blocks of 5 points
    total = 0.0
    for i in range(n):
        j = 4 * i
        weights = np.array([7, 32, 12, 32, 7])
        segment = y[j:j+5]
        total += np.dot(weights, segment)

    return (2 * h / 45) * total

def romberg(f, a, b, *, varname=None, tol=1e-8, max_iter=10, **kwargs):
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
        Name of the integration variable. If None, it's inferred automatically (only if f has one arg).
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

    # Infer varname if not provided
    if varname is None:
        try:
            sig = signature(f)
            pos_args = [p.name for p in sig.parameters.values()
                        if p.kind in (p.POSITIONAL_OR_KEYWORD, p.POSITIONAL_ONLY)]
            if len(pos_args) == 1:
                varname = pos_args[0]
            else:
                raise ValueError("Unable to infer variable of integration; provide 'varname' explicitly.")
        except Exception:
            raise ValueError("Could not inspect function signature. Pass 'varname' manually.")

    def eval_f(x):
        return f(**{varname: x}, **kwargs)

    # Romberg integration table
    R = np.zeros((max_iter, max_iter))
    h = b - a

    # First row: trapezoid rule
    R[0, 0] = 0.5 * h * (eval_f(a) + eval_f(b))

    for k in range(1, max_iter):
        h /= 2
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
    print(f"Warning: Romberg integration did not converge after {max_iter} iterations.")
    return R[max_iter - 1, max_iter - 1]

def newton(f, x0, *, fprime=None, varname=None, tol=1e-10, maxiter=50, dx=1e-6, **kwargs):
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
        Name of the variable with respect to which we take the root. Required if f has multiple arguments.
    tol : float, optional
        Absolute tolerance for convergence. Default is 1e-10.
    maxiter : int, optional
        Maximum number of iterations. Default is 50.
    dx : float, optional
        Step size for numerical differentiation. Default is 1e-6.
    **kwargs
        Additional keyword arguments passed to f (and fprime if provided).

    Returns
    -------
    float
        Approximated root of the function.

    Raises
    ------
    RuntimeError
        If the method fails to converge within `maxiter` iterations.
    """

    # Infer variable name if needed
    if varname is None:
        try:
            sig = signature(f)
            pos_args = [p.name for p in sig.parameters.values()
                        if p.kind in (p.POSITIONAL_OR_KEYWORD, p.POSITIONAL_ONLY)]
            if len(pos_args) == 1:
                varname = pos_args[0]
            else:
                raise ValueError("Unable to infer variable of integration; provide 'varname' explicitly.")
        except Exception:
            raise ValueError("Could not inspect function signature. Pass 'varname' manually.")

    def eval_f(x):
        return f(**{varname: x}, **kwargs)

    def eval_df(x):
        if fprime:
            return fprime(**{varname: x}, **kwargs)
        else:
            # Central finite difference
            return (eval_f(x + dx) - eval_f(x - dx)) / (2 * dx)

    x = x0
    for i in range(maxiter):
        fx = eval_f(x)
        dfx = eval_df(x)

        if dfx == 0:
            raise ZeroDivisionError(f"Derivative is zero at iteration {i} (x = {x}).")

        dx_newton = fx / dfx
        x_new = x - dx_newton

        if abs(dx_newton) < tol:
            return x_new

        x = x_new

    raise RuntimeError(f"Newton-Raphson did not converge after {maxiter} iterations.")