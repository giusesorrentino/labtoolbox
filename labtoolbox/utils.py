import math
import numpy as np
from .stats import samples
from astropy import units as u
from astropy.units import UnitConversionError
from inspect import signature
from ._helper import parse_unit

# --------------------------------------------------------------------------------

def PrintResult(value, err, name = "", unit = ""):
    """
    Returns a formatted string in the "mean ± sigma" format, with sigma to two significant figures,
    and the mean rounded consistently.

    Parameters
    ----------
    value : float
        Value of the variable.
    err : float
        Uncertainty of the variable considered.
    name : str, optional
        Name of the variable to display before the value. Default is an empty string.
    unit : str, optional
        Unit of measurement to display after the value in parentheses. Default is an empty string.

    Returns
    -------
    None
        Prints the formatted string directly.
    """

    # 1. Arrotonda sigma a due cifre significative
    if err == 0:
        raise ValueError("The uncertainty cannot be zero.")
        
    exponent = int(math.floor(math.log10(abs(err))))
    factor = 10**(exponent - 1)
    rounded_sigma = round(err / factor) * factor

    # 2. Arrotonda mean allo stesso ordine di grandezza di sigma
    rounded_mean = round(value, -exponent + 1)

    # 3. Converte in stringa mantenendo zeri finali
    fmt = f".{-exponent + 1}f" if exponent < 1 else "f"
    mean_str = f"{rounded_mean:.{max(0, -exponent + 1)}f}"
    sigma_str = f"{rounded_sigma:.{max(0, -exponent + 1)}f}"

    # 4. Crea la stringa risultante
    result = ""

    # Costruzione della parte numerica
    if unit != "":
        value_part = f"({mean_str} ± {sigma_str}) {unit}"
    else:
        value_part = f"{mean_str} ± {sigma_str}"

    # Aggiunta della percentuale relativa se possibile
    if rounded_mean != 0:
        nu = rounded_sigma / rounded_mean
        value_part += f" [{np.abs(nu) * 100:.2f}%]"

    # Aggiunta del nome della variabile, se fornito
    if name != "":
        result = f"{name} = {value_part}"
    else:
        result = value_part

    print(result)

def format_str(data, err):
    """
    Formats data and uncertainties into LaTeX strings of the form "$data \pm data_err$".

    Parameters
    ----------
    data : float or array-like
        Central values.
    err : float or array-like
        Uncertainties (must be same shape as `data`).

    Returns
    -------
    list of str
        LaTeX strings like "$data \pm data_err$" with proper rounding.
    """

    data = np.atleast_1d(data)
    err = np.atleast_1d(err)

    if data.shape != err.shape:
        raise ValueError("Shapes of 'data' and 'err' must match.")

    result = []

    for d, e in zip(data, err):
        if e == 0:
            result.append(f"${d}$")
        else:
            exponent = int(np.floor(np.log10(np.abs(e))))
            factor = 10**(exponent - 1)
            rounded_sigma = round(e / factor) * factor
            rounded_mean = round(d, -exponent + 1)

            digits = max(0, -exponent + 1)
            mean_str = f"{rounded_mean:.{digits}f}"
            sigma_str = f"{rounded_sigma:.{digits}f}"
            result.append(f"${mean_str} \\pm {sigma_str}$")

    return result

def latex_table(data, header, filename, caption="", label="", align="c"):
    """
    Writes a LaTeX-formatted table to file with caption, label, and custom styling.

    Parameters
    ----------
    data : list of lists
        The content of the table, organized as a list of columns (i.e., data[i][j] is value j of column i).
    header : list of str
        List of column names to appear in the header of the table.
    filename : str
        Path to the output `.tex` file (e.g., 'table.tex').
    caption : str, optional
        Caption text of the table.
    label : str, optional
        Label used for referencing the table in LaTeX.
    align : str, optional
        Column alignment string (e.g., "lcr"). If a single character ("l", "c", or "r") is given, it is repeated for all columns.
    
    Notes
    -----
    - Assumes all elements of `data` and `header` are convertible to string.
    - Does not escape LaTeX special characters.
    - Assumes `data` is column-oriented (i.e., each sublist is a column).
    """

    if not data or len(data) != len(header):
        raise ValueError("Length of 'header' must match number of columns in 'data'.")

    n_rows = len(data[0])
    n_cols = len(header)

    for col in data:
        if len(col) != n_rows:
            raise ValueError("All columns in 'data' must have the same length.")

    # Gestione formato colonne
    if len(align) == 1:
        col_format = align * n_cols
    elif len(align) == n_cols:
        col_format = align
    else:
        raise ValueError("Length of 'align' must be 1 or equal to number of columns.")

    with open(filename, 'w') as f:
        f.write("\\begin{table}[H]\n")

        if caption or label:
            caption_parts = []
            if label:
                caption_parts.append(f"\\label{{{label}}}")
            if caption:
                caption_parts.append(f"\\!\\!{caption}")
            line = f"\\caption{{"
            if caption:
                line += "\\large "
            line += " ".join(caption_parts) + "}\n"
            f.write(line)

        f.write("\\vspace{-0.7\\baselineskip}\n")
        f.write("\\centering\n")
        f.write(f"\\begin{{tabular}}{{{col_format}}}\n")
        f.write("\\hline\\hline\n")
        f.write("\\noalign{{\\vskip 1.5pt}}\n")
        f.write(" & ".join(header) + " \\\\\n")
        f.write("\\hline\n")
        f.write("\\noalign{{\\vskip 2pt}}\n")

        for i in range(n_rows):
            row = [str(data[j][i]) for j in range(n_cols)]
            f.write(" & ".join(row) + " \\\\\n")

        f.write("\\noalign{\\vskip 1.5pt}\n")
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    
def noise(n, std):
    warn("This function is deprecated and will be removed in a future release. Consider using scipy.stats", DeprecationWarning)
    return samples(n, 'norma', mu = 0, sigma = std)

def convert(value, from_unit: str, to_unit: str):
    """
    Converts a physical quantity between units, supporting SI prefixes, non-SI units and 
    compound units.

    Parameters
    ----------
    value : float or int
        Numerical value to be converted.
    from_unit : str
        Unit of the input quantity (e.g., 'erg', 'km/s', 'eV/Å^3').
    to_unit : str
        Desired target unit (e.g., 'J', 'm/s', 'GeV/fm^3').

    Returns
    -------
    float
        The value converted to the target unit.
    """

    try:
        parsed_from = parse_unit(from_unit)
        parsed_to = parse_unit(to_unit)
        quantity = value * u.Unit(parsed_from)
        converted = quantity.to(parsed_to)
        print(f"Input:  {value} [{from_unit}]")
        print(f"Output: {converted.value} [{to_unit}]")
        return converted.value
    except UnitConversionError as e:
        raise UnitConversionError(f"Cannot convert from '{from_unit}' to '{to_unit}': {e}")
    except ValueError as e:
        raise ValueError(f"Invalid unit specified: {e}")

# -------------> INTEGRAZIONE <------------- #

# def boole(f, a, b, n = None, varname = None, *, max_step = 0.1, **kwargs):
#     """
#     Approximate the definite integral of a function using Boole's Rule.

#     Parameters
#     ----------
#     f : callable
#         The function to integrate. Must accept the integration variable as a named argument.
#     a : float
#         Lower limit of integration.
#     b : float
#         Upper limit of integration.
#     n : int, optional
#         Number of Boole segments. Must be >=1. If not provided, an optimal value is estimated
#         to ensure segment width ≤ max_step.
#     varname : str, optional
#         Name of the integration variable as expected by `f`. If not provided and `f` is a lambda
#         or function with one positional argument, it is inferred automatically.
#     max_step : float, optional
#         Maximum width of a Boole segment. Only used if `n` is not provided. Default is 0.1.
#     **kwargs
#         Additional parameters passed to `f`.

#     Returns
#     -------
#     float
#         Approximation of the definite integral using Boole's Rule.
#     """

#     # Infer variable name if not given
#     if varname is None:
#         try:
#             sig = signature(f)
#             pos_params = [p.name for p in sig.parameters.values() if p.kind in (p.POSITIONAL_OR_KEYWORD, p.POSITIONAL_ONLY)]
#             if len(pos_params) == 1:
#                 varname = pos_params[0]
#             else:
#                 raise ValueError("Unable to infer integration variable name. Pass it explicitly via varname.")
#         except Exception:
#             raise ValueError("Function signature inspection failed. Pass varname explicitly.")

#     # Determine number of segments
#     if n is None:
#         total_length = abs(b - a)
#         n = max(1, int(np.ceil(total_length / (4 * max_step))))
#     if n < 1 or n != int(n):
#         raise ValueError("n must be a positive integer.")
#     n = int(n)

#     # Compute integration points
#     h = (b - a) / (4 * n)
#     x = np.linspace(a, b, 4 * n + 1)

#     # Evaluate f at the integration points
#     y = np.array([f(**{varname: xi}, **kwargs) for xi in x])

#     # Apply Boole's weights in blocks of 5 points
#     total = 0.0
#     for i in range(n):
#         j = 4 * i
#         weights = np.array([7, 32, 12, 32, 7])
#         segment = y[j:j+5]
#         total += np.dot(weights, segment)

#     return (2 * h / 45) * total

# def romberg(f, a, b, *, varname=None, tol=1e-8, max_iter=10, **kwargs):
#     """
#     Perform numerical integration using Romberg's method.

#     Parameters
#     ----------
#     f : callable
#         The function to integrate. Must accept the integration variable as a named argument.
#     a : float
#         Lower limit of integration.
#     b : float
#         Upper limit of integration.
#     varname : str, optional
#         Name of the integration variable. If None, it's inferred automatically (only if f has one arg).
#     tol : float, optional
#         Desired absolute tolerance. Default is 1e-8.
#     max_iter : int, optional
#         Maximum number of Romberg iterations. Default is 10.
#     **kwargs
#         Additional keyword arguments passed to `f`.

#     Returns
#     -------
#     float
#         Approximation of the integral.
#     """

#     # Infer varname if not provided
#     if varname is None:
#         try:
#             sig = signature(f)
#             pos_args = [p.name for p in sig.parameters.values()
#                         if p.kind in (p.POSITIONAL_OR_KEYWORD, p.POSITIONAL_ONLY)]
#             if len(pos_args) == 1:
#                 varname = pos_args[0]
#             else:
#                 raise ValueError("Unable to infer variable of integration; provide 'varname' explicitly.")
#         except Exception:
#             raise ValueError("Could not inspect function signature. Pass 'varname' manually.")

#     def eval_f(x):
#         return f(**{varname: x}, **kwargs)

#     # Romberg integration table
#     R = np.zeros((max_iter, max_iter))
#     h = b - a

#     # First row: trapezoid rule
#     R[0, 0] = 0.5 * h * (eval_f(a) + eval_f(b))

#     for k in range(1, max_iter):
#         h /= 2
#         # Composite Trapezoid Rule refinement
#         subtotal = sum(eval_f(a + (2 * i - 1) * h) for i in range(1, 2**(k-1)+1))
#         R[k, 0] = 0.5 * R[k - 1, 0] + h * subtotal

#         # Romberg extrapolation
#         for j in range(1, k + 1):
#             R[k, j] = (4**j * R[k, j - 1] - R[k - 1, j - 1]) / (4**j - 1)

#         # Convergence check
#         if abs(R[k, k] - R[k - 1, k - 1]) < tol:
#             return R[k, k]

#     # If it gets here, it didn't converge
#     print(f"Warning: Romberg integration did not converge after {max_iter} iterations.")
#     return R[max_iter - 1, max_iter - 1]