import math as _math
import warnings as _warnings 
import numpy as _np
from typing import Callable

# --------------------------------------------------------------------------------

def PrintResult(value, err, name = "", unit = ""):
    """
    Returns a formatted string in the "mean ± sigma" format, with sigma to two significant figures,
    and the mean rounded consistently.

    Parameters
    ----------
    value : scalar or array-like
        Value of the variable.
    err : scalar or array-like
        Uncertainty of the variable considered.
    name : str or list of str, optional
        Name of the variable to display before the value. Default is an empty string.
    unit : str or list of str, optional
        Unit of measurement to display after the value in parentheses. Default is an empty string.

    Returns
    -------
    None
        Prints the formatted string directly.
    """

    if _np.isscalar(value) and _np.isscalar(err):
        # 1. Arrotonda sigma a due cifre significative
        if not isinstance(value, (int, float)):
            raise TypeError("'value' must be a real number (int or float).")
        if not isinstance(err, (int, float)):
            raise TypeError("'err' must be a real number (int or float).")
        
        if err == 0:
            raise ValueError("'err' cannot be zero.")
        if err < 0:
            raise ValueError("'err' cannot be negative.")
            
        exponent = int(_math.floor(_math.log10(abs(err))))
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
            value_part += f" [{_np.abs(nu) * 100:.2f}%]"

        # Aggiunta del nome della variabile, se fornito
        if name != "":
            result = f"{name} = {value_part}"
        else:
            result = value_part

        print(result)
    else:
        value = _np.asarray(value)
        err = _np.asarray(err)

        if not all(isinstance(item, (int, float)) for item in value):
            raise TypeError("All elements in 'value' must be real numbers (int or float).")
        if not all(isinstance(item, (int, float)) for item in err):
            raise TypeError("All elements in 'err' must be real numbers (int or float).")
        
        if not isinstance(name, list):
            raise TypeError("'name' must be a list")
        if not all(isinstance(item, str) for item in name):
            raise TypeError("All elements in 'name' must be strings")
        
        if not isinstance(unit, list):
            raise TypeError("'unit' must be a list")
        if not all(isinstance(item, str) for item in unit):
            raise TypeError("All elements in 'unit' must be strings")

        if not (len(value) == len(err) == len(name) == len(unit)):
            raise ValueError("'value', 'err', 'name' and 'unit' must have the same length.")

        if value.size == 0 and err.size != 0:
            raise ValueError("'value' is an empty array.")
        if err.size == 0 and value.size != 0:
            raise ValueError("'err' is an empty array.")
        if value.size == 0 and err.size == 0:
            raise ValueError("'value' and 'err' are empty arrays.")
        
        if not _np.all(_np.isfinite(value)):
            raise ValueError("'value' contains non-finite values (NaN or inf).")
        if not _np.all(_np.isfinite(err)):
            raise ValueError("'err' contains non-finite values (NaN or inf).")

        for i in range(len(value)):
            PrintResult(value[i], err[i], name[i], unit[i])

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

    data = _np.atleast_1d(data)
    err = _np.atleast_1d(err)

    if not all(isinstance(item, (int, float)) for item in data):
        raise TypeError("All elements in 'data' must be real numbers (int or float).")
    if not all(isinstance(item, (int, float)) for item in err):
        raise TypeError("All elements in 'err' must be real numbers (int or float).")

    if data.shape != err.shape:
        raise ValueError("Shapes of 'data' and 'err' must match.")
    
    if data.size == 0:
            raise ValueError("'data' is an empty array.")
    if err.size == 0:
        raise ValueError("'err' is an empty array.")
    
    if not _np.all(_np.isfinite(data)):
        raise ValueError("'data' contains non-finite values (NaN or inf).")
    if not _np.all(_np.isfinite(err)):
        raise ValueError("'err' contains non-finite values (NaN or inf).")

    result = []

    for d, e in zip(data, err):
        if e == 0:
            result.append(f"${d}$")
        else:
            exponent = int(_np.floor(_np.log10(_np.abs(e))))
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
    data : list of array-like
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

    # Check that data is a list of NumPy arrays
    if not isinstance(data, list):
        raise TypeError("'data' must be a list of numpy.array.")
    if not all(isinstance(col, _np.ndarray) for col in data):
        raise TypeError("All elements in 'data' must be numpy.array.")
    
    if not isinstance(header, list):
        raise TypeError("'header' must be a list of numpy.array.")
    if not all(isinstance(col, str) for col in header):
        raise TypeError("All elements in 'header' must be strings.")
    
    if not _np.isscalar(caption):
        raise TypeError("'caption' must be a scalar.")
    if not _np.isscalar(label):
        raise TypeError("'label' must be a scalar.")
    if not _np.isscalar(align):
        raise TypeError("'align' must be a scalar.")
    
    if not isinstance(caption, str):
        raise TypeError("'caption' must be a string.")
    if not isinstance(label, str):
        raise TypeError("'label' must be a string.")
    if not isinstance(align, str):
        raise TypeError("'align' must be a string.")

    # Determine the length of each column
    lengths = [len(col) for col in data]
    max_length = max(lengths)

    # If columns have different lengths, pad the shorter ones
    if len(set(lengths)) != 1:
        warn("Columns in 'data' have different lengths. Padding shorter columns with empty values.")
        for i, col in enumerate(data):
            if len(col) < max_length:
                pad_value = _np.nan if _np.issubdtype(col.dtype, _np.number) else None
                padded_col = _np.concatenate([col, _np.full(max_length - len(col), pad_value, dtype=col.dtype)])
                data[i] = padded_col

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
    from .stats import samples
    return samples(n, 'normal', mu = 0, sigma = std)

def convert(value, from_unit: str, to_unit: str):
    """
    Converts a physical quantity between units, supporting SI prefixes, non-SI units and 
    compound units.

    Parameters
    ----------
    value : float or int or numpy.array
        Numerical value to be converted.
    from_unit : str or list
        Unit of the i_nput quantity (e.g., 'erg', 'km/s', 'eV/Å^3').
    to_unit : str or list
        Desired target unit (e.g., 'J', 'm/s', 'GeV/fm^3').

    Returns
    -------
    float or numpy.arrayf
        The value converted to the target unit.
    """

    from ._helper import parse_unit

    try:
        from astropy import units as u
        from astropy.units import UnitConversionError
    except ImportError:
        raise ImportError(
            "The 'astropy' package is not installed. "
            "Please install it by running 'pip install astropy'."
        )

    if _np.isscalar(value):
        if not isinstance(value, (int, float)):
            raise TypeError("'value' must be a real number (int or float).")
        try:
            parsed_from = parse_unit(from_unit)
            parsed_to = parse_unit(to_unit)
            quantity = value * u.Unit(parsed_from)
            converted = quantity.to(parsed_to)
            print(f"I_nput:  {value} [{from_unit}]")
            print(f"Output: {converted.value} [{to_unit}]")
            return converted.value
        except UnitConversionError as e:
            raise UnitConversionError(f"Cannot convert from '{from_unit}' to '{to_unit}': {e}")
        except ValueError as e:
            raise ValueError(f"Invalid unit specified: {e}")
    else:
        # Rigorous input validation
        if not isinstance(value, _np.ndarray):
            raise TypeError(f"'value' must be a numpy.ndarray, not {type(value).__name__}")
        
        if (not (_np.issubdtype(value.dtype, _np.floating) or _np.issubdtype(value.dtype, _np.integer))) or not _np.all(_np.isreal(value)):
            raise TypeError("'value' must contain only real numbers (int or float)")

        if not isinstance(from_unit, list) or not all(isinstance(u, str) for u in from_unit):
            raise TypeError("'from_unit' must be a list of strings.")

        if not isinstance(to_unit, list) or not all(isinstance(u, str) for u in to_unit):
            raise TypeError("'to_unit' must be a list of strings.")

        if not (len(value) == len(from_unit) == len(to_unit)):
            raise ValueError("'value', 'from_unit' and 'to_unit' must have the same length.")

        if not _np.all(_np.isfinite(value)):
            raise ValueError("'value' contains non-finite values (NaN or inf).")

        # Vectorized conversion
        converted_values = []
        for v, f_unit, t_unit in zip(value, from_unit, to_unit):
            try:
                parsed_f = parse_unit(f_unit)
                parsed_t = parse_unit(t_unit)
                q = v * u.Unit(parsed_f)
                c = q.to(parsed_t)
                print(f"I_nput:  {v} [{f_unit}]")
                print(f"Output: {c.value} [{t_unit}]")
                converted_values.append(c.value)
            except UnitConversionError as e:
                raise UnitConversionError(f"Cannot convert from '{f_unit}' to '{t_unit}': {e}")
            except ValueError as e:
                raise ValueError(f"Invalid unit specified for item: {e}")

        return _np.array(converted_values)

def genspace(start: float, stop: float, num: int, f: Callable[[float], float], 
             endpoint: bool = True) -> _np.ndarray:
    """
    Generate an array of points with spacing determined by a callable function.

    Similar to numpy.linspace, but the spacing between points is defined by the function f(x),
    which specifies the density of points.

    Parameters
    ----------
        start : float
            The starting value of the sequence.
        stop : float
            The end value of the sequence.
        num : int
            Number of points to generate. Must be positive.
        f : callable
            A function f(x) that defines the density of points. Must take a float and return a
            positive float. Higher values of f(x) result in denser points around x.
        endpoint : bool, optional
            If True, stop is the last point. Otherwise, it is excluded. Defaults to True.

    Returns
    -------
        numpy.ndarray
            A 1D array of num points from start to stop, spaced according to func.

    Examples
    --------
    >>> from special import genspace
    >>> import numpy as np
    >>> # Linear spacing (equivalent to np.linspace)
    >>> x = genspace(0, 1, 5, lambda x: 1.0)
    >>> print(x)  # [0.   0.25 0.5  0.75 1.  ]
    >>> # Denser points near x=0 with f(x) = 1/x
    >>> x = genspace(0.1, 1, 5, lambda x: 1/x)
    >>> print(x)  # Points closer together near 0.1
    """

    from scipy.optimize import newton
    from scipy.integrate import quad

    # Validate inputs
    if not isinstance(start, (int, float)) or not _np.isfinite(start):
        raise TypeError("start must be a finite float.")
    if not isinstance(stop, (int, float)) or not _np.isfinite(stop):
        raise TypeError("stop must be a finite float.")
    if not isinstance(num, int):
        raise TypeError("num must be an integer.")
    if num < 1:
        raise ValueError("num must be at least 1.")
    if not callable(f):
        raise TypeError("f must be a callable function.")
    if not isinstance(endpoint, bool):
        raise TypeError("endpoint must be a boolean.")
    if start == stop:
        raise ValueError("start and stop cannot be equal.")

    try:
        # Normalize func to create cumulative distribution
        def integrand(x: float) -> float:
            val = f(x)
            if not _np.isfinite(val) or val <= 0:
                raise ValueError("'f' must return positive finite values.")
            return val
        
        # Compute total integral for normalization
        total_integral, _ = quad(integrand, start, stop)
        if not _np.isfinite(total_integral) or total_integral <= 0:
            raise ValueError("Integral of 'f' over [start, stop] must be positive and finite.")
        
        # Cumulative distribution F(x) = ∫_start^x f(t) dt / total_integral
        def F(x: float) -> float:
            integral, _ = quad(integrand, start, x)
            return integral / total_integral
        
        # Generate uniform points in [0, 1]
        if endpoint:
            u = _np.linspace(0, 1, num)
        else:
            u = _np.linspace(0, 1 - 1/num, num)
        
        # Invert F to find points
        points = _np.zeros(num)
        points[0] = start
        if num > 1:
            for i in range(1, num):
                # Use Newton-Raphson to solve F(x) = u[i]
                def objective(x: float) -> float:
                    return F(x) - u[i]
                
                # Initial guess: linear interpolation
                x_guess = start + (stop - start) * u[i]
                points[i] = newton(objective, x_guess, tol=1e-10, maxiter=100)
        
        if not _np.all(_np.isfinite(points)):
            raise ValueError("Generated points contain non-finite values.")
        return points
    
    except Exception as e:
        raise ValueError(f"Error generating points: {str(e)}")