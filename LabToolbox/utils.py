from LabToolbox import math, np
from .stats import samples

def my_mean(x, w):
    return np.sum( x*w ) / np.sum( w )

def my_cov(x, y, w):
    return my_mean(x*y, w) - my_mean(x, w)*my_mean(y, w)

def my_var(x, w):
    return my_cov(x, x, w)

def my_line(x, m=1, c=0):
    return m*x + c

def y_estrapolato(x, m, c, sigma_m, sigma_c, cov_mc):
    y = m*x + c
    uy = np.sqrt((x * sigma_m)**2 + sigma_c**2 + 2 * x * cov_mc)
    return y, uy

def format_result(data, data_err):
    # 1. Arrotonda sigma a due cifre significative
    if data_err == 0:
        raise ValueError("The uncertainty cannot be zero.")
        
    exponent = int(math.floor(math.log10(abs(data_err))))
    factor = 10**(exponent - 1)
    rounded_sigma = round(data_err / factor) * factor

    # 2. Arrotonda mean allo stesso ordine di grandezza di sigma
    rounded_mean = round(data, -exponent + 1)

    # 3. Restituisce il valore numerico arrotondato
    return rounded_mean, rounded_sigma

def format_value_auto(val, err, unit=None, scale=0):
    if scale != 0:
        val /= 10**scale
        err /= 10**scale

    if err == 0 or np.isnan(err) or np.isinf(err):
        formatted = f"{val:.3g}"
        if unit:
            unit = unit.replace('$', '')
            formatted += f"\\,\\mathrm{{{unit}}}"
        return formatted

    err_exp = int(np.floor(np.log10(abs(err))))
    err_coeff = err / 10**err_exp

    if err_coeff < 1.5:
        err_exp -= 1
        err_coeff = err / 10**err_exp

    err_rounded = round(err, -err_exp + 1)
    val_rounded = round(val, -err_exp + 1)

    if abs(val_rounded) >= 1e4 or abs(val_rounded) < 1e-2:
        val_scaled = val_rounded / (10**err_exp)
        err_scaled = err_rounded / (10**err_exp)
        formatted = f"({val_scaled:.2f}\\pm{err_scaled:.2f})\\times 10^{{{err_exp}}}"
    else:
        ndecimals = max(0, -(err_exp - 1))
        fmt = f"{{:.{ndecimals}f}}"
        formatted = fmt.format(val_rounded) + "\\pm" + fmt.format(err_rounded)

    if unit:
        unit = unit.replace('$', '')
        formatted += f"\\,\\mathrm{{{unit}}}"

    return formatted

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
        Name of the variable to display before the value (default is an empty string).
    unit : str, optional
        Unit of measurement to display after the value in parentheses (default is an empty string).

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
    if unit != "":
        if rounded_mean != 0:
            nu = rounded_sigma / rounded_mean
            result = f"{name} = ({mean_str} ± {sigma_str}) {unit} [{np.abs(nu)*100:.2f}%]"
        else:
            result = f"{name} = ({mean_str} ± {sigma_str}) {unit}"
    else:
        if rounded_mean != 0:
            nu = rounded_sigma / rounded_mean
            result = f"{name} = ({mean_str} ± {sigma_str}) [{np.abs(nu)*100:.2f}%]"
        else:
            result = f"{name} = ({mean_str} ± {sigma_str})"

    print(result)

def to_latex_table(data, header, filename, caption, label):
    """
    Writes a LaTeX-formatted table to file with caption, label and predefined styling.

    Parameters
    ----------
    data : list of list of str or float
        The content of the table, organized as a list of rows.
    header : list of str
        List of column names to appear in the header of the table.
    filename : str
        Path to the output `.tex` file (e.g., 'table.tex').
    caption : str
        Caption text of the table.
    label : str
        Label used for referencing the table in LaTeX.
    
    Notes
    -----
    - Assumes that all elements of `data` and `header` are convertible to string.
    - Does not escape LaTeX special characters: input should already be LaTeX-safe.
    """

    n_cols = len(header)
    col_format = "c" * n_cols

    with open(filename, 'w') as f:
        f.write("\\begin{table}[H]\n")
        f.write(f"\\caption{{\\large \\label{{{label}}} {caption}}}\n")
        f.write("\\vspace{-0.7\\baselineskip}\n")
        f.write("\\centering\n")
        f.write(f"\\begin{{tabular}}{{{col_format}}}\n")
        f.write("\\hline\\hline\n")
        f.write("\\noalign{\\vskip 1.5pt}\n")
        f.write(" & ".join(header) + " \\\\\n")
        f.write("\\hline\n")
        f.write("\\noalign{\\vskip 2pt}\n")

        for row in data:
            row_str = " & ".join(str(cell) for cell in row)
            f.write(row_str + " \\\\\n")

        f.write("\\noalign{\\vskip 1.5pt}\n")
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    
def noise(n, std):
    return samples(n, 'norma', mu = 0, sigma = std)