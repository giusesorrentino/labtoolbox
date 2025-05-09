import numpy as np
import math
import re

# --------------------------------------------------------------------------------

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

def parse_unit(unit_str: str) -> str:
    """
    Converts human-readable unit strings (with '^' for powers and Unicode symbols)
    into astropy-compatible format.
    """
    # Sostituisce "^" con "**" per gli esponenti
    unit_str = re.sub(r"\^(\d+)", r"**\1", unit_str)
    # Sostituisce simboli Unicode comuni se necessario (es: Å → Angstrom)
    unit_str = unit_str.replace("Å", "Angstrom").replace("μ", "u")
    # Sostituisce · o * con spazio (entrambi compatibili)
    unit_str = unit_str.replace("·", " ").replace("*", " ")
    return unit_str

# --------------------------------------------------------------------------------

def format_result_helper(data, data_err):
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