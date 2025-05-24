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

def format_stokes(value, is_percentage=True):
    """
    Format a value (percentage or absolute) according to specified rules.
    
    Parameters
    ----------
    value : float
        The value to format.
    is_percentage : bool
        If True, treat value as a percentage (multiply by 100).
        If False, treat value as an absolute number (e.g., for I, ψ, χ).

    Returns
    -------
    str
        Formatted string representation of the value.
    """
    # Converti il valore in percentuale se necessario
    if is_percentage:
        display_value = value * 100  # Converti in percentuale
    else:
        display_value = value  # Valore assoluto (es. I, ψ, χ)

    # Usa il valore assoluto per determinare la formattazione
    p_value = abs(display_value)  # Valore in termini di percentuale o assoluto

    # Applica le regole di formattazione
    if p_value >= 10:  # Corrisponde a 0.10 se fosse normalizzato
        return f"= {display_value:.0f}" + ("%" if is_percentage else "")
    elif 0.05 < p_value < 10:  # Corrisponde a 0.005 < p_value < 0.10 se fosse normalizzato
        return f"= {display_value:.1f}" + ("%" if is_percentage else "")
    elif 0.05 < p_value <= 0.5:  # Corrisponde a 0.0005 < p_value <= 0.005 se fosse normalizzato
        return f"= {display_value:.2f}" + ("%" if is_percentage else "")
    else:
        return f"≃ 0" + ("%" if is_percentage else "")
    
# def format_BIC(value, is_percentage=True):
#     """
#     Format a value (percentage or absolute) according to specified rules.
    
#     Parameters
#     ----------
#     value : float
#         The value to format.
#     is_percentage : bool
#         If True, treat value as a percentage (multiply by 100).
#         If False, treat value as an absolute number (e.g., for I, ψ, χ).

#     Returns
#     -------
#     str
#         Formatted string representation of the value.
#     """
#     # Converti il valore in percentuale se necessario
#     if is_percentage:
#         display_value = value * 100  # Converti in percentuale
#     else:
#         display_value = value  # Valore assoluto (es. I, ψ, χ)

#     # Usa il valore assoluto per determinare la formattazione
#     p_value = abs(display_value)  # Valore in termini di percentuale o assoluto

#     # Applica le regole di formattazione
#     if p_value >= 10:  # Corrisponde a 0.10 se fosse normalizzato
#         return f"= {display_value:.0f}" + ("%" if is_percentage else "")
#     elif 0.05 < p_value < 10:  # Corrisponde a 0.005 < p_value < 0.10 se fosse normalizzato
#         return f"= {display_value:.2f}" + ("%" if is_percentage else "")
#     elif 0.05 < p_value <= 0.5:  # Corrisponde a 0.0005 < p_value <= 0.005 se fosse normalizzato
#         return f"= {display_value:.3f}" + ("%" if is_percentage else "")
#     else:
#         return f"≃ 0" + ("%" if is_percentage else "")

def format_smart(value, width=None, min_thresh=1e-3, max_thresh=1e6, equalsign=True):
    """
    Format a float for aligned table display with adaptive precision.

    Parameters
    ----------
    value : float
        The value to format.
    width : int, optional
        Width of the field.
    min_thresh : float, optional
        Lower threshold for using <min format.
    max_thresh : float, optional
        Upper threshold for using >max format.
    equalsign : bool, optional
        If True, prepend '=' to the formatted value; otherwise, omit it.

    Returns
    -------
    str
        Formatted string of specified width.
    """
    abs_val = abs(value)
    prefix = "= " if equalsign else ""

    if width is not None:
        if abs_val < min_thresh and value != 0:
            return f"< {min_thresh:.0e}".rjust(width)
        elif abs_val > max_thresh:
            return f"> {max_thresh:.0e}".rjust(width)
        elif abs_val >= 100:
            return f"{prefix}{value:>{width}.0f}"
        elif abs_val >= 10:
            return f"{prefix}{value:>{width}.1f}"
        elif abs_val >= 1:
            return f"{prefix}{value:>{width}.2f}"
        else:
            return f"{prefix}{value:>{width}.3f}"
    else:
        if abs_val < min_thresh and value != 0:
            return f"< {min_thresh:.0e}"
        elif abs_val > max_thresh:
            return f"> {max_thresh:.0e}"
        elif abs_val >= 100:
            return f"{prefix}{value:.0f}"
        elif abs_val >= 10:
            return f"{prefix}{value:.1f}"
        elif abs_val >= 1:
            return f"{prefix}{value:.2f}"
        else:
            return f"{prefix}{value:.3f}"    
    
def ispow2(n):
    return n > 0 and (n & (n-1)) == 0

def fft_cooley_tukey(data):
    # Caso base: se la lunghezza è 1, restituisci l'array invariato
    if data.size <= 1:
        return data
    
    # Dividi in sottosequenze pari e dispari
    pari = data[0::2]
    dispari = data[1::2]
    
    # Calcola ricorsivamente la FFT delle sottosequenze
    E = fft_cooley_tukey(pari)    # FFT degli elementi pari
    O = fft_cooley_tukey(dispari)  # FFT degli elementi dispari
    
    # Inizializza l'array risultato
    N = data.size
    X = np.zeros(N, dtype=np.complex128)
    
    # Combina i risultati usando i twiddle factors
    for k in range(N // 2):
        w = np.exp(-2j * np.pi * k / N)  # Twiddle factor
        t = w * O[k]
        X[k] = E[k] + t
        X[k + N // 2] = E[k] - t
    
    return X

def dft_direct(data):
    N = data.size
    X = np.zeros(N, dtype=np.complex128)
    for k in range(N):
        S = 0
        for n in range(N):
            w = np.exp(-2j * np.pi * k * n / N)
            S += data[n] * w
        X[k] = S
    return X

def ifft_cooley_tukey(data):
    """
    Calcola l'IFFT usando l'algoritmo di Cooley-Tukey.
    La IFFT è simile alla FFT, ma con segno opposto nell'esponente e divisione per N.
    """
    # Caso base: se la lunghezza è 1, restituisci l'array invariato
    if data.size <= 1:
        return data
    
    # Dividi in sottosequenze pari e dispari
    pari = data[0::2]
    dispari = data[1::2]
    
    # Calcola ricorsivamente la IFFT delle sottosequenze
    E = ifft_cooley_tukey(pari)    # IFFT degli elementi pari
    O = ifft_cooley_tukey(dispari)  # IFFT degli elementi dispari
    
    # Inizializza l'array risultato
    N = data.size
    x = np.zeros(N, dtype=np.complex128)
    
    # Combina i risultati usando i twiddle factors con segno opposto rispetto alla FFT
    for k in range(N // 2):
        w = np.exp(2j * np.pi * k / N)  # Twiddle factor (nota il segno positivo)
        t = w * O[k]
        x[k] = E[k] + t
        x[k + N // 2] = E[k] - t
    
    # Dividi per N per normalizzare
    x /= 1  # La normalizzazione può essere fatta qui o nel metodo idft_direct
    
    return x

def idft_direct(data):
    """
    Calcola la DFT inversa direttamente dalla definizione.
    """
    N = data.size
    x = np.zeros(N, dtype=np.complex128)
    for n in range(N):
        S = 0
        for k in range(N):
            w = np.exp(2j * np.pi * k * n / N)  # Nota il segno positivo
            S += data[k] * w
        x[n] = S / N  # Normalizzazione per 1/N
    return x