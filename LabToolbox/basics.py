from LabToolbox import np

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

# Prepara la funzione wrapper che accetta un vettore di variabili
# def wrapped_func(func, x_vector, params):
#     if params is not None:
#         return func(*[x_vector[i] for i in range(len(x_vector))], *params)
#     else:
#         return func(*[x_vector[i] for i in range(len(x_vector))])