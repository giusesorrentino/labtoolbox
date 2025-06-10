import numpy as np

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
    import re
    # Sostituisce "^" con "**" per gli esponenti
    unit_str = re.sub(r"\^(\d+)", r"**\1", unit_str)
    # Sostituisce simboli Unicode comuni se necessario (es: Å → Angstrom)
    unit_str = unit_str.replace("Å", "Angstrom").replace("μ", "u")
    # Sostituisce · o * con spazio (entrambi compatibili)
    unit_str = unit_str.replace("·", " ").replace("*", " ")
    return unit_str

# --------------------------------------------------------------------------------

def format_result_helper(data, data_err):
    import math
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

# --**-- uncertainty_class --**--

class uncert_prop:
    """ 
    Compute uncertainty of scalar function func(x). 
    
    Attributes:
    -----------
    func : callable
        - Should take a vector or real numbers x
        
    x : array
        - Array of variables

    cov_matrix : array or None, optional
        - The covariance matrix of the variables x. The default is `None` and equal to numpy.cov(x)
        
    method : str, optional
        - The desired method. There are 2 valid methods, `Delta` and `Monte_Carlo`. By default, method='Delta' 
        
    MC_sample_size : int, optional
        - The size of Monte Carlo sample. By default, MC_sample_size = 10000
        
    grad_dx : float, optional
        - The size of step to numerically compute gradient of func(x). By default, grad_dx = 1e-8
        
    Methods:
    --------
    x_MC_samples() : array
        - Array of sampled variables
        
    x_MC_dist_plot(contours = 15,cmap='jet',x_label=None, y_label=None, save_name=None) : array
        - 3D and 2D plots of x_MC_samples distribution  

    f_MC() : array
        - Array of func(x_MC_samples)
        
    f_MC_dist_plot(func_name='f',save_name=None) : str, optional
        - Plot of f_MC distribution 
        
    SEM() : float
        - the Standard Error of Mean
        
    confband(self,sample_size=None,conf=0.95) : tuple
        - Upper and lower confident bands of func(x). If sample_size = None then critical value is taken from normal distribution. Else, t-Student distribution is used.
   
    """
    import functools
    
    def __init__(self,func,x,cov_matrix=None,method='Delta',MC_sample_size=10000,grad_dx=1e-8):
        import sys
        self.func = func
        self.x = x
        try:
            if cov_matrix is list or (type(cov_matrix)==np.ndarray):
                self.cov_matrix =cov_matrix
            else: 
                self.cov_matrix = np.cov(self.x)
        except :
            print('Invalid list type')
            sys.exit()
        self.grad_dx = grad_dx
        self.MC_sample_size = int(np.floor(MC_sample_size))
        self.method = method
        if not ((method=='Delta') or method=='Monte_Carlo'):
            print('Incerted method is not valid.\nValid methods are: \n-- Delta-- \n--Monte_Carlo--.')
            sys.exit()
                            
    def __gradient(self):
        grad = np.zeros(len(self.x))
        for j in range(len(self.x)):  
            Dxj = (abs(self.x[j])*self.grad_dx if self.x[j] != 0 else self.grad_dx)
            x_plus = [(xi if k != j else xi + Dxj) for k, xi in enumerate(self.x)]
            grad[j] = (self.func(x_plus) - self.func(self.x))/Dxj
        return grad
            
    @functools.lru_cache(maxsize=128)
    def x_MC_samples(self):
        from scipy.stats import multivariate_normal
        if self.method=='Monte_Carlo':
            return multivariate_normal.rvs(self.x,self.cov_matrix,self.MC_sample_size)
        else:
            return print('x_MC_samples defined only for Monte_Carlo method')
            
    # Plot distribution from which popt_MC is sampled
    def x_MC_dist_plot(self,contours = 15,cmap='jet',xlabel=None, ylabel=None, save_name=None):
        import matplotlib.pyplot as plt
        from scipy.stats import multivariate_normal
        if self.method=='Monte_Carlo':
            if len(self.x)==2:
                # Create grid and multivariate normal
                x = np.linspace(min(self.x_MC_samples().T[0]),max(self.x_MC_samples().T[0]),500)
                y = np.linspace(min(self.x_MC_samples().T[1]),max(self.x_MC_samples().T[1]),500)
                X, Y = np.meshgrid(x,y)
                pos = np.empty(X.shape + (2,))
                pos[:, :, 0] = X; 
                pos[:, :, 1] = Y
                rv = multivariate_normal(self.x, self.cov_matrix)
                fig = plt.figure()
                ax1 = fig.add_subplot(1,2,1, projection='3d')
                ax1.plot_surface(X, Y, rv.pdf(pos),cmap=cmap,linewidth=0)
                ax2 = fig.add_subplot(1,2,2)
                ax2.contour(X,Y,rv.pdf(pos),contours,cmap=cmap)
                if xlabel != None:
                    ax2.set_xlabel(str(xlabel))
                else:
                    ax2.set_xlabel('$x_1$')
                if ylabel != None:
                    ax2.set_ylabel(str(ylabel))
                else:
                    ax2.set_ylabel('$x_2$')
                ax2.grid()
                if xlabel != None:
                    ax1.set_xlabel(str(xlabel))
                else:
                    ax1.set_xlabel('$x_1$')
                if ylabel != None:
                    ax1.set_ylabel(str(ylabel))
                else:
                    ax1.set_ylabel('$x_2$')
                ax1.set_zlabel('$Gaussian\ PDF$')
                ax1.view_init(elev=30, azim=-70)
                fig.set_size_inches((11.75,8.25), forward=False)
                if save_name != None:
                    fig.savefig(str(save_name)+'.png', dpi=300,bbox_inches='tight')
                del fig
            else:
                print('x_MC_dist_plot is only defined for x with 2 variables')
        else:
            print('x_MC_dist_plot is only defined for Monte_Carlo method')
            
    @functools.lru_cache(maxsize=128)
    def f_MC(self):
        if self.method=='Monte_Carlo':
            return [self.func(self.x_MC_samples()[i]) for i in range(self.MC_sample_size)]
        else:
            return print('f_MC defined only for Monte_Carlo method')
            
    # Plot distribution of f_MC
    def f_MC_dist_plot(self,func_name='f',save_name=None):
        import matplotlib.pyplot as plt
        try:
            import seaborn as sns
        except ImportError:
                raise ImportError(
                    "The 'seaborn' package is not installed. "
                    "Please install it by running 'pip install seaborn'."
                )
        from scipy import stats
        if self.method=='Monte_Carlo':
            fig,ax=plt.subplots(1)
            sns.distplot(self.f_MC(),kde=True,kde_kws={"color": "b", "lw": 1.5, "label": "Kernel Density Estimation"})
            f_MC_lnsp = np.linspace(min(self.f_MC()),max(self.f_MC()),200)  
            plt.plot(f_MC_lnsp, stats.norm.pdf(f_MC_lnsp,loc=np.array(self.f_MC()).mean(),scale=np.std(self.f_MC())),'r--',label="Gaussian distribution \nwith same mean \nand standard deviation \nas Monte Carlo sample") 
            plt.title('Distribution of '+str(func_name)+ ' after '+str(self.MC_sample_size)+' Monte Carlo Simulations')
            plt.xlabel(str(func_name))
            plt.ylabel('Probability Density')
            plt.legend()
            plt.grid() 
            if save_name != None:
                fig.set_size_inches((8.25,5.8), forward=False)
                fig.savefig(str(save_name)+'.png', dpi=300,bbox_inches='tight')                       
        else:
            print('f_MC_dist_plot is only defined for Monte_Carlo method')  
        
    def SEM(self):
        import sys
        if self.method =='Delta':
            return np.sqrt(self.__gradient().dot(self.cov_matrix).dot(self.__gradient().T))
        elif self.method == 'Monte_Carlo':
            return np.std(self.f_MC()) 
        else:
            print('Method is invalid')
            sys.exit()
        
    def confband(self,sample_size=None,conf=0.95):
        from scipy import stats
        alpha = 1.0 - conf    # significance
        var_n = len(self.x)  # number of parameters
        if not type(sample_size)==int or type(sample_size)==float:
            # Quantile of Normal distribution for p=(1-alpha/2)
            q = stats.norm.ppf(1.0 - alpha / 2.0)
        else:    
            # Quantile of Student's t distribution for p=(1-alpha/2)
            q = stats.t.ppf(1.0 - alpha / 2.0, sample_size - var_n)
        # Predicted values 
        yp = self.func(self.x)
        # Prediction band
        dy = q * self.SEM()
        # Upper & lower prediction bands.
        lcb, ucb = yp - dy, yp + dy
        return (lcb, ucb)
    
# --*-- Error Handling Class --*-- #

class GenericError(Exception):
    """
    A generic exception class for handling unexpected errors in any context.

    This exception is raised when an operation or process fails due to an unhandled or
    unexpected error. It provides a descriptive message, an optional context, the original
    exception (if any), and optional additional details for debugging purposes.

    Parameters
    ----------
    message : str
        A descriptive message explaining the error.
    context : str, optional
        A brief description of the situation where the error occurred (e.g., 'computing transform',
        'parsing input'). Defaults to 'unspecified context'.
    original_error : Exception, optional
        The original exception that triggered this error, if applicable. Defaults to None.
    details : dict, optional
        Additional details about the error, such as parameter values or input characteristics.
        Defaults to None.

    Attributes
    ----------
    message : str
        The error message.
    context : str
        The context description.
    original_error : Exception or None
        The original exception, if provided.
    details : dict or None
        Additional error details, if provided.

    Examples
    --------
    >>> try:
    ...     result = 1 / 0
    ... except Exception as e:
    ...     raise GenericError(
    ...         message="Failed to perform division",
    ...         context="computing reciprocal",
    ...         original_error=e,
    ...         details={"numerator": 1, "denominator": 0}
    ...     )
    Traceback (most recent call last):
      ...
    GenericError: Failed to perform division (context: computing reciprocal).
    Original error: division by zero.
    Details: {'numerator': 1, 'denominator': 0}.
    """

    def __init__(self, message: str, context: str = "unspecified context",
                 original_error: Exception = None, details: dict = None):
        self.message = message
        self.context = context
        self.original_error = original_error
        self.details = details
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        """
        Format the error message for display.

        Returns
        -------
        str
            A formatted string containing the message, context, original error (if any),
            and details (if any).
        """
        msg = f"{self.message} (context: {self.context})."
        if self.original_error:
            msg += f"\nOriginal error: {str(self.original_error)}."
        if self.details:
            msg += f"\nDetails: {self.details}."
        return msg

    def __str__(self) -> str:
        """
        Return the formatted error message.

        Returns
        -------
        str
            The formatted error message.
        """
        return self._format_message()