import warnings as _warnings
import numpy as _np
import matplotlib.pyplot as _plt

def fft(data, t = None, dt = None):
    """
    Compute the Fast Fourier Transform (FFT) of a signal.

    Parameters
    ----------
    data : array-like
        Input signal as a 1D NumPy array (real or complex).
    t : array-like, optional
        Time samples corresponding to the signal. If provided, must be a 1D NumPy
        array of the same length as `data`, monotonically increasing, and uniformly
        spaced. If None, only the FFT is returned unless `dt` is provided.
    dt : float, optional
        Time interval between samples. If provided, it is used to calculate frequency bins.
        Ignored if `t` is also provided.

    Returns
    -------
    X : numpy.array
        The FFT of the input signal, a 1D complex array of the same length as `data`.
    f : numpy.array, optional
        The frequency bins (in Hz) corresponding to the FFT coefficients, returned
        only if `t` or `dt` is provided. Same length as `data`.

    Raises
    ------
    ValueError
        If `t` is provided and:
        - Has a different length than `data`.
        - Is not monotonically increasing.
        - Is not uniformly spaced (equispaced).

    Notes
    -----
    - For lengths <= 16, a direct DFT is used.
    - The FFT is computed using the Cooley-Tukey algorithm for power-of-2 lengths.
    """

    from ._helper import ispow2, fft_cooley_tukey, dft_direct

    data = _np.asarray(data)

    if not (_np.issubdtype(data.dtype, _np.number)):
        raise TypeError("'data' must contain only numeric types (int, float, or complex).")
    
    if not _np.all(_np.isfinite(data)):
            raise ValueError("'data' contains non-finite values (NaN or inf).")

    if t is not None:
        t = _np.asarray(t)
        if (not (_np.issubdtype(t.dtype, _np.floating) or _np.issubdtype(t.dtype, _np.integer))) or not _np.all(_np.isreal(t)):
            raise TypeError("'t' must contain only real numbers (int or float).")
        
        if t.size != data.size:
            raise ValueError("'t' must have the same length as 'data'")
        if not _np.all(_np.diff(t) > 0):
            raise ValueError("'t' must be monotonically increasing")
        if not _np.allclose(_np.diff(t), _np.diff(t)[0], rtol=1e-5):
            raise ValueError("'t' must be uniformly spaced (equispaced)")

        if not _np.all(_np.isfinite(t)):
                raise ValueError("'t' contains non-finite values (NaN or inf).")
    else:
        if not isinstance(dt, (int, float)):
            raise TypeError("'dt' must be a real number (int or float).")
        if dt <= 0:
            raise ValueError("'dt' must be a positive scalar.")

    if data.size == 0:
        _warnings.warn("'data' is an empty array. Returning and empty numpy.array.", UserWarning)
        return _np.array([])
    if data.size == 1:
        _warnings.warn("'data' is a scalar. Returning 'data'.", UserWarning)
        return data
    elif data.size <= 16:
        X = dft_direct(data)
    elif ispow2(data.size):
        X = fft_cooley_tukey(data)
    else:
        M = int(2 ** _np.ceil(_np.log2(data.size)))
        padded = _np.pad(data, (0, M - data.size), mode="constant")
        X = fft_cooley_tukey(padded)
        X = X[:data.size]
    
    # Determina dt e gestisci errori
    if t is not None:
        dt_final = (t[-1] - t[0]) / (t.size - 1) if t.size > 1 else 1.0
    elif dt is not None:
        dt_final = dt
    else:
        return X  # Nessuna informazione temporale disponibile

    f = _np.fft.fftfreq(data.size, d=dt_final)

    # Controllo aliasing
    spectrum_magnitude = _np.abs(X)
    threshold = _np.max(spectrum_magnitude) * 0.05  # 5% del massimo
    freq_components = _np.abs(f[spectrum_magnitude > threshold])
    if freq_components.size > 0:
        f_max = _np.max(freq_components)
        fs = 1 / dt_final
        if f_max >= fs / 2:
            _warnings.warn(
                f"Potential aliasing detected: the signal contains frequency components up to {f_max:.2g}, "
                f"which exceeds the Nyquist frequency ({fs/2:.2g}) based on the current sampling rate. "
                "Please consider increasing the sampling frequency or applying an anti-aliasing filter prior to sampling. "
                "Proceeding may lead to distorted spectral analysis results."
            )

    # Riordina le frequenze e i coefficienti FFT
    order = _np.argsort(f)
    f_sorted = f[order]
    X_sorted = X[order]

    return X_sorted, f_sorted

def ifft(data, freq=None, df=None):
    """
    Compute the Inverse Fast Fourier Transform (IFFT) of a signal.

    Parameters
    ----------
    data : array-like
        Input frequency domain signal as a 1D NumPy array (complex).
    freq : array-like, optional
        Frequency samples corresponding to the signal. Must be a 1D NumPy array
        of the same length as `data`, monotonically increasing, and uniformly spaced.
    df : float, optional
        Frequency spacing (in Hz) of the input data. Used if `freq` is not provided.
        If both `freq` and `df` are None, time bins are not returned.

    Returns
    -------
    x : numpy.ndarray
        The IFFT of the input signal, a 1D complex array of the same length as `data`.
    t : numpy.ndarray, optional
        The time bins (in seconds) corresponding to the IFFT coefficients, returned
        only if `freq` or `df` is provided. Same length as `data`.

    Raises
    ------
    ValueError
        If `freq` is provided and:
        - Has a different length than `data`.
        - Is not monotonically increasing.
        - Is not uniformly spaced (equispaced).

    Notes
    -----
    - For lengths <= 16, a direct IDFT is used.
    - For non-power-of-2 lengths, the signal is zero-padded to the next power of 2.
    - The IFFT is computed using the Cooley-Tukey algorithm for power-of-2 lengths.
    """

    from ._helper import ispow2, idft_direct, ifft_cooley_tukey

    data = _np.asarray(data)

    if not (_np.issubdtype(data.dtype, _np.number)):
        raise TypeError("'data' must contain only numeric types (int, float, or complex).")
    
    if not _np.all(_np.isfinite(data)):
            raise ValueError("'data' contains non-finite values (NaN or inf).")

    if freq is not None:
        freq = _np.asarray(freq)

        if (not (_np.issubdtype(freq.dtype, _np.floating) or _np.issubdtype(freq.dtype, _np.integer))) or not _np.all(_np.isreal(freq)):
            raise TypeError("'freq' must contain only real numbers (int or float).")
        
        if freq.size != data.size:
                raise ValueError("'freq' must have the same length as 'data'")
        if not _np.all(_np.diff(freq) > 0):
            raise ValueError("'freq' must be monotonically increasing")
        if not _np.allclose(_np.diff(freq), _np.diff(freq)[0], rtol=1e-5):
            raise ValueError("'freq' must be uniformly spaced (equispaced)")

        if not _np.all(_np.isfinite(freq)):
                raise ValueError("'freq' contains non-finite values (NaN or inf).")
        
    else:
        if not isinstance(df, (int, float)):
            raise TypeError("'df' must be a real number (int or float).")
        if df <= 0:
            raise ValueError("'df' must be a positive scalar.")

    if data.size == 0:
        _warnings.warn("'data' is an empty array. Returning and empty numpy.array.", UserWarning)
        return _np.array([])
    if data.size == 1:
        _warnings.warn("'data' is a scalar. Returning 'data'.", UserWarning)
        return data
    elif data.size <= 16:
        x = idft_direct(data)
    elif ispow2(data.size):
        x = ifft_cooley_tukey(data)
    else:
        M = int(2 ** _np.ceil(_np.log2(data.size)))
        padded = _np.pad(data, (0, M - data.size), mode="constant")
        x = ifft_cooley_tukey(padded)
        x = x[:data.size]

    if freq is not None or df is not None:
        if freq is not None:
            df_value = (freq[-1] - freq[0]) / (freq.size - 1) if freq.size > 1 else 1.0
        else:
            df_value = df

        # fs = 1 / df_value  # Frequenza di campionamento temporale
        dt = 1 / (data.size * df_value)  # Intervallo di tempo (step)
        t = _np.arange(0, data.size) * dt  # Array dei tempi

        return x, t

    return x

def dfs(t, data, order, plot=True, showpanel = True, apply_filter = True, xlabel = "", ylabel = "", xscale = 0, yscale = 0, xlim = [], ylim = []):
    """
    Computes the discrete Fourier series approximation of a sampled function.

    Parameters
    ----------
    t : array-like
        1D array of sample points (must be uniformly spaced).
    data : array-like
        1D array of function values sampled at points t.
    order : int
        Order of the Fourier approximation (number of harmonics).
    plot : bool, optional
        If `True`, plots the original function and its Fourier approximation.
    showpanel : bool, optional
        If `True`, a top panel will display the superposition of sinusoidal basis functions
        (harmonics) used in the Fourier series decomposition. This panel visualizes the
        contributions of individual harmonics to the approximation of the input data.
    apply_filter : bool, optional
        If `True`, applies a basic low-pass filter to reduce high-frequency noise.
    xlabel : str, optional
        Label for the x-axis, including units in square brackets (e.g., "Time [s]").
    ylabel : str, optional
        Label for the y-axis, including units in square brackets (e.g., "Intensity [V]").
    xscale : int, optional
        Scaling factor for the x-axis (e.g., `xscale = -3` corresponds to 1e-3, to convert seconds to milliseconds).
    yscale : int, optional
        Scaling factor for the y-axis.
    xlim : tuple, optional
        Limits for the x-axis, in the form (xmin, xmax). The values should
        already be scaled with respect to `xscale`. If None or an empty tuple,
        the default limits will be automatically determined from the data.
    ylim : tuple, optional
        Limits for the y-axis, in the form (ymin, ymax). The values should
        already be scaled with respect to `yscale`. If None or an empty tuple,
        the default limits will be automatically determined from the data.

    Returns
    -------
    f_approx : numpy.array
        Array of the same shape as `t`, containing the values of the Fourier series approximation
        of the input function at each point in `t`.
    a0 : float
        Zeroth Fourier coefficient (mean value component of the function over the period). It 
        corresponds to the constant term of the Fourier series.
    a_n : numpy.array
        Array of cosine coefficients (Fourier coefficients of the even part of the function),
        corresponding to each harmonic up to the specified order (excluding a0).
    b_n : numpy.array
        Array of sine coefficients (Fourier coefficients of the odd part of the function),
        corresponding to each harmonic up to the specified order.

    Notes
    ----------
    The values of `xscale` and `yscale` affect only the axis scaling in the plot. All outputs are estimated using the original input data as provided.
    """

    from scipy.integrate import simpson

    data = _np.asarray(data)
    t = _np.asarray(t)

    if not _np.allclose(_np.diff(t), t[1] - t[0], rtol=1e-4):
        raise ValueError("'t' must be uniformly spaced.")

    if not isinstance(order, (int)):
        raise TypeError("'order' must be an integer.")
    if order <= 0:
        raise ValueError("'order' must be equal or greater than 1.")
    
    if t.size != data.size:
        raise ValueError("'t' must have the same length as 'data'")
    
    if not (_np.issubdtype(data.dtype, _np.number)):
        raise TypeError("'data' must contain only numeric types (int, float, or complex).")
    
    if (not (_np.issubdtype(t.dtype, _np.floating) or _np.issubdtype(t.dtype, _np.integer))) or not _np.all(_np.isreal(t)):
        raise TypeError("'t' must contain only real numbers (int or float).")
    
    if not _np.all(_np.isfinite(data)):
            raise ValueError("'data' contains non-finite values (NaN or inf).")
    if not _np.all(_np.isfinite(t)):
            raise ValueError("'t' contains non-finite values (NaN or inf).")

    if not isinstance(xscale, (int, float)):
        raise TypeError("'xscale' must be a real number (int or float).")
    if not isinstance(yscale, (int, float)):
        raise TypeError("'yscale' must be a real number (int or float).")

    if not isinstance(xlim, list):
        raise TypeError("'xlim' must be a list (either empty or containing two real numbers).")
    if len(xlim) != 0:
        if len(xlim) != 2:
            raise TypeError("'xlim' must be empty or a list of exactly two real numbers.")
        if not all(isinstance(u, (int, float)) and _np.isfinite(u) for u in xlim):
            raise TypeError("Both elements in 'xlim' must be finite real numbers (int or float).")
        
    if not isinstance(ylim, list):
        raise TypeError("'ylim' must be a list (either empty or containing two real numbers).")
    if len(ylim) != 0:
        if len(ylim) != 2:
            raise TypeError("'ylim' must be empty or a list of exactly two real numbers.")
        if not all(isinstance(u, (int, float)) and _np.isfinite(u) for u in ylim):
            raise TypeError("Both elements in 'ylim' must be finite real numbers (int or float).")
        
    if not isinstance(xlabel, (str)):
        raise TypeError("'xlabel' must be a string.")
    if not isinstance(ylabel, (str)):
        raise TypeError("'ylabel' must be a string.")
    
    xscale = 10**xscale
    yscale = 10**yscale

    N = len(t)
    dt = t[1] - t[0]
    fs = 1 / dt                  # Sampling frequency
    f_nyquist = fs / 2          # Nyquist frequency
    max_freq = order / (t[-1] - t[0])  # Maximum frequency in Fourier expansion

    if max_freq > f_nyquist:
        _warnings.warn(
            f"Potential aliasing detected: the signal contains frequency components up to {max_freq:.2g}, "
            f"which exceeds the Nyquist frequency ({f_nyquist:.2g}) based on the current sampling rate. "
            "Please consider increasing the sampling frequency or applying an anti-aliasing filter prior to sampling. "
            "Proceeding may lead to distorted spectral analysis results."
        )

    a = t[0]
    b = t[-1]
    T = b - a  # Assumed period of the signal

    # Zeroth coefficient
    a0 = 2 * simpson(data, t) / T

    a_n = []
    b_n = []

    for n in range(1, order):
        cos_term = _np.cos(2 * _np.pi * n * t / T)
        sin_term = _np.sin(2 * _np.pi * n * t / T)

        an = 2 * simpson(data * cos_term, t) / T
        bn = 2 * simpson(data * sin_term, t) / T

        if apply_filter:
            # Apply exponential decay to filter high-frequency components
            decay = _np.exp(- (n / order)**2)
            an *= decay
            bn *= decay

        a_n.append(an)
        b_n.append(bn)

    # Construct approximation
    f_approx = _np.full_like(t, a0 / 2)

    for n in range(1, order):
        f_approx += (
            a_n[n - 1] * _np.cos(2 * _np.pi * n * t / T) +
            b_n[n - 1] * _np.sin(2 * _np.pi * n * t / T)
        )

    if plot:

        fig = _plt.figure(figsize=(6.4, 4.8))

        if showpanel:
            gs = fig.add_gridspec(2, hspace=0, height_ratios=[0.1, 0.9])
            axs = gs.subplots(sharex=True)

            # Pannello superiore: componenti armoniche
            total_harmonic = _np.zeros_like(t, dtype=float)
            for n in range(1, order):
                harmonic = (a_n[n - 1] * _np.cos(2 * _np.pi * n * t / T) +
                            b_n[n - 1] * _np.sin(2 * _np.pi * n * t / T))
                total_harmonic += harmonic
                axs[0].plot(t / xscale, harmonic / yscale, lw=0.8)

            # Calcola l'ampiezza massima della somma delle armoniche
            max_harmonic_amplitude = _np.max(_np.abs(total_harmonic / yscale))
            axs[0].set_ylim(-1.5 * max_harmonic_amplitude, 1.5 * max_harmonic_amplitude)
            axs[0].tick_params(labelbottom=False)
            axs[0].set_yticklabels('')
            # axs[0].set_yticks([])
        else:
            gs = fig.add_gridspec(2, hspace=0, height_ratios=[0, 1])
            axs = gs.subplots(sharex=True)
            #axs = gs.subplots()
            axs[0].remove()  # Rimuovi axs[0], axs[1] rimane valido

        # Pannello inferiore: dati originali e approssimazione
        # axs[1].plot(t / xscale, data / yscale, label="Input data", lw=1, color="blue")
        # axs[1].plot(t / xscale, f_approx / yscale, label=f"Fourier approx. (order={order})", lw=1, color="red")
        # axs[1].set_xlabel(xlabel)
        # axs[1].set_ylabel(ylabel)
        # axs[1].legend()

        # Pannello inferiore: dati originali e approssimazione
        axs[1].plot(
            t / xscale, data / yscale, label= "Input data", lw=1,
            color =  "mediumblue", marker=''
        )

        axs[1].plot(
            t / xscale, f_approx / yscale, label=f"Partial Fourier series\nNo. of terms = {order}", lw=1,
            color = "crimson", linestyle='-'
        )

        # Imposta limiti se forniti
        if xlim:
            if showpanel:
                axs[0].set_xlim(xlim)
            axs[1].set_xlim(xlim)
        if ylim:
            axs[1].set_ylim(ylim)

        axs[1].set_xlabel(xlabel)
        axs[1].set_ylabel(ylabel)
        axs[1].legend()

    return f_approx, a0, a_n, b_n

def fourier_series(f, interval, order, num_points=1000, xlabel = "x [ux]", ylabel = "y [uy]", xscale = 0, yscale = 0):
    """
    Computes the Fourier series approximation of a function f(x)
    
    Parameters
    ----------
    f : callable
        Function to approximate.
    interval : list of float
        The interval (a, b) over which to compute the Fourier series.
    order : int
        Number of Fourier modes (n) to use in the approximation.
    num_points : int, optional
        Number of points for plotting (default is 1000).
        
    Returns
    -------
    x : numpy.array
        Array of shape (N,), representing the uniformly spaced sample points over one period.
        These are the evaluation points at which both the original function and the Fourier
        approximation are computed.
    f_original : numpy.array
        Array of shape (N,), representing the values of the original input function evaluated at 
        the sample points `x`. This is the reference signal used for comparison with the Fourier 
        approximation.
    f_approx : numpy.array
        Array of shape (N,), containing the values of the truncated Fourier series evaluated at 
        the same sample points `x`. This is the approximation of `f_original` using a finite number 
        of harmonics (up to the specified order).
    a0 : float
        Zeroth Fourier coefficient (mean value component of the function over the period). It 
        corresponds to the constant term of the Fourier series.
    a_n : numpy.array
        Array of cosine coefficients (Fourier coefficients of the even part of the function),
        corresponding to each harmonic up to the specified order (excluding a0).
    b_n : numpy.array
        Array of sine coefficients (Fourier coefficients of the odd part of the function),
        corresponding to each harmonic up to the specified order.

    Notes
    ----------
    The values of `xscale` and `yscale` affect only the axis scaling in the plot. All parameters are estimated using the original input data as provided.
    """

    from scipy.integrate import quad

    if not callable(f):
        raise TypeError("'f' must be a callable function.")
    
    if not isinstance(interval, list):
        raise TypeError("'interval' must be a list.")
    if len(interval) != 2:
        raise ValueError("'interval' must have exactly two elements.")
    
    a, b = interval
    
    if not (_np.isscalar(a) and _np.isscalar(b)):
        raise TypeError("'interval' elements must be scalars.")
    
    if not (isinstance(a, (int, float)) and isinstance(b, (int, float))):
        raise TypeError("'interval' elements must be real numbers (int or float).")

    if b < a:
        a, b = b, a
        _warnings.warn("Integration limits 'a' and 'b' have been swapped.", UserWarning)

    if a == b:
        raise ValueError("'a' must be not equal to 'b'.")

    if not isinstance(order, (int)):
        raise TypeError("'order' must be an integer.")
    if order <= 0:
        raise ValueError("'order' must be equal or greater than 1.")
    
    if not isinstance(xlabel, (str)):
        raise TypeError("'xlabel' must be a string.")
    if not isinstance(ylabel, (str)):
        raise TypeError("'ylabel' must be a string.")
    
    if num_points <= 0:
        raise ValueError("'num_points' must be equal or greater than 1.")
    
    if not isinstance(xscale, (int, float)):
        raise TypeError("'xscale' must be a real number (int or float).")
    if not isinstance(yscale, (int, float)):
        raise TypeError("'yscale' must be a real number (int or float).")

    xscale = 10**xscale
    yscale = 10**yscale

    # aggiungere nota che la yscale è solo a livello grafico

    T = b - a  # Period
    L = T / 2
    x = _np.linspace(a, b, num_points)
    
    # Compute a0 separately
    a0, _ = quad(lambda x_: f(x_), a, b)
    a0 /= L

    # Compute coefficients a_n and b_n
    a_n = []
    b_n = []

    for n in range(1, order + 1):
        an, _ = quad(lambda x_: f(x_) * _np.cos(n * _np.pi * (x_ - a) / L), a, b)
        bn, _ = quad(lambda x_: f(x_) * _np.sin(n * _np.pi * (x_ - a) / L), a, b)
        a_n.append(an / L)
        b_n.append(bn / L)

    # Build the Fourier approximation
    f_approx = _np.full_like(x, a0 / 2)

    for n in range(1, order + 1):
        f_approx += (
            a_n[n - 1] * _np.cos(n * _np.pi * (x - a) / L) +
            b_n[n - 1] * _np.sin(n * _np.pi * (x - a) / L)
        )

    f_original = f(x)

    # Plot
    _plt.plot(x / xscale, f_original / yscale, label="Input function", lw=0.8, color = "blue")
    _plt.plot(x / xscale, f_approx / yscale, '--', label=f"Fourier approx. (order = {order})", lw=0.8, color = "red")
    _plt.xlabel(xlabel)
    _plt.ylabel(ylabel)
    _plt.legend()

    return f_original, f_approx, a0, a_n, b_n

def psd(data, t = None, plot = True, spectrogram = False, log = None, logcs = False, psd_unit = None, time_unit = "s", freq_unit = "Hz", color = "k"):
    """
    Compute the Power Spectral Density (PSD) of a signal using the FFT.

    Parameters
    ----------
    data : numpy.array
        Input signal as a 1D NumPy array (real or complex).
    t : numpy.array, optional
        Time samples corresponding to the signal. If provided, must be a 1D NumPy
        array of the same length as `data`, monotonically increasing, and uniformly
        spaced. Units are specified by `time_unit`. If None, only the PSD is returned
        without frequencies.
    plot : bool, optional
        Whether to plot the PSD or spectrogram.
    spectrogram : bool, optional
        If True, compute and plot a spectrogram (time-frequency representation).
        Requires `t` to be provided.
    log : str, optional
        If set to 'x' or 'y', the corresponding axis is plotted on a logarithmic scale;
        if 'xy', both axes. Default is `None`.
    logcs : bool, optional
        If `True` (and `spectrogram = True`), the color scale is set to logarithmic.
        Default is `False`.
    psd_unit : str, optional
        Unit of the power spectral density (e.g., V^2/Hz). Default is `None`.
    time_unit : str, optional
        Unit of time for `t` (e.g., 's' for seconds, 'ms' for milliseconds).
        Default is `'s'`.
    freq_unit : str, optional
        Unit of frequency for the frequency axis (e.g., 'Hz', 'kHz').
        Default is `'Hz'`.
    color : str, optional
        Color passed to `matplotlib.pyplot.plot` when plotting the PSD. Default is `k` (black).

    Returns
    -------
    S : numpy.ndarray
        The PSD of the input signal, a 1D real array of length `floor(N/2) + 1`
        (positive frequencies only).
    freqs : numpy.ndarray
        The frequency bins (in Hz) corresponding to the FFT coefficients, returned
        only if `t` is provided. Same length as `S`.

    Raises
    ------
    ValueError
        If `t` is provided and:
        - Has a different length than `data`.
        - Is not monotonically increasing.
        - Is not uniformly spaced (equispaced).
        If `log` is not one of 'x', 'y', 'xy', 'cs', or None.
    """

    from matplotlib.colors import LogNorm

    data = _np.asarray(data)
    if t is not None:
        t = _np.asarray(t)

    # --- Validazioni iniziali ---
    if log is not None and log not in ("x", "y", "xy", "cs"):
        raise ValueError("'log' must be one of: 'x', 'y', 'xy', 'cs' or None.")
    
    if logcs and not spectrogram:
        _warnings.warn("'logcs = True' has no effect unless 'spectrogram = True'.", RuntimeWarning)
    
    if spectrogram and t is None:
        raise ValueError("Parameter 't' must be provided if 'spectrogram = True'.")
    
    if logcs and t is None:
        _warnings.warn("'logcs = True' requires 't' to be defined.", RuntimeWarning)

    if data.size == 0:
        return _np.array([]), _np.array([])

    if psd_unit == None or psd_unit == "":
        label_psd = f"Power Spectral Density"
    else:
        label_psd = f"Power Spectral Density [{psd_unit}]"

    # --- Calcolo FFT e frequenze ---
    X, freqs = fft(data, t)
    
    N = data.size
    S = _np.abs(X)**2
    # S = S[:N//2 + 1]       # Frequenze positive
    # freqs = freqs[:N//2 + 1]

    # --- Calcolo fs e normalizzazione ---
    fs = 1.0
    if t is not None:
        if t.size != N:
            raise ValueError("'t' must have the same length as 'data'.")
        if not _np.all(_np.diff(t) > 0):
            raise ValueError("'t' must be strictly increasing.")
        if not _np.allclose(_np.diff(t), _np.diff(t)[0], rtol=1e-5):
            raise ValueError("'t' must be uniformly spaced.")
        dt = (t[-1] - t[0]) / (N - 1)
        fs = 1.0 / dt

    # S[1:] *= 2 / (fs * N)   # Per frequenze non-DC
    # S[0] /= (fs * N)        # Per frequenza zero

    # --- Plot se richiesto ---
    if plot and t is not None:
        if spectrogram:
            # Parametri automatici
            N_total = data.size
            fs = 1.0 / (t[1] - t[0])
            window_size = min(512, N_total // 20)
            step = window_size // 4

            t_spec = []
            S_list = []

            for start in range(0, N_total - window_size + 1, step):
                stop = start + window_size
                seg = data[start:stop]
                seg_t = t[start:stop]
                X = fft(seg, t=None)
                N = seg.size
                fs_seg = 1.0 / (seg_t[1] - seg_t[0])
                freqs_full = _np.fft.fftfreq(N, d=1/fs_seg)
                pos_mask = freqs_full >= 0
                freqs = freqs_full[pos_mask][:N//2 + 1]
                S_seg = _np.abs(X)[pos_mask][:N//2 + 1] ** 2
                S_seg[1:] *= 2 / (fs_seg * N)
                S_seg[0] /= (fs_seg * N)
                S_list.append(S_seg)
                t_spec.append(_np.mean(seg_t))

            S_array = _np.array(S_list).T
            t_spec = _np.array(t_spec)

            if not _np.all(_np.diff(t_spec) > 0):
                raise ValueError("'t_spec' is not monotonically increasing.")
            if not _np.all(_np.diff(freqs) > 0):
                raise ValueError("'freqs' is not monotonically increasing.")

            fig, ax = _plt.subplots()
            if log in ("x", "xy"):
                ax.set_xscale("log")
            if log in ("y", "xy"):
                ax.set_yscale("log")

            if logcs:
                norm = LogNorm(vmin=S_array[S_array > 0].min(), vmax=S_array.max())
            else:
                norm = None

            T, F = _np.meshgrid(t_spec, freqs)
            t_edges = _np.concatenate([t_spec - _np.diff(t_spec, append=t_spec[-1])/2, [t_spec[-1] + (t_spec[-1] - t_spec[-2])/2]])
            f_edges = _np.concatenate([freqs - _np.diff(freqs, append=freqs[-1])/2, [freqs[-1] + (freqs[-1] - freqs[-2])/2]])
            if logcs:
                pcm = ax.pcolormesh(t_edges, f_edges, S_array, shading='auto', cmap='plasma', norm=norm)
            else:
                pcm = ax.pcolormesh(t_edges, f_edges, S_array, shading='auto', cmap='plasma', norm=None)
            fig.colorbar(pcm, ax=ax, label=label_psd, pad = 0)
            # --- Grafico standard della PSD ---
            # Gestione dei limiti per evitare problemi con logscale
            if log in ("x", "xy"):
                t_mask = t_spec > 0  # Esclude tempi <= 0 solo se scala log su x
                t_spec_filtered = t_spec[t_mask]
                t_min = t_spec_filtered[0] if t_spec_filtered.size > 0 else t_spec[0]
                t_max = t_spec_filtered[-1] if t_spec_filtered.size > 0 else t_spec[-1]
            else:
                t_min = t_spec[0]
                t_max = t_spec[-1]

            if log in ("y", "xy"):
                f_mask = freqs > 0  # Esclude frequenze <= 0 solo se scala log su y
                freqs_filtered = freqs[f_mask]
                f_min = freqs_filtered[0] if freqs_filtered.size > 0 else freqs[0]
                f_max = freqs_filtered[-1] if freqs_filtered.size > 0 else freqs[-1]
            else:
                f_min = freqs[0]
                f_max = freqs[-1]

            # Imposta i limiti degli assi
            ax.set_xlim(t_min, t_max)
            ax.set_ylim(f_min, f_max)
            ax.set_xlabel(f"Time [{time_unit}]")
            ax.set_ylabel(f"Frequency [{freq_unit}]")
        else:
            # --- Grafico standard della PSD ---
            mask = freqs > 0 if log in ("x", "xy", "y") else slice(None)
            f_plot = freqs[mask]
            S_plot = S[mask]

            if log in ("x", "xy"):
                _plt.xscale("log")

            if log in ("y", "xy"):
                _plt.yscale("log")

            _plt.plot(f_plot, S_plot, lw = 0.8, color = color)

            if f_plot.size > 0:
                _plt.xlim(f_plot[0], f_plot[-1])
            _plt.xlabel(f"Frequency [{freq_unit}]")
            _plt.ylabel(label_psd)

    if t is not None:
        return S, freqs
    return S

def harmonic(t, y, prominence = 0.05, n_max = None, verbose = True):
    """
    Identifies the dominant harmonics present in a real-valued signal.

    This function computes the FFT of the input signal y(t), finds the peaks
    in its amplitude spectrum, and returns their frequencies, amplitudes, and phases.
    The phase of each harmonic is calculated relative to the phase of the first
    (most prominent) harmonic. It does *not* reconstruct the time-domain components—only
    reports what harmonics are strongest.

    Parameters
    ----------
    t : array-like
        Time array.
    y : array-like
        Signal samples corresponding to `t`.
    prominence : float, optional
        Minimum prominence of peaks in the power spectrum. Default is 0.05.
    n_max : int or None, optional
        If given, return at most `n_max` harmonics.
    verbose : bool, optional
        If `True`, prints a formatted table with the frequency, amplitude, and phase of the harmonics.

    Returns
    -------
    harmonics : list of dict
        Each dict contains frequency, amplitude, and phase (rad, relative to the first harmonic) of a harmonic component.
    """
    from scipy.signal import find_peaks

    t = _np.asarray(t)
    y = _np.asarray(y)

    if t.size != y.size:
        raise ValueError("'t' must have the same length as 'y'")

    if not (_np.issubdtype(y.dtype, _np.number)):
        raise TypeError("'y' must contain only numeric types (int, float, or complex).")
    
    if (not (_np.issubdtype(t.dtype, _np.floating) or _np.issubdtype(t.dtype, _np.integer))) or not _np.all(_np.isreal(t)):
        raise TypeError("'t' must contain only real numbers (int or float).")
    
    if not _np.all(_np.isfinite(y)):
            raise ValueError("'data' contains non-finite values (NaN or inf).")
    if not _np.all(_np.isfinite(t)):
            raise ValueError("'t' contains non-finite values (NaN or inf).")
    
    if not isinstance(prominence, (int, float)):
        raise TypeError("'prominence' must be a real number (int or float).")

    N = len(y)

    yf, freqs = fft(y, t) 

    pos_mask = freqs > 0
    freqs = freqs[pos_mask]
    yf = yf[pos_mask]

    power = _np.abs(yf)
    phases = _np.angle(yf)

    peaks, _ = find_peaks(power, prominence=prominence * _np.max(power))
    sorted_peaks = peaks[_np.argsort(power[peaks])[::-1]]

    if n_max is not None:
        if not isinstance(prominence, (int)):
            raise TypeError("'prominence' must be an integer.")
        sorted_peaks = sorted_peaks[:n_max]

    harmonics = []
    if sorted_peaks.size > 0:
        # Phase of the first (most prominent) harmonic
        reference_phase = phases[sorted_peaks[0]]
    else:
        reference_phase = 0  # Fallback if no peaks are found

    for idx in sorted_peaks:
        harmonics.append({
            "frequency": freqs[idx],
            "amplitude": 2 * power[idx] / N,
            "phase": phases[idx] - reference_phase  # Phase relative to the first harmonic
        })

    if verbose:
        def format_dynamic_number(val):
            if val == 0:
                return "0"
            abs_val = abs(val)
            if abs_val >= 1e4 or abs_val <= 1e-3:
                return f"{val:.3e}"  # Scientific format
            elif abs_val >= 1:
                return f"{val:.4f}".rstrip('0').rstrip('.')  # Decimal up to 4 places
            else:
                return f"{val:.4g}"  # Significant figures

        headers = ["Harmonic", "Frequency", "Amplitude", "Phase"]

        # Prepare rows with formatted values
        rows = []
        for i, h in enumerate(harmonics, 1):
            row = [
                f"H#{i}",
                format_dynamic_number(h["frequency"]),
                format_dynamic_number(h["amplitude"]),
                format_dynamic_number(h["phase"])
            ]
            rows.append(row)

        # Calculate maximum width for each column
        col_widths = []
        for col_idx, header in enumerate(headers):
            max_data_width = max(len(row[col_idx]) for row in rows) if rows else len(header)
            col_width = max(len(header), max_data_width)
            col_widths.append(col_width)

        # Build header line
        header_line = f"{headers[0]:<{col_widths[0]}} | " + " | ".join(
            f"{headers[i]:<{col_widths[i]}}" for i in range(1, len(headers))
        )
        divider = "-" * len(header_line)

        # Pretty print
        print()
        print("=" * len(header_line))
        print(header_line)
        print(divider)
        for row in rows:
            line = f"{row[0]:<{col_widths[0]}} | " + " | ".join(
                f"{row[i]:<{col_widths[i]}}" for i in range(1, len(row))
            )
            print(line)
        print("=" * len(header_line))

    return harmonics

def decompose(t, y, freqs, verbose = True):
    """
    Reconstructs specified sinusoidal components from a real-valued signal.

    Given a set of target frequencies, this function fits the time‑domain data
    y(t) to a sum of sinusoids at those frequencies (via linear least squares),
    and returns the amplitude and phase of each component. Use this when
    you already know which harmonics to extract.

    Parameters
    ----------
    t : array-like
        Time array.
    y : array-like
        Signal samples.
    freqs : array like
        Frequencies (Hz) of the components to extract.
    verbose : bool, optional
        If `True`, prints a formatted table of the components.

    Returns
    -------
    components : list of dict
        Each dict contains frequency (Hz), amplitude, and phase (rad).
    """

    from numpy.linalg import lstsq

    t = _np.asarray(t)
    y = _np.asarray(y)
    freqs = _np.asarray(freqs)

    if t.size != y.size:
        raise ValueError("'t' must have the same length as 'y'")
    
    if not (_np.issubdtype(y.dtype, _np.number)):
        raise TypeError("'y' must contain only numeric types (int, float, or complex).")
    
    if (not (_np.issubdtype(t.dtype, _np.floating) or _np.issubdtype(t.dtype, _np.integer))) or not _np.all(_np.isreal(t)):
        raise TypeError("'t' must contain only real numbers (int or float).")
    
    if (not (_np.issubdtype(freqs.dtype, _np.floating) or _np.issubdtype(freqs.dtype, _np.integer))) or not _np.all(_np.isreal(freqs)):
        raise TypeError("'freqs' must contain only real numbers (int or float).")
    
    if not _np.all(_np.isfinite(y)):
            raise ValueError("'data' contains non-finite values (NaN or inf).")
    if not _np.all(_np.isfinite(t)):
            raise ValueError("'t' contains non-finite values (NaN or inf).")

    A = []
    for f in freqs:
        A.append(_np.sin(2 * _np.pi * f * t))
        A.append(_np.cos(2 * _np.pi * f * t))
    A = _np.vstack(A).T

    coeffs, _, _, _ = lstsq(A, y, rcond=None)
    components = []

    for i, f in enumerate(freqs):
        sin_coef = coeffs[2 * i]
        cos_coef = coeffs[2 * i + 1]
        amplitude = _np.hypot(sin_coef, cos_coef)
        phase = _np.arctan2(cos_coef, sin_coef)
        components.append({
            "frequency": f,
            "amplitude": amplitude,
            "phase": phase
        })

    if verbose:
        def format_dynamic_number(val):
            if val == 0:
                return "0"
            abs_val = abs(val)
            if abs_val >= 1e4 or abs_val <= 1e-3:
                return f"{val:.3e}"  # formato scientifico
            elif abs_val >= 1:
                return f"{val:.4f}".rstrip('0').rstrip('.')  # decimali fino a 4 cifre
            else:
                return f"{val:.4g}"  # cifre significative


        headers = ["Frequency", "Amplitude", "Phase"]

            # Prepara righe con valori formattati
        rows = []
        for i, h in enumerate(components, 1):
            row = [
                format_dynamic_number(h["frequency"]),
                format_dynamic_number(h["amplitude"]),
                format_dynamic_number(h["phase"])
            ]
            rows.append(row)

        # Calcola larghezza massima per ogni colonna
        col_widths = []
        for col_idx, header in enumerate(headers):
            max_data_width = max(len(row[col_idx]) for row in rows)
            col_width = max(len(header), max_data_width)
            col_widths.append(col_width)

        # Costruisci la riga dell'intestazione
        header_line = f"{headers[0]:<{col_widths[0]}} | " + " | ".join(
            f"{headers[i]:<{col_widths[i]}}" for i in range(1, len(headers))
        )
        divider = "-" * len(header_line)

        # Stampa elegante
        print()
        print("=" * len(header_line))
        print(header_line)
        print(divider)
        for row in rows:
            line = f"{row[0]:<{col_widths[0]}} | " + " | ".join(
                f"{row[i]:<{col_widths[i]}}" for i in range(1, len(row))
            )
            print(line)
        print("=" * len(header_line))

    return components

def convolve(a, v, mode='full', use_fft=True, axis=-1):
    """
    Compute the discrete, linear convolution of two input arrays with optional FFT acceleration.

    Parameters
    ----------
    a : array-like
        First input array.
    v : array-like
        Second input array (the filter or kernel).
    mode : {'full', 'valid', 'same'}, optional
        Indicates the size of the output:
        - 'full': (default) return the full convolution.
        - 'same': return output of the same size as `a`.
        - 'valid': return only those parts where `v` fully overlaps `a`.
    use_fft : bool, optional
        If True, use FFT-based convolution when it is more efficient.
        For small arrays, direct method is used.
    axis : int, optional
        Axis along which to perform the convolution.

    Returns
    -------
    output : ndarray
        Convolution of `a` with `v`, along the specified axis.

    Raises
    ------
    ValueError
        If `mode` is invalid or inputs are not broadcastable.

    Notes
    -----
    This function extends `np.convolve` by supporting n-dimensional input along a specific axis,
    automatic dispatch between FFT and direct convolution, and better handling of numeric types.
    """
    from scipy.signal import fftconvolve

    a = _np.asarray(a)
    v = _np.asarray(v)

    if not (_np.issubdtype(a.dtype, _np.number)):
        raise TypeError("'a' must contain only numeric types (int, float, or complex).")

    if not (_np.issubdtype(b.dtype, _np.number)):
        raise TypeError("'b' must contain only numeric types (int, float, or complex).")
    
    if not _np.all(_np.isfinite(a)):
        raise ValueError("'a' contains non-finite values (NaN or inf).")
    if not _np.all(_np.isfinite(b)):
        raise ValueError("'b' contains non-finite values (NaN or inf).")

    if mode not in ('full', 'same', 'valid'):
        raise ValueError("`mode` must be one of {'full', 'same', 'valid'}.")

    if a.ndim == 1 and v.ndim == 1:
        if use_fft and (a.size + v.size > 500):  # empirical threshold
            return fftconvolve(a, v, mode=mode)
        else:
            return _np.convolve(a, v, mode=mode)

    # If axis is not last, move it to the end
    a = _np.moveaxis(a, axis, -1)
    result_shape = list(a.shape)
    result_shape[-1] = {
        'full': a.shape[-1] + v.shape[-1] - 1,
        'same': a.shape[-1],
        'valid': max(a.shape[-1] - v.shape[-1] + 1, 0)
    }[mode]

    output = _np.empty(result_shape, dtype=_np.result_type(a, v))
    it = _np.nditer(a[..., 0], flags=['multi_index'])

    for _ in it:
        idx = it.multi_index
        slice_a = a[idx + (slice(None),)]
        if use_fft and (slice_a.size + v.size > 500):
            output[idx + (slice(None),)] = fftconvolve(slice_a, v, mode=mode)
        else:
            output[idx + (slice(None),)] = _np.convolve(slice_a, v, mode=mode)

    return np.moveaxis(output, -1, axis)

def envelope(signal, method='hilbert', mode='upper', filter_size=31, fs=1.0):
    """
    Compute the envelope of a 1D real-valued signal.

    Parameters
    ----------
    signal : array_like
        The input real-valued signal.
    method : {'hilbert', 'abs_filt'}, optional
        The method to compute the envelope:
        - 'hilbert': uses the analytic signal via the Hilbert transform.
        - 'abs_filt': uses |signal| filtered with a moving average or median.
    mode : {'upper', 'lower', 'both'}, optional
        Select which envelope to return:
        - 'upper': the upper envelope (default).
        - 'lower': the lower envelope.
        - 'both' : returns a tuple (upper, lower).
    filter_size : int, optional
        Size of the smoothing filter (only used for 'abs_filt' method). Must be odd.
    fs : float, optional
        Sampling frequency (only relevant for some future extensions or plotting).

    Returns
    -------
    envelope : ndarray or tuple of ndarrays
        The computed envelope(s), depending on `mode`.

    Notes
    -----
    - The Hilbert method provides a mathematically correct envelope for bandpass signals.
    - The abs_filt method is more robust to noise and useful when the signal is not analytic.
    - The 'lower' envelope is the negative of the envelope of -signal.

    References
    ----------
    - Oppenheim, A. V., & Schafer, R. W. (2010). Discrete-Time Signal Processing.
    - Boashash, B. (1992). Estimating and interpreting the instantaneous frequency of a signal.
    """
    from scipy.signal import hilbert, medfilt

    signal = _np.asarray(signal)

    if signal.ndim != 1:
        raise ValueError("Input signal must be one-dimensional.")
    if filter_size % 2 == 0:
        raise ValueError("filter_size must be an odd integer.")

    if (not (_np.issubdtype(signal.dtype, _np.floating) or _np.issubdtype(signal.dtype, _np.integer))) or not _np.all(_np.isreal(signal)):
        raise TypeError("'signal' must contain only real numbers (int or float).")
    
    if not _np.all(_np.isfinite(signal)):
            raise ValueError("'signal' contains non-finite values (NaN or inf).")

    if method == 'hilbert':
        analytic = hilbert(signal)
        env = _np.abs(analytic)

    elif method == 'abs_filt':
        abs_sig = _np.abs(signal)
        env = medfilt(abs_sig, kernel_size=filter_size)

    else:
        raise ValueError("Invalid method. Choose 'hilbert' or 'abs_filt'.")

    if mode == 'upper':
        return env
    elif mode == 'lower':
        return -envelope(-signal, method=method, mode='upper', filter_size=filter_size, fs=fs)
    elif mode == 'both':
        upper = env
        lower = -envelope(-signal, method=method, mode='upper', filter_size=filter_size, fs=fs)
        return upper, lower
    else:
        raise ValueError("Invalid mode. Choose 'upper', 'lower', or 'both'.")