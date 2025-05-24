import warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from numpy.linalg import lstsq
from scipy.integrate import quad, simpson
import scipy.fft as four
from scipy.signal import find_peaks
from ._helper import ispow2, fft_cooley_tukey, dft_direct, idft_direct, ifft_cooley_tukey

def fft(data, t = None):
    """
    Compute the Fast Fourier Transform (FFT) of a signal.

    Parameters
    ----------
    data : numpy.ndarray
        Input signal as a 1D NumPy array (real or complex).
    t : numpy.ndarray, optional
        Time samples corresponding to the signal. If provided, must be a 1D NumPy
        array of the same length as `data`, monotonically increasing, and uniformly
        spaced. If None, only the FFT is returned.

    Returns
    -------
    X : numpy.ndarray
        The FFT of the input signal, a 1D complex array of the same length as `data`.
    f : numpy.ndarray, optional
        The frequency bins (in Hz) corresponding to the FFT coefficients, returned
        only if `t` is provided. Same length as `data`.

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
    - For non-power-of-2 lengths, the signal is zero-padded to the next power of 2.
    - The FFT is computed using the Cooley-Tukey algorithm for power-of-2 lengths.
    """

    if data.size == 0:
        return np.array([])
    if data.size <= 1:
        return data
    elif data.size <= 16:
        X = dft_direct(data)
    elif ispow2(data.size):
        X = fft_cooley_tukey(data)
    else:
        M = int(2 ** np.ceil(np.log2(data.size)))
        padded = np.pad(data, (0, M - data.size), mode="constant")
        X = fft_cooley_tukey(padded)
        X = X[:data.size]
    
    if t is not None:
        if t.size != data.size:
            raise ValueError("t must have the same length as data")
        if not np.all(np.diff(t) > 0):
            raise ValueError("t must be monotonically increasing")
        if not np.allclose(np.diff(t), np.diff(t)[0], rtol=1e-5):
            raise ValueError("t must be uniformly spaced (equispaced)")

        dt = (t[-1] - t[0]) / (t.size - 1) if t.size > 1 else 1.0
        f = np.fft.fftfreq(data.size, d=dt)

        # Riordina le frequenze e i coefficienti FFT
        order = np.argsort(f)
        f_sorted = f[order]
        X_sorted = X[order]

        return X_sorted, f_sorted
    return X

def ifft(data, freq=None):
    """
    Compute the Inverse Fast Fourier Transform (IFFT) of a signal.

    Parameters
    ----------
    data : numpy.ndarray
        Input frequency domain signal as a 1D NumPy array (complex).
    freq : numpy.ndarray, optional
        Frequency samples corresponding to the signal. If provided, must be a 1D NumPy
        array of the same length as `data`, monotonically increasing, and uniformly
        spaced. If None, only the IFFT is returned.

    Returns
    -------
    x : numpy.ndarray
        The IFFT of the input signal, a 1D complex array of the same length as `data`.
    t : numpy.ndarray, optional
        The time bins (in seconds) corresponding to the IFFT coefficients, returned
        only if `freq` is provided. Same length as `data`.

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

    if data.size == 0:
        return np.array([])
    if data.size <= 1:
        return data
    elif data.size <= 16:
        x = idft_direct(data)
    elif ispow2(data.size):
        x = ifft_cooley_tukey(data)
    else:
        M = int(2 ** np.ceil(np.log2(data.size)))
        padded = np.pad(data, (0, M - data.size), mode="constant")
        x = ifft_cooley_tukey(padded)
        x = x[:data.size]
    
    if freq is not None:
        if freq.size != data.size:
            raise ValueError("freq must have the same length as data")
        if not np.all(np.diff(freq) > 0):
            raise ValueError("freq must be monotonically increasing")
        if not np.allclose(np.diff(freq), np.diff(freq)[0], rtol=1e-5):
            raise ValueError("freq must be uniformly spaced (equispaced)")
        
        # Calcola l'intervallo di campionamento nel dominio temporale
        df = (freq[-1] - freq[0]) / (freq.size - 1) if freq.size > 1 else 1.0
        fs = 1 / df  # Frequenza di campionamento
        dt = 1 / (freq.size * df)  # Intervallo di tempo
        t = np.arange(0, data.size) * dt  # Array dei tempi
        return x, t
    return x

def dfs(t, data, order, plot=True, apply_filter=True, xlabel = "x [ux]", ylabel = "y [uy]", xscale = 0, yscale = 0):
    """
    Computes the discrete Fourier series approximation of a sampled function.

    Parameters
    ----------
    t : ndarray
        Array of sample points (must be uniformly spaced).
    data : ndarray
        Array of function values sampled at points t.
    order : int
        Order of the Fourier approximation (number of harmonics).
    plot : bool, optional
        If True, plots the original function and its Fourier approximation.
    apply_filter : bool, optional
        If True, applies a basic low-pass filter to reduce high-frequency noise.
    xlabel : str, optional
        Label for the x-axis, including units in square brackets (e.g., "Time [s]").
    ylabel : str, optional
        Label for the y-axis, including units in square brackets (e.g., "Intensity [V]").
    xscale : int, optional
        Scaling factor for the x-axis (e.g., `xscale = -3` corresponds to 1e-3, to convert seconds to milliseconds).
    yscale : int, optional
        Scaling factor for the y-axis.

    Returns
    -------
    f_approx : ndarray
        Array of the same shape as `t`, containing the values of the Fourier series approximation
        of the input function at each point in `t`.
    a0 : float
        Zeroth Fourier coefficient (mean value component of the function over the period). It 
        corresponds to the constant term of the Fourier series.
    a_n : ndarray
        Array of cosine coefficients (Fourier coefficients of the even part of the function),
        corresponding to each harmonic up to the specified order (excluding a0).
    b_n : ndarray
        Array of sine coefficients (Fourier coefficients of the odd part of the function),
        corresponding to each harmonic up to the specified order.

    Notes
    ----------
    The values of `xscale` and `yscale` affect only the axis scaling in the plot. All outputs are estimated using the original input data as provided.
    """

    xscale = 10**xscale
    yscale = 10**yscale

    if not np.allclose(np.diff(t), t[1] - t[0], rtol=1e-4):
        raise ValueError("Input array t must be uniformly spaced.")

    N = len(t)
    dt = t[1] - t[0]
    fs = 1 / dt                  # Sampling frequency
    f_nyquist = fs / 2          # Nyquist frequency
    max_freq = order / (t[-1] - t[0])  # Maximum frequency in Fourier expansion

    if max_freq > f_nyquist:
        warnings.warn(
            f"Aliasing may occur: max Fourier frequency ({max_freq:.2f} Hz) "
            f"exceeds Nyquist frequency ({f_nyquist:.2f} Hz).", RuntimeWarning
        )

    a = t[0]
    b = t[-1]
    T = b - a  # Assumed period of the signal

    # Zeroth coefficient
    a0 = 2 * simpson(data, t) / T

    a_n = []
    b_n = []

    for n in range(1, order):
        cos_term = np.cos(2 * np.pi * n * t / T)
        sin_term = np.sin(2 * np.pi * n * t / T)

        an = 2 * simpson(data * cos_term, t) / T
        bn = 2 * simpson(data * sin_term, t) / T

        if apply_filter:
            # Apply exponential decay to filter high-frequency components
            decay = np.exp(- (n / order)**2)
            an *= decay
            bn *= decay

        a_n.append(an)
        b_n.append(bn)

    # Construct approximation
    f_approx = np.full_like(t, a0 / 2)

    for n in range(1, order):
        f_approx += (
            a_n[n - 1] * np.cos(2 * np.pi * n * t / T) +
            b_n[n - 1] * np.sin(2 * np.pi * n * t / T)
        )

    if plot:
        plt.plot(t / xscale, data / yscale, label="Input data", lw=1, color = "blue")
        plt.plot(t / xscale, f_approx / yscale, label=f"Fourier approx. (order = {order})", lw=1, color = "red")
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()

    return f_approx, a0, a_n, b_n

def fourier_series(f, interval, order, num_points=1000, xlabel = "x [ux]", ylabel = "y [uy]", xscale = 0, yscale = 0):
    """
    Computes the Fourier series approximation of a function f(x)
    
    Parameters
    ----------
    f : callable
        Function to approximate.
    interval : tuple of float
        The interval (a, b) over which to compute the Fourier series.
    order : int
        Number of Fourier modes (n) to use in the approximation.
    num_points : int, optional
        Number of points for plotting (default is 1000).
        
    Returns
    -------
    x : ndarray
        Array of shape (N,), representing the uniformly spaced sample points over one period.
        These are the evaluation points at which both the original function and the Fourier
        approximation are computed.
    f_original : ndarray
        Array of shape (N,), representing the values of the original input function evaluated at 
        the sample points `x`. This is the reference signal used for comparison with the Fourier 
        approximation.
    f_approx : ndarray
        Array of shape (N,), containing the values of the truncated Fourier series evaluated at 
        the same sample points `x`. This is the approximation of `f_original` using a finite number 
        of harmonics (up to the specified order).
    a0 : float
        Zeroth Fourier coefficient (mean value component of the function over the period). It 
        corresponds to the constant term of the Fourier series.
    a_n : ndarray
        Array of cosine coefficients (Fourier coefficients of the even part of the function),
        corresponding to each harmonic up to the specified order (excluding a0).
    b_n : ndarray
        Array of sine coefficients (Fourier coefficients of the odd part of the function),
        corresponding to each harmonic up to the specified order.

    Notes
    ----------
    The values of `xscale` and `yscale` affect only the axis scaling in the plot. All parameters are estimated using the original input data as provided.
    """

    xscale = 10**xscale
    yscale = 10**yscale

    # aggiungere nota che la yscale è solo a livello grafico

    a, b = interval
    T = b - a  # Period
    L = T / 2
    x = np.linspace(a, b, num_points)
    
    # Compute a0 separately
    a0, _ = quad(lambda x_: f(x_), a, b)
    a0 /= L

    # Compute coefficients a_n and b_n
    a_n = []
    b_n = []

    for n in range(1, order + 1):
        an, _ = quad(lambda x_: f(x_) * np.cos(n * np.pi * (x_ - a) / L), a, b)
        bn, _ = quad(lambda x_: f(x_) * np.sin(n * np.pi * (x_ - a) / L), a, b)
        a_n.append(an / L)
        b_n.append(bn / L)

    # Build the Fourier approximation
    f_approx = np.full_like(x, a0 / 2)

    for n in range(1, order + 1):
        f_approx += (
            a_n[n - 1] * np.cos(n * np.pi * (x - a) / L) +
            b_n[n - 1] * np.sin(n * np.pi * (x - a) / L)
        )

    f_original = f(x)

    # Plot
    plt.plot(x / xscale, f_original / yscale, label="Input function", lw=0.8, color = "blue")
    plt.plot(x / xscale, f_approx / yscale, '--', label=f"Fourier approx. (order = {order})", lw=0.8, color = "red")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()

    return f_original, f_approx, a0, a_n, b_n

def psd(data, t = None, plot = True, spectrogram = False, log = None, logcs = False, psd_unit = None, time_unit = "s", freq_unit = "Hz", color = "k"):
    """
    Compute the Power Spectral Density (PSD) of a signal using the FFT.

    Parameters
    ----------
    data : numpy.ndarray
        Input signal as a 1D NumPy array (real or complex).
    t : numpy.ndarray, optional
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

    # --- Validazioni iniziali ---
    if log is not None and log not in ("x", "y", "xy", "cs"):
        raise ValueError("log must be one of: 'x', 'y', 'xy', 'cs' or None.")
    
    if logcs and not spectrogram:
        warnings.warn("logcs=True has no effect unless spectrogram=True.", RuntimeWarning)
    
    if spectrogram and t is None:
        raise ValueError("Parameter 't' must be provided if spectrogram=True.")
    
    if logcs and t is None:
        warnings.warn("logcs=True requires 't' to be defined.", RuntimeWarning)

    if data.size == 0:
        return np.array([]), np.array([])

    if psd_unit == None or psd_unit == "":
        label_psd = f"Power Spectral Density"
    else:
        label_psd = f"Power Spectral Density [{psd_unit}]"

    # --- Calcolo FFT e frequenze ---
    X, freqs = fft(data, t)
    
    N = data.size
    S = np.abs(X)**2
    # S = S[:N//2 + 1]       # Frequenze positive
    # freqs = freqs[:N//2 + 1]

    # --- Calcolo fs e normalizzazione ---
    fs = 1.0
    if t is not None:
        if t.size != N:
            raise ValueError("Array 't' must have the same length as 'data'.")
        if not np.all(np.diff(t) > 0):
            raise ValueError("Array 't' must be strictly increasing.")
        if not np.allclose(np.diff(t), np.diff(t)[0], rtol=1e-5):
            raise ValueError("Array 't' must be uniformly spaced.")
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
                freqs_full = np.fft.fftfreq(N, d=1/fs_seg)
                pos_mask = freqs_full >= 0
                freqs = freqs_full[pos_mask][:N//2 + 1]
                S_seg = np.abs(X)[pos_mask][:N//2 + 1] ** 2
                S_seg[1:] *= 2 / (fs_seg * N)
                S_seg[0] /= (fs_seg * N)
                S_list.append(S_seg)
                t_spec.append(np.mean(seg_t))

            S_array = np.array(S_list).T
            t_spec = np.array(t_spec)

            if not np.all(np.diff(t_spec) > 0):
                raise ValueError("t_spec is not monotonically increasing.")
            if not np.all(np.diff(freqs) > 0):
                raise ValueError("freqs is not monotonically increasing.")

            fig, ax = plt.subplots()
            if log in ("x", "xy"):
                ax.set_xscale("log")
            if log in ("y", "xy"):
                ax.set_yscale("log")

            if logcs:
                norm = LogNorm(vmin=S_array[S_array > 0].min(), vmax=S_array.max())
            else:
                norm = None

            T, F = np.meshgrid(t_spec, freqs)
            t_edges = np.concatenate([t_spec - np.diff(t_spec, append=t_spec[-1])/2, [t_spec[-1] + (t_spec[-1] - t_spec[-2])/2]])
            f_edges = np.concatenate([freqs - np.diff(freqs, append=freqs[-1])/2, [freqs[-1] + (freqs[-1] - freqs[-2])/2]])
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
                plt.xscale("log")

            if log in ("y", "xy"):
                plt.yscale("log")

            plt.plot(f_plot, S_plot, lw = 0.8, color = color)

            if f_plot.size > 0:
                plt.xlim(f_plot[0], f_plot[-1])
            plt.xlabel(f"Frequency [{freq_unit}]")
            plt.ylabel(label_psd)

    if t is not None:
        return S, freqs
    return S

def harmonic(t, y, prominence=0.05, n_max=None):
    """
    Identifies the dominant harmonics present in a real-valued signal.

    This function computes the FFT of the input signal y(t), finds the peaks
    in its amplitude spectrum, and returns their frequencies, amplitudes, and phases.
    It does *not* reconstruct the time‑domain components—only reports what
    harmonics are strongest.

    Parameters
    ----------
    t : array_like
        Time array.
    y : array_like
        Signal samples corresponding to `t`.
    prominence : float, optional
        Minimum prominence of peaks in the power spectrum. Default is 0.05.
    n_max : int or None, optional
        If given, return at most `n_max` harmonics.

    Returns
    -------
    harmonics : list of dict
        Each dict contains frequency (Hz), amplitude, and phase (rad) of a harmonic component.
    """
    t = np.asarray(t)
    y = np.asarray(y)
    dt = t[1] - t[0]
    N = len(y)

    yf = four.fft(y)
    freqs = four.fftfreq(N, d=dt)

    pos_mask = freqs > 0
    freqs = freqs[pos_mask]
    yf = yf[pos_mask]

    power = np.abs(yf)
    phases = np.angle(yf)

    peaks, _ = find_peaks(power, prominence=prominence * np.max(power))
    sorted_peaks = peaks[np.argsort(power[peaks])[::-1]]

    if n_max is not None:
        sorted_peaks = sorted_peaks[:n_max]

    harmonics = []
    for idx in sorted_peaks:
        harmonics.append({
            "frequency": freqs[idx],
            "amplitude": 2 * power[idx] / N,
            "phase": phases[idx]
        })

    return harmonics

def decompose(t, y, freqs):
    """
    Reconstructs specified sinusoidal components from a real-valued signal.

    Given a set of target frequencies, this function fits the time‑domain data
    y(t) to a sum of sinusoids at those frequencies (via linear least squares),
    and returns the amplitude and phase of each component. Use this when
    you already know which harmonics to extract.

    Parameters
    ----------
    t : array_like
        Time array.
    y : array_like
        Signal samples.
    freqs : array_like
        Frequencies (Hz) of the components to extract.

    Returns
    -------
    components : list of dict
        Each dict contains frequency (Hz), amplitude, and phase (rad).
    """
    t = np.asarray(t)
    y = np.asarray(y)
    freqs = np.asarray(freqs)

    A = []
    for f in freqs:
        A.append(np.sin(2 * np.pi * f * t))
        A.append(np.cos(2 * np.pi * f * t))
    A = np.vstack(A).T

    coeffs, _, _, _ = lstsq(A, y, rcond=None)
    components = []

    for i, f in enumerate(freqs):
        sin_coef = coeffs[2 * i]
        cos_coef = coeffs[2 * i + 1]
        amplitude = np.hypot(sin_coef, cos_coef)
        phase = np.arctan2(cos_coef, sin_coef)
        components.append({
            "frequency": f,
            "amplitude": amplitude,
            "phase": phase
        })

    return components

def quality(t, signal, nev, responsivity = None):
    """
    Calculate the Signal-to-Noise Ratio (S/N) and Noise Equivalent Power (NEP).

    Parameters:
    -----------
    t : array-like
        Array of time values (in seconds).
    signal : array-like
        Array of signal amplitudes (e.g., in volts).
    nev : float
        Noise Equivalent Voltage (in volts).
    responsivity : float, optional
        System responsivity (in V/W). If None, NEP is not calculated.

    Returns:
    --------
    snr : float
        Signal-to-Noise Ratio (dimensionless).
    nep : float or None
        Noise Equivalent Power (in W/sqrt(Hz)) or None if responsivity is not provided.
    bandwidth : float
        Effective signal bandwidth (in Hz).
    """
    
    # Convert inputs to numpy arrays
    t = np.array(t)
    signal = np.array(signal)
    
    # Calculate time step and sampling frequency
    dt = np.mean(np.diff(t))
    fs = 1 / dt  # Sampling frequency
    
    # Calculate RMS of the signal
    signal_rms = np.sqrt(np.mean(signal**2))
    
    # Calculate S/N
    snr = signal_rms / nev
    
    # Calculate bandwidth using FFT
    N = len(signal)
    yf = fft(signal)
    xf = four.fftfreq(N, dt)[:N//2]  # Positive frequencies
    power_spectrum = np.abs(yf[:N//2])**2 / N  # Power spectrum
    
    # Calculate effective (RMS) bandwidth
    power_total = np.sum(power_spectrum)
    freq_weighted = np.sum(xf * power_spectrum) / power_total
    freq_squared_weighted = np.sum(xf**2 * power_spectrum) / power_total
    bandwidth = np.sqrt(freq_squared_weighted - freq_weighted**2)
    
    # Calculate NEP
    nep = None
    if responsivity is not None:
        nep = nev / responsivity / np.sqrt(bandwidth)

    print(f"S/N:\t\t{snr:.2f}")
    print(f"Bandwidth:\t{bandwidth:.2f} Hz")
    if nep is not None:
        print(f"NEP:\t\t{nep:.2e} W/sqrt(Hz)")
    else:
        print("NEP not calculated: provide responsivity.")
    
    return snr, nep, bandwidth