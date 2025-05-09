import warnings
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import lstsq
from scipy.integrate import quad, simpson
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks

# def power_spectrum(t, y, interval = None, xscale = 0, yscale = 0, normalize=True, logscale=True, density=False, unit=""):
#     """
#     Computes and plots the power spectrum or power spectral density (PSD)
#     of a real-valued discrete signal.

#     Parameters
#     ----------
#     t : array_like
#         Time array corresponding to the samples of the signal.
#     y : array_like
#         Input signal, assumed to be real and sampled at times given by `t`.
#     interval : tuple of float or None, optional
#         The interval (a, b) over which to view the plot. Default is `None.`
#     xscale : int, optional
#         If non-zero, rescale frequency axis by 10**(-xscale) and update label.
#     xscale : int, optional
#         If non-zero, rescale y-axis by 10**(-yscale) and update label.
#     normalize : bool, optional
#         If True, normalize the spectrum so that its sum equals 1.
#     logscale : bool, optional
#         If True, display the power in log scale (dB). Default is True.
#     density : bool, optional
#         If True, return the PSD (power per Hz). Default is False.
#     unit : str, optional
#         Physical unit of the signal y (e.g. "V", "A", "counts"). This will appear
#         (squared) in the y‑axis label when `logscale = False`.

#     Returns
#     -------
#     freqs : ndarray
#         Array of frequencies (Hz).
#     power : ndarray
#         Power spectrum or PSD (normalized if normalize=True).

#     Notes
#     -------
#     LaTeX formatting is already embedded in the strings used to display the units of the y-axis. You do not need to use "$...$".
#     """

#     t = np.asarray(t)
#     y = np.asarray(y)
#     if len(t) != len(y):
#         raise ValueError("Time and signal arrays must have the same length.")

#     dt = np.mean(np.diff(t))
#     N = len(y)
#     yf = np.fft.fft(y)
#     freqs = np.fft.fftfreq(N, d=dt)

#     idx = freqs > 0
#     freqs = freqs[idx]
#     power = np.abs(yf[idx])**2

#     if density:
#         power *= 2 * dt / N
#     else:
#         power *= 2 / N**2

#     if normalize:
#         power /= np.sum(power)

#     if logscale:
#         power_plot = 10 * np.log10(power + 1e-20)
#     else:
#         power_plot = power

#     if xscale != 0:
#         freqs_scaled = freqs * 10**(-xscale)
#         freq_label = f"Frequency [$\\times 10^{{{xscale}}}$ Hz]"
#     else:
#         freqs_scaled = freqs
#         freq_label = "Frequency [Hz]"

#     if yscale != 0:
#         factor = f"\\times 10^{{{yscale}}} "
#     else:
#         factor = ""

#     plt.plot(freqs_scaled, power_plot / 10**yscale, color="blue")
#     if interval is not None:
#         plt.xlim(*interval)
#     plt.xlabel(freq_label)

#     # Costruzione dell'etichetta y includendo l'unità
#     if logscale:
#         y_label = f"Power spectral density [{factor}dB$\,$Hz$^{{-1}}$]" if density else f"Power spectrum [{factor}dB]"
#     else:
#         # se c'è un'unità, la eleviamo al quadrato
#         unit_sq = f"$\\mathrm{{{unit}}}^2$" if unit else ""
#         y_label = f"Power spectral density [{factor}{unit_sq}$\,$Hz$^{{-1}}$]" if density else f"Power spectrum [{factor}{unit_sq}]"

#     plt.ylabel(y_label)

#     return freqs, power

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

    yf = fft(y)
    freqs = fftfreq(N, d=dt)

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