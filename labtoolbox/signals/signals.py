import numpy as _np
import matplotlib.pyplot as _plt
import warnings
from typing import Union, Tuple, Optional, Callable, List
from numpy.typing import ArrayLike
from .._helper import GenericError

# __all__ = ["fft", "ifft", "dfs", "fourier_series", "decompose", "harmonic", "envelope"]

def fft(
    data: _np.ndarray,
    t: Optional[Union[_np.ndarray, Tuple[_np.ndarray, _np.ndarray]]] = None,
    dt: Optional[Union[float, Tuple[float, float]]] = None,
    oversample: int = 2
) -> Union[_np.ndarray, Tuple[_np.ndarray, _np.ndarray], Tuple[_np.ndarray, _np.ndarray, _np.ndarray]]:
    """
    Compute the Fast Fourier Transform (FFT) of a signal.

    For 1D signals, supports both uniformly and non-uniformly sampled data using FFT or
    Non-Uniform FFT (NUFFT). For 2D signals, supports only uniformly sampled data using
    2D FFT applied along rows and columns.

    Parameters
    ----------
    data : ndarray
        Input signal as a 1D or 2D NumPy array (real or complex). For 1D, shape `(N,)`.
        For 2D, shape `(N, M)`.
    t : ndarray or tuple of ndarray, optional
        Time samples:

        - For 1D: 1D array of shape (N,), monotonically increasing.
        - For 2D: Tuple `(t1, t2)` where `t1` and `t2` are 1D arrays, monotonically increasing.

        If `None`, only `X` is returned unless `dt` is provided. Defaults to `None`.
    dt : float or tuple of float, optional
        Time interval(s) for uniform sampling:

        - For 1D: Single float `dt_interval`.
        - For 2D: Tuple `(dt1, dt2)` for intervals along axes.

        Ignored if `t` is provided. Used to calculate frequency bins if `t` is `None`.
        Defaults to `None`.
    oversample : int, optional
        Oversampling factor for NUFFT interpolation in non-uniform 1D case. Must be >= 1.
        Defaults to `2`. Ignored for 2D data.

    Returns
    -------
    X : ndarray
        The FFT or NUFFT of the input signal. For 1D, shape `(N,)`. For 2D, shape `(N, M)`.
        Complex-valued.
    f : ndarray, optional
        For 1D: Frequency bins, shape `(N,)`. Returned only if `t` or `dt` is provided.
    f1, f2 : ndarray, optional
        For 2D: Frequency bins along axes 0 and 1, shapes `(N,)` and `(M,)`, respectively.
        Returned only if `t` or `dt` is provided.

    Notes
    -----
    - For 1D uniform sampling with N ≤ 16, a direct DFT is used. For power-of-2 lengths, the Cooley-Tukey FFT algorithm is used.
    - For 1D non-uniform sampling, a NUFFT algorithm with Gaussian interpolation is used, with O(N log N + M) complexity.
    - For 2D signals, applies 1D FFTs along rows and columns for uniform sampling.
    """
    try:
        from .._helper import ispow2, fft_cooley_tukey, dft_direct, GenericError
        import scipy.fft as spf
        
        # Validate inputs
        data = _np.asarray(data, dtype=complex)
        if data.ndim not in (1, 2):
            raise ValueError("'data' must be a 1D or 2D array.")
        if not _np.all(_np.isfinite(data)):
            raise ValueError("'data' contains non-finite values (NaN or inf).")

        if not isinstance(oversample, int) or oversample < 1:
            raise TypeError("'oversample' must be a positive integer.")

        if data.ndim == 1:
            N = len(data)
            if N == 0:
                warnings.warn("'data' is empty. Returning empty array.", UserWarning)
                return _np.array([])
            if N == 1:
                warnings.warn("'data' is a scalar. Returning 'data'.", UserWarning)
                return data, _np.array([0.0]) if (t is not None or dt is not None) else data

            # Validate t for 1D
            is_uniform = True
            if t is not None:
                t = _np.asarray(t, dtype=float)
                if t.ndim != 1 or t.size != N:
                    raise ValueError("'t' must be a 1D array of length 'N' for 1D data.")
                if not _np.all(_np.isreal(t)):
                    raise TypeError("'t' must contain only real numbers.")
                if not _np.all(_np.isfinite(t)):
                    raise ValueError("'t' contains non-finite values (NaN or inf).")
                if not _np.all(_np.diff(t) > 0):
                    raise ValueError("'t' must be monotonically increasing.")
                # Check uniformity
                if t.size > 1 and not _np.allclose(_np.diff(t), _np.diff(t)[0], rtol=1e-5):
                    warnings.warn("Non-uniform sampling detected. Using NUFFT algorithm.", UserWarning)
                    is_uniform = False

            # Validate dt for 1D
            if dt is not None:
                if not isinstance(dt, (int, float)):
                    raise TypeError("'dt' must be a real number for 1D data.")
                if not _np.isfinite(dt) or dt <= 0:
                    raise ValueError("'dt' must be a positive finite value.")

            # 1D Uniform sampling
            if is_uniform:
                if N <= 16:
                    X = dft_direct(data)
                elif ispow2(N):
                    X = fft_cooley_tukey(data)
                else:
                    M = int(2 ** _np.ceil(_np.log2(N)))
                    padded = _np.pad(data, (0, M - N), mode="constant")
                    X = fft_cooley_tukey(padded)
                    X = X[:N]

                # Determine dt and frequencies
                if t is not None:
                    dt_final = (t[-1] - t[0]) / (N - 1) if N > 1 else 1.0
                elif dt is not None:
                    dt_final = dt
                else:
                    return X

                f = _np.fft.fftfreq(N, d=dt_final)

                # Check aliasing
                spectrum_magnitude = _np.abs(X)
                threshold = _np.max(spectrum_magnitude) * 0.05
                freq_components = _np.abs(f[spectrum_magnitude > threshold])
                if freq_components.size > 0:
                    f_max = _np.max(freq_components)
                    fs = 1 / dt_final
                    if f_max >= fs / 2:
                        warnings.warn(
                            f"Potential aliasing detected: the signal contains frequency components up to {f_max:.2g}, "
                            f"which exceeds the Nyquist frequency ({fs/2:.2g}). "
                            "Consider increasing the sampling frequency or applying an anti-aliasing filter.",
                            UserWarning
                        )

                order = _np.argsort(f)
                return X[order], f[order]

            # 1D Non-uniform sampling: NUFFT
            points = t
            M = N
            t_min, t_max = points.min(), points.max()
            if t_max == t_min:
                raise ValueError("'t' must span a non-zero interval.")
            points_norm = (points - t_min) / (t_max - t_min) - 0.5

            # Generate frequencies
            dt = _np.mean(_np.diff(points)) if N > 1 else 1.0
            fs = 1.0 / dt
            frequencies = _np.linspace(-fs / 2, fs / 2 - fs / M, M)

            # Uniform grid for FFT
            N_grid = oversample * max(N, M)
            h = 1.0 / N_grid
            grid = _np.linspace(-0.5, 0.5 - h, N_grid)

            # Gaussian interpolation kernel
            sigma = 2.0 / oversample
            spread = int(6 * sigma * N_grid)
            spread = max(1, spread // 2 * 2 + 1)
            kernel = _np.exp(-_np.arange(-spread//2 + 1, spread//2 + 1)**2 / (2 * sigma**2))
            kernel /= _np.sum(kernel)

            # Interpolate to uniform grid
            signal_grid = _np.zeros(N_grid, dtype=complex)
            for n in range(N):
                idx = int((points_norm[n] + 0.5) * N_grid)
                for i in range(-spread//2 + 1, spread//2 + 1):
                    j = (idx + i) % N_grid
                    dist = (points_norm[n] - grid[j]) * N_grid
                    signal_grid[j] += data[n] * _np.exp(-dist**2 / (2 * sigma**2))

            # FFT on uniform grid
            fft_grid = spf.fft(signal_grid)

            # Interpolate to target frequencies
            result = _np.zeros(M, dtype=complex)
            freq_norm = frequencies / (t_max - t_min)
            for m in range(M):
                for i in range(N_grid):
                    phase = _np.exp(-2j * _np.pi * freq_norm[m] * grid[i])
                    result[m] += fft_grid[i] * phase * h

            if not _np.all(_np.isfinite(result)):
                raise ValueError("Computed NUFFT contains non-finite values.")

            scaled_frequencies = frequencies * (t_max - t_min)
            return result, scaled_frequencies

        else:  # 2D data
            N, M = data.shape
            if N == 0 or M == 0:
                warnings.warn("'data' is empty. Returning empty array.", UserWarning)
                return _np.array([])
            if N == 1 and M == 1:
                warnings.warn("'data' is a scalar. Returning 'data'.", UserWarning)
                return data, _np.array([0.0]), _np.array([0.0]) if (t is not None or dt is not None) else data

            # Validate t for 2D
            t1, t2 = None, None
            if t is not None:
                if not isinstance(t, tuple) or len(t) != 2:
                    raise TypeError("'t' must be a tuple of two 1D NumPy arrays for 2D data.")
                t1, t2 = _np.asarray(t[0], dtype=float), _np.asarray(t[1], dtype=float)
                if t1.ndim != 1 or t2.ndim != 1 or t1.shape[0] != N or t2.shape[0] != M:
                    raise ValueError("'t1' and 't2' must be 1D arrays of lengths 'N' and 'M', respectively.")
                if not _np.all(_np.isreal(t1)) or not _np.all(_np.isreal(t2)):
                    raise TypeError("'t1' and 't2' must contain only real numbers.")
                if not _np.all(_np.isfinite(t1)) or not _np.all(_np.isfinite(t2)):
                    raise ValueError("'t1' or 't2' contains non-finite values (NaN or inf).")
                if not _np.all(_np.diff(t1) > 0) or not _np.all(_np.diff(t2) > 0):
                    raise ValueError("'t1' and 't2' must be monotonically increasing.")
                # Check uniformity for 2D (required by docstring)
                if t1.size > 1 and not _np.allclose(_np.diff(t1), _np.diff(t1)[0], rtol=1e-5):
                    raise ValueError("Non-uniform sampling not supported for 2D data: 't1' must be uniformly spaced.")
                if t2.size > 1 and not _np.allclose(_np.diff(t2), _np.diff(t2)[0], rtol=1e-5):
                    raise ValueError("Non-uniform sampling not supported for 2D data: 't2' must be uniformly spaced.")

            # Validate dt for 2D
            if dt is not None:
                if not isinstance(dt, tuple) or len(dt) != 2:
                    raise TypeError("'dt' must be a tuple of two floats for 2D data.")
                dt1, dt2 = dt
                if not isinstance(dt1, (int, float)) or not isinstance(dt2, (int, float)):
                    raise TypeError("'dt1' and 'dt2' must be real numbers.")
                if not _np.isfinite(dt1) or not _np.isfinite(dt2) or dt1 <= 0 or dt2 <= 0:
                    raise ValueError("'dt1' and 'dt2' must be positive finite values.")

            # Suppress aliasing warnings during 1D FFT calls
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Potential aliasing detected", category=UserWarning)
                # Initialize output
                X = _np.zeros((N, M), dtype=complex)

                # FFT along rows (axis 1, using t2)
                for i in range(N):
                    row_t = t2 if t is not None else None
                    row_dt = dt2 if dt is not None else None
                    X[i, :], _ = fft(data[i, :], t=row_t, dt=row_dt, oversample=oversample)

                # FFT along columns (axis 0, using t1)
                X = X.T.copy()
                for j in range(M):
                    col_t = t1 if t is not None else None
                    col_dt = dt1 if dt is not None else None
                    X[j, :], _ = fft(X[j, :], t=col_t, dt=col_dt, oversample=oversample)
                X = X.T

            # Compute frequencies
            if t is not None:
                dt1_final = (t1[-1] - t1[0]) / (N - 1) if t1 is not None and N > 1 else 1.0
                dt2_final = (t2[-1] - t2[0]) / (M - 1) if t2 is not None and M > 1 else 1.0
            elif dt is not None:
                dt1_final, dt2_final = dt
            else:
                return X

            f1 = _np.fft.fftfreq(N, d=dt1_final)
            f2 = _np.fft.fftfreq(M, d=dt2_final)

            # Check aliasing
            spectrum_magnitude = _np.abs(X)
            threshold = _np.max(spectrum_magnitude) * 0.05
            freq_components = _np.abs(f1[_np.any(spectrum_magnitude > threshold, axis=1)])
            if freq_components.size > 0:
                f1_max = _np.max(freq_components)
                fs1 = 1 / dt1_final
                if f1_max >= fs1 / 2:
                    warnings.warn(
                        f"Potential aliasing detected on axis 0: the signal contains frequency components up to {f1_max:.2g}, "
                        f"which exceeds the Nyquist frequency ({fs1/2:.2g}). "
                        "Consider increasing the sampling frequency or applying an anti-aliasing filter.",
                        UserWarning
                    )
            freq_components = _np.abs(f2[_np.any(spectrum_magnitude > threshold, axis=0)])
            if freq_components.size > 0:
                f2_max = _np.max(freq_components)
                fs2 = 1 / dt2_final
                if f2_max >= fs2 / 2:
                    warnings.warn(
                        f"Potential aliasing detected on axis 1: the signal contains frequency components up to {f2_max:.2g}, "
                        f"which exceeds the Nyquist frequency ({fs2/2:.2g}). "
                        "Consider increasing the sampling frequency or applying an anti-aliasing filter.",
                        UserWarning
                    )

            order1, order2 = _np.argsort(f1), _np.argsort(f2)
            X = X[order1][:, order2]
            return X, f1[order1], f2[order2]

    except Exception as e:
        raise GenericError(
            message=f"Error computing FFT: {str(e)}",
            context="executing fft",
            original_error=e,
            details={"data_shape": data.shape, "t_type": type(t), "dt_type": type(dt)}
        )

def ifft(data: ArrayLike, freq: Optional[Union[ArrayLike, Tuple[ArrayLike, ArrayLike]]] = None,
        df: Optional[Union[float, Tuple[float, float]]] = None, 
        oversample: int = 2) -> Union[ArrayLike, Tuple[ArrayLike, ArrayLike], Tuple[ArrayLike, ArrayLike, ArrayLike]]:
    """
    Compute the Inverse Fast Fourier Transform (IFFT) of a signal.

    For 1D signals, supports both uniformly and non-uniformly sampled frequency data using
    IFFT or Non-Uniform IFFT (NUIFFT). For 2D signals, supports only uniformly sampled data
    using 2D IFFT applied along rows and columns.

    Parameters
    ----------
    data : ndarray
        Input frequency domain signal as a 1D or 2D NumPy array (complex). For 1D, shape `(N,)`.
        For 2D, shape `(N, M)`.
    freq : ndarray or tuple of ndarray, optional
        Frequency samples:

        - For 1D: 1D array of shape `(N,)`, monotonically increasing.
        - For 2D: Tuple `(f1, f2)` where `f1` (shape `(N,)`) and `f2` (shape `(M,)`) are 1D arrays, monotonically increasing.

        If `None`, time bins are returned only if `df` is provided. Defaults to `None`.
    df : float or tuple of float, optional
        Frequency spacing(s) for uniform sampling:

        - For 1D: Single float `df_interval`.
        - For 2D: Tuple `(df1, df2)` for spacings along axes.

        Ignored if `freq` is provided. Defaults to `None`.
    oversample : int, optional
        Oversampling factor for NUIFFT interpolation in non-uniform 1D case. Must be >= 1.
        Defaults to 2. Ignored for 2D data.

    Returns
    -------
    x : ndarray
        The IFFT or NUIFFT of the input signal. For 1D, shape `(N,)`. For 2D, shape `(N, M)`.
        Complex-valued.
    t : ndarray, optional
        For 1D: Time bins, shape `(N,)`. Returned only if `freq` or `df` is provided.
    t1, t2 : ndarray, optional
        For 2D: Time bins along axes 0 and 1, shapes `(N,)` and `(M,)`, respectively.
        Returned only if `freq` or `df` is provided.

    Notes
    -----

    - For 1D uniform sampling with N <= 16, a direct IDFT is used. For power-of-2 lengths, the Cooley-Tukey IFFT algorithm is used.
    - For 1D non-uniform sampling, a NUIFFT algorithm with Gaussian interpolation is used, with O(N log N + M) complexity.
    - For 2D signals, applies 1D IFFTs along rows and columns for uniform sampling.

    """
    from .._helper import ispow2, idft_direct, ifft_cooley_tukey
    import scipy.fft as spf

    # Validate inputs
    data = _np.asarray(data, dtype=complex)
    if data.ndim not in (1, 2):
        raise ValueError("'data' must be a 1D or 2D array.")
    if not _np.all(_np.isfinite(data)):
        raise ValueError("'data' contains non-finite values (NaN or inf).")

    if not isinstance(oversample, int) or oversample < 1:
        raise TypeError("'oversample' must be a positive integer.")

    if data.ndim == 1:
        N = len(data)
        if N == 0:
            warnings.warn("'data' is empty. Returning empty array.", UserWarning)
            return _np.array([])
        if N == 1:
            warnings.warn("'data' is a scalar. Returning 'data'.", UserWarning)
            return data, _np.array([0.0]) if (freq is not None or df is not None) else data

        # Validate freq for 1D
        is_uniform = True
        if freq is not None:
            freq = _np.asarray(freq, dtype=float)
            if freq.ndim != 1 or freq.size != N:
                raise ValueError("'freq' must be a 1D array of length 'N' for 1D data.")
            if not _np.all(_np.isreal(freq)):
                raise TypeError("'freq' must contain only real numbers.")
            if not _np.all(_np.isfinite(freq)):
                raise ValueError("'freq' contains non-finite values (NaN or inf).")
            if not _np.all(_np.diff(freq) > 0):
                raise ValueError("'freq' must be monotonically increasing.")
            # Check uniformity
            if freq.size > 1 and not _np.allclose(_np.diff(freq), _np.diff(freq)[0], rtol=1e-5):
                warnings.warn("Non-uniform frequency sampling detected. Using NUIFFT algorithm.", UserWarning)
                is_uniform = False

        # Validate df for 1D
        if df is not None:
            if not isinstance(df, (int, float)):
                raise TypeError("'df' must be a real number for 1D data.")
            if not _np.isfinite(df) or df <= 0:
                raise ValueError("'df' must be a positive finite value.")

        # 1D Uniform sampling
        if is_uniform:
            if N <= 16:
                x = idft_direct(data)
            elif ispow2(N):
                x = ifft_cooley_tukey(data)
            else:
                M = int(2 ** _np.ceil(_np.log2(N)))
                padded = _np.pad(data, (0, M - N), mode="constant")
                x = ifft_cooley_tukey(padded)
                x = x[:N]

            if freq is not None or df is not None:
                df_value = (freq[-1] - freq[0]) / (N - 1) if freq is not None and N > 1 else df
                dt = 1 / (N * df_value) if df_value is not None else 1.0
                t = _np.arange(0, N) * dt
                return x, t
            return x

        # 1D Non-uniform sampling: NUIFFT
        try:
            points = freq
            M = N
            f_min, f_max = points.min(), points.max()
            if f_max == f_min:
                raise ValueError("'freq' must span a non-zero interval.")
            points_norm = (points - f_min) / (f_max - f_min) - 0.5

            # Generate time points
            df_value = _np.mean(_np.diff(points)) if N > 1 else 1.0
            fs = 1.0 / df_value
            times = _np.linspace(-fs / 2, fs / 2 - fs / M, M)

            # Uniform grid for FFT
            N_grid = oversample * max(N, M)
            h = 1.0 / N_grid
            grid = _np.linspace(-0.5, 0.5 - h, N_grid)

            # Gaussian interpolation kernel
            sigma = 2.0 / oversample
            spread = int(6 * sigma * N_grid)
            spread = max(1, spread // 2 * 2 + 1)
            kernel = _np.exp(-_np.arange(-spread//2 + 1, spread//2 + 1)**2 / (2 * sigma**2))
            kernel /= _np.sum(kernel)

            # Interpolate to uniform grid
            signal_grid = _np.zeros(N_grid, dtype=complex)
            for n in range(N):
                idx = int((points_norm[n] + 0.5) * N_grid)
                for i in range(-spread//2 + 1, spread//2 + 1):
                    j = (idx + i) % N_grid
                    dist = (points_norm[n] - grid[j]) * N_grid
                    signal_grid[j] += data[n] * _np.exp(-dist**2 / (2 * sigma**2))

            # IFFT on uniform grid
            ifft_grid = spf.ifft(signal_grid)

            # Interpolate to target times
            result = _np.zeros(M, dtype=complex)
            time_norm = times / (f_max - f_min)
            for m in range(M):
                for i in range(N_grid):
                    phase = _np.exp(2j * _np.pi * time_norm[m] * grid[i])
                    result[m] += ifft_grid[i] * phase * h

            if not _np.all(_np.isfinite(result)):
                raise ValueError("Computed NUIFFT contains non-finite values.")

            scaled_times = times * (f_max - f_min)
            return result, scaled_times

        except Exception as e:
            raise ValueError(f"Error computing NUIFFT: {str(e)}")

    else:  # 2D data
        N, M = data.shape
        if N == 0 or M == 0:
            warnings.warn("'data' is empty. Returning empty array.", UserWarning)
            return _np.array([])
        if N == 1 and M == 1:
            warnings.warn("'data' is a scalar. Returning 'data'.", UserWarning)
            return data, _np.array([0.0]), _np.array([0.0]) if (freq is not None or df is not None) else data

        # Validate freq for 2D
        f1, f2 = None, None
        if freq is not None:
            if not isinstance(freq, tuple) or len(freq) != 2:
                raise TypeError("'freq' must be a tuple of two 1D NumPy arrays for 2D data.")
            f1, f2 = _np.asarray(freq[0], dtype=float), _np.asarray(freq[1], dtype=float)
            if f1.ndim != 1 or f2.ndim != 1 or f1.shape[0] != N or f2.shape[0] != M:
                raise ValueError("'f1' and 'f2' must be 1D arrays of lengths 'N' and 'M', respectively.")
            if not _np.all(_np.isreal(f1)) or not _np.all(_np.isreal(f2)):
                raise TypeError("'f1' and 'f2' must contain only real numbers.")
            if not _np.all(_np.isfinite(f1)) or not _np.all(_np.isfinite(f2)):
                raise ValueError("'f1' or 'f2' contains non-finite values (NaN or inf).")
            if not _np.all(_np.diff(f1) > 0) or not _np.all(_np.diff(f2) > 0):
                raise ValueError("'f1' and 'f2' must be monotonically increasing.")

        # Validate df for 2D
        if df is not None:
            if not isinstance(df, tuple) or len(df) != 2:
                raise TypeError("'df' must be a tuple of two floats for 2D data.")
            df1, df2 = df
            if not isinstance(df1, (int, float)) or not isinstance(df2, (int, float)):
                raise TypeError("'df1' and 'df2' must be real numbers.")
            if not _np.isfinite(df1) or not _np.isfinite(df2) or df1 <= 0 or df2 <= 0:
                raise ValueError("'df1' and 'df2' must be positive finite values.")

        try:
            # Initialize output
            x = _np.zeros((N, M), dtype=complex)

            # IFFT along rows (axis 1, using f2)
            for i in range(N):
                row_f = f2 if f2 is not None else None
                row_df = df2 if df is not None else None
                x[i, :], _ = ifft(data[i, :], freq=row_f, df=row_df, oversample=oversample)

            # IFFT along columns (axis 0, using f1)
            x = x.T.copy()
            for j in range(M):
                col_f = f1 if f1 is not None else None
                col_df = df1 if df is not None else None
                x[j, :], _ = ifft(x[j, :], freq=col_f, df=col_df, oversample=oversample)
            x = x.T

            # Compute times
            if freq is not None:
                df1_final = (f1[-1] - f1[0]) / (N - 1) if f1 is not None and N > 1 else 1.0
                df2_final = (f2[-1] - f2[0]) / (M - 1) if f2 is not None and M > 1 else 1.0
            elif df is not None:
                df1_final, df2_final = df
            else:
                return x

            dt1 = 1 / (N * df1_final)
            dt2 = 1 / (M * df2_final)
            t1 = _np.arange(0, N) * dt1
            t2 = _np.arange(0, M) * dt2
            return x, t1, t2

        except Exception as e:
            raise GenericError(f"Error computing IFFT2: {str(e)}")

def dfs(
    t: ArrayLike,
    data: ArrayLike,
    order: int,
    plot: bool = True,
    showlegend: bool = True,
    xlabel: str = "",
    ylabel: str = "",
    xscale: int = 0,
    yscale: int = 0,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None
) -> Tuple[ArrayLike, float, ArrayLike, ArrayLike]:
    """
    Computes the discrete Fourier series approximation of a sampled signal.

    Parameters
    ----------
    t : array-like
        1D array of sample points (must be uniformly spaced).
    data : array-like
        1D array of sampled signal.
    order : int
        Order of the Fourier approximation (number of harmonics).
    plot : bool, optional
        If `True`, plots the original function and its Fourier approximation.
    showlegend : bool, optional
        If `True`, ...
    xlabel : str, optional
        Label for the x-axis, including units in square brackets (e.g., "Time [s]").
    ylabel : str, optional
        Label for the y-axis, including units in square brackets (e.g., "Intensity [V]").
    xscale : int, optional
        Scaling factor for the x-axis (e.g., `xscale = -3` corresponds to 1e-3, to convert seconds to milliseconds).
    yscale : int, optional
        Scaling factor for the y-axis.
    xlim : tuple, optional
        Limits for the x-axis, in the form `(xmin, xmax)`. The values should
        already be scaled with respect to `xscale`. If `None` or an empty tuple,
        the default limits will be automatically determined from the data.
    ylim : tuple, optional
        Limits for the y-axis, in the form `(ymin, ymax)`. The values should
        already be scaled with respect to `yscale`. If `None` or an empty tuple,
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
    try:
        from scipy.integrate import simpson
        # import matplotlib.cm as _cm
        # Validazione input
        data = _np.asarray(data)
        t = _np.asarray(t)

        if not _np.allclose(_np.diff(t), t[1] - t[0], rtol=1e-4):
            raise ValueError("'t' must be uniformly spaced.")
        if not isinstance(order, int):
            raise TypeError("'order' must be an integer.")
        if order <= 0:
            raise ValueError("'order' must be equal or greater than 1.")
        if t.size != data.size:
            raise ValueError("'t' must have the same length as 'data'")
        if not _np.issubdtype(data.dtype, _np.number):
            raise TypeError("'data' must contain only numeric types (int, float, or complex).")
        if not (_np.issubdtype(t.dtype, _np.floating) or _np.issubdtype(t.dtype, _np.integer)) or not _np.all(_np.isreal(t)):
            raise TypeError("'t' must contain only real numbers (int or float).")
        if not _np.all(_np.isfinite(data)):
            raise ValueError("'data' contains non-finite values (NaN or inf).")
        if not _np.all(_np.isfinite(t)):
            raise ValueError("'t' contains non-finite values (NaN or inf).")
        if not isinstance(xscale, (int, float)):
            raise TypeError("'xscale' must be a real number (int or float).")
        if not isinstance(yscale, (int, float)):
            raise TypeError("'yscale' must be a real number (int or float).")
        if xlim is not None:
            if not isinstance(xlim, (list, tuple)):
                raise TypeError("'xlim' must be a list or tuple (either empty or containing two real numbers).")
            if len(xlim) != 0 and (len(xlim) != 2 or not all(isinstance(u, (int, float)) and _np.isfinite(u) for u in xlim)):
                raise TypeError("'xlim' must be empty or a list/tuple of exactly two finite real numbers.")
        if ylim is not None:
            if not isinstance(ylim, (list, tuple)):
                raise TypeError("'ylim' must be a list or tuple (either empty or containing two real numbers).")
            if len(ylim) != 0 and (len(ylim) != 2 or not all(isinstance(u, (int, float)) and _np.isfinite(u) for u in ylim)):
                raise TypeError("'ylim' must be empty or a list/tuple of exactly two finite real numbers.")
        if not isinstance(xlabel, str):
            raise TypeError("'xlabel' must be a string.")
        if not isinstance(ylabel, str):
            raise TypeError("'ylabel' must be a string.")

        xscale = 10**xscale
        yscale = 10**yscale

        N = len(t)
        dt = t[1] - t[0]
        fs = 1 / dt
        f_nyquist = fs / 2
        max_freq = order / (t[-1] - t[0])

        if max_freq > f_nyquist:
            warnings.warn(
                f"Potential aliasing detected: the signal contains frequency components up to {max_freq:.2g}, "
                f"which exceeds the Nyquist frequency ({f_nyquist:.2g}). "
                "Consider increasing the sampling frequency or applying an anti-aliasing filter."
            )

        a = t[0]
        b = t[-1]
        T = b - a

        # Zeroth coefficient
        a0 = 2 * simpson(data, t) / T

        a_n = []
        b_n = []

        for n in range(1, order):
            cos_term = _np.cos(2 * _np.pi * n * t / T)
            sin_term = _np.sin(2 * _np.pi * n * t / T)
            an = 2 * simpson(data * cos_term, t) / T
            bn = 2 * simpson(data * sin_term, t) / T
            # if apply_filter:
            #     decay = _np.exp(- (n / order)**2)
            #     an *= decay
            #     bn *= decay
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

            # Plot armoniche con alpha ridotto se showpanel=True
            # if showharmonics:
            #     if shift:
            #         i = a0
            #     max_harmonics = min(order - 1, 5)  # Limita a 10 armoniche per chiarezza
            #     # cmap = _cm.viridis
            #     # norm = _plt.Normalize(1, max_harmonics)
            #     for n in range(1, max_harmonics + 1):
            #         harmonic = (
            #             a_n[n-1] * _np.cos(2 * _np.pi * n * t / T) +
            #             b_n[n-1] * _np.sin(2 * _np.pi * n * t / T)
            #         )
            #         _plt.plot(
            #             t / xscale, ((harmonic + i) / yscale),
            #             color="dodgerblue", lw=0.8, alpha=0.3
            #         )

            # color=cmap(norm(n))
            
            # Plot dati originali e approssimazione
            _plt.plot(
                t / xscale, data / yscale, label="Input data",
                lw=1, color="mediumblue", marker=''
            )
            _plt.plot(
                t / xscale, f_approx / yscale, label=f"Partial Fourier series\nNo. of terms = {order}",
                lw=1, color="crimson", linestyle='-'
            )

            # Imposta limiti e etichette
            if xlim:
                _plt.xlim(xlim)
            if ylim:
                _plt.ylim(ylim)
            _plt.xlabel(xlabel)
            _plt.ylabel(ylabel)
            if showlegend:
                _plt.legend()

        return f_approx, a0, _np.array(a_n), _np.array(b_n)

    except Exception as e:
        raise GenericError(
            message="Unexpected error in discrete Fourier series computation",
            context="executing dfs",
            original_error=e,
            details={"t_shape": t.shape, "data_shape": data.shape, "order": order}
        )

def fourier_series(f: Callable[[Union[float, ArrayLike]], Union[float, ArrayLike]], interval: Tuple[float, float], order: int, 
                   num_points: int = 1000, xlabel: str = "x [ux]", ylabel: str = "y [uy]", 
                   xscale: int = 0, yscale: int = 0) -> Tuple[ArrayLike, ArrayLike, ArrayLike, float, ArrayLike, ArrayLike]:
    """
    Computes the Fourier series approximation of a function f(x)
    
    Parameters
    ----------
    f : callable
        Function to approximate.
    interval : tuple of float
        The interval `(a, b)` over which to compute the Fourier series.
    order : int
        Number of Fourier modes to use in the approximation.
    num_points : int, optional
        Number of points for plotting. Default is 1000).
        
    Returns
    -------
    x : array-like
        1D array representing the uniformly spaced sample points over one period.
        These are the evaluation points at which both the original function and the Fourier
        approximation are computed.
    f_original : array-like
        1D array representing the values of the original input function evaluated at 
        the sample points `x`. This is the reference signal used for comparison with the Fourier 
        approximation.
    f_approx : array-like
        1D Array containing the values of the truncated Fourier series evaluated at 
        the same sample points `x`. This is the approximation of `f_original` using a finite number 
        of harmonics (up to the specified order).
    a0 : float
        Zeroth Fourier coefficient (mean value component of the function over the period). It 
        corresponds to the constant term of the Fourier series.
    a_n : array-like
        Array of cosine coefficients (Fourier coefficients of the even part of the function),
        corresponding to each harmonic up to the specified order (excluding a0).
    b_n : array-like
        Array of sine coefficients (Fourier coefficients of the odd part of the function),
        corresponding to each harmonic up to the specified order.

    Notes
    ----------
    The values of `xscale` and `yscale` affect only the axis scaling in the plot. All parameters are estimated using the original input data as provided.
    """

    from scipy.integrate import quad

    if not callable(f):
        raise TypeError("'f' must be a callable function.")
    
    if not isinstance(interval, (list, tuple)):
        raise TypeError("'interval' must be a list or a tuple.")
    if len(interval) != 2:
        raise ValueError("'interval' must have exactly two elements.")
    
    a, b = interval
    
    if not (_np.isscalar(a) and _np.isscalar(b)):
        raise TypeError("'interval' elements must be scalars.")
    
    if not (isinstance(a, (int, float)) and isinstance(b, (int, float))):
        raise TypeError("'interval' elements must be real numbers (int or float).")

    if b < a:
        a, b = b, a
        warnings.warn("Integration limits 'a' and 'b' have been swapped.", UserWarning)

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

def harmonic(t: ArrayLike, y: ArrayLike, prominence: float = 0.05, n_max: Optional[int] = None, verbose: bool = True) -> List[dict]:
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

def decompose(t: ArrayLike, y: ArrayLike, freqs: ArrayLike, verbose: bool = True) -> List[dict]:
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
    freqs : array-like
        Frequencies of the components to extract.
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

def envelope(signal: ArrayLike, 
             method: str = 'peaks', 
             mode: str = 'upper', 
             filter_size: int = 31, 
             fs: float = 1.0, 
             remove_mean: bool = False) -> Union[ArrayLike, Tuple[ArrayLike, ArrayLike]]:
    """
    Compute the envelope of a 1D real-valued signal.

    Parameters
    ----------
        signal : array-like
            The input real-valued signal (1D array).
        method : {'hilbert', 'peaks', 'adaptive'}, optional
            The method to compute the envelope:

            - 'hilbert': Uses the analytic signal via the Hilbert transform (best for bandpass signals).
            - 'peaks': Interpolates a smooth curve through signal peaks (versatile for general signals).
            - 'adaptive': Applies a variable-width filter to abs(signal) (robust to noise).

            Defaults to 'peaks'.
        mode : {'upper', 'lower', 'both'}, optional
            Select which envelope to return:

            - 'upper': The upper envelope (default).
            - 'lower': The lower envelope.
            - 'both': Returns a tuple (upper, lower).

        filter_size : int, optional
            Size of the smoothing filter for 'adaptive' method. Must be odd and positive.
            Defaults to `31`.
        fs : float, optional
            Sampling frequency in Hz, used for frequency estimation in 'adaptive' method.
            Defaults to `1.0`.
        remove_mean : bool, optional
            If True, removes the signal mean before computing the envelope.
            Defaults to `False`.

    Returns
    -------
        ndarray or tuple of ndarrays
            The computed envelope(s):

            - If mode='upper' or 'lower', returns a 1D array.
            - If mode='both', returns a tuple (upper, lower).

    Notes
    -----

        - The 'hilbert' method is mathematically rigorous for bandpass signals but may fail for broadband or non-oscillatory signals.
        - The 'peaks' method is versatile, interpolating through local maxima/minima, and works well for most signals.
        - The 'adaptive' method adjusts filter width based on local signal frequency, ideal for noisy or irregular signals.

    """
    from scipy.signal import hilbert, find_peaks, medfilt
    from scipy.interpolate import UnivariateSpline
    # Validate signal
    signal = _np.asarray(signal, dtype=float)
    if signal.ndim != 1:
        raise ValueError("signal must be one-dimensional.")
    if not _np.issubdtype(signal.dtype, _np.floating) or not _np.all(_np.isreal(signal)):
        raise TypeError("signal must contain real numbers (float).")
    if not _np.all(_np.isfinite(signal)):
        raise ValueError("signal contains non-finite values (NaN or inf).")

    # Validate method
    valid_methods = ['hilbert', 'peaks', 'adaptive']
    if method not in valid_methods:
        raise ValueError(f"method must be one of {valid_methods}.")

    # Validate mode
    valid_modes = ['upper', 'lower', 'both']
    if mode not in valid_modes:
        raise ValueError(f"mode must be one of {valid_modes}.")

    # Validate filter_size
    if not isinstance(filter_size, int):
        raise TypeError("filter_size must be an integer.")
    if filter_size % 2 == 0 or filter_size < 1:
        raise ValueError("filter_size must be a positive odd integer.")

    # Validate fs
    if not isinstance(fs, (int, float)) or fs <= 0:
        raise ValueError("fs must be a positive number.")

    # Validate remove_mean
    if not isinstance(remove_mean, bool):
        raise TypeError("remove_mean must be a boolean.")

    # Remove mean if requested
    if remove_mean:
        signal = signal - _np.mean(signal)

    try:
        if method == 'hilbert':
            # Compute envelope using Hilbert transform
            analytic = hilbert(signal)
            env = _np.abs(analytic)

        elif method == 'peaks':
            # Find peaks for upper envelope
            peaks, _ = find_peaks(signal, distance=filter_size // 2)
            if len(peaks) < 2:
                # Fallback to constant envelope if too few peaks
                env = _np.full_like(signal, _np.max(_np.abs(signal)))
            else:
                # Interpolate through peaks
                x = _np.arange(len(signal))
                spline = UnivariateSpline(peaks, signal[peaks], k=3, s=0, ext='const')
                env = spline(x)
                env = _np.maximum(env, 0)  # Ensure non-negative envelope

        elif method == 'adaptive':
            # Compute absolute signal
            abs_sig = _np.abs(signal)
            # Estimate local frequency using zero-crossings
            zero_crossings = _np.where(_np.diff(_np.sign(signal)))[0]
            if len(zero_crossings) > 1:
                mean_period = _np.mean(_np.diff(zero_crossings))
                adaptive_size = max(3, int(mean_period / 2) * 2 + 1)  # Ensure odd
            else:
                adaptive_size = filter_size
            # Apply median filter with adaptive size
            env = medfilt(abs_sig, kernel_size=adaptive_size)

        # Handle mode
        if mode == 'upper':
            return env
        elif mode == 'lower':
            # Compute lower envelope as upper envelope of -signal
            lower_env = envelope(-signal, method=method, mode='upper', 
                                filter_size=filter_size, fs=fs, remove_mean=False)
            return -lower_env
        elif mode == 'both':
            upper = env
            lower_env = envelope(-signal, method=method, mode='upper', 
                                filter_size=filter_size, fs=fs, remove_mean=False)
            return upper, -lower_env

    except Exception as e:
        raise GenericError(f"Error computing envelope: {str(e)}")