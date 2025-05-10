This module provides functions for signal processing and analysis, including power spectrum computation, discrete Fourier series approximation, and harmonic decomposition of real-valued signals. It leverages libraries such as `numpy`, `scipy`, and `matplotlib` for efficient computation and visualization.

> **Note**: Coming in future updates!

## `LabToolbox.signals.dfs`

**Description**:  
Computes the discrete Fourier series approximation of a sampled function, optionally applying a low-pass filter and plotting the results. This function is useful for reconstructing periodic signals from discrete data.

**Parameters**:  
- `t` (*ndarray*): Array of sample points (must be uniformly spaced).  
- `data` (*ndarray*): Array of function values sampled at points `t`.  
- `order` (*int*): Order of the Fourier approximation (number of harmonics).  
- `plot` (*bool, optional*): If `True`, plots the original function and its Fourier approximation. Default is `True`.  
- `apply_filter` (*bool, optional*): If `True`, applies a basic low-pass filter to reduce high-frequency noise. Default is `True`.  
- `xlabel` (*str, optional*): Label for the x-axis, including units in square brackets (e.g., "Time [s]"). Default is `"x [ux]"`.  
- `ylabel` (*str, optional*): Label for the y-axis, including units in square brackets (e.g., "Intensity [V]"). Default is `"y [uy]"`.  
- `xscale` (*int, optional*): Scaling factor for the x-axis (e.g., `xscale = -3` corresponds to $10^{-3}$, to convert seconds to milliseconds). Default is `0`.  
- `yscale` (*int, optional*): Scaling factor for the y-axis. Default is `0`.

**Returns**:  
- `f_approx` (*ndarray*): Array of the same shape as `t`, containing the values of the Fourier series approximation of the input function at each point in `t`.  
- `a0` (*float*): Zeroth Fourier coefficient (mean value component of the function over the period). It corresponds to the constant term of the Fourier series.  
- `a_n` (*ndarray*): Array of cosine coefficients (Fourier coefficients of the even part of the function), corresponding to each harmonic up to the specified order (excluding `a0`).  
- `b_n` (*ndarray*): Array of sine coefficients (Fourier coefficients of the odd part of the function), corresponding to each harmonic up to the specified order.

> **Note**: The values of `xscale` and `yscale` affect only the axis scaling in the plot. All outputs are estimated using the original input data as provided.

---

## `LabToolbox.signals.fourier_series`

**Description**:  
Computes the Fourier series approximation of a continuous function $f(x)$ over a specified interval, with optional plotting of the original and approximated functions.

**Parameters**:  
- `f` (*callable*): Function to approximate.  
- `interval` (*tuple of float*): The interval $(a, b)$ over which to compute the Fourier series.  
- `order` (*int*): Number of Fourier modes ($n$) to use in the approximation.  
- `num_points` (*int, optional*): Number of points for plotting (default is `1000`).  
- `xlabel` (*str, optional*): Label for the x-axis, including units in square brackets (e.g., "Time [s]"). Default is `"x [ux]"`.  
- `ylabel` (*str, optional*): Label for the y-axis, including units in square brackets (e.g., "Intensity [V]"). Default is `"y [uy]"`.  
- `xscale` (*int, optional*): Scaling factor for the x-axis (e.g., `xscale = -2` corresponds to $10^{-2}$). Default is `0`.  
- `yscale` (*int, optional*): Scaling factor for the y-axis. Default is `0`.

**Returns**:  
- `x` (*ndarray*): Array of shape $(N,)$, representing the uniformly spaced sample points over one period. These are the evaluation points at which both the original function and the Fourier approximation are computed.  
- `f_original` (*ndarray*): Array of shape $(N,)$, representing the values of the original input function evaluated at the sample points `x`. This is the reference signal used for comparison with the Fourier approximation.  
- `f_approx` (*ndarray*): Array of shape $(N,)$, containing the values of the truncated Fourier series evaluated at the same sample points `x`. This is the approximation of `f_original` using a finite number of harmonics (up to the specified order).  
- `a0` (*float*): Zeroth Fourier coefficient (mean value component of the function over the period). It corresponds to the constant term of the Fourier series.  
- `a_n` (*ndarray*): Array of cosine coefficients (Fourier coefficients of the even part of the function), corresponding to each harmonic up to the specified order (excluding `a0`).  
- `b_n` (*ndarray*): Array of sine coefficients (Fourier coefficients of the odd part of the function), corresponding to each harmonic up to the specified order.

> **Note**: The values of `xscale` and `yscale` affect only the axis scaling in the plot. All parameters are estimated using the original input data as provided.

---

## `LabToolbox.signals.harmonic`

**Description**:  
Identifies the dominant harmonics present in a real-valued signal by computing the FFT and detecting peaks in the amplitude spectrum. This function does not reconstruct the time-domain components but reports the strongest harmonics.

**Parameters**:  
- `t` (*array_like*): Time array.  
- `y` (*array_like*): Signal samples corresponding to `t`.  
- `prominence` (*float, optional*): Minimum prominence of peaks in the power spectrum. Default is `0.05`.  
- `n_max` (*int or None, optional*): If given, return at most `n_max` harmonics. Default is `None`.

**Returns**:  
- `harmonics` (*list of dict*): Each dict contains `frequency` (Hz), `amplitude`, and `phase` (rad) of a harmonic component.

---

## `LabToolbox.signals.decompose`

**Description**:  
Reconstructs specified sinusoidal components from a real-valued signal using linear least squares to fit the time-domain data to a sum of sinusoids at given frequencies.

**Parameters**:  
- `t` (*array_like*): Time array.  
- `y` (*array_like*): Signal samples.  
- `freqs` (*array_like*): Frequencies (Hz) of the components to extract.

**Returns**:  
- `components` (*list of dict*): Each dict contains `frequency` (Hz), `amplitude`, and `phase` (rad).

---

This documentation will be updated as additional modules and functions are added to the `LabToolbox` package. For contributions or issues, please refer to the GitHub repository.