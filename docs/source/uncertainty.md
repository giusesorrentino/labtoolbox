This module provides functions for propagating uncertainties in experimental data using numerical methods, Monte Carlo simulations, and integration with the `uncertainty_class` library. It is designed to handle both single-point and array-based calculations with various uncertainty propagation techniques.

## `LabToolbox.uncertainty.numerical`

**Description**:  
Computes uncertainty propagation via numerical derivatives for a given function, evaluating the partial derivatives at each point using a central difference approximation. This method is suitable for functions with multiple input variables and optional parameters.

**Parameters**:  
- `f` (*callable*): Function `f(x1, ..., xn; a1, ..., am)` that returns an array of shape `(N,)`, where `x1, ..., xn` are input variables and `a1, ..., am` are optional constant parameters.  
- `x_val` (*list of np.ndarray*): List of input arrays `x1, ..., xn`, each with shape `(N,)` representing the independent variables.  
- `x_err` (*list of np.ndarray*): List of uncertainty arrays corresponding to each `x_i`, with shape `(N,)` matching the corresponding `x_val` array.  
- `params` (*tuple, optional*): Tuple of constant parameters `(a1, ..., am)` to be passed to the function. Defaults to an empty tuple `()`.

**Returns**:  
- `f_val` (*np.ndarray*): Central values of the function, shape `(N,)`.  
- `f_err` (*np.ndarray*): Propagated uncertainty, shape `(N,)`.

---

## `LabToolbox.uncertainty.propagate`

**Description**:  
Propagates uncertainty from input arrays to a generic function using the `uncertainty_class` library, supporting both Delta method and Monte Carlo approaches. This function is versatile, handling multiple variables and optional parameters with customizable methods.

**Parameters**:  
- `func` (*callable*): The base function, in the form `f(x, a)` where:  
  - `x` is a vector of variables.  
  - `a` is a vector of optional parameters.  
- `x_val` (*list of numpy.ndarray*): List containing the input variable arrays `[x1, x2, ..., xn]`. Each `xi` must have the same length.  
- `x_err` (*list or numpy.ndarray*): List of uncertainties for each variable, or a full covariance matrix. If a list of uncertainties `sigma` is provided, a diagonal covariance matrix is constructed.  
- `params` (*list or numpy.ndarray, optional*): List or array of constant parameters `[a1, a2, ..., am]`. Defaults to `None`.  
- `method` (*str, optional*): The uncertainty propagation method (`'Delta'` or `'Monte_Carlo'`). Defaults to `'Delta'`.  
- `MC_sample_size` (*int, optional*): Sample size for the Monte Carlo method. Defaults to `10000`.

**Returns**:  
- `f_values` (*numpy.ndarray*): Values of the function calculated at each point `j`.  
- `f_uncertainties` (*numpy.ndarray*): Propagated uncertainties on the output function for each point `j`.  
- `confidence_bands` (*tuple of numpy.ndarray*): Lower and upper confidence bands for each point `j`.

> **Note**: See the external library documentation at [this link](https://github.com/yiorgoskost/Uncertainty-Propagation/tree/master) for further details on the `uncertainty_class` implementation.

---

## ` LabToolbox.uncertainty.montecarlo`

**Description**:  
Estimates the propagated uncertainty on a function of `N` variables using Monte Carlo simulation. This method samples input variables from normal distributions and evaluates the function to compute mean and standard deviation.

**Parameters**:  
- `func` (*callable*): The function to evaluate. Must accept the same number of arguments as the length of `values`.  
- `value` (*array-like*): Central values of the input variables. Must be of the same length as `uncertainties`.  
- `err` (*array-like*): Standard deviations (1-sigma uncertainties) of the input variables.  
- `N` (*int, optional*): Number of Monte Carlo samples to generate. Defaults to `10000`.  
- `seed` (*int or None, optional*): Seed for the random number generator, for reproducibility. Defaults to `None`.

**Returns**:  
- `mean` (*float*): Mean value of the function evaluated over the sampled inputs.  
- `std` (*float*): Standard deviation (uncertainty) of the function output.

**Example**:  
```python
>>> def f(x, y): return x * y
>>> montecarlo(f, [2.0, 3.0], [0.1, 0.2])
(6.00..., 0.42...)
```
---

This documentation will be updated as additional modules and functions are added to the `LabToolbox` package. For contributions or issues, please refer to the GitHub repository.