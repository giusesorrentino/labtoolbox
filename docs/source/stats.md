This module provides statistical analysis tools, including histogram generation with Gaussianity tests, residual analysis, synthetic data generation, outlier removal, and Bayesian parameter estimation using MCMC methods. It integrates with libraries such as `numpy`, `matplotlib`, `scipy`, `statsmodels`, `emcee`, and `lmfit`.

## `LabToolbox.stats.hist`

**Description**:  
Plots the histogram of a dataset and assess its Gaussianity using statistical indicators and a Shapiro-Wilk test. The function visualizes the empirical distribution, overlays a Gaussian curve, and reports mean, standard deviation, skewness, kurtosis, and a normality test p-value.

**Parameters**:  
- `data` (*array-like*): Numerical data representing the variable of interest.  
- `data_err` (*array-like or None*): Array of uncertainties associated with each element of `data`. If `None`, uncertainties are not included in the computation of the effective standard deviation.  
- `scale` (*int, optional*): Scaling exponent for `data` and `sigma_data` (default is `0`). For example, `scale = -2` rescales the inputs by `10**2`.
- `bins` (*int, optional*): Number of bins or binning strategy passed to `matplotlib.pyplot.hist`. Defaults to `"auto"`.
- `label` (*str, optional*): Label for the x-axis, typically the name of the variable.  
- `unit` (*str, optional*): Unit of measurement for the x-axis variable (e.g., "cm"). If provided, it will be displayed in the axis label and summary output.

**Returns**:  
- `mean` (*float*): Arithmetic mean of the scaled data.  
- `std` (*float*): Effective standard deviation of the distribution, accounting for both the empirical spread and uncertainties (if provided).  
- `skewness` (*float*): Skewness of the distribution.  
- `kurtosis` (*float*): Excess kurtosis (fourth standardized moment minus 3) of the distribution.  
- `p_value` (*float*): p-value from the Shapiro-Wilk test for normality.

**Example**:  
```python
>>> x = np.random.normal(loc=5, scale=0.2, size=100)
>>> sigmax = np.full_like(x, 0.05)
>>> hist(x, sigmax, scale=-2, label="Length", unit="cm")
```

> **Notes**:  
> - The effective standard deviation is computed as $\sqrt{\text{data.std()}^2 + \sum \text{data err}^2 / \text{len(data)}}$ if `data_err` is provided.  
> - The function rescales both `data` and `data_err` by $10^\text{scale}$ for display purposes, but all statistics are computed on the scaled data.
> - The normal distribution is referred to as $\mathcal{N}(\mu, \sigma^2)$.  
-----

## `LabToolbox.stats.analyze_residuals`

**Description**:  
Analyzes and visualizes the residuals of a fit, including histogram, Gaussianity test, and autocorrelation test (Durbin-Watson statistic). The function plots residuals with uncertainty bands and provides statistical metrics.

**Parameters**:  
- `data` (*array-like*): Measured data points.  
- `expected_data` (*array-like*): Expected values to compare with `data` (e.g., from a model, theoretical prediction, or fit).  
- `data_err` (*array-like*): Uncertainties associated with each data point in `data`.  
- `scale` (*int, optional*): Scaling exponent applied to all quantities (e.g., `scale = -2` scales meters to centimeters). Default is `0`.  
- `unit` (*str, optional*): Unit of measurement of the data (e.g., `"cm"` or `"s"`). Used for labeling axes. Default is an empty string.  
- `bins` (*int, optional*): Number of bins or binning strategy passed to `matplotlib.pyplot.hist`. Defaults to`"auto"`.  
- `confidence` (*float, optional*): Confidence factor for visualizing bounds (e.g., `confidence = 2` draws ±2σ bounds). Default is `2`.  
- `norm` (*bool, optional*): If `True`, residuals in the upper panel will be normalized. Default is `False`.

**Returns**:  
- `mean` (*float*): Mean value of the residuals, after applying the specified scale.  
- `sigma` (*float*): Estimated standard deviation of the residuals, weighted by `sigma_data`.  
- `skewness` (*float*): Skewness (third standardized moment) of the residual distribution.  
- `kurtosis` (*float*): Excess kurtosis (fourth standardized moment minus 3) of the residual distribution.  
- `p_value` (*float*): p-value from the Shapiro–Wilk normality test.  
- `dw` (*float*): Durbin–Watson statistic for testing autocorrelation in the residuals.

> **Notes**:  
> - The residuals are computed as $\text{resid} = \text{data} - \text{expected data}$, and scaled by $10^\text{scale}$.  
> - The standard deviation is computed as $\sqrt{\text{resid.std()}^2 + \sum \text{data err}^2 / \text{len(data err)}}$.   
> - The normal distribution is referred to as $\mathcal{N}(\mu, \sigma^2)$.  

---

## `LabToolbox.stats.samples`

**Description**:  
Generates synthetic data from common probability distributions, supporting a variety of statistical models for simulation purposes.

**Parameters**:  
- `n` (*int*): Number of data points to generate.  
- `distribution` (*{'normal', 'uniform', 'exponential', 'poisson', 'binomial', 'gamma', 'beta', 'lognormal', 'weibull', 'chi2', 't'}, optional*): Type of distribution to sample from. Default is `'normal'`.  
- `**params` (*dict*): Distribution-specific parameters:  
  - `normal`: `mu` (mean), `sigma` (stddev)  
  - `uniform`: `low`, `high`  
  - `exponential`: `scale` (1/lambda)  
  - `poisson`: `lam` (expected rate)  
  - `binomial`: `n` (number of trials), `p` (success probability)  
  - `gamma`: `shape`, `scale`  
  - `beta`: `alpha`, `beta`  
  - `lognormal`: `mean`, `sigma`  
  - `weibull`: `shape`  
  - `chi2`: `df` (degrees of freedom)  
  - `t`: `df` (degrees of freedom)

**Returns**:  
- `data` (*ndarray*): Array of length `n` with samples drawn from the specified distribution.

**Examples**:  
```python
>>> samples(1000, 'normal', mu=0, sigma=1)
array([...])
>>> samples(500, 'uniform', low=0, high=10)
array([...])
>>> samples(200, 'poisson', lam=3)
array([...])
```

## `LabToolbox.stats.remove_outliers`

**Description**:  
Removes outliers from a data array according to the specified method, with options to compare against expected values.

**Parameters**:  
- `data` (*array-like*): Observed data.  
- `data_err` (*array-like, optional*): Uncertainties on the data. Necessary if comparing with `'expected'`.  
- `expected` (*array-like, optional*): Expected values for the data. If provided, the `'zscore'` method is automatically used.  
- `method` (*str, optional*): Method to use (`"zscore"`, `"mad"`, or `"iqr"`). Default: `"zscore"`.  
- `threshold` (*float, optional*): Threshold value to identify outliers. Default: `3.0`.

**Returns**:  
- `data_clean` (*ndarray*): Data without outliers.

---

## `LabToolbox.stats.posterior`

**Description**:  
Performs a Bayesian analysis using the `emcee` package to fit a user-defined model function to data via Markov Chain Monte Carlo (MCMC). It first estimates the parameters with a classical least-squares fit and then samples the posterior distribution using MCMC. It reports both the median and Maximum Likelihood Estimation (MLE) values, and visualizes the posterior distribution with a corner plot.

**Parameters**:  
- `x` (*array-like*): Measured values for the independent variable.  
- `y` (*array-like*): Measured values for the dependent variable (to be fitted to the model).  
- `y_err` (*array-like*): Uncertainties on the measurements of the dependent variable.  
- `f` (*function*): Model function to be fitted to the data. The function must accept an independent variable `x` as first argument and free parameters as subsequent arguments.  
- `p0` (*list*): Initial guess for the model parameters. Example: `[a0, b0, c0]`, where each value is the initial estimate of a parameter.  
- `burn` (*int, optional*): Number of burn-in steps to discard from the beginning of each MCMC chain (default: 1000).  
- `steps` (*int, optional*): Total number of steps in the MCMC chains (default: 5000).  
- `thin` (*int, optional*): Thinning factor to reduce autocorrelation in the chain (default: 10).  
- `maxfev` (*int, optional*): Maximum number of iterations allowed for the initial least-squares optimization (`curve_fit`, default: 5000).  
- `names` (*list of str, optional*): Parameter names. If `None`, use `['p0', 'p1', ...]`. Used for labeling the corner plot.  
- `prior_bounds` (*list of tuple, optional*): List of `(min, max)` bounds for each parameter. If `None`, uses non-informative priors (only enforces positivity).

**Returns**:  
- `params` (*lmfit.Parameters*): Parameters object with best-fit values (MLE) after optimization.  
- `flat_samples` (*np.ndarray*): Flattened MCMC chain after burn-in and thinning, shape `(n_samples, n_parameters)`.

**Example**:
```python
>>> def model(x, a, b):
...     return a * x + b
>>> x = np.linspace(0, 10, 50)
>>> y = model(x, 2.5, 1.0) + np.random.normal(0, 0.5, size=x.size)
>>> sy = 0.5 * np.ones_like(y)
>>> posterior(x, y, sy, model, [1, 1])
```

> **Note**: Only uniform priors are currently supported; parameters are rejected if outside the user-specified bounds or if negative (in the absence of bounds).  

---

This documentation will be updated as additional modules and functions are added to the `LabToolbox` package. For contributions or issues, please refer to the GitHub repository.