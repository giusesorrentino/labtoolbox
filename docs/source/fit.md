This module contains functions for performing linear and non-linear curve fitting, including weighted least squares (WLS) and ordinary least squares (OLS) regression, general-purpose model fitting, and bootstrap analysis. It integrates with libraries such as `numpy`, `scipy`, `statsmodels`, and `matplotlib` for robust statistical analysis and visualization.

## `LabToolbox.fit.lin_fit`

**Description**:  
Performs a linear fit (Weighted Least Squares or Ordinary Least Squares) on experimental data and generates a plot displaying the regression line, uncertainty band, and optionally residuals. The function handles scaling of axes and parameters, supports custom units, and provides statistical metrics such as reduced chi-square and p-value.

**Parameters**:  
- `x` (*array-like*): Values of the independent variable.  
- `y` (*array-like*): Values of the dependent variable.  
- `y_err` (*array-like*): Uncertainties associated with `y` values.  
- `x_err` (*array-like, optional*): Uncertainties associated with `x` values. Defaults to `None`.  
- `fitmodel` (*str, optional*): Fitting model, either `"wls"` (Weighted Least Squares) or `"ols"` (Ordinary Least Squares). Defaults to `"wls"`.  
- `xlabel` (*str, optional*): Label for the x-axis, including units in square brackets (e.g., `"x [m]"`). Defaults to `"x [ux]"`.  
- `ylabel` (*str, optional*): Label for the y-axis, including units in square brackets (e.g., `"y [s]"`). Defaults to `"y [uy]"`.  
- `showlegend` (*bool, optional*): If `True`, displays a legend with the values of slope `m` and intercept `c` on the plot. Defaults to `True`.  
- `legendloc` (*str, optional*): Location of the legend on the plot (e.g., `'upper right'`, `'lower left'`, `'upper center'`). Defaults to `None`.  
- `xscale` (*int, optional*): Scaling factor for the x-axis (e.g., `xscale = -2` corresponds to $10^{-2}$, converting meters to centimeters). Defaults to `0`.  
- `yscale` (*int, optional*): Scaling factor for the y-axis. Defaults to `0`.  
- `mscale` (*int, optional*): Scaling factor for the slope `m`. Defaults to `0`.  
- `cscale` (*int, optional*): Scaling factor for the intercept `c`. Defaults to `0`.  
- `m_units` (*str, optional*): Unit of measurement for `m` (e.g., `"m/s"`), ensuring consistency with `x`, `y`, and scale factors. Defaults to `""`.  
- `c_units` (*str, optional*): Unit of measurement for `c`, ensuring consistency with `x`, `y`, and scale factors. Defaults to `""`.  
- `confidence` (*int, optional*): Residual confidence interval to display, i.e., `[-confidence, +confidence]`. Defaults to `2`.  
- `confidencerange` (*bool, optional*): If `True`, shows the 1σ uncertainty band around the fit line. Defaults to `True`.
- `log` (*str, optional*): If set to `'x'` or `'y'`, the corresponding axis is plotted on a logarithmic scale; if `'xy'`, both axes. Defaul is `None`.  
- `residuals` (*bool, optional*): If `True`, adds an upper panel showing fit residuals. Defaults to `True`.  
- `norm` (*bool, optional*): If `True`, residuals in the upper panel will be normalized by uncertainties. Defaults to `True`.  
- `result` (*bool, optional*): If `True`, prints the output of the fit (e.g., `wls_fit` or `ols_fit`) to the screen. Defaults to `False`.

**Returns**:  
- `m` (*float*): Slope of the regression line.  
- `c` (*float*): Intercept of the regression line.  
- `sigma_m` (*float*): Uncertainty on the slope.  
- `sigma_c` (*float*): Uncertainty on the intercept.  
- `chi2_red` (*float*): Reduced chi-square value.  
- `p_value` (*float*): Fit p-value (probability that the observed chi-square is compatible with the model).

> **Notes**:
> - The values of `xscale` and `yscale` affect only the axis scaling in the plot and do not impact the fitting computation, which uses the original input data.  
> - LaTeX formatting is embedded in the strings used to display the units of `m` and `c`; manual `$...$` is not required.  

---

## `LabToolbox.fit.model_fit`

**Description**:  
Performs a general-purpose fit of multi-parameter functions to experimental data, with an option to display residuals. This function leverages `scipy.optimize.curve_fit` and supports custom bounds, logarithmic scales, and uncertainty bands.

**Parameters**:  
- `x` (*array-like*): Measured values of the independent variable.  
- `y` (*array-like*): Measured values of the dependent variable.  
- `y_err` (*array-like*): Uncertainties associated with the dependent variable.  
- `f` (*function*): Function of one variable (first argument of `f`) with `N` free parameters.  
- `p0` (*list*): Initial guess for the model parameters, in the form `[a, ..., z]`.  
- `x_err` (*array-like, optional*): Uncertainties associated with the independent variable. Defaults to `None`.  
- `xlabel` (*str, optional*): Label (and units) for the independent variable. Defaults to `"x [ux]"`.  
- `ylabel` (*str, optional*): Label (and units) for the dependent variable. Defaults to `"y [uy]"`.  
- `showlegend` (*bool, optional*): If `True`, displays a legend with the reduced chi-square and p-value in the plot. Defaults to `True`.  
- `legendloc` (*str, optional*): Location of the legend in the plot (e.g., `'upper right'`, `'lower left'`, `'upper center'`). Defaults to `None`.  
- `bounds` (*2-tuple of array-like, optional*): Tuple `([lower_bounds], [upper_bounds])` specifying bounds for the parameters. Defaults to `None`.  
- `confidencerange` (*bool, optional*): If `True`, displays the 1σ uncertainty band around the best-fit curve. Defaults to `True`.  
- `log` (*str, optional*): If set to `'x'` or `'y'`, the corresponding axis is plotted on a logarithmic scale; if `'xy'`, both axes. Defaults to `None`.  
- `maxfev` (*int, optional*): Maximum number of iterations allowed by `curve_fit`. Defaults to `5000`.  
- `xscale` (*int, optional*): Scaling factor for the x-axis (e.g., `xscale = -2` corresponds to $10^{-2}$, converting meters to centimeters). Defaults to `0`.  
- `yscale` (*int, optional*): Scaling factor for the y-axis. Defaults to `0`.  
- `confidence` (*int, optional*): Residual confidence interval to display, i.e., `[-confidence, +confidence]`. Defaults to `2`.  
- `residuals` (*bool, optional*): If `True`, adds an upper panel showing fit residuals. Defaults to `True`.  
- `norm` (*bool, optional*): If `True`, residuals in the upper panel will be normalized. Defaults to `True`.

**Returns**:  
- `popt` (*array-like*): Array of optimal parameters estimated from the fit.  
- `perr` (*array-like*): Uncertainties on the optimal parameters.  
- `chi2_red` (*float*): Reduced chi-square value.  
- `p_value` (*float*): Fit p-value (probability that the observed chi-square is compatible with the model).

> **Note**: The values of `xscale` and `yscale` affect only the axis scaling in the plot and do not impact the fitting computation, which uses the original input data.

---

## `LabToolbox.fit.bootstrap_fit`

**Description**:  
Performs a bootstrap analysis of the fit to estimate the parameter distributions, optionally accounting for uncertainties in the dependent variable. This method is useful for assessing the robustness of fit parameters through resampling.

**Parameters**:  
- `func` (*callable*): Model function to be fitted, in the form `func(x, *params)`.  
- `xdata` (*array_like*): Independent data (x-values).  
- `ydata` (*array_like*): Dependent data (y-values).  
- `y_err` (*array_like, optional*): Uncertainties associated with `ydata`. If provided, a weighted fit will be performed. Defaults to `None`.  
- `p0` (*array_like, optional*): Initial guess for the fit parameters. Defaults to `None`.  
- `punits` (*list of str, optional*): List of strings specifying the units of each parameter. Use an empty string `""` for dimensionless parameters. Defaults to `None`.  
- `n_iter` (*int, optional*): Number of bootstrap iterations. Defaults to `1000`.  
- `bounds` (*2-tuple of arrays, optional*): Lower and upper bounds for the fit parameters. Defaults to `(-np.inf, np.inf)`.

**Returns**:  
- `popt_mean` (*array*): Mean values of the parameters obtained from the bootstrap samples.  
- `popt_std` (*array*): Standard deviations of the parameters (as uncertainty estimates).  
- `all_popt` (*array*): Full array of all parameter estimates (shape: `[n_iter, n_params]`).

---

This documentation will be updated as additional modules and functions are added to the `LabToolbox` package. For contributions or issues, please refer to the GitHub repository.