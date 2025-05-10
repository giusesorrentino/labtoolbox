This module provides utility functions for statistical calculations, data formatting, and visualization support within the `LabToolbox` package. It includes custom implementations of mean, covariance, and variance, as well as tools for formatting results and generating LaTeX tables.

## `LabToolbox.utils.PrintResult`

**Description**:  
Prints a formatted string in the "mean ± sigma" format, with sigma rounded to two significant figures and mean rounded consistently.

**Parameters**:  
- `value` (*float*): Value of the variable.  
- `err` (*float*): Uncertainty of the variable.  
- `name` (*str, optional*): Name of the variable to display before the value. Defaults to an empty string.  
- `unit` (*str, optional*): Unit of measurement to display after the value. Defaults to an empty string.

**Returns**:  
- (*None*): Prints the formatted string directly.

---

## `LabToolbox.utils.format_str`

**Description**:  
Formats values and their uncertainties into LaTeX-compatible strings of the form "$data \pm data_err$". The function ensures proper rounding, handling both single values and arrays.

> **Note**: Coming in future updates!

**Parameters**:  
- `data` (*float or array-like*): Central values to be formatted.  
- `err` (*float or array-like*): Uncertainties corresponding to each central value (must have the same shape as `data`).

**Returns**:  
- (*list of str or str*): LaTeX strings in the format "$data \pm data_err$". If a single value is provided, returns a single string; otherwise, returns a list of strings.

> **Notes**:  
> - The function converts inputs to arrays for consistent handling of both scalar and array-like inputs.  
> - For each pair of `data` and `err`:  
    - If the uncertainty (`err`) is zero, the output is formatted as $\text{data}$ without the $\pm$ term.  
    - Otherwise, the uncertainty is rounded to two significant figures, and the central value is rounded to the same order of magnitude.  
> - The output strings are enclosed in LaTeX math mode (e.g., `$1.23 \pm 0.04$`).  

**Example**:  
```python
>>> format_str(1.23456, 0.04321)
'$1.235 \\pm 0.043$'
>>> format_str([1.23456, 2.345], [0.04321, 0.012])
['$1.235 \\pm 0.043$', '$2.355 \\pm 0.012$']
>>> format_str(5.0, 0.0)
'$5.0$'
```

---

## `LabToolbox.utils.latex_table`

**Description**:  
Writes a LaTeX-formatted table to a file with a caption, label, and predefined styling.

**Parameters**:  
- `data` (*list of list of str or float*): The content of the table, organized as a list of rows.  
- `header` (*list of str*): List of column names to appear in the header of the table.  
- `filename` (*str*): Path to the output `.tex` file (e.g., `'table.tex'`).  
- `caption` (*str*): Caption text of the table.  
- `label` (*str*): Label used for referencing the table in LaTeX.
- `align` (*str, optional*): Column alignment string (e.g., "lcr"). If a single character ("l", "c", or "r") is given, it is repeated for all columns. Defaults to "c".

**Example**:  
```python
>>> # Example data: measurements with uncertainties
>>> measurements = np.array([1.23456, 2.34567, 3.45678])
>>> uncertainties = np.array([0.04321, 0.01234, 0.05678])
>>> # Format data and uncertainties using format_str
>>> formatted_data = format_str(measurements, uncertainties)
>>> # Create table data with formatted values and a second column for notes
>>> table_data = [
...     [formatted_data[0], "Sample 1"],
...     [formatted_data[1], "Sample 2"],
...     [formatted_data[2], "Sample 3"]
... ]
>>> header = ["Measurement", "Description"]
>>> # Write LaTeX table to file
>>> latex_table(
...     data = table_data,
...     header = header,
...     filename = "measurements.tex",
...     caption = "Measurements with uncertainties from experiment.",
...     label = "tab:measurements"
... )
```
---

## `LabToolbox.utils.noise`

**Description**:  
Generates an array of random noise samples from a normal distribution.

**Parameters**:  
- `n` (*int*): Number of samples to generate.  
- `std` (*float*): Standard deviation of the normal distribution.

**Returns**:  
- (*numpy.ndarray*): Array of `n` random samples with mean 0 and standard deviation `std`.

---

## `LabToolbox.utils.convert`

**Description**:  
Converts a physical quantity between units, supporting SI prefixes, non-SI units, and compound units. The function uses `astropy.units` to handle unit conversions, providing accurate transformations and informative output.

> **Note**: Coming in future updates!

**Parameters**:  
- `value` (*float or int*): Numerical value to be converted.  
- `from_unit` (*str*): Unit of the input quantity (e.g., `'erg'`, `'km/s'`, `'eV/Å^3'`).  
- `to_unit` (*str*): Desired target unit (e.g., `'J'`, `'m/s'`, `'GeV/fm^3'`).

**Returns**:  
- (*float*): The value converted to the target unit.

**Example**:  
```python
>>> convert(1000, "erg", "J")
Input:  1000 [erg]
Output: 0.0001 [J]
0.0001
>>> convert(60, "km/h", "m/s")
Input:  60 [km/h]
Output: 16.666666666666668 [m/s]
16.666666666666668
>>> convert(1, "eV/Å^3", "GeV/fm^3")
Input:  1 [eV/Å^3]
Output: 0.1602176634 [GeV/fm^3]
0.1602176634
```

---

This documentation will be updated as additional modules and functions are added to the `LabToolbox` package. For contributions or issues, please refer to the GitHub repository.