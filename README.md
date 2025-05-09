<!-- # LabToolbox -->

[![PyPI - Version](https://img.shields.io/pypi/v/LabToolbox?label=PyPI)](https://pypi.org/project/LabToolbox/)
![Python Versions](https://img.shields.io/pypi/pyversions/LabToolbox)
![PyPI - Downloads](https://img.shields.io/pypi/dm/LabToolbox)
[![License](https://img.shields.io/pypi/l/LabToolbox)](https://github.com/giusesorrentino/LabToolbox/blob/main/LICENSE.txt)
[![GitHub Issues](https://img.shields.io/github/issues/giusesorrentino/LabToolbox)](https://github.com/giusesorrentino/LabToolbox/issues)
[![GitHub Pull Requests](https://img.shields.io/github/issues-pr/giusesorrentino/LabToolbox)](https://github.com/giusesorrentino/LabToolbox/pulls)
![GitHub Repo stars](https://img.shields.io/github/stars/giusesorrentino/LabToolbox)
![GitHub Forks](https://img.shields.io/github/forks/giusesorrentino/LabToolbox)

<p align="left">
  <picture>
    <source srcset="https://github.com/giusesorrentino/LabToolbox/raw/main/docs/logo_dark.png" media="(prefers-color-scheme: dark)">
    <img src="https://github.com/giusesorrentino/LabToolbox/raw/main/docs/logo2.png" alt="LabToolbox logo" width="700">
  </picture>
</p>

**LabToolbox** is a Python package that provides a collection of useful tools for laboratory data analysis. It offers intuitive and optimized functions for curve fitting, uncertainty propagation, data handling, and graphical visualization, enabling a faster and more rigorous approach to experimental data processing. Designed for students, researchers, and anyone working with experimental data, it combines ease of use with methodological accuracy.

<!-- The `example.ipynb` notebook, available on the package's [GitHub page](https://github.com/giusesorrentino/LabToolbox/blob/main/example.ipynb), includes usage examples for the main functions of `LabToolbox`. -->

## Installation

You can install **LabToolbox** from PyPI using `pip`:

```bash
pip install LabToolbox
```
<!-- 
Alternatively, you can install the latest development version directly from GitHub:

```bash
pip install git+https://github.com/giusesorrentino/LabToolbox.git
``` -->

<!-- If you prefer to clone the repository and install manually: -->
Alternatively, you can clone the repository and install it manually:

```bash
git clone https://github.com/giusesorrentino/LabToolbox.git
cd LabToolbox
pip install .
```

## Dependencies

LabToolbox relies on a set of well-established scientific Python libraries. When installed via `pip`, these dependencies are automatically handled. However, for reference or manual setup, here is the list of core dependencies:

- **numpy** – fundamental package for numerical computing.
- **scipy** – scientific and technical computing tools.
- **matplotlib** – for plotting and data visualization.
- **statsmodels** – statistical modeling and inference.
- **emcee** – affine-invariant ensemble sampler for MCMC.
- **corner** – corner plots for visualizing multidimensional distributions.
- **lmfit** – flexible curve-fitting with parameter constraints.
- **astropy** – core astronomy library for Python.

> **Note**: Up to version 2.0.3, the package was tested and validated on Python 3.9.6. Starting from version 3.0.0, it has been tested only on Python 3.13.3. While compatibility with earlier Python versions (≥ 3.9.6) is still expected, it is no longer officially guaranteed. The minimum required version remains Python 3.9.

## Library Structure

The **LabToolbox** package is organized into multiple submodules, each dedicated to a specific aspect of experimental data analysis:

<!-- ### `LabToolbox.utils`
A collection of helper functions for tasks like data formatting and general-purpose utilities used throughout the package.

### `LabToolbox.stats`
Statistical tools for experimental data analysis, including generation of synthetic datasets, histogram construction, outlier removal, residual analysis (normality, skewness, kurtosis), and likelihood/posterior computation for parametric models.

### `LabToolbox.fit`
Routines for linear and non-linear curve fitting, with support for uncertainty-aware methods.

### `LabToolbox.uncertainty`
Methods for estimating and propagating uncertainties in experimental contexts, allowing quantification of how input errors affect model outputs.

### `LabToolbox.signals`
Signal analysis tools tailored for laboratory experiments, featuring frequency domain analysis and post-processing of acquired data. -->
- **LabToolbox.fit**: Routines for linear and non-linear curve fitting.

- **LabToolbox.signals**: Signal analysis tools tailored for laboratory experiments, featuring frequency domain analysis and post-processing of acquired data.

- **LabToolbox.stats**: Statistical tools for experimental data analysis, including generation of synthetic datasets, histogram construction, outlier removal, residual analysis (normality, skewness, kurtosis), and likelihood/posterior computation for parametric models.

- **LabToolbox.uncertainty**: Methods for estimating and propagating uncertainties in experimental contexts, allowing quantification of how input errors affect model outputs.

- **LabToolbox.utils**: A collection of helper functions for tasks like data formatting and general-purpose utilities used throughout the package.

## Documentation

Detailed documentation for all modules and functions is available in the [GitHub Wiki](https://github.com/giusesorrentino/LabToolbox/wiki). The wiki includes function descriptions, usage examples, and practical guidance to help you get the most out of the library.

## Citation

If you use this software, please cite it using the metadata in [CITATION.cff](https://github.com/giusesorrentino/LabToolbox/blob/main/CITATION.cff). You can also use GitHub’s “Cite this repository” feature (available in the sidebar of the repository page).

<!-- ## License 

MIT License – See the [LICENSE.txt](https://github.com/giusesorrentino/LabToolbox/blob/main/LICENSE.txt) file. -->

## Code of Conduct

This project includes a [Code of Conduct](https://github.com/giusesorrentino/LabToolbox/blob/main/CODE_OF_CONDUCT.md), which all users and contributors are expected to read and follow.

Additionally, the Code of Conduct contains a section titled “Author’s Ethical Requests” outlining the author's personal expectations regarding responsible and respectful use, especially in commercial or large-scale contexts. While not legally binding, these principles reflect the spirit in which this software was developed, and users are kindly asked to consider them when using the project.

## Disclaimer

LabToolbox makes use of the **uncertainty_class** package, available on [GitHub](https://github.com/yiorgoskost/Uncertainty-Propagation/tree/master), which provides functionality for uncertainty propagation in calculations. Manual installation is not required, as it is included as a module within LabToolbox.

Some utility functions — namely `my_cov`, `my_var`, `my_mean`, `my_line`, and `y_estrapolato` — available in the **LabToolbox.utils** module, are adapted from the [**my_lib_santanastasio**](https://baltig.infn.it/LabMeccanica/PythonJupyter) package, originally developed by F. Santanastasio for the *Laboratorio di Meccanica* course at the University of Rome “La Sapienza”.

Additionally, the `lin_fit` and `model_fit` functions provide the option to visualize fit residuals. This feature draws inspiration from the [**VoigtFit**](https://github.com/jkrogager/VoigtFit) library, with the relevant portions of code clearly annotated within the source.