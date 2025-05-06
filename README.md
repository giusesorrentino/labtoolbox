# LabToolbox

[![PyPI - Version](https://img.shields.io/pypi/v/LabToolbox?label=PyPI)](https://pypi.org/project/LabToolbox/)
![Python Versions](https://img.shields.io/pypi/pyversions/LabToolbox)
![PyPI - Downloads](https://img.shields.io/pypi/dm/LabToolbox)
[![License](https://img.shields.io/pypi/l/LabToolbox)](https://github.com/giusesorrentino/LabToolbox/blob/main/LICENSE.txt)
[![GitHub Issues](https://img.shields.io/github/issues/giusesorrentino/LabToolbox)](https://github.com/giusesorrentino/LabToolbox/issues)
[![GitHub Pull Requests](https://img.shields.io/github/issues-pr/giusesorrentino/LabToolbox)](https://github.com/giusesorrentino/LabToolbox/pulls)
![GitHub Repo stars](https://img.shields.io/github/stars/giusesorrentino/LabToolbox)
![GitHub Forks](https://img.shields.io/github/forks/giusesorrentino/LabToolbox)

<p align="left">
  <img src="https://raw.githubusercontent.com/giusesorrentino/LabToolbox/main/docs/logo2.png" width="700">
</p>

**LabToolbox** is a Python package that provides a collection of useful tools for laboratory data analysis. It offers intuitive and optimized functions for curve fitting, uncertainty propagation, data handling, and graphical visualization, enabling a faster and more rigorous approach to experimental data processing. Designed for students, researchers, and anyone working with experimental data, it combines ease of use with methodological accuracy.

The `example.ipynb` notebook, available on the package's [GitHub page](https://github.com/giusesorrentino/LabToolbox/blob/main/example.ipynb), includes usage examples for the main functions of `LabToolbox`.

## Installation

You can install **LabToolbox** easily using `pip`:

```bash
pip install LabToolbox
```

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
- `LabToolbox.utils`: A collection of helper functions for tasks like data formatting and general-purpose utilities used throughout the package.

- `LabToolbox.stats`: Statistical tools for experimental data analysis, including generation of synthetic datasets, histogram construction, outlier removal, residual analysis (normality, skewness, kurtosis), and likelihood/posterior computation for parametric models.

- `LabToolbox.fit`: Routines for linear and non-linear curve fitting, with support for uncertainty-aware methods.

- `LabToolbox.uncertainty`: Methods for estimating and propagating uncertainties in experimental contexts, allowing quantification of how input errors affect model outputs.

## Documentation

Detailed documentation for all modules and functions is available in the [GitHub Wiki](https://github.com/giusesorrentino/LabToolbox/wiki). The wiki includes function descriptions, usage examples, and practical guidance to help you get the most out of the library.

## Citation

If you use this software, please cite it using the metadata in [CITATION.cff](https://github.com/giusesorrentino/LabToolbox/blob/main/CITATION.cff). You can also use GitHub’s “Cite this repository” feature (available in the sidebar of the repository page).

## License 

MIT License – See the [LICENSE.txt](https://github.com/giusesorrentino/LabToolbox/blob/main/LICENSE.txt) file.

## Code of Conduct

This project includes a [Code of Conduct](https://github.com/giusesorrentino/LabToolbox/blob/main/CODE_OF_CONDUCT.md), which all users and contributors are expected to read and follow.

Additionally, the Code of Conduct contains a section titled “Author’s Ethical Requests” outlining the author's personal expectations regarding responsible and respectful use, especially in commercial or large-scale contexts. While not legally binding, these principles reflect the spirit in which this software was developed, and users are kindly asked to consider them when using the project.

## Disclaimer

This package makes use of the `uncertainty_class` library, available on [GitHub](https://github.com/yiorgoskost/Uncertainty-Propagation/tree/master), which provides functionality for uncertainty propagation in calculations. Manual installation is not required, as it is included as a module within `LabToolbox`.

The functions `my_cov`, `my_var`, `my_mean`, `my_line` and `y_estrapolato`, found in the modules `LabToolbox.utils` and `LabToolbox.fit`, originate from the `my_lib_santanastasio` library, developed by F. Santanastasio (professor of the *Laboratorio di Meccanica* course at the University of Rome “La Sapienza”), available at [this link](https://baltig.infn.it/LabMeccanica/PythonJupyter).

Tools such as `lin_fit` and `model_fit` include an option to display fit residuals. This functionality incorporates elements from the [**VoigtFit**](https://github.com/jkrogager/VoigtFit) library. The relevant portions of code are clearly marked in the source with a dedicated comment.