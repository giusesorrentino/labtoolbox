[![PyPI - Version](https://img.shields.io/pypi/v/labtoolbox?label=PyPI)](https://pypi.org/project/labtoolbox/)
[![PyPI Downloads](https://static.pepy.tech/badge/labtoolbox)](https://pepy.tech/projects/labtoolbox)
[![License](https://img.shields.io/pypi/l/labtoolbox)](https://github.com/giusesorrentino/labtoolbox/blob/master/LICENSE.txt)
[![GitHub Issues](https://img.shields.io/github/issues/giusesorrentino/labtoolbox)](https://github.com/giusesorrentino/labtoolbox/issues)
[![GitHub Pull Requests](https://img.shields.io/github/issues-pr/giusesorrentino/labtoolbox)](https://github.com/giusesorrentino/labtoolbox/pulls)
![GitHub Repo stars](https://img.shields.io/github/stars/giusesorrentino/labtoolbox)
![GitHub Forks](https://img.shields.io/github/forks/giusesorrentino/labtoolbox)

<p align="left">
  <picture>
    <source srcset="https://github.com/giusesorrentino/labtoolbox/raw/master/docs/logo_dark.png" media="(prefers-color-scheme: dark)">
    <img src="https://github.com/giusesorrentino/labtoolbox/raw/master/docs/logo_light.png" alt="LabToolbox logo" width="700">
  </picture>
</p>

**Labtoolbox** (stylized as *LabToolbox*) is a Python package that provides a collection of useful tools for laboratory data analysis. It offers intuitive and optimized functions for curve fitting, uncertainty propagation, data handling, and graphical visualization, enabling a faster and more rigorous approach to experimental data processing. Designed for students, researchers, and anyone working with experimental data, it combines ease of use with methodological accuracy.

## Installation

You can install `labtoolbox` from PyPI using `pip`:

```bash
pip install labtoolbox
```
<!-- 
Alternatively, you can install the latest development version directly from GitHub:

```bash
pip install git+https://github.com/giusesorrentino/LabToolbox.git
``` -->

<!-- If you prefer to clone the repository and install manually: -->
Alternatively, you can clone the repository and install it manually:

```bash
git clone https://github.com/giusesorrentino/labtoolbox.git
cd labtoolbox
pip install .
```

As of now, the package is not available on conda-forge. If you’re using Anaconda or Jupyter Notebook, you can still install `labtoolbox` by running the following code in the first cell of your notebook (only once per environment, not every time or for each `.ipynb` file):

```bash
!pip install labtoolbox
```

and then you can import the library.

### Important Note

As of version 3.1.0, the library must be imported using lowercase:

```python
import labtoolbox
```

The previous form with capitalization (`import LabToolbox`) is no longer supported.

If you have already installed a previous version of the package, you may encounter an issue where only the old import works, while the new one fails. This is typically caused by residual files from the previous installation. To resolve this, navigate to your site-packages directory (where Python packages are installed), then either:

-	Rename the old package folder (e.g., from `LabToolbox/` to `labtoolbox/`);
- Delete the old folder along with any associated metadata directories (e.g., .egg-info) and reinstall the new version using `pip`.

## Dependencies

LabToolbox relies on a set of well-established scientific Python libraries. When installed via `pip`, these dependencies are automatically handled. However, for reference or manual setup, here is the list of core dependencies:

- **numpy** – fundamental package for numerical computing.
- **scipy** – scientific and technical computing tools.
- **matplotlib** – for plotting and data visualization.
<!-- - **statsmodels** – statistical modeling and inference.
- **emcee** – affine-invariant ensemble sampler for MCMC.
- **corner** – corner plots for visualizing multidimensional distributions.
- **lmfit** – flexible curve-fitting with parameter constraints.
- **astropy** – core astronomy library for Python. -->

> **Note**: Up to version 2.0.3, the package was tested and validated on Python 3.9. Starting from version 3.1.0, it has been tested only on Python 3.11. While compatibility with earlier Python versions (≥ 3.9.6) is still expected, it is no longer officially guaranteed. The minimum required version remains Python 3.9.6.

## Library Structure

The `labtoolbox` package is organized into multiple submodules, each dedicated to a specific aspect of experimental data analysis. Below is an overview of the submodules and their functionalities:


| Subpackage    | Description                                                                                                          |
|----------------|----------------------------------------------------------------------------------------------------------------------|
| `numerical`    | General-purpose numerical routines, such as numerical integration and root finding        |
| `signals`      | Tools for post-processing and analysis of acquired signals. |
| `special`      | Special mathematical functions. |
| `stats`        | Tools for statistical analysis and data modeling. |
| `utils`        | A collection of general-purpose utilities used throughout the package. |
<!-- | `fit`        | Routines for linear and non-linear curve fitting, with support for uncertainty-aware methods. | -->
<!-- | `uncertainty`| Methods for estimating and propagating uncertainties in experimental contexts, allowing quantification of how input errors affect model outputs. | -->
<!-- | `optics`       | Optics-related tools, including polarization modeling, Jones calculus, and waveplate simulations.                   |
| `linalg`       | Tools for linear algebra operations, including matrix manipulations, eigensystems, and coordinate transformations. | -->

## Documentation

Detailed documentation for all modules and functions is available in the [GitHub Wiki](https://github.com/giusesorrentino/labtoolbox/wiki). The wiki includes function descriptions, usage examples, and practical guidance to help you get the most out of the library.

## Citation

If you use this software, please cite it using the metadata in [CITATION.cff](https://github.com/giusesorrentino/labtoolbox/blob/main/CITATION.cff). You can also use GitHub’s “Cite this repository” feature (available in the sidebar of the repository page).

## Code of Conduct

This project includes a [Code of Conduct](https://github.com/giusesorrentino/labtoolbox/blob/main/CODE_OF_CONDUCT.md), which all users and contributors are expected to read and follow.

Additionally, the Code of Conduct contains a section titled “Author’s Ethical Requests” outlining the author's personal expectations regarding responsible and respectful use, especially in commercial or large-scale contexts. While not legally binding, these principles reflect the spirit in which this software was developed, and users are kindly asked to consider them when using the project.

## Disclaimer

Labtoolbox makes use of the `uncertainty_class` package, available on [GitHub](https://github.com/yiorgoskost/Uncertainty-Propagation/tree/master), which provides functionality for uncertainty propagation in calculations. Manual installation is not required, as it is included as a module within LabToolbox.

Some utility functions are adapted from the `my_lib_santanastasio` package, available at [this link](https://baltig.infn.it/LabMeccanica/PythonJupyter), originally developed by F. Santanastasio for the *Laboratorio di Meccanica* course at the University of Rome “La Sapienza”.

Additionally, the `lin_fit` and `model_fit` functions provide the option to visualize fit residuals. This feature draws inspiration from the `VoigtFit` package, available on [GitHub](https://github.com/jkrogager/VoigtFit), with the relevant portions of code clearly annotated within the source.