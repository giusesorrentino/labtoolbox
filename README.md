# LabToolbox

[![PyPI - Version](https://img.shields.io/pypi/v/LabToolbox)](https://pypi.org/project/LabToolbox/)
![Python Versions](https://img.shields.io/pypi/pyversions/LabToolbox)
![PyPI - Downloads](https://img.shields.io/pypi/dm/LabToolbox)
[![License](https://img.shields.io/pypi/l/LabToolbox)](https://github.com/giusesorrentino/LabToolbox/blob/main/LICENSE.txt)
[![GitHub Issues](https://img.shields.io/github/issues/giusesorrentino/LabToolbox)](https://github.com/giusesorrentino/LabToolbox/issues)
[![GitHub Pull Requests](https://img.shields.io/github/issues-pr/giusesorrentino/LabToolbox)](https://github.com/giusesorrentino/LabToolbox/pulls)
![GitHub Repo stars](https://img.shields.io/github/stars/giusesorrentino/LabToolbox)
![GitHub Forks](https://img.shields.io/github/forks/giusesorrentino/LabToolbox)
<!-- ```text
    __          __  ______            ____              
   / /   ____ _/ /_/_  __/___  ____  / / /_  ____  _  __
  / /   / __ `/ __ \/ / / __ \/ __ \/ / __ \/ __ \| |/_/
 / /___/ /_/ / /_/ / / / /_/ / /_/ / / /_/ / /_/ />  <  
/_____/\__,_/_.___/_/  \____/\____/_/_.___/\____/_/|_|  
``` -->
<p align="left">
  <img src="docs/logo.png" alt="LabToolbox Logo" width="700"/>
</p>

**LabToolbox** is a Python library that provides a collection of useful tools for laboratory data analysis. It offers intuitive and optimized functions for curve fitting, uncertainty propagation, data handling, and graphical visualization, enabling a faster and more rigorous approach to experimental data processing. Designed for students, researchers, and anyone working with experimental data, it combines ease of use with methodological accuracy.

The `example.ipynb` notebook, available on the library's [GitHub page](https://github.com/giusesorrentino/LabToolbox/blob/main/example.ipynb), includes usage examples for the main functions of `LabToolbox`.

## Installation

You can install **LabToolbox** easily using `pip`:

```bash
pip install LabToolbox
```

## Library Structure

The **LabToolbox** library is organized into multiple submodules, each dedicated to a specific aspect of experimental data analysis:

- `LabToolbox.utils`  
  A collection of helper functions for general tasks that support various parts of the library, such as data formatting and other utility operations.

- `LabToolbox.stats`
  Provides statistical tools for experimental data analysis. This includes functions for generating synthetic datasets from known distributions, creating histograms, removing outliers, performing residual analysis with statistical tests (e.g., normality, skewness, kurtosis), and computing likelihoods and posterior probabilities for parametric models.

- `LabToolbox.fit`  
  Provides routines for linear and non-linear curve fitting, including uncertainty-aware methods.

- `LabToolbox.uncertainty`  
  Provides tools for estimating and propagating uncertainties in experimental data and models, helping to quantify the effect of input errors on the final results.

## Citation

If you use this software, please cite it using the metadata in [CITATION.cff](https://github.com/giusesorrentino/LabToolbox/blob/main/CITATION.cff). You can also use GitHub’s “Cite this repository” feature (available in the sidebar of the repository page).

## License 

MIT License – See the [LICENSE.txt](https://github.com/giusesorrentino/LabToolbox/blob/main/LICENSE.txt) file.

## Code of Conduct and Ethical Use

This project includes a [Code of Conduct](https://github.com/giusesorrentino/LabToolbox/blob/main/CODE_OF_CONDUCT.md), which all users and contributors are expected to read and follow.

Additionally, the Code of Conduct contains a section titled “Author’s Ethical Requests” outlining the author's personal expectations regarding responsible and respectful use, especially in commercial or large-scale contexts. While not legally binding, these principles reflect the spirit in which this software was developed, and users are kindly asked to consider them when using the project.

## Disclaimer

This package makes use of the `uncertainty_class` library, available on [GitHub](https://github.com/yiorgoskost/Uncertainty-Propagation/tree/master), which provides functionality for uncertainty propagation in calculations. Manual installation is not required, as it is included as a module within `LabToolbox`.

The functions `my_cov`, `my_var`, `my_mean`, `my_line` and `y_estrapolato`, found in the modules `LabToolbox.basics` and `LabToolbox.fit`, originate from the `my_lib_santanastasio` library, developed by F. Santanastasio (professor of the *Laboratorio di Meccanica* course at the University of Rome “La Sapienza”), available at [this link](https://baltig.infn.it/LabMeccanica/PythonJupyter).

Tools such as `lin_fit` and `model_fit` include an option to display fit residuals. This functionality incorporates elements from the [**VoigtFit**](https://github.com/jkrogager/VoigtFit) library. The relevant portions of code are clearly marked in the source with a dedicated comment.