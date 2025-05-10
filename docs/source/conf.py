# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'LabToolbox'
copyright = '2025, Giuseppe Sorrentino'
author = 'Giuseppe Sorrentino'
release = '3.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = []

templates_path = ['_templates']
exclude_patterns = []

import os
import sys
sys.path.insert(0, os.path.abspath('..'))  # o percorso alla tua libreria

# Estensioni
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',  # per Google/Numpy docstring
    'sphinx.ext.viewcode',
    'numpydoc',             # stile NumPy
]

# Tema
html_theme = 'sphinx_rtd_theme'

# Opzioni per il tema
html_theme_options = {
    'navigation_depth': 4,
    'collapse_navigation': False,
    'sticky_navigation': True,
}

# Percorso statico (se vuoi aggiungere CSS personalizzato)
html_static_path = ['_static']