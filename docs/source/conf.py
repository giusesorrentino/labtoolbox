# Configuration file for the Sphinx documentation builder.
# For full details, see: https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

import os
import sys
sys.path.insert(0, os.path.abspath('..'))  # Assicura l'import della libreria

# -- Project information -----------------------------------------------------

project = 'LabToolbox'
author = 'Giuseppe Sorrentino'
copyright = '2025, Giuseppe Sorrentino'
release = '3.0.0'
version = release

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',     # Documentazione automatica da docstring
    'sphinx.ext.napoleon',    # Supporto per Google e NumPy style docstrings
    'sphinx.ext.viewcode',    # Mostra sorgente Python linkabile
    'numpydoc',               # Analisi avanzata di docstring in stile NumPy
    'myst_parser',          # Decommenta se vuoi supportare Markdown
]

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------

html_theme = 'sphinx_rtd_theme'

html_theme_options = {
    'navigation_depth': 4,
    'collapse_navigation': False,
    'sticky_navigation': True,
}

html_static_path = ['_static']