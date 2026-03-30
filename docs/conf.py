import os
import sys
sys.path.insert(0, os.path.abspath('..'))  # punta alla root del progetto

# -- Project information
project = 'labtoolbox'
copyright = '2026, Giuseppe Sorrentino'
author = 'Giuseppe Giuseppe'
release = '3.1.0'

# -- Extensions
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.autosummary',
]

# -- Autodoc configuration
autodoc_typehints = 'none'

# -- Tema
html_theme = 'pydata_sphinx_theme'

html_static_path = ['_static']

html_theme_options = {
    "logo": {
        "image_light": "_static/logo_light.png",
        "image_dark": "_static/logo_dark.png",
    },
    "show_toc_level": 2,
}