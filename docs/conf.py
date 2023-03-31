# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Unum · UNSW'
copyright = '2023, Unum'
author = 'Unum'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'breathe', 'm2r2',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon',
    'sphinxcontrib.googleanalytics']

exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '*.md']


googleanalytics_id = '341385789'
googleanalytics_enabled = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_logo = '../assets/unum.png'
html_theme = 'furo'
html_static_path = ['_static']
html_css_files = []
html_js_files = []


breathe_projects = {'UNSW': '../build/xml'}
breathe_default_project = 'UNSW'
