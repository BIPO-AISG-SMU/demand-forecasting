# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
# -- Path setup --------------------------------------------------------------
# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

sys.path.insert(0, os.path.abspath("../src"))


# -- Project information -----------------------------------------------------

project = "BIPO Demand Forecasting"
author = "AI Singapore"
copyright = "2023, AI Singapore"

# The full version, including alpha/beta/rc tags
release = "v1.0.0"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx_autodoc_typehints",
    "sphinx.ext.doctest",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.ifconfig",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "nbsphinx",
    "sphinx_copybutton",
    "myst_parser",
    "sphinx.ext.intersphinx",
]

# intersphinx mapping for external libraries
intersphinx_mapping = {
    'numpy': ('https://numpy.org/doc/stable/', None),
    'pandas': ('https://pandas.pydata.org/pandas-docs/stable/', None),
    'py': ('https://docs.python.org/3/', None),
    'sklearn': ('https://scikit-learn.org/stable/', None),
    'lightweightmmm': ('https://lightweight-mmm.readthedocs.io/en/latest/', None),
    'tsfresh': ('https://tsfresh.readthedocs.io/', None),
    'statsmodels': ('https://www.statsmodels.org/stable/', None),
    'interpretml': ('https://interpret.ml/docs/', None),
    'mlflow': ('https://mlflow.org/docs/latest/', None),
    'kedro': ('https://docs.kedro.org/', None),
    'pytest': ('https://docs.pytest.org/en/7.4.x/', None),
    'pydantic': ('https://docs.pydantic.dev/latest/', None),
}

myst_enable_extensions = [
    "html_image",
]

myst_heading_anchors = 4

autodoc_inherit_docstrings = False

# enable autosummary plugin (table of contents for modules/classes/class methods)
autosummary_generate = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The suffix(es) of source filenames
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}

# The master toctree document.
master_doc = "index"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_book_theme"

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
# html_theme_options = {"collapse_navigation": False, "style_external_links": True}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["static"]

html_show_sourcelink = False

# -- Options for HTMLHelp output ---------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = "bipo_demand_forecasting doc"

# -- Options for manual page output ------------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    (
        master_doc,
        "bipo_demand_forecasting",
        "bipo_demand_forecasting Documentation",
        [author],
        1,
    )
]

# -- Extension configuration -------------------------------------------------

# nbsphinx_prolog = """
# see here for prolog/epilog details:
# https://nbsphinx.readthedocs.io/en/0.3.1/prolog-and-epilog.html
# """

# -- NBconvert kernel config -------------------------------------------------
nbsphinx_kernel_name = "python3"


def remove_arrows_in_examples(lines):
    for i, line in enumerate(lines):
        lines[i] = line.replace(">>> ", "")


def autodoc_process_docstring(app, what, name, obj, options, lines):
    remove_arrows_in_examples(lines)
