"""Sphinx configuration file for the ML Switcheroo documentation.

This script configures the Sphinx documentation builder for the ML Switcheroo project
and its associated subprojects. It dynamically configures the Python path to include
target subprojects for autodoc generation, sets up extensions like autosummary and
myst_parser, and configures the Furo HTML theme.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from ml_playground_directive import setup as setup_directive

# All target directories relative to the current conf.py
projects: object = [
    "../ml-switcheroo-compiler",
    "../ml-switcheroo-ir",
    "../zero-chex",
    "../zero-flax",
    "../zero-grain",
    "../zero-jax",
    "../zero-keras",
    "../zero-mlx",
    "../zero-optax",
    "../zero-orbax",
    "../zero-pax",
    "../zero-pytorch",
    "../zero-tensorflow",
]

fast_build: object = os.environ.get("FAST_BUILD", "0") == "1"

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

if not fast_build:
    for p in projects:
        project_root: object = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", p))
        sys.path.insert(0, project_root)
        src_dir: object = os.path.join(project_root, "src")
        if os.path.exists(src_dir):
            sys.path.insert(0, src_dir)

# -- Project information -----------------------------------------------------

project: object = "ML Switcheroo"
copyright: object = "2026, ML Switcheroo Authors"
author: object = "ML Switcheroo Authors"
version: object = "0.1"
release: object = "0.1"

# -- General configuration ---------------------------------------------------

extensions: object = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx_copybutton",
    "sphinx_design",
    "myst_parser",
]

if not fast_build:
    autosummary_generate: object = True

templates_path: object = ["_templates"]
exclude_patterns: object = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------

html_theme: object = "furo"
html_static_path: object = ["_static"]


def setup(app: object) -> object:
    """Initializes the Sphinx extension by registering custom directives.

    This function is called by Sphinx when the extension is loaded. It imports and sets
    up the custom playground directive for the documentation.

    Args:
    app (object): The Sphinx application instance used to register extensions and
    directives.

    Returns:
    object: The result of the directive setup, typically None.
    """
    setup_directive(app)
