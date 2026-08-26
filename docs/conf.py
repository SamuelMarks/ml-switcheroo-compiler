"""Sphinx configuration file for the ML Switcheroo documentation.

This script configures the Sphinx documentation builder for the ML Switcheroo project
and its associated subprojects. It dynamically configures the Python path to include
target subprojects for autodoc generation, sets up extensions like autosummary and
myst_parser, and configures the Furo HTML theme.
"""

import os
import sys
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    import sphinx.application  # type: ignore[import-untyped]

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from ml_playground_directive import setup as setup_directive

# All target directories relative to the current conf.py
projects: list[str] = [
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

fast_build: bool = os.environ.get("FAST_BUILD", "0") == "1"

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

if not fast_build:
    for p in projects:
        project_root: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", p))
        sys.path.insert(0, project_root)
        src_dir: str = os.path.join(project_root, "src")
        if os.path.exists(src_dir):
            sys.path.insert(0, src_dir)

# -- Project information -----------------------------------------------------

project: str = "ML Switcheroo"
copyright: str = "2026, ML Switcheroo Authors"
author: str = "ML Switcheroo Authors"
version: str = "0.1"
release: str = "0.1"

# -- General configuration ---------------------------------------------------

extensions: list[str] = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx_copybutton",
    "sphinx_design",
    "myst_parser",
]

if not fast_build:
    autosummary_generate: bool = True

templates_path: list[str] = ["_templates"]
exclude_patterns: list[str] = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------

html_theme: str = "furo"
html_static_path: list[str] = ["_static"]


def setup(app: "sphinx.application.Sphinx") -> dict[str, Union[str, bool]]:
    """Initializes the Sphinx extension by registering custom directives.

    This function is called by Sphinx when the extension is loaded. It imports and sets
    up the custom playground directive for the documentation.

    Args:
    app: The Sphinx application instance used to register extensions and
    directives.

    Returns:
    The result of the directive setup, typically a metadata dictionary.
    """
    return setup_directive(app)
