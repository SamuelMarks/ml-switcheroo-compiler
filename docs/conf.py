import os
import sys

# All target directories relative to the current conf.py
projects = [
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

fast_build = os.environ.get("FAST_BUILD", "0") == "1"

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

if not fast_build:
    for p in projects:
        sys.path.insert(
            0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", p))
        )

# -- Project information -----------------------------------------------------

project = "ML Switcheroo"
copyright = "2026, ML Switcheroo Authors"
author = "ML Switcheroo Authors"
version = "0.1"
release = "0.1"

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx_copybutton",
    "sphinx_design",
    "myst_parser",
]

if not fast_build:
    autosummary_generate = True

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------

html_theme = "furo"
html_static_path = ["_static"]


def setup(app):
    from ml_playground_directive import setup as setup_directive

    setup_directive(app)
