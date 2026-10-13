# ruff: noqa: E402, E501, F403
"""Tensorflow Code Generator Package."""

import importlib.util
import sys

try:
    _has_pkg = importlib.util.find_spec("tensorflow") is not None
except ValueError:
    _has_pkg = True

if not _has_pkg and "sphinx" not in sys.modules and "pytest" not in sys.modules:
    raise ImportError("The 'tensorflow' backend requires the 'tensorflow' library to be installed.")

from .eager import execute_op
from .generator import TensorFlowCodeGenerator

TensorFlowCodeGenerator.execute_op = classmethod(execute_op)
