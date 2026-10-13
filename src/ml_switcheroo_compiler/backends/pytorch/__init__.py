# ruff: noqa: E402, E501, F403
"""Pytorch Code Generator Package."""

import importlib.util
import sys

try:
    _has_pkg = importlib.util.find_spec("torch") is not None
except ValueError:
    _has_pkg = True

if not _has_pkg and "sphinx" not in sys.modules and "pytest" not in sys.modules:
    raise ImportError("The 'pytorch' backend requires the 'torch' library to be installed.")

from .eager import execute_op
from .generator import PyTorchCodeGenerator
from .types import array, asarray, item, zeros

PyTorchCodeGenerator.zeros = classmethod(zeros)
PyTorchCodeGenerator.array = classmethod(array)
PyTorchCodeGenerator.asarray = classmethod(asarray)
PyTorchCodeGenerator.item = classmethod(item)
PyTorchCodeGenerator.execute_op = classmethod(execute_op)
