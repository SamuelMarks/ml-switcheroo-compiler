"""Tensorflow Code Generator Package."""

from __future__ import annotations

import importlib.util
import sys
import typing

from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol

try:
    _has_pkg = importlib.util.find_spec("tensorflow") is not None
except ValueError:
    _has_pkg = True

if not _has_pkg and "sphinx" not in sys.modules and "pytest" not in sys.modules:
    raise ImportError("The 'tensorflow' backend requires the 'tensorflow' library to be installed.")

from .eager import execute_op
from .generator import TensorFlowCodeGenerator

type.__setattr__(TensorFlowCodeGenerator, "execute_op", typing.cast("EagerCallableProtocol", classmethod)(execute_op))
