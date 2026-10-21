"""Keras Code Generator Package."""

from __future__ import annotations

import importlib.util
import sys

from ml_switcheroo_compiler.backends.eager_registry import as_callable

try:
    _has_pkg = importlib.util.find_spec("keras") is not None
except ValueError:
    _has_pkg = True

if not _has_pkg and "sphinx" not in sys.modules and "pytest" not in sys.modules:
    raise ImportError("The 'keras' backend requires the 'keras' library to be installed.")

from .eager import execute_op
from .generator import KerasCodeGenerator

type.__setattr__(KerasCodeGenerator, "execute_op", as_callable(classmethod)(execute_op))
