"""Jax Code Generator Package."""

import importlib.util
import sys
from typing import Callable, cast

from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol

try:
    _has_pkg = importlib.util.find_spec("jax") is not None
except ValueError:
    _has_pkg = True

if not _has_pkg and "sphinx" not in sys.modules and "pytest" not in sys.modules:
    raise ImportError("The 'jax' backend requires the 'jax' and 'jaxlib' libraries to be installed.")

from .eager import execute_op
from .generator import JAXCodeGenerator

type.__setattr__(JAXCodeGenerator, "execute_op", cast("Callable[..., EagerTensorProtocol]", classmethod)(execute_op))
