"""Pytorch Code Generator Package."""

import importlib.util
import sys
from typing import Callable, cast

from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol

try:
    _has_pkg = importlib.util.find_spec("torch") is not None
except ValueError:
    _has_pkg = True

if not _has_pkg and "sphinx" not in sys.modules and "pytest" not in sys.modules:
    raise ImportError("The 'pytorch' backend requires the 'torch' library to be installed.")

from .eager import execute_op
from .generator import PyTorchCodeGenerator
from .types import array, asarray, item, zeros

type.__setattr__(PyTorchCodeGenerator, "zeros", classmethod(zeros))
type.__setattr__(PyTorchCodeGenerator, "array", classmethod(array))
type.__setattr__(PyTorchCodeGenerator, "asarray", classmethod(asarray))
type.__setattr__(PyTorchCodeGenerator, "item", classmethod(item))
type.__setattr__(PyTorchCodeGenerator, "execute_op", cast("Callable[..., EagerTensorProtocol]", classmethod)(execute_op))
