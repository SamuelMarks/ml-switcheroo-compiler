"""Pytorch Code Generator Package."""

import importlib.util
import sys
from typing import cast

from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol

try:
    _has_pkg = importlib.util.find_spec("torch") is not None
except ValueError:
    _has_pkg = True

if not _has_pkg and "sphinx" not in sys.modules and "pytest" not in sys.modules:
    raise ImportError("The 'pytorch' backend requires the 'torch' library to be installed.")

from .distributed_collectives import (
    pytorch_all_gather,
    pytorch_all_reduce,
    pytorch_broadcast,
    pytorch_reduce_scatter,
)
from .eager import execute_op
from .generator import PyTorchCodeGenerator
from .types import array, asarray, item, zeros

type.__setattr__(PyTorchCodeGenerator, "zeros", classmethod(zeros))
type.__setattr__(PyTorchCodeGenerator, "array", classmethod(array))
type.__setattr__(PyTorchCodeGenerator, "asarray", classmethod(asarray))
type.__setattr__(PyTorchCodeGenerator, "item", classmethod(item))
type.__setattr__(PyTorchCodeGenerator, "execute_op", cast("EagerCallableProtocol", classmethod)(execute_op))
type.__setattr__(PyTorchCodeGenerator, "all_reduce", staticmethod(pytorch_all_reduce))
type.__setattr__(PyTorchCodeGenerator, "all_gather", staticmethod(pytorch_all_gather))
type.__setattr__(PyTorchCodeGenerator, "reduce_scatter", staticmethod(pytorch_reduce_scatter))
type.__setattr__(PyTorchCodeGenerator, "broadcast", staticmethod(pytorch_broadcast))
