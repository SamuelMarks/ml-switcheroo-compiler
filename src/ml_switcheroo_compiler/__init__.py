"""Compiler for ml-switcheroo."""

from __future__ import annotations

from ml_switcheroo_compiler import random as random
from ml_switcheroo_compiler.core.ragged_tensor import RaggedTensor
from ml_switcheroo_compiler.core.sparse_tensor import SparseTensor
from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.core.tensor_array import TensorArray
from ml_switcheroo_compiler.ops import control_flow as control_flow

__version__ = "0.0.1"

__all__ = ["RaggedTensor", "SparseTensor", "Tensor", "TensorArray", "__version__"]
