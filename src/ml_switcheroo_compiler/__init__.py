"""Compiler for ml-switcheroo."""

from ml_switcheroo_compiler import random as random
from ml_switcheroo_compiler.core.ragged_tensor import RaggedTensor
from ml_switcheroo_compiler.core.sparse_tensor import SparseTensor
from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.core.tensor_array import TensorArray
from ml_switcheroo_compiler.ops import control_flow as control_flow

__all__ = [
    "RaggedTensor",
    "SparseTensor",
    "Tensor",
    "TensorArray",
]
