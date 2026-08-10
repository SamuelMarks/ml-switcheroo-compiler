# ruff: noqa: E402, D100, D103, D104, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, D101, D102, D107, E701, E722, F403, E711, E712, PLR0913, PLR0915
from typing import Any

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
