"""Sparse COO Code Generator and Execution Package."""

import importlib.util
import sys

try:
    _has_pkg = importlib.util.find_spec("sparse") is not None
except (ValueError, ModuleNotFoundError):
    _has_pkg = True

if not _has_pkg and "sphinx" not in sys.modules and "pytest" not in sys.modules:
    raise ImportError("The 'sparse' backend requires the 'sparse' library to be installed.")

from . import eager, generator, kernels, types
from .eager import execute_op
from .generator import SparseGenerator
from .kernels import (
    csc_add,
    csr_add,
    dense_spmm,
    sparse_conv2d_mask,
    sparse_mask,
    spgemm,
    spgemm_grad,
    spmm,
    spmm_grad,
)
from .types import (
    COOTensor,
    CSCTensor,
    CSRTensor,
    array,
    asarray,
    coo_to_csc,
    coo_to_csr,
    csc_to_coo,
    csc_to_csr,
    csr_to_coo,
    csr_to_csc,
    item,
    zeros,
)

SparseGenerator.zeros = classmethod(zeros)
SparseGenerator.array = classmethod(array)
SparseGenerator.asarray = classmethod(asarray)
SparseGenerator.item = classmethod(item)
SparseGenerator.execute_op = classmethod(execute_op)

__all__ = [
    "COOTensor",
    "CSCTensor",
    "CSRTensor",
    "SparseGenerator",
    "array",
    "asarray",
    "coo_to_csc",
    "coo_to_csr",
    "csc_add",
    "csc_to_coo",
    "csc_to_csr",
    "csr_add",
    "csr_to_csc",
    "csr_to_coo",
    "dense_spmm",
    "eager",
    "execute_op",
    "generator",
    "item",
    "kernels",
    "sparse_conv2d_mask",
    "sparse_mask",
    "spgemm",
    "spgemm_grad",
    "spmm",
    "spmm_grad",
    "types",
    "zeros",
]
