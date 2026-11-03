"""Apache Arrow Compute Backend Package for zero-copy columnar execution."""

from __future__ import annotations

from . import eager, generator, types
from .eager import execute_op
from .generator import PyArrowComputeGenerator
from .types import (
    array,
    arrow_tensor_to_tensor,
    asarray,
    check_parity,
    chunked_array,
    empty,
    from_arrow,
    is_arrow_object,
    item,
    ones,
    record_batch,
    record_batch_to_tensor,
    slice_record_batch,
    table,
    tensor,
    tensor_to_arrow_tensor,
    tensor_to_record_batch,
    to_arrow,
    to_table,
    zeros,
)

__all__ = [
    "PyArrowComputeGenerator",
    "array",
    "arrow_tensor_to_tensor",
    "asarray",
    "check_parity",
    "chunked_array",
    "eager",
    "empty",
    "execute_op",
    "from_arrow",
    "generator",
    "is_arrow_object",
    "item",
    "ones",
    "record_batch",
    "record_batch_to_tensor",
    "slice_record_batch",
    "table",
    "tensor",
    "tensor_to_arrow_tensor",
    "tensor_to_record_batch",
    "to_arrow",
    "to_table",
    "types",
    "zeros",
]
