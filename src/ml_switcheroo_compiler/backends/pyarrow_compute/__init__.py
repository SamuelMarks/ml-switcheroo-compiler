"""Apache Arrow Compute Backend Package for zero-copy columnar execution."""

from __future__ import annotations

from typing import cast

from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol

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

type.__setattr__(PyArrowComputeGenerator, "zeros", staticmethod(zeros))
type.__setattr__(PyArrowComputeGenerator, "ones", staticmethod(ones))
type.__setattr__(PyArrowComputeGenerator, "empty", staticmethod(empty))
type.__setattr__(PyArrowComputeGenerator, "array", staticmethod(array))
type.__setattr__(PyArrowComputeGenerator, "asarray", staticmethod(asarray))
type.__setattr__(PyArrowComputeGenerator, "chunked_array", staticmethod(chunked_array))
type.__setattr__(PyArrowComputeGenerator, "record_batch", staticmethod(record_batch))
type.__setattr__(PyArrowComputeGenerator, "slice_record_batch", staticmethod(slice_record_batch))
type.__setattr__(PyArrowComputeGenerator, "to_table", staticmethod(to_table))
type.__setattr__(PyArrowComputeGenerator, "table", staticmethod(table))
type.__setattr__(PyArrowComputeGenerator, "tensor", staticmethod(tensor))
type.__setattr__(PyArrowComputeGenerator, "check_parity", staticmethod(check_parity))
type.__setattr__(PyArrowComputeGenerator, "item", staticmethod(item))
type.__setattr__(PyArrowComputeGenerator, "execute_op", cast("EagerCallableProtocol", staticmethod)(execute_op))
type.__setattr__(PyArrowComputeGenerator, "tensor_to_arrow_tensor", staticmethod(tensor_to_arrow_tensor))
type.__setattr__(PyArrowComputeGenerator, "arrow_tensor_to_tensor", staticmethod(arrow_tensor_to_tensor))
type.__setattr__(PyArrowComputeGenerator, "tensor_to_record_batch", staticmethod(tensor_to_record_batch))
type.__setattr__(PyArrowComputeGenerator, "record_batch_to_tensor", staticmethod(record_batch_to_tensor))
type.__setattr__(PyArrowComputeGenerator, "is_arrow_object", staticmethod(is_arrow_object))
type.__setattr__(PyArrowComputeGenerator, "to_arrow", staticmethod(to_arrow))
type.__setattr__(PyArrowComputeGenerator, "from_arrow", staticmethod(from_arrow))

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
