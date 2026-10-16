"""Apache Arrow Compute Backend Package for zero-copy columnar execution."""

from __future__ import annotations

from typing import Callable, cast

from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol

from . import eager, generator, types
from .eager import execute_op
from .generator import PyArrowComputeGenerator
from .types import (
    array,
    arrow_tensor_to_tensor,
    asarray,
    chunked_array,
    from_arrow,
    is_arrow_object,
    item,
    record_batch,
    record_batch_to_tensor,
    slice_record_batch,
    tensor_to_arrow_tensor,
    tensor_to_record_batch,
    to_arrow,
    to_table,
    zeros,
)

type.__setattr__(PyArrowComputeGenerator, "zeros", classmethod(zeros))
type.__setattr__(PyArrowComputeGenerator, "array", classmethod(array))
type.__setattr__(PyArrowComputeGenerator, "asarray", classmethod(asarray))
type.__setattr__(PyArrowComputeGenerator, "chunked_array", classmethod(chunked_array))
type.__setattr__(PyArrowComputeGenerator, "record_batch", classmethod(record_batch))
type.__setattr__(PyArrowComputeGenerator, "slice_record_batch", classmethod(slice_record_batch))
type.__setattr__(PyArrowComputeGenerator, "to_table", classmethod(to_table))
type.__setattr__(PyArrowComputeGenerator, "item", classmethod(item))
type.__setattr__(PyArrowComputeGenerator, "execute_op", cast("Callable[..., EagerTensorProtocol]", classmethod)(execute_op))
type.__setattr__(PyArrowComputeGenerator, "tensor_to_arrow_tensor", classmethod(tensor_to_arrow_tensor))
type.__setattr__(PyArrowComputeGenerator, "arrow_tensor_to_tensor", classmethod(arrow_tensor_to_tensor))
type.__setattr__(PyArrowComputeGenerator, "tensor_to_record_batch", classmethod(tensor_to_record_batch))
type.__setattr__(PyArrowComputeGenerator, "record_batch_to_tensor", classmethod(record_batch_to_tensor))
type.__setattr__(PyArrowComputeGenerator, "is_arrow_object", classmethod(is_arrow_object))
type.__setattr__(PyArrowComputeGenerator, "to_arrow", classmethod(to_arrow))
type.__setattr__(PyArrowComputeGenerator, "from_arrow", classmethod(from_arrow))

__all__ = [
    "PyArrowComputeGenerator",
    "array",
    "arrow_tensor_to_tensor",
    "asarray",
    "chunked_array",
    "eager",
    "execute_op",
    "from_arrow",
    "generator",
    "is_arrow_object",
    "item",
    "record_batch",
    "record_batch_to_tensor",
    "slice_record_batch",
    "tensor_to_arrow_tensor",
    "tensor_to_record_batch",
    "to_arrow",
    "to_table",
    "types",
    "zeros",
]
