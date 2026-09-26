"""Apache Arrow Compute Backend Package for zero-copy columnar execution."""

from __future__ import annotations

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

PyArrowComputeGenerator.zeros = classmethod(zeros)
PyArrowComputeGenerator.array = classmethod(array)
PyArrowComputeGenerator.asarray = classmethod(asarray)
PyArrowComputeGenerator.chunked_array = classmethod(chunked_array)
PyArrowComputeGenerator.record_batch = classmethod(record_batch)
PyArrowComputeGenerator.slice_record_batch = classmethod(slice_record_batch)
PyArrowComputeGenerator.to_table = classmethod(to_table)
PyArrowComputeGenerator.item = classmethod(item)
PyArrowComputeGenerator.execute_op = classmethod(execute_op)
PyArrowComputeGenerator.tensor_to_arrow_tensor = classmethod(tensor_to_arrow_tensor)
PyArrowComputeGenerator.arrow_tensor_to_tensor = classmethod(arrow_tensor_to_tensor)
PyArrowComputeGenerator.tensor_to_record_batch = classmethod(tensor_to_record_batch)
PyArrowComputeGenerator.record_batch_to_tensor = classmethod(record_batch_to_tensor)
PyArrowComputeGenerator.is_arrow_object = classmethod(is_arrow_object)
PyArrowComputeGenerator.to_arrow = classmethod(to_arrow)
PyArrowComputeGenerator.from_arrow = classmethod(from_arrow)

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
