"""Apache Arrow Compute Backend Package for zero-copy columnar execution."""

from __future__ import annotations

from . import eager, generator, types
from .eager import execute_op
from .generator import PyArrowComputeGenerator
from .types import (
    array,
    asarray,
    chunked_array,
    item,
    record_batch,
    slice_record_batch,
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

__all__ = [
    "PyArrowComputeGenerator",
    "array",
    "asarray",
    "chunked_array",
    "eager",
    "execute_op",
    "generator",
    "item",
    "record_batch",
    "slice_record_batch",
    "to_table",
    "types",
    "zeros",
]
