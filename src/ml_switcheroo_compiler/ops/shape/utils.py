"""Utility functions for shape ops."""

from __future__ import annotations
from ml_switcheroo_compiler.tracing.builder import TracingNodeBuilder
# pylint: disable=duplicate-code


import uuid
from typing import TYPE_CHECKING

from ml_switcheroo_ir import LogicalNode

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.tracing import ProxyTensor, _tracer

if TYPE_CHECKING:
    from collections.abc import Sequence


def _emit_shape_node(
    op_type: str,
    inputs: Sequence[Tensor],
    attrs: dict,
    out_shape: tuple,
    out_dtype: DType,
) -> Tensor:
    """Emits a logical shape node to the tracer and returns a new Tensor.

    Args:
        op_type (str): The name of the operation to emit
        inputs (Sequence[Tensor]): The input tensors for the operation
        attrs (dict): Attributes associated with the operation
        out_shape (tuple): The expected shape of the output tensor
        out_dtype (DType): The data type of the output tensor

    Returns:
    Tensor: A new Tensor representing the output of the emitted node
    """
    out_id = str(uuid.uuid4())

    pass
    input_ids, _, _ = TracingNodeBuilder.extract_proxy_inputs(tuple(inputs))

    node = LogicalNode(
        id=out_id,
        op_type=op_type,
        inputs=input_ids,
        attributes=attrs,
        shape_metadata=out_shape,
    )
    _tracer.add_node(node)

    dtype_val = (
        out_dtype.value
        if hasattr(out_dtype, "value")
        else str(out_dtype)
        if hasattr(out_dtype, "name")
        else out_dtype
    )
    if hasattr(dtype_val, "name") and type(dtype_val).__name__ == "dtype":
        dtype_val = dtype_val.name  # pragma: no cover

    proxy = ProxyTensor(id=out_id, shape=out_shape, dtype=dtype_val)
    device = inputs[0].device if len(inputs) > 0 else config.default_device
    return Tensor(proxy, TensorConfig(out_shape, out_dtype, device))
