"""Utility functions for shape ops."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from ml_switcheroo_ir import LogicalNode

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor
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
    node = LogicalNode(
        id=out_id,
        op_type=op_type,
        inputs=[inp.data.id for inp in inputs],
        attributes=attrs,
        shape_metadata=out_shape,
    )
    _tracer.add_node(node)
    proxy = ProxyTensor(id=out_id, shape=out_shape, dtype=out_dtype.value)
    device = inputs[0].device if len(inputs) > 0 else config.default_device
    return Tensor(data=proxy, shape=out_shape, dtype=out_dtype, device=device)
