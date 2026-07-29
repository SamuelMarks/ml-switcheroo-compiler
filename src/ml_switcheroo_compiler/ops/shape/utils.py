"""Utility functions for shape ops."""

from __future__ import annotations

# pylint: disable=duplicate-code
import uuid
from collections.abc import Sequence

from ml_switcheroo_ir import LogicalNode

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.registry import register_util
from ml_switcheroo_compiler.tracing import ProxyTensor, global_tracing_state
from ml_switcheroo_compiler.tracing.builder import TracingNodeBuilder


@register_util("_emit_shape_node")
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

    input_ids, _, _ = TracingNodeBuilder.extract_proxy_inputs(tuple(inputs))

    node = LogicalNode(
        id=out_id,
        op_type=op_type,
        inputs=input_ids,
        attributes=attrs,
        shape_metadata=out_shape,
    )
    global_tracing_state.add_node(node)

    dtype_val = out_dtype.value if hasattr(out_dtype, "value") else str(out_dtype) if hasattr(out_dtype, "name") else out_dtype

    proxy = ProxyTensor(id=out_id, shape=out_shape, dtype=dtype_val)
    device = getattr(inputs[0], "device", config.default_device) if len(inputs) > 0 else config.default_device
    return Tensor(proxy, TensorConfig(out_shape, out_dtype, device))


def compute_reduction_shape(x_shape: tuple[int, ...], axes: tuple[int, ...], keepdims: bool) -> tuple[int, ...]:
    """Reusable utility to compute the shape after a reduction operation."""
    if keepdims:
        return tuple(1 if i in axes else s for i, s in enumerate(x_shape))
    return tuple(s for i, s in enumerate(x_shape) if i not in axes)
