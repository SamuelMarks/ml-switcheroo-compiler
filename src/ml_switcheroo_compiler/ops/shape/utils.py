"""Utility functions for shape ops."""

from __future__ import annotations

# pylint: disable=duplicate-code
import uuid
from collections.abc import Mapping, Sequence
from typing import Callable, cast

from ml_switcheroo_ir import AttributeValue, LogicalNode

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.shape import TensorOrScalar
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.interpreter.evaluator import TensorLike
from ml_switcheroo_compiler.ops.registry import register_util
from ml_switcheroo_compiler.tracing import ProxyTensor, global_tracing_state
from ml_switcheroo_compiler.tracing.builder import TracingNodeBuilder


@register_util("_emit_shape_node")
def _emit_shape_node(
    op_type: str,
    inputs: Sequence[Tensor | TensorLike | bytes | TensorOrScalar],
    attrs: Mapping[str, TensorOrScalar] | None,
    out_shape: Sequence[int | str],
    out_dtype: DType | str | None,
) -> Tensor:
    """Emit a logical shape node to the tracer and returns a new Tensor.

    Args:
        op_type (str): The op_type parameter.
        inputs (Sequence): The inputs parameter.
        attrs (dict): The attrs parameter.
        out_shape (tuple): The out_shape parameter.
        out_dtype (DType): The out_dtype parameter.

    Returns:
        Tensor: Result.
    """
    out_id = str(uuid.uuid4())

    input_ids, _, _ = TracingNodeBuilder.extract_proxy_inputs(tuple(inputs))

    node = cast("Callable[..., LogicalNode]", LogicalNode)(
        id=out_id,
        op_type=op_type,
        inputs=input_ids,
        attributes=cast("dict[str, AttributeValue]", attrs or {}),
        shape_metadata=out_shape,
    )
    global_tracing_state.add_node(node)

    dtype_val = out_dtype.value if isinstance(out_dtype, DType) else str(out_dtype or "float32")

    proxy = ProxyTensor(id=out_id, shape=tuple(out_shape), dtype=dtype_val)
    device = getattr(inputs[0], "device", config.default_device) if len(inputs) > 0 else config.default_device
    return Tensor(proxy, TensorConfig(tuple(out_shape), out_dtype, device))


def compute_reduction_shape(x_shape: tuple[int, ...], axes: tuple[int, ...], keepdims: bool) -> tuple[int, ...]:
    """Reusable utility to compute the shape after a reduction operation.

    Args:
        x_shape (tuple[int, ...]): The x_shape parameter.
        axes (tuple[int, ...]): The axes parameter.
        keepdims (bool): The keepdims parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    if keepdims:
        return tuple(1 if i in axes else s for i, s in enumerate(x_shape))
    return tuple(s for i, s in enumerate(x_shape) if i not in axes)
