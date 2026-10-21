"""Utility functions for shape ops."""

from __future__ import annotations

# pylint: disable=duplicate-code
import uuid
from collections.abc import Mapping, Sequence

from ml_switcheroo_ir import LogicalNode

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
    inputs: Sequence[Tensor | TensorLike | bytes | TensorOrScalar] | TensorOrScalar,
    attrs: Mapping[str, TensorOrScalar] | None,
    out_shape: Sequence[int | str | None] | TensorOrScalar,
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

    seq_inputs = tuple(inputs) if isinstance(inputs, (list, tuple)) else (inputs,)
    input_ids, _, _ = TracingNodeBuilder.extract_proxy_inputs(seq_inputs)
    flat_inputs: list[str] = [str(sub) for item in input_ids for sub in (item if isinstance(item, (list, tuple)) else (item,))]
    from ml_switcheroo_ir import SymInt, SymNode

    raw_out_shape: Sequence[TensorOrScalar] = out_shape if isinstance(out_shape, (list, tuple)) else ()
    first_elem: TensorOrScalar = raw_out_shape[0] if len(raw_out_shape) > 0 else None
    flat_shape: Sequence[TensorOrScalar] = first_elem if isinstance(first_elem, (tuple, list)) else raw_out_shape
    norm_shape: tuple[int | str | SymNode | SymInt, ...] = tuple(int(d) if isinstance(d, float) else (d if isinstance(d, (int, str, SymNode, SymInt)) and not isinstance(d, bool) else ("?" if d is None else str(d))) for d in flat_shape)

    node = LogicalNode(
        id=out_id,
        op_type=op_type,
        inputs=flat_inputs,
        attributes={},
        shape_metadata=norm_shape,
    )
    if attrs:
        object.__setattr__(node, "attributes", dict(attrs))
    global_tracing_state.add_node(node)

    from ml_switcheroo_compiler.backends.registry import as_shape

    dtype_val = out_dtype.value if isinstance(out_dtype, DType) else str(out_dtype or "float32")
    shape_tuple = as_shape(tuple(raw_out_shape))

    proxy = ProxyTensor(id=out_id, shape=shape_tuple, dtype=dtype_val)
    device = getattr(seq_inputs[0], "device", config.default_device) if len(seq_inputs) > 0 else config.default_device
    return Tensor(proxy, TensorConfig(shape_tuple, out_dtype, device))


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
