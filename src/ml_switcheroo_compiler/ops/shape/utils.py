"""Utility functions for shape ops."""

from __future__ import annotations

# pylint: disable=duplicate-code
import uuid
from collections.abc import Mapping, Sequence

from ml_switcheroo_ir import LogicalNode

from ml_switcheroo_compiler.backends.registry import coerce_int
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.shape import (
    DTypeHolderProtocol,
    OpValue,
    extract_dtype,
    extract_shape,
)
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.interpreter.evaluator import TensorLike
from ml_switcheroo_compiler.ir.core import ensure_attrs
from ml_switcheroo_compiler.ops.registry import register_util
from ml_switcheroo_compiler.tracing import ProxyTensor, global_tracing_state
from ml_switcheroo_compiler.tracing.builder import TracingNodeBuilder


def _extract_shape_or_seq(inp: OpValue) -> tuple[int, ...]:
    """Extract integer shape tuple from `.shape`, `.shape_metadata`, or a sequence.

    Args:
        inp (OpValue): Input tensor, node, or shape sequence.

    Returns:
        tuple[int, ...]: Extracted integer shape tuple.
    """
    return extract_shape(inp, tuple(coerce_int(d) for d in inp) if isinstance(inp, (list, tuple)) else ())


def _extract_shape_dtype(
    obj: OpValue,
    default: DType | str | None = None,
) -> DType | str | None:
    """Safely extract `.dtype` from `obj`, falling back to `default`.

    Args:
        obj (OpValue): Input tensor or array.
        default (DType | str | None): Default dtype if `obj` has no dtype.

    Returns:
        DType | str | None: Extracted dtype or `default`.
    """
    return extract_dtype(obj) if isinstance(obj, DTypeHolderProtocol) and obj.dtype is not None else default


@register_util("_emit_shape_node")
def _emit_shape_node(
    op_type: str,
    inputs: Sequence[Tensor | TensorLike | bytes | OpValue] | OpValue,
    attrs: Mapping[str, OpValue] | None,
    out_shape: Sequence[int | str | None] | OpValue,
    out_dtype: DType | str | OpValue | None,
) -> Tensor:
    """Emit a logical shape node to the tracer and returns a new Tensor.

    Args:
        op_type (str): Input `op_type` argument (op type).
        inputs (Sequence[Tensor | TensorLike | bytes | OpValue] | OpValue):
            Input `inputs` argument (inputs).
        attrs (Mapping[str, OpValue] | None): Input `attrs` argument (attrs).
        out_shape (Sequence[int | str | None] | OpValue): Input `out_shape` argument (out shape).
        out_dtype (DType | str | OpValue | None): Input `out_dtype` argument (out dtype).

    Returns:
        Tensor: Computed `Tensor` output.
    """
    out_id = str(uuid.uuid4())

    seq_inputs = tuple(inputs) if isinstance(inputs, (list, tuple)) else (inputs,)
    input_ids, _, _ = TracingNodeBuilder.extract_proxy_inputs(seq_inputs)
    flat_inputs: list[str] = [
        str(sub) for item in input_ids for sub in (item if isinstance(item, (list, tuple)) else (item,))
    ]
    from ml_switcheroo_compiler.ir import SymInt, SymNode

    raw_out_shape: Sequence[OpValue] = out_shape if isinstance(out_shape, (list, tuple)) else ()
    first_elem: OpValue = raw_out_shape[0] if len(raw_out_shape) > 0 else None
    flat_shape: Sequence[OpValue] = first_elem if isinstance(first_elem, (tuple, list)) else raw_out_shape
    norm_shape: tuple[int | str | SymNode | SymInt, ...] = tuple(
        int(d)
        if isinstance(d, float)
        else (d if isinstance(d, (int, str, SymNode, SymInt)) and not isinstance(d, bool) else str(d))
        for d in flat_shape
    )

    node = LogicalNode(
        id=out_id,
        op_type=op_type,
        inputs=flat_inputs,
        attributes=ensure_attrs(dict(attrs)) if attrs else {},
        shape_metadata=norm_shape,
    )
    global_tracing_state.add_node(node)

    dtype_val = out_dtype.value if isinstance(out_dtype, DType) else str(out_dtype or "float32")

    proxy = ProxyTensor(id=out_id, shape=norm_shape, dtype=dtype_val)
    first_in = seq_inputs[0] if len(seq_inputs) > 0 else None
    device = first_in.device if isinstance(first_in, Tensor) else config.default_device
    return Tensor(proxy, TensorConfig(norm_shape, out_dtype, device))


def compute_reduction_shape(x_shape: tuple[int, ...], axes: tuple[int, ...], keepdims: bool) -> tuple[int, ...]:
    """Reusable utility to compute the shape after a reduction operation.

    Args:
        x_shape (tuple[int, ...]): Input `x_shape` argument (x shape).
        axes (tuple[int, ...]): Input `axes` argument (axes).
        keepdims (bool): Input `keepdims` argument (keepdims).

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    if keepdims:
        return tuple(1 if i in axes else s for i, s in enumerate(x_shape))
    return tuple(s for i, s in enumerate(x_shape) if i not in axes)
