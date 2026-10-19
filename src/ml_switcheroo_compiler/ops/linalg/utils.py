"""Implementation of utils operations and utilities for the ops subsystem (`ml_switcheroo_compiler.ops.linalg.utils`)."""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from typing import TYPE_CHECKING, Callable, cast

from ml_switcheroo_ir import AttributeValue, LogicalNode

if TYPE_CHECKING:
    from ml_switcheroo_compiler.ir.shape_system import SymInt, SymNode

from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.shape import TensorOrScalar
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.tracing import ProxyTensor, global_tracing_state
from ml_switcheroo_compiler.tracing.builder import TracingNodeBuilder


def _build_linalg_output_tensors(
    out_ids: list[str],
    out_shapes: Sequence[Sequence[int | str | SymNode | SymInt | None]],
    out_dtypes: Sequence[DType | str | None],
    device: Device | str | None,
) -> list[Tensor]:
    """Compute the build linalg output tensors tensor operation.

    Args:
        out_ids (list[str]): The out_ids parameter.
        out_shapes (Sequence): The out_shapes parameter.
        out_dtypes (Sequence): The out_dtypes parameter.
        device: The device parameter.

    Returns:
        list[Tensor]: Result.
    """
    tensors: list[Tensor] = []
    for out_id, shape, dtype in zip(out_ids, out_shapes, out_dtypes):
        dt = dtype or DType.Float32
        clean_shape: tuple[int | str, ...] = tuple(int(s) if isinstance(s, int) else str(s) for s in shape)
        proxy = ProxyTensor(id=out_id, shape=clean_shape, dtype=dt.value if isinstance(dt, DType) else str(dt))
        tensors.append(Tensor(proxy, TensorConfig(clean_shape, dt, device)))
    return tensors


def _emit_linalg_node(
    op_type: str,
    inputs: Sequence[Tensor | TensorOrScalar],
    attrs: dict[str, TensorOrScalar] | None,
    out_shapes: Sequence[Sequence[int | str | SymNode | SymInt | None]],
    out_dtypes: Sequence[DType | str | None],
) -> Tensor | tuple[Tensor, ...]:
    """Emit a linear algebra operation node to the tracing IR graph.

    Args:
        op_type (str): The op_type parameter.
        inputs (Sequence): The inputs parameter.
        attrs (dict): The attrs parameter.
        out_shapes (Sequence): The out_shapes parameter.
        out_dtypes (Sequence): The out_dtypes parameter.

    Returns:
            tuple[int, ...]: Result.

    Raises:
        RuntimeError: An exception.
    """
    if not global_tracing_state.is_tracing:
        msg = f"Cannot emit {op_type} node outside of a tracing context."
        raise RuntimeError(msg)

    out_ids = [str(uuid.uuid4()) for _ in out_shapes]
    shape_meta = cast(
        "Sequence[int | str]",
        tuple(out_shapes[0]) if len(out_shapes) == 1 else tuple(tuple(s) for s in out_shapes),
    )
    input_ids, _, _ = TracingNodeBuilder.extract_proxy_inputs(tuple(inputs))
    node = cast("Callable[..., LogicalNode]", LogicalNode)(
        id=out_ids[0],
        op_type=op_type,
        inputs=input_ids,
        attributes=cast("dict[str, AttributeValue]", attrs or {}),
        shape_metadata=shape_meta,
    )
    global_tracing_state.add_node(node)
    device = getattr(inputs[0], "device", "cpu") if inputs else "cpu"
    tensors = _build_linalg_output_tensors(out_ids, out_shapes, out_dtypes, device)
    return tensors[0] if len(tensors) == 1 else tuple(tensors)
