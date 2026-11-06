"""Implementation of utils operations and utilities for the ops subsystem (`ml_switcheroo_compiler.ops.linalg.utils`)."""

from __future__ import annotations

import typing
import uuid
from collections.abc import Sequence

from ml_switcheroo_ir import LogicalNode

from ml_switcheroo_compiler.backends.registry import ensure_int
from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.shape import (
    DTypeHolderProtocol,
    TensorOrScalar,
    extract_device,
    extract_dtype,
    extract_shape,
)
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ir.core import ensure_attrs
from ml_switcheroo_compiler.tracing import ProxyTensor, global_tracing_state
from ml_switcheroo_compiler.tracing.builder import TracingNodeBuilder

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.ir.shape_system import SymInt, SymNode


def _extract_linalg_shape(inp: TensorOrScalar) -> tuple[int, ...]:
    """Extract integer shape tuple from `.shape`, `.shape_metadata`, or a sequence.

    Args:
        inp (TensorOrScalar): Input tensor, node, or shape sequence.

    Returns:
        tuple[int, ...]: Extracted integer shape tuple.
    """
    return extract_shape(inp, tuple(ensure_int(d) for d in inp) if isinstance(inp, (list, tuple)) else ())


def _extract_linalg_dtype(
    obj: TensorOrScalar,
    default: DType | str | None = None,
) -> DType | str | None:
    """Safely extract `.dtype` from `obj`, falling back to `default`.

    Args:
        obj (TensorOrScalar): Input tensor or array.
        default (DType | str | None): Default dtype if `obj` has no dtype.

    Returns:
        DType | str | None: Extracted dtype or `default`.
    """
    return extract_dtype(obj) if isinstance(obj, DTypeHolderProtocol) and obj.dtype is not None else default


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
    shape_meta: tuple[int | str, ...] = tuple("?" if d is None else (d if isinstance(d, int) else str(d)) for d in out_shapes[0]) if out_shapes else ()
    input_ids, _, _ = TracingNodeBuilder.extract_proxy_inputs(tuple(inputs))
    node = LogicalNode(
        id=out_ids[0],
        op_type=op_type,
        inputs=[str(i) for i in input_ids],
        attributes=ensure_attrs(attrs or {}),
        shape_metadata=shape_meta,
    )
    global_tracing_state.add_node(node)
    device = extract_device(inputs[0], "cpu") if inputs else "cpu"
    tensors = _build_linalg_output_tensors(out_ids, out_shapes, out_dtypes, device)
    return tensors[0] if len(tensors) == 1 else tuple(tensors)
