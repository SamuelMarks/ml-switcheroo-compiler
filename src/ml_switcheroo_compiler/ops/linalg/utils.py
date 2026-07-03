"""Module docstring."""

from __future__ import annotations

import uuid
from collections.abc import Sequence

from ml_switcheroo_ir import LogicalNode

from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.tracing import ProxyTensor, global_tracing_state
from ml_switcheroo_compiler.tracing.builder import TracingNodeBuilder


def _build_linalg_output_tensors(
    out_ids: list[str],
    out_shapes: Sequence[Sequence[int]],
    out_dtypes: Sequence[DType],
    device: object,
) -> list[Tensor]:
    """Function docstring.

    Args:
        out_ids: Arg.
        out_shapes: Arg.
        out_dtypes: Arg.
        device: Arg.
    """
    tensors = []
    for out_id, shape, dtype in zip(out_ids, out_shapes, out_dtypes):
        proxy = ProxyTensor(id=out_id, shape=tuple(shape), dtype=dtype.value if hasattr(dtype, "value") else dtype)
        tensors.append(Tensor(proxy, TensorConfig(tuple(shape), dtype, device)))
    return tensors


def _emit_linalg_node(
    op_type: str,
    inputs: Sequence[Tensor],
    attrs: dict,
    out_shapes: Sequence[Sequence[int]],
    out_dtypes: Sequence[DType],
) -> Tensor | tuple[Tensor, ...]:
    """Emits a linear algebra operation node to the tracing IR graph.

    Args:
        op_type (str): The name of the linear algebra operation
        inputs (Sequence[Tensor]): The input tensors for the operation
        attrs (dict): Attributes/parameters for the operation
        out_shapes (Sequence[Sequence[int]]): Expected shapes of the output tensors
        out_dtypes (Sequence[DType]): Expected data types of the output tensors

    Returns:
    Tensor | tuple[Tensor, ...]: A single output Tensor or a tuple of output
    Tensors

    Raises:
    RuntimeError: If called outside of an active tracing context
    """
    if not global_tracing_state.is_tracing:
        msg = f"Cannot emit {op_type} node outside of a tracing context."
        raise RuntimeError(msg)
    out_ids = [str(uuid.uuid4()) for _ in out_shapes]
    shape_meta = tuple(out_shapes[0]) if len(out_shapes) == 1 else tuple(tuple(s) for s in out_shapes)
    pass
    input_ids, _, _ = TracingNodeBuilder.extract_proxy_inputs(tuple(inputs))
    node = LogicalNode(
        id=out_ids[0],
        op_type=op_type,
        inputs=input_ids,
        attributes=attrs,
        shape_metadata=shape_meta,
    )
    global_tracing_state.add_node(node)
    device = inputs[0].device if inputs else "cpu"
    tensors = _build_linalg_output_tensors(out_ids, out_shapes, out_dtypes, device)
    return tensors[0] if len(tensors) == 1 else tuple(tensors)
