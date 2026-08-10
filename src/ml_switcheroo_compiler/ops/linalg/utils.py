from __future__ import annotations

# ruff: noqa: E402, D100, D103, D104, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, D101, D102, D107, E701, E722, F403, E711, E712, PLR0913, PLR0915

"""Core abstractions and logic definitions for utils.py."""
import uuid
from collections.abc import Sequence
from typing import Any

from ml_switcheroo_ir import LogicalNode

from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.tracing import ProxyTensor, global_tracing_state
from ml_switcheroo_compiler.tracing.builder import TracingNodeBuilder


def _build_linalg_output_tensors(
    out_ids: list[str],
    out_shapes: Sequence[Sequence[int]],
    out_dtypes: Sequence[DType],
    device: Any,
) -> list[Tensor]:
    """Evaluate _build_linalg_output_tensors operation.

    Args:
        out_ids (object): The out_ids parameter.
        out_shapes (object): The out_shapes parameter.
        out_dtypes (object): The out_dtypes parameter.
        device (object): The device parameter.

    Returns: Any: Result.
    """
    tensors = []
    for out_id, shape, dtype in zip(out_ids, out_shapes, out_dtypes):
        proxy = ProxyTensor(id=out_id, shape=tuple(shape), dtype=dtype.value if hasattr(dtype, "value") else dtype)  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism
        tensors.append(Tensor(proxy, TensorConfig(tuple(shape), dtype, device)))
    return tensors


def _emit_linalg_node(
    op_type: str,
    inputs: Sequence[Tensor],
    attrs: dict,
    out_shapes: Sequence[Sequence[int]],
    out_dtypes: Sequence[DType],
) -> Tensor | tuple[Tensor, ...]:
    """Emit a linear algebra operation node to the tracing IR graph.

    Args:
        op_type (str): The op_type parameter.
        inputs (Sequence): The inputs parameter.
        attrs (dict): The attrs parameter.
        out_shapes (Sequence): The out_shapes parameter.
        out_dtypes (Sequence): The out_dtypes parameter.

    Returns: Any: Result.

    Raises:
        RuntimeError: An exception.
    """
    if not global_tracing_state.is_tracing:
        msg = f"Cannot emit {op_type} node outside of a tracing context."
        raise RuntimeError(msg)
    out_ids = [str(uuid.uuid4()) for _ in out_shapes]
    shape_meta = tuple(out_shapes[0]) if len(out_shapes) == 1 else tuple(tuple(s) for s in out_shapes)  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism
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
