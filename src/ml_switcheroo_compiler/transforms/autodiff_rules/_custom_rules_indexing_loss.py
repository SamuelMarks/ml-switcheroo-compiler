"""Autodiff rules for indexing, scatter/update, ZerosLike, and LogPoissonLoss operations."""

from __future__ import annotations

from collections.abc import Sequence

from ml_switcheroo_ir import AttributeValue, IRGraph, IRNode

from ml_switcheroo_compiler.transforms.autodiff_rules._custom_rules_common import (
    _node_shape_meta,
)
from ml_switcheroo_compiler.transforms.autodiff_rules.jvp_registry import register_jvp
from ml_switcheroo_compiler.transforms.autodiff_rules.vjp_registry import register_vjp


@register_vjp("GetItem")
def getitem_vjp(graph: IRGraph, node: IRNode, cotangent: str) -> tuple[str, ...]:
    """VJP for GetItem indexing operation.

    Args:
        graph (IRGraph): Target computation graph.
        node (IRNode): GetItem node.
        cotangent (str): Upstream cotangent node ID.

    Returns:
        tuple[str, ...]: Adjoint for input array.
    """
    import uuid

    from ml_switcheroo_ir import LogicalNode

    inp_id = node.inputs[0] if node.inputs else ""
    key = node.attributes.get("key", 0)
    inp_shape = _node_shape_meta(graph, inp_id)

    zero_in_id = f"zero_in_{uuid.uuid4().hex[:6]}"
    graph.nodes[zero_in_id] = LogicalNode(
        id=zero_in_id,
        op_type="ZerosLike",
        inputs=[inp_id] if inp_id else [],
        shape_metadata=inp_shape,
    )

    put_id = f"index_put_{uuid.uuid4().hex[:6]}"
    graph.nodes[put_id] = LogicalNode(
        id=put_id,
        op_type="IndexPut",
        inputs=[zero_in_id, cotangent],
        attributes={"key": key},
        shape_metadata=inp_shape,
    )
    return (put_id,)


@register_jvp("IndexPut")
def index_put_jvp(graph: IRGraph, node: IRNode, tangents: Sequence[str] | str) -> str:
    """JVP for IndexPut operation.

    Args:
        graph (IRGraph): Target computation graph.
        node (IRNode): IndexPut node.
        tangents (Sequence[str] | str): Tangent identifiers.

    Returns:
        str: Tangent node ID.
    """
    import uuid

    from ml_switcheroo_ir import LogicalNode

    tangents_list: list[str] = [str(t) for t in tangents] if isinstance(tangents, (tuple, list)) else [str(tangents)]
    t_target = tangents_list[0] if len(tangents_list) > 0 else "0"
    t_val = tangents_list[1] if len(tangents_list) > 1 else "0"
    out_id = f"index_put_tan_{uuid.uuid4().hex[:6]}"
    graph.nodes[out_id] = LogicalNode(
        id=out_id,
        op_type="IndexPut",
        inputs=[t_target, t_val],
        attributes=dict(node.attributes or {}),
        shape_metadata=node.shape_metadata,
    )
    return out_id


@register_jvp("ZerosLike")
def zeros_like_jvp(graph: IRGraph, node: IRNode, tangents: Sequence[str] | str) -> str:
    """JVP for ZerosLike operation.

    Args:
        graph (IRGraph): Target computation graph.
        node (IRNode): ZerosLike node.
        tangents (Sequence[str] | str): Tangent identifiers.

    Returns:
        str: Tangent node ID.
    """
    import uuid

    from ml_switcheroo_ir import LogicalNode

    zero_id = f"zeros_like_tan_{uuid.uuid4().hex[:6]}"
    graph.nodes[zero_id] = LogicalNode(
        id=zero_id,
        op_type="ZerosLike",
        inputs=list(node.inputs),
        attributes=dict(node.attributes or {}),
        shape_metadata=node.shape_metadata,
    )
    return zero_id


@register_vjp("LogPoissonLoss")
def log_poisson_loss_vjp(graph: IRGraph, node: IRNode, cotangent: str) -> tuple[str, ...]:
    """VJP for LogPoissonLoss operation.

    Args:
        graph (IRGraph): Target computation graph.
        node (IRNode): LogPoissonLoss node.
        cotangent (str): Upstream cotangent node ID.

    Returns:
        tuple[str, ...]: Adjoint node IDs for targets and log_input arguments.
    """
    import uuid

    from ml_switcheroo_ir import LogicalNode

    targets_id: str = node.inputs[0] if len(node.inputs) > 0 else ""
    log_input_id: str = node.inputs[1] if len(node.inputs) > 1 else ""
    attrs: dict[str, AttributeValue] = dict(node.attributes or {})
    log_input_flag: bool = bool(attrs.get("log_input", True))

    exp_id = f"exp_{uuid.uuid4().hex[:6]}"
    sub_id = f"sub_{uuid.uuid4().hex[:6]}"
    mul_id = f"mul_{uuid.uuid4().hex[:6]}"

    if log_input_flag:
        graph.nodes[exp_id] = LogicalNode(
            id=exp_id, op_type="Exp", inputs=[log_input_id], shape_metadata=node.shape_metadata
        )
        graph.nodes[sub_id] = LogicalNode(
            id=sub_id, op_type="Sub", inputs=[exp_id, targets_id], shape_metadata=node.shape_metadata
        )
        graph.nodes[mul_id] = LogicalNode(
            id=mul_id, op_type="Mul", inputs=[sub_id, cotangent], shape_metadata=node.shape_metadata
        )
    else:
        one_id = f"one_{uuid.uuid4().hex[:6]}"
        div_id = f"div_{uuid.uuid4().hex[:6]}"
        graph.nodes[one_id] = LogicalNode(
            id=one_id, op_type="Constant", attributes={"value": 1.0}, shape_metadata=node.shape_metadata
        )
        graph.nodes[div_id] = LogicalNode(
            id=div_id, op_type="Div", inputs=[targets_id, log_input_id], shape_metadata=node.shape_metadata
        )
        graph.nodes[sub_id] = LogicalNode(
            id=sub_id, op_type="Sub", inputs=[one_id, div_id], shape_metadata=node.shape_metadata
        )
        graph.nodes[mul_id] = LogicalNode(
            id=mul_id, op_type="Mul", inputs=[sub_id, cotangent], shape_metadata=node.shape_metadata
        )

    target_zero_id = f"zero_{uuid.uuid4().hex[:6]}"
    graph.nodes[target_zero_id] = LogicalNode(
        id=target_zero_id,
        op_type="ZerosLike",
        inputs=[targets_id] if targets_id else [],
        shape_metadata=node.shape_metadata,
    )
    return (target_zero_id, mul_id)


@register_jvp("LogPoissonLoss")
def log_poisson_loss_jvp(graph: IRGraph, node: IRNode, tangents: Sequence[str] | str) -> str:
    """JVP for LogPoissonLoss operation.

    Args:
        graph (IRGraph): Target computation graph.
        node (IRNode): LogPoissonLoss node.
        tangents (Sequence[str] | str): Tangent identifiers.

    Returns:
        str: Tangent node ID.
    """
    import uuid

    from ml_switcheroo_ir import LogicalNode

    targets_id: str = node.inputs[0] if len(node.inputs) > 0 else ""
    log_input_id: str = node.inputs[1] if len(node.inputs) > 1 else ""
    t_tangent: str = tangents[1] if len(tangents) > 1 else tangents[0]

    attrs: dict[str, AttributeValue] = dict(node.attributes or {})
    log_input_flag: bool = bool(attrs.get("log_input", True))

    exp_id = f"exp_tan_{uuid.uuid4().hex[:6]}"
    sub_id = f"sub_tan_{uuid.uuid4().hex[:6]}"
    mul_id = f"mul_tan_{uuid.uuid4().hex[:6]}"

    if log_input_flag:
        graph.nodes[exp_id] = LogicalNode(
            id=exp_id, op_type="Exp", inputs=[log_input_id], shape_metadata=node.shape_metadata
        )
        graph.nodes[sub_id] = LogicalNode(
            id=sub_id, op_type="Sub", inputs=[exp_id, targets_id], shape_metadata=node.shape_metadata
        )
        graph.nodes[mul_id] = LogicalNode(
            id=mul_id, op_type="Mul", inputs=[sub_id, t_tangent], shape_metadata=node.shape_metadata
        )
    else:
        one_id = f"one_tan_{uuid.uuid4().hex[:6]}"
        div_id = f"div_tan_{uuid.uuid4().hex[:6]}"
        graph.nodes[one_id] = LogicalNode(
            id=one_id, op_type="Constant", attributes={"value": 1.0}, shape_metadata=node.shape_metadata
        )
        graph.nodes[div_id] = LogicalNode(
            id=div_id, op_type="Div", inputs=[targets_id, log_input_id], shape_metadata=node.shape_metadata
        )
        graph.nodes[sub_id] = LogicalNode(
            id=sub_id, op_type="Sub", inputs=[one_id, div_id], shape_metadata=node.shape_metadata
        )
        graph.nodes[mul_id] = LogicalNode(
            id=mul_id, op_type="Mul", inputs=[sub_id, t_tangent], shape_metadata=node.shape_metadata
        )

    return mul_id
