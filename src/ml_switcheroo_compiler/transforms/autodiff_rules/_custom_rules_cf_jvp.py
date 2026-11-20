"""Forward-mode (JVP) autodiff rules for structured control flow operations."""

from __future__ import annotations

from collections.abc import Sequence

from ml_switcheroo_ir import IRGraph, IRNode

from ml_switcheroo_compiler.core.shape import OpValue
from ml_switcheroo_compiler.transforms.autodiff_rules._custom_rules_common import (
    _is_subgraph_branch,
    _make_node_attrs,
)
from ml_switcheroo_compiler.transforms.autodiff_rules.jvp_registry import register_jvp


@register_jvp("If")
def _if_jvp(graph: IRGraph, node: IRNode, tangents: Sequence[str] | str) -> str:
    """JVP for If operation.

    Args:
        graph (IRGraph): Input `graph` argument (graph).
        node (IRNode): Input `node` argument (node).
        tangents (Sequence[str] | str): Input `tangents` argument (tangents).

    Returns:
        str: Tangent output node ID.
    """
    from ml_switcheroo_ir import IRNode

    from ml_switcheroo_compiler.transforms.autodiff import jvp

    attrs = node.attributes or {}
    if "then_branch" in attrs and "else_branch" in attrs:
        tb = attrs["then_branch"]
        eb = attrs["else_branch"]
        assert _is_subgraph_branch(tb) and _is_subgraph_branch(eb)
        tangents_seq: list[str] = [str(t) for t in tangents] if isinstance(tangents, (tuple, list)) else [str(tangents)]
        primals = []
        tangent_ids = []
        for i, in_id in enumerate(node.inputs):
            primals.append(in_id)
            if i < len(tangents_seq):
                tangent_ids.append(tangents_seq[i])

        then_jvp = jvp(tb, primals, tangent_ids, tb.outputs)
        else_jvp = jvp(eb, primals, tangent_ids, eb.outputs)

        new_id = f"{node.id}_jvp"
        new_node = IRNode(
            id=new_id,
            op_type="If",
            inputs=list(node.inputs),
            attributes=_make_node_attrs({"then_branch": then_jvp, "else_branch": else_jvp}),
        )
        graph.nodes[new_id] = new_node
        return new_id

    return "mock_tangent"


@register_jvp("Cond")
def _cond_jvp(graph: IRGraph, node: IRNode, tangents: Sequence[str] | str) -> str:
    """JVP for Cond operation.

    Args:
        graph (IRGraph): The computation graph.
        node (IRNode): The primal Cond node.
        tangents (Sequence[str] | str): Input tangent identifiers.

    Returns:
        str: Tangent output node identifier.
    """
    from ml_switcheroo_ir import IRNode

    from ml_switcheroo_compiler.transforms.autodiff import jvp

    attrs = node.attributes or {}
    tb = attrs.get("then_branch", attrs.get("true_branch"))
    eb = attrs.get("else_branch", attrs.get("false_branch"))

    if tb is not None and eb is not None and _is_subgraph_branch(tb) and _is_subgraph_branch(eb):
        primals = list(node.inputs)
        tangent_ids = [str(t) for t in tangents] if isinstance(tangents, (tuple, list)) else [str(tangents)]
        then_jvp = jvp(tb, primals, tangent_ids, tb.outputs)
        else_jvp = jvp(eb, primals, tangent_ids, eb.outputs)

        new_id = f"{node.id}_jvp"
        new_node = IRNode(
            id=new_id,
            op_type="Cond",
            inputs=list(node.inputs),
            attributes=_make_node_attrs(
                {
                    "then_branch": then_jvp,
                    "else_branch": else_jvp,
                    "true_branch": then_jvp,
                    "false_branch": else_jvp,
                }
            ),
            shape_metadata=node.shape_metadata,
        )
        graph.nodes[new_id] = new_node
        return new_id

    return "mock_tangent"


@register_jvp("WhileLoop")
def _while_loop_jvp(graph: IRGraph, node: IRNode, tangents: Sequence[str] | str) -> str:
    """JVP for WhileLoop operation.

    Args:
        graph (IRGraph): The computation graph.
        node (IRNode): The primal WhileLoop node.
        tangents (Sequence[str] | str): Input tangent identifiers.

    Returns:
        str: Tangent output node identifier.
    """
    from ml_switcheroo_ir import IRNode

    from ml_switcheroo_compiler.transforms.autodiff import jvp

    attrs = node.attributes or {}
    body = attrs.get("body", attrs.get("body_subgraph"))
    if body is not None and _is_subgraph_branch(body) and body.inputs and body.outputs:
        primals = list(node.inputs)
        tangent_ids = [str(t) for t in tangents] if isinstance(tangents, (tuple, list)) else [str(tangents)]
        body_jvp = jvp(body, primals, tangent_ids, body.outputs)

        new_id = f"{node.id}_jvp"
        new_node = IRNode(
            id=new_id,
            op_type="WhileLoop",
            inputs=list(node.inputs),
            attributes=_make_node_attrs(
                {
                    "body": body_jvp,
                    "body_subgraph": body_jvp,
                    "cond": attrs.get("cond"),
                }
            ),
            shape_metadata=node.shape_metadata,
        )
        graph.nodes[new_id] = new_node
        return new_id

    return "mock_tangent"


@register_jvp("Loop")
def _loop_jvp(graph: IRGraph, node: OpValue, tangents: OpValue) -> str:
    """JVP for Loop operation.

    Args:
        graph (IRGraph): Input `graph` argument (graph).
        node (OpValue): Input `node` argument (node).
        tangents (OpValue): Input `tangents` argument (tangents).

    Returns:
        str: Tangent node ID.
    """
    _ = (graph, node, tangents)
    return ""


@register_jvp("Scan")
def _scan_jvp(graph: IRGraph | None, node: IRNode | None, tangents: Sequence[str] | str) -> str:
    """JVP for Scan operation.

    Args:
        graph (IRGraph | None): The computation graph.
        node (IRNode | None): The primal Scan node.
        tangents (Sequence[str] | str): Tangents.

    Returns:
        str: Tangent node ID.
    """
    from ml_switcheroo_ir import IRNode

    from ml_switcheroo_compiler.transforms.autodiff import jvp

    if node is None or graph is None:
        return ""

    attrs = node.attributes or {}
    body = attrs.get("body", attrs.get("body_subgraph"))
    if body is not None and _is_subgraph_branch(body) and body.inputs and body.outputs:
        primals = list(node.inputs)
        tangent_ids = [str(t) for t in tangents] if isinstance(tangents, (tuple, list)) else [str(tangents)]
        body_jvp = jvp(body, primals, tangent_ids, body.outputs)

        new_id = f"{node.id}_jvp"
        new_node = IRNode(
            id=new_id,
            op_type="Scan",
            inputs=list(node.inputs),
            attributes=_make_node_attrs(
                {
                    "body": body_jvp,
                    "body_subgraph": body_jvp,
                    "reverse": attrs.get("reverse", False),
                }
            ),
            shape_metadata=node.shape_metadata,
        )
        graph.nodes[new_id] = new_node
        return new_id

    return ""


@register_jvp("AssociativeScan")
def _assoc_scan_jvp(graph: IRGraph, node: OpValue, tangents: OpValue) -> str:
    """JVP for AssociativeScan operation.

    Args:
        graph (IRGraph): Input `graph` argument (graph).
        node (OpValue): Input `node` argument (node).
        tangents (OpValue): Input `tangents` argument (tangents).

    Returns:
        str: Tangent node ID.
    """
    _ = (graph, node, tangents)
    return ""
