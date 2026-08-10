# ruff: noqa: E402, D100, D103, D104, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, D101, D102, D107, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Shape rules for misc."""

from typing import Any

from ml_switcheroo_compiler.ops.base import emit_ir_node


def _generic_shape_vjp(graph: Any, node: Any, cotangent: str) -> tuple:
    """VJP for generic shape transformations.

    Args:
        graph (Any): The IR graph.
        node (Any): The IR node.
        cotangent (str): The cotangent ID.

    Returns:
        tuple: Tuple containing the reshaped cotangent.
    """
    x = node.inputs[0]
    return (emit_ir_node(graph, "Reshape", [cotangent], graph.nodes[x].shape_metadata),)


def _generic_shape_jvp(graph: Any, node: Any, tangent: str) -> str:
    """JVP for generic shape transformations.

    Args:
        graph (Any): The IR graph.
        node (Any): The IR node.
        tangent (str): The tangent ID.

    Returns:
        str: Tangent node ID.
    """
    return emit_ir_node(graph, node.op_type, [tangent], graph.nodes[node.id].shape_metadata, attributes=node.attributes)
