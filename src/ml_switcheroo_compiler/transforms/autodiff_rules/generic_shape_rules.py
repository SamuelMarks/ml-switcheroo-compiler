"""Shape rules for misc."""

from ml_switcheroo_compiler.ops.base import emit_ir_node


def _generic_shape_vjp(graph: object, node: object, cotangent: str) -> tuple:
    """VJP for generic shape transformations.

    Args:
        graph (object): The IR graph.
        node (object): The IR node.
        cotangent (str): The cotangent ID.

    Returns:
        tuple: Tuple containing the reshaped cotangent.
    """
    x = node.inputs[0]
    return (emit_ir_node(graph, "Reshape", [cotangent], graph.nodes[x].shape_metadata),)


def _generic_shape_jvp(graph: object, node: object, tangent: str) -> str:
    """JVP for generic shape transformations.

    Args:
        graph (object): The IR graph.
        node (object): The IR node.
        tangent (str): The tangent ID.

    Returns:
        str: Tangent node ID.
    """
    return emit_ir_node(graph, node.op_type, [tangent], graph.nodes[node.id].shape_metadata, attributes=node.attributes)
