"""Defines Vector-Jacobian Product (VJP) and Jacobian-Vector Product (JVP) rules for.

reduction operations
"""

from ml_switcheroo_compiler.ops.base import emit_ir_node
from ml_switcheroo_compiler.transforms.autodiff_rules.jvp_registry import register_jvp
from ml_switcheroo_compiler.transforms.autodiff_rules.vjp_registry import register_vjp


@register_vjp("Sum")
def sum_vjp(graph: object, node: object, cotangent: str) -> tuple:
    """Computes the Vector-Jacobian Product (VJP) for the Sum operation.

    This rule broadcasts the incoming cotangent back to the shape of the original
    input
    tensor of the Sum operation

    Args:
        graph (object): The computation graph
        node (object): The Sum operation node
        cotangent (str): The identifier of the cotangent tensor

    Returns:
    tuple: A tuple containing the identifier of the broadcasted cotangent tensor
    """
    x = node.inputs[0]
    return (
        emit_ir_node(
            graph,
            "BroadcastTo",
            [cotangent],
            graph.nodes[x].shape_metadata,
            attributes={"shape": graph.nodes[x].shape_metadata},
        ),
    )


@register_jvp("Sum")
def sum_jvp(graph: object, node: object, tangent: str) -> str:
    """Computes the JVP for the Sum operation.

    Args:
        graph (object): The computation graph
        node (object): The Sum operation node
        tangent (str): The identifier of the tangent tensor

    Returns:
    str: The identifier of the computed JVP node
    """
    return emit_ir_node(graph, "Sum", [tangent], node.shape_metadata, node.attributes)


@register_vjp("Mean")
def mean_vjp(graph: object, node: object, cotangent: str) -> tuple:
    """Computes the Vector-Jacobian Product (VJP) for the Mean operation.

    Args:
        graph (object): The computation graph
        node (object): The Mean operation node
        cotangent (str): The identifier of the cotangent tensor

    Returns:
    tuple: A tuple containing the identifier of the broadcasted cotangent tensor
    """
    x = node.inputs[0]
    bcast = emit_ir_node(
        graph,
        "BroadcastTo",
        [cotangent],
        graph.nodes[x].shape_metadata,
        {"shape": graph.nodes[x].shape_metadata},
    )

    import math

    n = 1.0
    if graph.nodes[x].shape_metadata and node.shape_metadata:
        n = math.prod(graph.nodes[x].shape_metadata) / (
            math.prod(node.shape_metadata) if node.shape_metadata else 1.0
        )

    n_id = emit_ir_node(graph, "Constant", [], graph.nodes[x].shape_metadata, {"value": float(n)})
    res = emit_ir_node(graph, "TrueDivide", [bcast, n_id], graph.nodes[x].shape_metadata)
    return (res,)


@register_jvp("Mean")
def mean_jvp(graph: object, node: object, tangent: str) -> str:
    """Computes the JVP for the Mean operation.

    Args:
        graph (object): The computation graph
        node (object): The Mean operation node
        tangent (str): The identifier of the tangent tensor

    Returns:
    str: The identifier of the computed JVP node
    """
    return emit_ir_node(graph, "Mean", [tangent], node.shape_metadata, node.attributes)


@register_vjp("Max")
def max_vjp(graph: object, node: object, cotangent: str) -> tuple:
    """Computes the Vector-Jacobian Product (VJP) for the Max operation.

    Args:
        graph (object): The computation graph
        node (object): The Max operation node
        cotangent (str): The identifier of the cotangent tensor

    Returns:
    tuple: A tuple containing the identifier of the broadcasted cotangent tensor
    """
    x = node.inputs[0]
    bcast = emit_ir_node(
        graph,
        "BroadcastTo",
        [cotangent],
        graph.nodes[x].shape_metadata,
        {"shape": graph.nodes[x].shape_metadata},
    )
    return (bcast,)


@register_jvp("Max")
def max_jvp(graph: object, node: object, tangent: str) -> str:
    """Computes the JVP for the Max operation.

    Args:
        graph (object): The computation graph
        node (object): The Max operation node
        tangent (str): The identifier of the tangent tensor

    Returns:
    str: The identifier of the computed JVP node
    """
    return emit_ir_node(graph, "Max", [tangent], node.shape_metadata, node.attributes)


@register_vjp("Min")
def min_vjp(graph: object, node: object, cotangent: str) -> tuple:
    """Computes the Vector-Jacobian Product (VJP) for the Min operation.

    Args:
        graph (object): The computation graph
        node (object): The Min operation node
        cotangent (str): The identifier of the cotangent tensor

    Returns:
    tuple: A tuple containing the identifier of the broadcasted cotangent tensor
    """
    x = node.inputs[0]
    bcast = emit_ir_node(
        graph,
        "BroadcastTo",
        [cotangent],
        graph.nodes[x].shape_metadata,
        {"shape": graph.nodes[x].shape_metadata},
    )
    return (bcast,)


@register_jvp("Min")
def min_jvp(graph: object, node: object, tangent: str) -> str:
    """Computes the JVP for the Min operation.

    Args:
        graph (object): The computation graph
        node (object): The Min operation node
        tangent (str): The identifier of the tangent tensor

    Returns:
    str: The identifier of the computed JVP node
    """
    return emit_ir_node(graph, "Min", [tangent], node.shape_metadata, node.attributes)
