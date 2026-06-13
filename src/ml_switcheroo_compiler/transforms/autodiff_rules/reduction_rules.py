"""Defines Vector-Jacobian Product (VJP) and Jacobian-Vector Product (JVP) rules for.

reduction operations
"""

from ml_switcheroo_compiler.core.errors import UnimplementedMathError
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
    """Computes the Jacobian-Vector Product (JVP) for the Sum operation.

    Args:
    graph (object): The computation graph
    node (object): The Sum operation node
    tangent (str): The identifier of the tangent tensor

    Returns:
    str: The identifier of the computed tangent tensor

    Raises:
    UnimplementedMathError: This operation is currently not implemented
    """
    msg = "JVP not implemented for Sum"
    raise UnimplementedMathError(msg)


@register_vjp("Mean")
def mean_vjp(graph: object, node: object, cotangent: str) -> tuple:
    """Computes the Vector-Jacobian Product (VJP) for the Mean operation.

    Args:
    graph (object): The computation graph
    node (object): The Mean operation node
    cotangent (str): The identifier of the cotangent tensor

    Returns:
    tuple: A tuple containing the identifier of the input cotangent tensor

    Raises:
    UnimplementedMathError: This operation is currently not implemented
    """
    msg = "VJP not implemented for Mean"
    raise UnimplementedMathError(msg)


@register_jvp("Mean")
def mean_jvp(graph: object, node: object, tangent: str) -> str:
    """Computes the Jacobian-Vector Product (JVP) for the Mean operation.

    Args:
    graph (object): The computation graph
    node (object): The Mean operation node
    tangent (str): The identifier of the tangent tensor

    Returns:
    str: The identifier of the computed tangent tensor

    Raises:
    UnimplementedMathError: This operation is currently not implemented
    """
    msg = "JVP not implemented for Mean"
    raise UnimplementedMathError(msg)


@register_vjp("Max")
def max_vjp(graph: object, node: object, cotangent: str) -> tuple:
    """Computes the Vector-Jacobian Product (VJP) for the Max operation.

    Args:
    graph (object): The computation graph
    node (object): The Max operation node
    cotangent (str): The identifier of the cotangent tensor

    Returns:
    tuple: A tuple containing the identifier of the input cotangent tensor

    Raises:
    UnimplementedMathError: This operation is currently not implemented
    """
    msg = "VJP not implemented for Max"
    raise UnimplementedMathError(msg)


@register_jvp("Max")
def max_jvp(graph: object, node: object, tangent: str) -> str:
    """Computes the Jacobian-Vector Product (JVP) for the Max operation.

    Args:
    graph (object): The computation graph
    node (object): The Max operation node
    tangent (str): The identifier of the tangent tensor

    Returns:
    str: The identifier of the computed tangent tensor

    Raises:
    UnimplementedMathError: This operation is currently not implemented
    """
    msg = "JVP not implemented for Max"
    raise UnimplementedMathError(msg)


@register_vjp("Min")
def min_vjp(graph: object, node: object, cotangent: str) -> tuple:
    """Computes the Vector-Jacobian Product (VJP) for the Min operation.

    Args:
    graph (object): The computation graph
    node (object): The Min operation node
    cotangent (str): The identifier of the cotangent tensor

    Returns:
    tuple: A tuple containing the identifier of the input cotangent tensor

    Raises:
    UnimplementedMathError: This operation is currently not implemented
    """
    msg = "VJP not implemented for Min"
    raise UnimplementedMathError(msg)


@register_jvp("Min")
def min_jvp(graph: object, node: object, tangent: str) -> str:
    """Computes the Jacobian-Vector Product (JVP) for the Min operation.

    Args:
    graph (object): The computation graph
    node (object): The Min operation node
    tangent (str): The identifier of the tangent tensor

    Returns:
    str: The identifier of the computed tangent tensor

    Raises:
    UnimplementedMathError: This operation is currently not implemented
    """
    msg = "JVP not implemented for Min"
    raise UnimplementedMathError(msg)
