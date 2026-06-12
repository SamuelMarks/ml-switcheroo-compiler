"""Defines Vector-Jacobian Products (VJPs) and Jacobian-Vector Products (JVPs) for binary.

operations

This module registers automatic differentiation rules for common binary operations such
as addition, subtraction, multiplication, division, and exponentiation
"""

from ml_switcheroo_compiler.transforms.autodiff_rules.vjp_registry import register_vjp
from ml_switcheroo_compiler.transforms.autodiff_rules.jvp_registry import register_jvp
from ml_switcheroo_compiler.core.errors import UnimplementedMathError
from ml_switcheroo_compiler.ops.base import emit_ir_node


@register_vjp("Add")
def add_vjp(graph: object, node: object, cotangent: str) -> tuple:
    """Computes the Vector-Jacobian Product (VJP) for the addition operation.

    For z = x + y, the gradients with respect to x and y are both equal to the
    incoming cotangent (gradient of the loss with respect to z)

    Args:
    graph (object): The computation graph containing the nodes
    node (object): The addition node in the computation graph
    cotangent (str): The name or identifier of the incoming cotangent

    Returns:
    tuple: A tuple containing the cotangents for the two inputs (cotangent_x,
    cotangent_y)
    """
    return (cotangent, cotangent)


@register_jvp("Add")
def add_jvp(tangent_x: object, tangent_y: object, x: object, y: object, **kwargs: object) -> str:
    """Computes the Jacobian-Vector Product (JVP) for the addition operation.

    For z = x + y, the tangent of z is tangent_x + tangent_y

    Args:
    tangent_x (object): The tangent of the first input x
    tangent_y (object): The tangent of the second input y
    x (object): The primal value of the first input
    y (object): The primal value of the second input
    **kwargs (object): Additional keyword arguments

    Returns:
    str: A string representation of the tangent expression
    """
    return f"({tangent_x} + {tangent_y})"


@register_vjp("Subtract")
def subtract_vjp(graph: object, node: object, cotangent: str) -> tuple:
    """Computes the Vector-Jacobian Product (VJP) for the subtraction operation.

    For z = x - y, the gradient with respect to x is the incoming cotangent,
    and the gradient with respect to y is the negative of the incoming cotangent

    Args:
    graph (object): The computation graph containing the nodes
    node (object): The subtraction node in the computation graph
    cotangent (str): The name or identifier of the incoming cotangent

    Returns:
    tuple: A tuple containing the cotangents for the two inputs (cotangent_x,
    cotangent_y)
    """
    neg_adj = emit_ir_node(graph, "Negative", [cotangent], graph.nodes[cotangent].shape_metadata)
    return (cotangent, neg_adj)


@register_jvp("Subtract")
def subtract_jvp(
    tangent_x: object, tangent_y: object, x: object, y: object, **kwargs: object
) -> str:
    """Computes the Jacobian-Vector Product (JVP) for the subtraction operation.

    For z = x - y, the tangent of z is tangent_x - tangent_y

    Args:
    tangent_x (object): The tangent of the first input x
    tangent_y (object): The tangent of the second input y
    x (object): The primal value of the first input
    y (object): The primal value of the second input
    **kwargs (object): Additional keyword arguments

    Returns:
    str: A string representation of the tangent expression
    """
    return f"({tangent_x} - {tangent_y})"


@register_vjp("Multiply")
def multiply_vjp(graph: object, node: object, cotangent: str) -> tuple:
    """Computes the Vector-Jacobian Product (VJP) for the multiplication operation.

    For z = x * y, the gradient with respect to x is cotangent * y,
    and the gradient with respect to y is cotangent * x

    Args:
    graph (object): The computation graph containing the nodes
    node (object): The multiplication node in the computation graph
    cotangent (str): The name or identifier of the incoming cotangent

    Returns:
    tuple: A tuple containing the cotangents for the two inputs (cotangent_x,
    cotangent_y)
    """
    x, y = node.inputs
    adj_x = emit_ir_node(graph, "Multiply", [cotangent, y], graph.nodes[x].shape_metadata)
    adj_y = emit_ir_node(graph, "Multiply", [cotangent, x], graph.nodes[y].shape_metadata)
    return (adj_x, adj_y)


@register_jvp("Multiply")
def multiply_jvp(
    tangent_x: object, tangent_y: object, x: object, y: object, **kwargs: object
) -> str:
    """Computes the Jacobian-Vector Product (JVP) for the multiplication operation.

    For z = x * y, the tangent of z is tangent_x * y + x * tangent_y

    Args:
    tangent_x (object): The tangent of the first input x
    tangent_y (object): The tangent of the second input y
    x (object): The primal value of the first input
    y (object): The primal value of the second input
    **kwargs (object): Additional keyword arguments

    Returns:
    str: A string representation of the tangent expression
    """
    return f"({tangent_x} * {y} + {x} * {tangent_y})"


@register_vjp("Divide")
def divide_vjp(graph: object, node: object, cotangent: str) -> tuple:
    """Computes the Vector-Jacobian Product (VJP) for the division operation.

    Currently not implemented

    Args:
    graph (object): The computation graph containing the nodes
    node (object): The division node in the computation graph
    cotangent (str): The name or identifier of the incoming cotangent

    Returns:
    tuple: A tuple containing the cotangents for the two inputs

    Raises:
    UnimplementedMathError: Always raised as VJP is not implemented for Divide
    """
    raise UnimplementedMathError("VJP not implemented for Divide")


@register_jvp("Divide")
def divide_jvp(tangent_x: object, tangent_y: object, x: object, y: object, **kwargs: object) -> str:
    """Computes the Jacobian-Vector Product (JVP) for the division operation.

    Currently not implemented

    Args:
    tangent_x (object): The tangent of the first input x
    tangent_y (object): The tangent of the second input y
    x (object): The primal value of the first input
    y (object): The primal value of the second input
    **kwargs (object): Additional keyword arguments

    Returns:
    str: A string representation of the tangent expression

    Raises:
    UnimplementedMathError: Always raised as JVP is not implemented for Divide
    """
    raise UnimplementedMathError("JVP not implemented for Divide")


@register_vjp("Power")
def power_vjp(graph: object, node: object, cotangent: str) -> tuple:
    """Computes the Vector-Jacobian Product (VJP) for the power operation.

    Currently not implemented

    Args:
    graph (object): The computation graph containing the nodes
    node (object): The power node in the computation graph
    cotangent (str): The name or identifier of the incoming cotangent

    Returns:
    tuple: A tuple containing the cotangents for the two inputs

    Raises:
    UnimplementedMathError: Always raised as VJP is not implemented for Power
    """
    raise UnimplementedMathError("VJP not implemented for Power")


@register_jvp("Power")
def power_jvp(tangent_x: object, tangent_y: object, x: object, y: object, **kwargs: object) -> str:
    """Computes the Jacobian-Vector Product (JVP) for the power operation.

    Currently not implemented

    Args:
    tangent_x (object): The tangent of the first input x
    tangent_y (object): The tangent of the second input y
    x (object): The primal value of the first input
    y (object): The primal value of the second input
    **kwargs (object): Additional keyword arguments

    Returns:
    str: A string representation of the tangent expression

    Raises:
    UnimplementedMathError: Always raised as JVP is not implemented for Power
    """
    raise UnimplementedMathError("JVP not implemented for Power")
