"""Defines Vector-Jacobian Product (VJP) and Jacobian-Vector Product (JVP) rules for unary.

operations

This module registers autodiff rules for common unary mathematical operations such as
Sine, Cosine, Exponential, and Natural Logarithm, allowing backpropagation and forward-
mode differentiation through the computation graph
"""

from ml_switcheroo_compiler.core.errors import UnimplementedMathError
from ml_switcheroo_compiler.ops.base import emit_ir_node
from ml_switcheroo_compiler.transforms.autodiff_rules.jvp_registry import register_jvp
from ml_switcheroo_compiler.transforms.autodiff_rules.vjp_registry import register_vjp


@register_vjp("Sin")
def sin_vjp(graph: object, node: object, cotangent: str) -> tuple:
    """Computes the Vector-Jacobian Product (VJP) for the Sine operation.

    Args:
        graph (object): The computation graph containing the nodes
        node (object): The IR node representing the Sine operation
        cotangent (str): The identifier of the incoming cotangent vector

    Returns:
    tuple: A single-element tuple containing the identifier of the computed VJP node
    (cotangent * cos(x))
    """
    x = node.inputs[0]
    cos_x = emit_ir_node(graph, "Cos", [x], graph.nodes[x].shape_metadata)
    return (emit_ir_node(graph, "Multiply", [cotangent, cos_x], graph.nodes[x].shape_metadata),)


@register_jvp("Sin")
def sin_jvp(graph: object, node: object, tangent: str) -> str:
    """Computes the Jacobian-Vector Product (JVP) for the Sine operation.

    Args:
        graph (object): The computation graph containing the nodes
        node (object): The IR node representing the Sine operation
        tangent (str): The identifier of the incoming tangent vector

    Returns:
    str: The identifier of the computed JVP node

    Raises:
    UnimplementedMathError: Always raised as JVP is not implemented for Sin
    """
    msg = "JVP not implemented for Sin"
    raise UnimplementedMathError(msg)


@register_vjp("Cos")
def cos_vjp(graph: object, node: object, cotangent: str) -> tuple:
    """Computes the Vector-Jacobian Product (VJP) for the Cosine operation.

    Args:
        graph (object): The computation graph containing the nodes
        node (object): The IR node representing the Cosine operation
        cotangent (str): The identifier of the incoming cotangent vector

    Returns:
    tuple: A single-element tuple containing the identifier of the computed VJP node
    (-cotangent * sin(x))
    """
    x = node.inputs[0]
    sin_x = emit_ir_node(graph, "Sin", [x], graph.nodes[x].shape_metadata)
    neg_sin = emit_ir_node(graph, "Negative", [sin_x], graph.nodes[x].shape_metadata)
    return (emit_ir_node(graph, "Multiply", [cotangent, neg_sin], graph.nodes[x].shape_metadata),)


@register_jvp("Cos")
def cos_jvp(graph: object, node: object, tangent: str) -> str:
    """Computes the Jacobian-Vector Product (JVP) for the Cosine operation.

    Args:
        graph (object): The computation graph containing the nodes
        node (object): The IR node representing the Cosine operation
        tangent (str): The identifier of the incoming tangent vector

    Returns:
    str: The identifier of the computed JVP node

    Raises:
    UnimplementedMathError: Always raised as JVP is not implemented for Cos
    """
    msg = "JVP not implemented for Cos"
    raise UnimplementedMathError(msg)


@register_vjp("Exp")
def exp_vjp(graph: object, node: object, cotangent: str) -> tuple:
    """Computes the Vector-Jacobian Product (VJP) for the Exponential operation.

    Args:
        graph (object): The computation graph containing the nodes
        node (object): The IR node representing the Exponential operation
        cotangent (str): The identifier of the incoming cotangent vector

    Returns:
    tuple: A single-element tuple containing the identifier of the computed VJP node
    (cotangent * exp(x))
    """
    return (
        emit_ir_node(
            graph,
            "Multiply",
            [cotangent, node.id],
            graph.nodes[node.inputs[0]].shape_metadata,
        ),
    )


@register_jvp("Exp")
def exp_jvp(graph: object, node: object, tangent: str) -> str:
    """Computes the Jacobian-Vector Product (JVP) for the Exponential operation.

    Args:
        graph (object): The computation graph containing the nodes
        node (object): The IR node representing the Exponential operation
        tangent (str): The identifier of the incoming tangent vector

    Returns:
    str: The identifier of the computed JVP node

    Raises:
    UnimplementedMathError: Always raised as JVP is not implemented for Exp
    """
    msg = "JVP not implemented for Exp"
    raise UnimplementedMathError(msg)


@register_vjp("Log")
def log_vjp(graph: object, node: object, cotangent: str) -> tuple:
    """Computes the Vector-Jacobian Product (VJP) for the Natural Logarithm operation.

    Args:
        graph (object): The computation graph containing the nodes
        node (object): The IR node representing the Logarithm operation
        cotangent (str): The identifier of the incoming cotangent vector

    Returns:
    tuple: A single-element tuple containing the identifier of the computed VJP node
    (cotangent / x)
    """
    x = node.inputs[0]
    recip = emit_ir_node(graph, "Reciprocal", [x], graph.nodes[x].shape_metadata)
    return (emit_ir_node(graph, "Multiply", [cotangent, recip], graph.nodes[x].shape_metadata),)


@register_jvp("Log")
def log_jvp(graph: object, node: object, tangent: str) -> str:
    """Computes the Jacobian-Vector Product (JVP) for the Natural Logarithm operation.

    Args:
        graph (object): The computation graph containing the nodes
        node (object): The IR node representing the Logarithm operation
        tangent (str): The identifier of the incoming tangent vector

    Returns:
    str: The identifier of the computed JVP node

    Raises:
    UnimplementedMathError: Always raised as JVP is not implemented for Log
    """
    msg = "JVP not implemented for Log"
    raise UnimplementedMathError(msg)
