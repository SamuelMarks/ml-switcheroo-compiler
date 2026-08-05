"""Unary rules for nn."""

from ml_switcheroo_compiler.ops.base import emit_ir_node
from ml_switcheroo_compiler.transforms.autodiff_rules.jvp_registry import register_jvp
from ml_switcheroo_compiler.transforms.autodiff_rules.vjp_registry import register_vjp


@register_vjp("Sigmoid")
def sigmoid_vjp(graph: object, node: object, cotangent: str) -> tuple:
    """Compute the Vector-Jacobian Product (VJP) for Sigmoid.

    Args:
        graph (object): The graph parameter.
        node (object): The node parameter.
        cotangent (str): The cotangent parameter.

    Returns:
        tuple: Result.
    """
    x = node.inputs[0]
    z = emit_ir_node(graph, "Sigmoid", [x], graph.nodes[x].shape_metadata)
    one = emit_ir_node(graph, "Constant", [], None, attributes={"value": 1.0})
    one_minus_z = emit_ir_node(graph, "Subtract", [one, z], graph.nodes[x].shape_metadata)
    z_times_one_minus_z = emit_ir_node(graph, "Multiply", [z, one_minus_z], graph.nodes[x].shape_metadata)
    dx = emit_ir_node(graph, "Multiply", [cotangent, z_times_one_minus_z], graph.nodes[x].shape_metadata)
    return (dx,)


@register_jvp("Sigmoid")
def sigmoid_jvp(graph: object, node: object, tangents: tuple) -> str:
    """Compute the Jacobian-Vector Product (JVP) for Sigmoid.

    Args:
        graph (object): The graph parameter.
        node (object): The node parameter.
        tangents (tuple): The tangents parameter.

    Returns:
        str: Result.
    """
    x = node.inputs[0]
    t_x = tangents[0]
    z = emit_ir_node(graph, "Sigmoid", [x], graph.nodes[x].shape_metadata)
    one = emit_ir_node(graph, "Constant", [], None, attributes={"value": 1.0})
    one_minus_z = emit_ir_node(graph, "Subtract", [one, z], graph.nodes[x].shape_metadata)
    z_times_one_minus_z = emit_ir_node(graph, "Multiply", [z, one_minus_z], graph.nodes[x].shape_metadata)
    dx = emit_ir_node(graph, "Multiply", [t_x, z_times_one_minus_z], graph.nodes[x].shape_metadata)
    return dx


@register_vjp("LogSigmoid")
def logsigmoid_vjp(graph: object, node: object, cotangent: str) -> tuple:
    """Compute the Vector-Jacobian Product (VJP) for LogSigmoid.

    Args:
        graph (object): The graph parameter.
        node (object): The node parameter.
        cotangent (str): The cotangent parameter.

    Returns:
        tuple: Result.
    """
    x = node.inputs[0]
    z = emit_ir_node(graph, "Sigmoid", [x], graph.nodes[x].shape_metadata)
    one = emit_ir_node(graph, "Constant", [], None, attributes={"value": 1.0})
    one_minus_z = emit_ir_node(graph, "Subtract", [one, z], graph.nodes[x].shape_metadata)
    dx = emit_ir_node(graph, "Multiply", [cotangent, one_minus_z], graph.nodes[x].shape_metadata)
    return (dx,)


@register_jvp("LogSigmoid")
def logsigmoid_jvp(graph: object, node: object, tangents: tuple) -> str:
    """Compute the Jacobian-Vector Product (JVP) for LogSigmoid.

    Args:
        graph (object): The graph parameter.
        node (object): The node parameter.
        tangents (tuple): The tangents parameter.

    Returns:
        str: Result.
    """
    x = node.inputs[0]
    t_x = tangents[0]
    z = emit_ir_node(graph, "Sigmoid", [x], graph.nodes[x].shape_metadata)
    one = emit_ir_node(graph, "Constant", [], None, attributes={"value": 1.0})
    one_minus_z = emit_ir_node(graph, "Subtract", [one, z], graph.nodes[x].shape_metadata)
    dx = emit_ir_node(graph, "Multiply", [t_x, one_minus_z], graph.nodes[x].shape_metadata)
    return dx


@register_vjp("Logit")
def logit_vjp(graph: object, node: object, cotangent: str) -> tuple:
    """Compute the Vector-Jacobian Product (VJP) for Logit.

    Args:
        graph (object): The graph parameter.
        node (object): The node parameter.
        cotangent (str): The cotangent parameter.

    Returns:
        tuple: Result.
    """
    x = node.inputs[0]
    one = emit_ir_node(graph, "Constant", [], None, attributes={"value": 1.0})
    one_minus_x = emit_ir_node(graph, "Subtract", [one, x], graph.nodes[x].shape_metadata)
    x_times_one_minus_x = emit_ir_node(graph, "Multiply", [x, one_minus_x], graph.nodes[x].shape_metadata)
    dx = emit_ir_node(graph, "Divide", [cotangent, x_times_one_minus_x], graph.nodes[x].shape_metadata)
    return (dx,)


@register_jvp("Logit")
def logit_jvp(graph: object, node: object, tangents: tuple) -> str:
    """Compute the Jacobian-Vector Product (JVP) for Logit.

    Args:
        graph (object): The graph parameter.
        node (object): The node parameter.
        tangents (tuple): The tangents parameter.

    Returns:
        str: Result.
    """
    x = node.inputs[0]
    t_x = tangents[0]
    one = emit_ir_node(graph, "Constant", [], None, attributes={"value": 1.0})
    one_minus_x = emit_ir_node(graph, "Subtract", [one, x], graph.nodes[x].shape_metadata)
    x_times_one_minus_x = emit_ir_node(graph, "Multiply", [x, one_minus_x], graph.nodes[x].shape_metadata)
    dx = emit_ir_node(graph, "Divide", [t_x, x_times_one_minus_x], graph.nodes[x].shape_metadata)
    return dx
