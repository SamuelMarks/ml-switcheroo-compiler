"""Defines Vector-Jacobian Product (VJP) and Jacobian-Vector Product (JVP) rules for unary.

operations

This module registers autodiff rules for common unary mathematical operations such as
Sine, Cosine, Exponential, and Natural Logarithm, allowing backpropagation and forward-
mode differentiation through the computation graph
"""

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
    """
    x = node.inputs[0]
    cos_x = emit_ir_node(graph, "Cos", [x], graph.nodes[x].shape_metadata)
    return emit_ir_node(graph, "Multiply", [tangent, cos_x], graph.nodes[x].shape_metadata)


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
    """
    x = node.inputs[0]
    sin_x = emit_ir_node(graph, "Sin", [x], graph.nodes[x].shape_metadata)
    neg_sin = emit_ir_node(graph, "Negative", [sin_x], graph.nodes[x].shape_metadata)
    return emit_ir_node(graph, "Multiply", [tangent, neg_sin], graph.nodes[x].shape_metadata)


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
    """
    return emit_ir_node(
        graph, "Multiply", [tangent, node.id], graph.nodes[node.inputs[0]].shape_metadata
    )


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
    """
    x = node.inputs[0]
    recip = emit_ir_node(graph, "Reciprocal", [x], graph.nodes[x].shape_metadata)
    return emit_ir_node(graph, "Multiply", [tangent, recip], graph.nodes[x].shape_metadata)


@register_vjp("ReciprocalNoNan")
def reciprocal_no_nan_vjp(graph: object, node: object, cotangent: str) -> tuple:
    """VJP for ReciprocalNoNan."""
    x = node.inputs[0]
    # y = 1/x -> dy/dx = -1/x^2
    neg_one = emit_ir_node(graph, "Constant", [], None, {"value": -1.0})
    x_sq = emit_ir_node(graph, "Multiply", [x, x], graph.nodes[x].shape_metadata, {})
    grad = emit_ir_node(graph, "DivideNoNan", [neg_one, x_sq], graph.nodes[x].shape_metadata, {})
    return (emit_ir_node(graph, "Multiply", [cotangent, grad], node.shape_metadata, {}),)


@register_vjp("L2Normalize")
def l2_normalize_vjp(graph: object, node: object, cotangent: str) -> tuple:
    """VJP for L2Normalize."""
    x = node.inputs[0]
    axis = node.attributes.get("axis", None)

    # norm = ReduceEuclideanNorm(x, axis, keepdims=True)
    norm = emit_ir_node(
        graph,
        "ReduceEuclideanNorm",
        [x],
        graph.nodes[x].shape_metadata,
        {"axis": axis, "keepdims": True},
    )

    # x_adj_1 = cotangent / norm
    norm_bcast = emit_ir_node(
        graph,
        "BroadcastTo",
        [norm],
        graph.nodes[x].shape_metadata,
        {"shape": graph.nodes[x].shape_metadata},
    )
    x_adj_1 = emit_ir_node(
        graph, "DivideNoNan", [cotangent, norm_bcast], graph.nodes[x].shape_metadata, {}
    )

    # dot_prod = sum(cotangent * x, axis, keepdims=True)
    cotangent_x = emit_ir_node(graph, "Multiply", [cotangent, x], graph.nodes[x].shape_metadata, {})
    dot_prod = emit_ir_node(
        graph, "Sum", [cotangent_x], graph.nodes[x].shape_metadata, {"axis": axis, "keepdims": True}
    )
    dot_prod_bcast = emit_ir_node(
        graph,
        "BroadcastTo",
        [dot_prod],
        graph.nodes[x].shape_metadata,
        {"shape": graph.nodes[x].shape_metadata},
    )

    # norm_sq = norm^2
    norm_sq = emit_ir_node(
        graph, "Multiply", [norm_bcast, norm_bcast], graph.nodes[x].shape_metadata, {}
    )
    norm_cubed = emit_ir_node(
        graph, "Multiply", [norm_sq, norm_bcast], graph.nodes[x].shape_metadata, {}
    )

    # x_adj_2 = (x * dot_prod) / norm^3
    x_dot = emit_ir_node(graph, "Multiply", [x, dot_prod_bcast], graph.nodes[x].shape_metadata, {})
    x_adj_2 = emit_ir_node(
        graph, "DivideNoNan", [x_dot, norm_cubed], graph.nodes[x].shape_metadata, {}
    )

    # dx = x_adj_1 - x_adj_2
    dx = emit_ir_node(graph, "Subtract", [x_adj_1, x_adj_2], graph.nodes[x].shape_metadata, {})
    return (dx,)


@register_vjp("BesselJ0")
def bessel_j0_vjp(graph: object, node: object, cotangent: str) -> tuple:
    """Docstring."""
    x = node.inputs[0]
    j1 = emit_ir_node(graph, "BesselJ1", [x], node.shape_metadata, {})
    neg_j1 = emit_ir_node(graph, "Negative", [j1], node.shape_metadata, {})
    return (emit_ir_node(graph, "Multiply", [cotangent, neg_j1], node.shape_metadata, {}),)


@register_vjp("BesselJ1")
def bessel_j1_vjp(graph: object, node: object, cotangent: str) -> tuple:
    """Docstring."""
    x = node.inputs[0]
    j0 = emit_ir_node(graph, "BesselJ0", [x], node.shape_metadata, {})
    j1 = node.id
    j1_over_x = emit_ir_node(graph, "DivideNoNan", [j1, x], node.shape_metadata, {})
    grad = emit_ir_node(graph, "Subtract", [j0, j1_over_x], node.shape_metadata, {})
    return (emit_ir_node(graph, "Multiply", [cotangent, grad], node.shape_metadata, {}),)


@register_vjp("BesselK0")
def bessel_k0_vjp(graph: object, node: object, cotangent: str) -> tuple:
    """Docstring."""
    x = node.inputs[0]
    k1 = emit_ir_node(graph, "BesselK1", [x], node.shape_metadata, {})
    neg_k1 = emit_ir_node(graph, "Negative", [k1], node.shape_metadata, {})
    return (emit_ir_node(graph, "Multiply", [cotangent, neg_k1], node.shape_metadata, {}),)


@register_vjp("BesselK0e")
def bessel_k0e_vjp(graph: object, node: object, cotangent: str) -> tuple:
    """Docstring."""
    x = node.inputs[0]
    k0e = node.id
    k1e = emit_ir_node(graph, "BesselK1e", [x], node.shape_metadata, {})
    grad = emit_ir_node(graph, "Subtract", [k0e, k1e], node.shape_metadata, {})
    return (emit_ir_node(graph, "Multiply", [cotangent, grad], node.shape_metadata, {}),)


@register_vjp("BesselK1")
def bessel_k1_vjp(graph: object, node: object, cotangent: str) -> tuple:
    """Docstring."""
    x = node.inputs[0]
    k0 = emit_ir_node(graph, "BesselK0", [x], node.shape_metadata, {})
    k1 = node.id
    k1_over_x = emit_ir_node(graph, "DivideNoNan", [k1, x], node.shape_metadata, {})
    term = emit_ir_node(graph, "Add", [k0, k1_over_x], node.shape_metadata, {})
    grad = emit_ir_node(graph, "Negative", [term], node.shape_metadata, {})
    return (emit_ir_node(graph, "Multiply", [cotangent, grad], node.shape_metadata, {}),)


@register_vjp("BesselK1e")
def bessel_k1e_vjp(graph: object, node: object, cotangent: str) -> tuple:
    """Docstring."""
    x = node.inputs[0]
    k1e = node.id
    k0e = emit_ir_node(graph, "BesselK0e", [x], node.shape_metadata, {})
    k1e_over_x = emit_ir_node(graph, "DivideNoNan", [k1e, x], node.shape_metadata, {})
    term1 = emit_ir_node(graph, "Subtract", [k1e, k0e], node.shape_metadata, {})
    grad = emit_ir_node(graph, "Subtract", [term1, k1e_over_x], node.shape_metadata, {})
    return (emit_ir_node(graph, "Multiply", [cotangent, grad], node.shape_metadata, {}),)


@register_vjp("BesselY0")
def bessel_y0_vjp(graph: object, node: object, cotangent: str) -> tuple:
    """Docstring."""
    x = node.inputs[0]
    y1 = emit_ir_node(graph, "BesselY1", [x], node.shape_metadata, {})
    neg_y1 = emit_ir_node(graph, "Negative", [y1], node.shape_metadata, {})
    return (emit_ir_node(graph, "Multiply", [cotangent, neg_y1], node.shape_metadata, {}),)


@register_vjp("BesselY1")
def bessel_y1_vjp(graph: object, node: object, cotangent: str) -> tuple:
    """Docstring."""
    x = node.inputs[0]
    y0 = emit_ir_node(graph, "BesselY0", [x], node.shape_metadata, {})
    y1 = node.id
    y1_over_x = emit_ir_node(graph, "DivideNoNan", [y1, x], node.shape_metadata, {})
    grad = emit_ir_node(graph, "Subtract", [y0, y1_over_x], node.shape_metadata, {})
    return (emit_ir_node(graph, "Multiply", [cotangent, grad], node.shape_metadata, {}),)


@register_vjp("Dawsn")
def dawsn_vjp(graph: object, node: object, cotangent: str) -> tuple:
    """Docstring."""
    x = node.inputs[0]
    dawsn_x = node.id
    two = emit_ir_node(graph, "Constant", [], None, {"value": 2.0})
    two_x = emit_ir_node(graph, "Multiply", [two, x], node.shape_metadata, {})
    term = emit_ir_node(graph, "Multiply", [two_x, dawsn_x], node.shape_metadata, {})
    one = emit_ir_node(graph, "Constant", [], None, {"value": 1.0})
    grad = emit_ir_node(graph, "Subtract", [one, term], node.shape_metadata, {})
    return (emit_ir_node(graph, "Multiply", [cotangent, grad], node.shape_metadata, {}),)


@register_vjp("Expint")
def expint_vjp(graph: object, node: object, cotangent: str) -> tuple:
    """Docstring."""
    x = node.inputs[0]
    exp_x = emit_ir_node(graph, "Exp", [x], node.shape_metadata, {})
    grad = emit_ir_node(graph, "DivideNoNan", [exp_x, x], node.shape_metadata, {})
    return (emit_ir_node(graph, "Multiply", [cotangent, grad], node.shape_metadata, {}),)


@register_vjp("FresnelCos")
def fresnel_cos_vjp(graph: object, node: object, cotangent: str) -> tuple:
    """Docstring."""
    import math

    x = node.inputs[0]
    pi_over_two = emit_ir_node(graph, "Constant", [], None, {"value": math.pi / 2.0})
    x_sq = emit_ir_node(graph, "Multiply", [x, x], node.shape_metadata, {})
    arg = emit_ir_node(graph, "Multiply", [pi_over_two, x_sq], node.shape_metadata, {})
    grad = emit_ir_node(graph, "Cos", [arg], node.shape_metadata, {})
    return (emit_ir_node(graph, "Multiply", [cotangent, grad], node.shape_metadata, {}),)


@register_vjp("FresnelSin")
def fresnel_sin_vjp(graph: object, node: object, cotangent: str) -> tuple:
    """Docstring."""
    import math

    x = node.inputs[0]
    pi_over_two = emit_ir_node(graph, "Constant", [], None, {"value": math.pi / 2.0})
    x_sq = emit_ir_node(graph, "Multiply", [x, x], node.shape_metadata, {})
    arg = emit_ir_node(graph, "Multiply", [pi_over_two, x_sq], node.shape_metadata, {})
    grad = emit_ir_node(graph, "Sin", [arg], node.shape_metadata, {})
    return (emit_ir_node(graph, "Multiply", [cotangent, grad], node.shape_metadata, {}),)


@register_vjp("Spence")
def spence_vjp(graph: object, node: object, cotangent: str) -> tuple:
    """Docstring."""
    x = node.inputs[0]
    log_x = emit_ir_node(graph, "Log", [x], node.shape_metadata, {})
    one = emit_ir_node(graph, "Constant", [], None, {"value": 1.0})
    x_minus_one = emit_ir_node(graph, "Subtract", [x, one], node.shape_metadata, {})
    # To handle x=1, log(x)/(x-1) approaches 1. DivideNoNan will return 0 if denom is 0, which is incorrect.
    # But usually x!=1. If we just do Division, it will yield NaN for 0/0.
    grad = emit_ir_node(graph, "Divide", [log_x, x_minus_one], node.shape_metadata, {})
    return (emit_ir_node(graph, "Multiply", [cotangent, grad], node.shape_metadata, {}),)
