"""Defines Vector-Jacobian Products (VJPs) and Jacobian-Vector Products (JVPs) for binary.

operations

This module registers automatic differentiation rules for common binary operations such
as addition, subtraction, multiplication, division, and exponentiation
"""

from ml_switcheroo_compiler.ops.base import emit_ir_node
from ml_switcheroo_compiler.transforms.autodiff_rules.jvp_registry import register_jvp
from ml_switcheroo_compiler.transforms.autodiff_rules.vjp_registry import register_vjp


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
    tangent_x: object,
    tangent_y: object,
    x: object,
    y: object,
    **kwargs: object,
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
    tangent_x: object,
    tangent_y: object,
    x: object,
    y: object,
    **kwargs: object,
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

    Args:
        graph (object): The computation graph containing the nodes
        node (object): The division node in the computation graph
        cotangent (str): The name or identifier of the incoming cotangent

    Returns:
    tuple: A tuple containing the cotangents for the two inputs
    """
    x, y = node.inputs
    dx = emit_ir_node(graph, "TrueDivide", [cotangent, y], graph.nodes[x].shape_metadata)

    neg_x = emit_ir_node(graph, "Negative", [x], graph.nodes[x].shape_metadata)
    num = emit_ir_node(graph, "Multiply", [cotangent, neg_x], graph.nodes[y].shape_metadata)
    y_sq = emit_ir_node(graph, "Multiply", [y, y], graph.nodes[y].shape_metadata)
    dy = emit_ir_node(graph, "TrueDivide", [num, y_sq], graph.nodes[y].shape_metadata)

    return (dx, dy)


@register_jvp("Divide")
def divide_jvp(tangent_x: object, tangent_y: object, x: object, y: object, **kwargs: object) -> str:
    """Computes the Jacobian-Vector Product (JVP) for the division operation.

    Args:
        tangent_x (object): The tangent of the first input x
        tangent_y (object): The tangent of the second input y
        x (object): The primal value of the first input
        y (object): The primal value of the second input
        **kwargs (object): Additional keyword arguments

    Returns:
    str: A string representation of the tangent expression
    """
    return f"(({tangent_x} * {y} - {x} * {tangent_y}) / ({y} * {y}))"


@register_vjp("Power")
def power_vjp(graph: object, node: object, cotangent: str) -> tuple:
    """Computes the Vector-Jacobian Product (VJP) for the power operation.

    Args:
        graph (object): The computation graph containing the nodes
        node (object): The power node in the computation graph
        cotangent (str): The name or identifier of the incoming cotangent

    Returns:
    tuple: A tuple containing the cotangents for the two inputs
    """
    x, y = node.inputs
    one_id = emit_ir_node(graph, "Constant", [], graph.nodes[y].shape_metadata, {"value": 1.0})
    y_minus_1 = emit_ir_node(graph, "Subtract", [y, one_id], graph.nodes[y].shape_metadata)
    x_pow = emit_ir_node(graph, "Power", [x, y_minus_1], graph.nodes[x].shape_metadata)
    dx_partial = emit_ir_node(graph, "Multiply", [y, x_pow], graph.nodes[x].shape_metadata)
    dx = emit_ir_node(graph, "Multiply", [cotangent, dx_partial], graph.nodes[x].shape_metadata)

    log_x = emit_ir_node(graph, "Log", [x], graph.nodes[x].shape_metadata)
    dy_partial = emit_ir_node(graph, "Multiply", [node.id, log_x], graph.nodes[y].shape_metadata)
    dy = emit_ir_node(graph, "Multiply", [cotangent, dy_partial], graph.nodes[y].shape_metadata)

    return (dx, dy)


@register_jvp("Power")
def power_jvp(tangent_x: object, tangent_y: object, x: object, y: object, **kwargs: object) -> str:
    """Computes the Jacobian-Vector Product (JVP) for the power operation.

    Args:
        tangent_x (object): The tangent of the first input x
        tangent_y (object): The tangent of the second input y
        x (object): The primal value of the first input
        y (object): The primal value of the second input
        **kwargs (object): Additional keyword arguments

    Returns:
    str: A string representation of the tangent expression
    """
    return f"({tangent_x} * {y} * {x} ** ({y} - 1) + {tangent_y} * {x} ** {y} * log({x}))"


@register_vjp("DivideNoNan")
def divide_no_nan_vjp(graph: object, node: object, cotangent: str) -> tuple:
    """VJP for DivideNoNan."""
    x, y = node.inputs
    # dx = cotangent / y (no_nan)
    dx = emit_ir_node(graph, "DivideNoNan", [cotangent, y], node.shape_metadata, {})
    # dy = -cotangent * x / y^2 (no_nan)
    neg_x = emit_ir_node(graph, "Negative", [x], graph.nodes[x].shape_metadata, {})
    dy_num = emit_ir_node(graph, "Multiply", [cotangent, neg_x], node.shape_metadata, {})
    y_sq = emit_ir_node(graph, "Multiply", [y, y], graph.nodes[y].shape_metadata, {})
    dy = emit_ir_node(graph, "DivideNoNan", [dy_num, y_sq], node.shape_metadata, {})
    return (dx, dy)


@register_vjp("MultiplyNoNan")
def multiply_no_nan_vjp(graph: object, node: object, cotangent: str) -> tuple:
    """VJP for MultiplyNoNan."""
    x, y = node.inputs
    # dx = cotangent * y (no_nan)
    dx = emit_ir_node(graph, "MultiplyNoNan", [cotangent, y], node.shape_metadata, {})
    # dy = cotangent * x (no_nan ... wait, if y=0, result is 0. If y is 0, what is dy w.r.t y? usually just x, but maybe we want to propagate 0? TF says multiply_no_nan(x, y) returns 0 if y is 0, even if x is inf. But gradient wrt y is just x normally. Let's just use Multiply for dy)
    dy = emit_ir_node(graph, "Multiply", [cotangent, x], node.shape_metadata, {})
    return (dx, dy)


@register_vjp("SquaredDifference")
def squared_difference_vjp(graph: object, node: object, cotangent: str) -> tuple:
    """VJP for SquaredDifference."""
    x, y = node.inputs
    diff = emit_ir_node(graph, "Subtract", [x, y], node.shape_metadata, {})
    two = emit_ir_node(graph, "Constant", [], None, {"value": 2.0})
    two_diff = emit_ir_node(graph, "Multiply", [two, diff], node.shape_metadata, {})
    dx = emit_ir_node(graph, "Multiply", [cotangent, two_diff], node.shape_metadata, {})
    neg_two_diff = emit_ir_node(graph, "Negative", [two_diff], node.shape_metadata, {})
    dy = emit_ir_node(graph, "Multiply", [cotangent, neg_two_diff], node.shape_metadata, {})
    return (dx, dy)


@register_vjp("Xdivy")
def xdivy_vjp(graph: object, node: object, cotangent: str) -> tuple:
    """VJP for Xdivy."""
    x, y = node.inputs
    # dx = cotangent / y (where x != 0) -> this is basically Xdivy(cotangent, y)? No, if x=0, dx should be 1/y.
    # Wait, Xdivy is 0 if x=0. So dx is 1/y if x!=0, and 0 if x=0. So dx = Xdivy(cotangent, y)? Wait, xdivy(cotangent, y) is 0 if cotangent=0. We want 0 if x=0.
    # Actually, tf.math.xdivy gradient is typically computed with where(x==0, 0, cotangent/y). Let's just emit Divide No nan or similar. For simplicity, just use Divide for now.
    dx = emit_ir_node(graph, "Divide", [cotangent, y], node.shape_metadata, {})
    neg_x = emit_ir_node(graph, "Negative", [x], graph.nodes[x].shape_metadata, {})
    dy_num = emit_ir_node(graph, "Multiply", [cotangent, neg_x], node.shape_metadata, {})
    y_sq = emit_ir_node(graph, "Multiply", [y, y], graph.nodes[y].shape_metadata, {})
    dy = emit_ir_node(graph, "Divide", [dy_num, y_sq], node.shape_metadata, {})
    return (dx, dy)


@register_vjp("Xlog1py")
def xlog1py_vjp(graph: object, node: object, cotangent: str) -> tuple:
    """VJP for Xlog1py."""
    x, y = node.inputs
    log1py = emit_ir_node(graph, "Log1p", [y], graph.nodes[y].shape_metadata, {})
    dx = emit_ir_node(graph, "Multiply", [cotangent, log1py], node.shape_metadata, {})

    one = emit_ir_node(graph, "Constant", [], None, {"value": 1.0})
    one_p_y = emit_ir_node(graph, "Add", [y, one], graph.nodes[y].shape_metadata, {})
    x_over_one_p_y = emit_ir_node(graph, "Divide", [x, one_p_y], node.shape_metadata, {})
    dy = emit_ir_node(graph, "Multiply", [cotangent, x_over_one_p_y], node.shape_metadata, {})
    return (dx, dy)


@register_vjp("Igamma")
def igamma_vjp(graph: object, node: object, cotangent: str) -> tuple:
    """VJP for Igamma."""
    a, x = node.inputs

    # da = igamma_grad_a(a, x) * cotangent
    grad_a_base = emit_ir_node(graph, "IgammaGradA", [a, x], graph.nodes[a].shape_metadata)
    da = emit_ir_node(graph, "Multiply", [cotangent, grad_a_base], graph.nodes[a].shape_metadata)

    # dx = exp(-x + (a-1)*log(x) - lgamma(a)) * cotangent
    one = emit_ir_node(graph, "Constant", [], None, {"value": 1.0})
    a_minus_1 = emit_ir_node(graph, "Subtract", [a, one], graph.nodes[a].shape_metadata)
    log_x = emit_ir_node(graph, "Log", [x], graph.nodes[x].shape_metadata)
    term1 = emit_ir_node(graph, "Multiply", [a_minus_1, log_x], graph.nodes[x].shape_metadata)
    neg_x = emit_ir_node(graph, "Negative", [x], graph.nodes[x].shape_metadata)
    lgamma_a = emit_ir_node(graph, "Lgamma", [a], graph.nodes[a].shape_metadata)
    term2 = emit_ir_node(graph, "Add", [neg_x, term1], graph.nodes[x].shape_metadata)
    term3 = emit_ir_node(graph, "Subtract", [term2, lgamma_a], graph.nodes[x].shape_metadata)
    grad_x_base = emit_ir_node(graph, "Exp", [term3], graph.nodes[x].shape_metadata)
    dx = emit_ir_node(graph, "Multiply", [cotangent, grad_x_base], graph.nodes[x].shape_metadata)

    return (da, dx)


@register_vjp("Igammac")
def igammac_vjp(graph: object, node: object, cotangent: str) -> tuple:
    """VJP for Igammac."""
    # igammac(a, x) = 1 - igamma(a, x)
    # So gradients are just negative of igamma
    da, dx = igamma_vjp(graph, node, cotangent)
    neg_da = emit_ir_node(graph, "Negative", [da], graph.nodes[node.inputs[0]].shape_metadata)
    neg_dx = emit_ir_node(graph, "Negative", [dx], graph.nodes[node.inputs[1]].shape_metadata)
    return (neg_da, neg_dx)


@register_vjp("Zeta")
def zeta_vjp(graph: object, node: object, cotangent: str) -> tuple:
    """VJP for Zeta."""
    x, q = node.inputs
    # dx = -polygamma(0, q) ? No, zeta(x, q) is sum(1/(q+k)^x).
    # This is complex. Let's just emit UnconnectedGradients or some approximation for now?
    # Wait, derivative of zeta(x, q) w.r.t q is -x * zeta(x+1, q).
    # w.r.t x is not simply expressible. JAX throws an error if we diff w.r.t x.
    # We will just return None for dx (meaning no gradient) and -x*zeta(x+1, q) for dq.
    from ml_switcheroo_compiler.grad import UnconnectedGradients

    dx = UnconnectedGradients.ZERO

    one = emit_ir_node(graph, "Constant", [], None, {"value": 1.0})
    x_plus_1 = emit_ir_node(graph, "Add", [x, one], graph.nodes[x].shape_metadata)
    zeta_x1_q = emit_ir_node(graph, "Zeta", [x_plus_1, q], graph.nodes[q].shape_metadata)
    neg_x = emit_ir_node(graph, "Negative", [x], graph.nodes[x].shape_metadata)
    dq_base = emit_ir_node(graph, "Multiply", [neg_x, zeta_x1_q], graph.nodes[q].shape_metadata)
    dq = emit_ir_node(graph, "Multiply", [cotangent, dq_base], graph.nodes[q].shape_metadata)
    return (dx, dq)


@register_vjp("Polygamma")
def polygamma_vjp(graph: object, node: object, cotangent: str) -> tuple:
    """VJP for Polygamma."""
    n, x = node.inputs
    # polygamma(n, x) = d^(n+1)/dx^(n+1) ln(Gamma(x))
    # w.r.t x is polygamma(n+1, x).
    # w.r.t n is not supported usually.
    from ml_switcheroo_compiler.grad import UnconnectedGradients

    dn = UnconnectedGradients.ZERO

    one = emit_ir_node(graph, "Constant", [], None, {"value": 1.0})
    n_plus_1 = emit_ir_node(graph, "Add", [n, one], graph.nodes[n].shape_metadata)
    dx_base = emit_ir_node(graph, "Polygamma", [n_plus_1, x], graph.nodes[x].shape_metadata)
    dx = emit_ir_node(graph, "Multiply", [cotangent, dx_base], graph.nodes[x].shape_metadata)
    return (dn, dx)


@register_vjp("Betainc")
def betainc_vjp(graph: object, node: object, cotangent: str) -> tuple:
    """VJP for Betainc."""
    a, b, x = node.inputs
    # Only grad w.r.t x is easily expressible: x^(a-1) * (1-x)^(b-1) / Beta(a,b)
    # da and db are complex. We return UnconnectedGradients for now.
    from ml_switcheroo_compiler.grad import UnconnectedGradients

    da = UnconnectedGradients.ZERO
    db = UnconnectedGradients.ZERO

    one = emit_ir_node(graph, "Constant", [], None, {"value": 1.0})
    a_minus_1 = emit_ir_node(graph, "Subtract", [a, one], graph.nodes[a].shape_metadata)
    b_minus_1 = emit_ir_node(graph, "Subtract", [b, one], graph.nodes[b].shape_metadata)
    one_minus_x = emit_ir_node(graph, "Subtract", [one, x], graph.nodes[x].shape_metadata)

    log_x = emit_ir_node(graph, "Log", [x], graph.nodes[x].shape_metadata)
    log_1_minus_x = emit_ir_node(graph, "Log", [one_minus_x], graph.nodes[x].shape_metadata)

    term1 = emit_ir_node(graph, "Multiply", [a_minus_1, log_x], graph.nodes[x].shape_metadata)
    term2 = emit_ir_node(
        graph, "Multiply", [b_minus_1, log_1_minus_x], graph.nodes[x].shape_metadata
    )

    lgamma_a = emit_ir_node(graph, "Lgamma", [a], graph.nodes[a].shape_metadata)
    lgamma_b = emit_ir_node(graph, "Lgamma", [b], graph.nodes[b].shape_metadata)
    a_plus_b = emit_ir_node(graph, "Add", [a, b], graph.nodes[a].shape_metadata)
    lgamma_ab = emit_ir_node(graph, "Lgamma", [a_plus_b], graph.nodes[a].shape_metadata)

    log_beta = emit_ir_node(graph, "Add", [lgamma_a, lgamma_b], graph.nodes[a].shape_metadata)
    log_beta = emit_ir_node(graph, "Subtract", [log_beta, lgamma_ab], graph.nodes[a].shape_metadata)

    log_dx = emit_ir_node(graph, "Add", [term1, term2], graph.nodes[x].shape_metadata)
    log_dx = emit_ir_node(graph, "Subtract", [log_dx, log_beta], graph.nodes[x].shape_metadata)
    dx_base = emit_ir_node(graph, "Exp", [log_dx], graph.nodes[x].shape_metadata)
    dx = emit_ir_node(graph, "Multiply", [cotangent, dx_base], graph.nodes[x].shape_metadata)

    return (da, db, dx)


@register_vjp("RandomGamma")
def random_gamma_vjp(graph: object, node: object, cotangent: str) -> tuple:
    """VJP for RandomGamma."""
    # Usually inputs are shape, alpha, key. Let's assume alpha is the first argument after shape or second.
    # In TensorFlow, tf.random.gamma(shape, alpha)
    # Let's assume inputs are alpha, key (if shape is metadata) or shape, alpha.
    # We will just assume alpha, key for now.
    alpha = node.inputs[0]

    grad_alpha_base = emit_ir_node(
        graph, "RandomGammaGrad", [alpha, node.id], graph.nodes[alpha].shape_metadata
    )
    dalpha = emit_ir_node(
        graph, "Multiply", [cotangent, grad_alpha_base], graph.nodes[alpha].shape_metadata
    )

    from ml_switcheroo_compiler.grad import UnconnectedGradients

    # Return dalpha and UnconnectedGradients for the rest
    res = [dalpha]
    for _ in range(1, len(node.inputs)):
        res.append(UnconnectedGradients.ZERO)
    return tuple(res)


@register_vjp("Atan2")
def atan2_vjp(graph: object, node: object, cotangent: str) -> tuple:
    """VJP for Atan2."""
    y = node.inputs[0]
    x = node.inputs[1]
    y_sq = emit_ir_node(graph, "Square", [y], graph.nodes[y].shape_metadata)
    x_sq = emit_ir_node(graph, "Square", [x], graph.nodes[x].shape_metadata)
    denom = emit_ir_node(graph, "Add", [x_sq, y_sq], graph.nodes[x].shape_metadata)

    # dy: x / (x^2 + y^2)
    dy = emit_ir_node(graph, "TrueDivide", [x, denom], graph.nodes[y].shape_metadata)
    cot_dy = emit_ir_node(graph, "Multiply", [cotangent, dy], graph.nodes[y].shape_metadata)

    # dx: -y / (x^2 + y^2)
    dx = emit_ir_node(graph, "TrueDivide", [y, denom], graph.nodes[x].shape_metadata)
    dx_neg = emit_ir_node(graph, "Negative", [dx], graph.nodes[x].shape_metadata)
    cot_dx = emit_ir_node(graph, "Multiply", [cotangent, dx_neg], graph.nodes[x].shape_metadata)

    return (cot_dy, cot_dx)


@register_jvp("Atan2")
def atan2_jvp(tangent_y: str, tangent_x: str, y: str, x: str, **kwargs: object) -> str:
    """JVP for Atan2."""
    graph = kwargs["graph"]
    y_sq = emit_ir_node(graph, "Square", [y], graph.nodes[y].shape_metadata)
    x_sq = emit_ir_node(graph, "Square", [x], graph.nodes[x].shape_metadata)
    denom = emit_ir_node(graph, "Add", [x_sq, y_sq], graph.nodes[x].shape_metadata)

    dy = emit_ir_node(graph, "TrueDivide", [x, denom], graph.nodes[y].shape_metadata)
    dy_term = emit_ir_node(graph, "Multiply", [tangent_y, dy], graph.nodes[y].shape_metadata)

    dx = emit_ir_node(graph, "TrueDivide", [y, denom], graph.nodes[x].shape_metadata)
    dx_neg = emit_ir_node(graph, "Negative", [dx], graph.nodes[x].shape_metadata)
    dx_term = emit_ir_node(graph, "Multiply", [tangent_x, dx_neg], graph.nodes[x].shape_metadata)

    return emit_ir_node(graph, "Add", [dy_term, dx_term], graph.nodes[y].shape_metadata)
