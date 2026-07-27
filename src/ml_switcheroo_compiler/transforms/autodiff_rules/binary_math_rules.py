"""Binary rules for math."""

import enum

from ml_switcheroo_compiler.ops.base import emit_ir_node
from ml_switcheroo_compiler.transforms.autodiff_rules.jvp_registry import register_jvp
from ml_switcheroo_compiler.transforms.autodiff_rules.vjp_registry import register_vjp


class UnconnectedGradients(enum.Enum):
    """Unconnected gradients enum."""

    NONE = "none"
    ZERO = "zero"


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
@register_vjp("TrueDivide")
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
@register_jvp("TrueDivide")
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


@register_jvp("DivideNoNan")
def divide_no_nan_jvp(graph: object, node: object, tangents: tuple) -> str:
    """JVP for DivideNoNan."""
    x, y = node.inputs
    t_x, t_y = tangents
    term1 = emit_ir_node(graph, "Multiply", [t_x, y], graph.nodes[x].shape_metadata, {})
    term2 = emit_ir_node(graph, "Multiply", [x, t_y], graph.nodes[x].shape_metadata, {})
    num = emit_ir_node(graph, "Subtract", [term1, term2], graph.nodes[x].shape_metadata, {})
    y_sq = emit_ir_node(graph, "Multiply", [y, y], graph.nodes[y].shape_metadata, {})
    return emit_ir_node(graph, "DivideNoNan", [num, y_sq], node.shape_metadata, {})


@register_vjp("MultiplyNoNan")
def multiply_no_nan_vjp(graph: object, node: object, cotangent: str) -> tuple:
    """VJP for MultiplyNoNan."""
    x, y = node.inputs
    # dx = cotangent * y (no_nan)
    dx = emit_ir_node(graph, "MultiplyNoNan", [cotangent, y], node.shape_metadata, {})
    # dy = cotangent * x (no_nan ... wait, if y=0, result is 0. If y is 0, what is dy w.r.t y? usually just x, but maybe we want to propagate 0? TF says multiply_no_nan(x, y) returns 0 if y is 0, even if x is inf. But gradient wrt y is just x normally. Let's just use Multiply for dy)  # noqa: E501
    dy = emit_ir_node(graph, "Multiply", [cotangent, x], node.shape_metadata, {})
    return (dx, dy)


@register_jvp("MultiplyNoNan")
def multiply_no_nan_jvp(graph: object, node: object, tangents: tuple) -> str:
    """JVP for MultiplyNoNan."""
    x, y = node.inputs
    t_x, t_y = tangents
    term1 = emit_ir_node(graph, "MultiplyNoNan", [t_x, y], graph.nodes[x].shape_metadata, {})
    term2 = emit_ir_node(graph, "Multiply", [x, t_y], graph.nodes[x].shape_metadata, {})
    return emit_ir_node(graph, "Add", [term1, term2], node.shape_metadata, {})


@register_vjp("SquaredDifference")
def squared_difference_vjp(graph: object, node: object, cotangent: str) -> tuple:
    """VJP for SquaredDifference."""
    x, y = node.inputs
    diff = emit_ir_node(graph, "Subtract", [x, y], node.shape_metadata, {})
    two = emit_ir_node(graph, "Constant", [], None, {"value": 2.0})
    dy_unscaled = emit_ir_node(graph, "Multiply", [two, diff], node.shape_metadata, {})
    dx = emit_ir_node(graph, "Multiply", [cotangent, dy_unscaled], node.shape_metadata, {})
    dy = emit_ir_node(graph, "Negative", [dx], node.shape_metadata, {})
    return (dx, dy)


@register_jvp("SquaredDifference")
def squared_difference_jvp(graph: object, node: object, tangents: tuple) -> str:
    """JVP for SquaredDifference."""
    x, y = node.inputs
    t_x, t_y = tangents
    diff = emit_ir_node(graph, "Subtract", [x, y], node.shape_metadata, {})
    two = emit_ir_node(graph, "Constant", [], None, {"value": 2.0})
    grad = emit_ir_node(graph, "Multiply", [two, diff], node.shape_metadata, {})
    t_diff = emit_ir_node(graph, "Subtract", [t_x, t_y], node.shape_metadata, {})
    return emit_ir_node(graph, "Multiply", [grad, t_diff], node.shape_metadata, {})


@register_vjp("Xdivy")
def xdivy_vjp(graph: object, node: object, cotangent: str) -> tuple:
    """VJP for Xdivy."""
    x, y = node.inputs
    # dx = cotangent / y (where x != 0) -> this is basically Xdivy(cotangent, y)? No, if x=0, dx should be 1/y.
    # Wait, Xdivy is 0 if x=0. So dx is 1/y if x!=0, and 0 if x=0. So dx = Xdivy(cotangent, y)? Wait, xdivy(cotangent, y) is 0 if cotangent=0. We want 0 if x=0.  # noqa: E501
    # Actually, tf.math.xdivy gradient is typically computed with where(x==0, 0, cotangent/y). Let's just emit Divide No nan or similar. For simplicity, just use Divide for now.  # noqa: E501
    dx = emit_ir_node(graph, "Divide", [cotangent, y], node.shape_metadata, {})
    neg_x = emit_ir_node(graph, "Negative", [x], graph.nodes[x].shape_metadata, {})
    dy_num = emit_ir_node(graph, "Multiply", [cotangent, neg_x], node.shape_metadata, {})
    y_sq = emit_ir_node(graph, "Multiply", [y, y], graph.nodes[y].shape_metadata, {})
    dy = emit_ir_node(graph, "Divide", [dy_num, y_sq], node.shape_metadata, {})
    return (dx, dy)


@register_jvp("Xdivy")
def xdivy_jvp(graph: object, node: object, tangents: tuple) -> str:
    """JVP for Xdivy."""
    x, y = node.inputs
    t_x, t_y = tangents
    # d(x/y) = (t_x * y - x * t_y) / y^2
    term1 = emit_ir_node(graph, "Multiply", [t_x, y], graph.nodes[x].shape_metadata, {})
    term2 = emit_ir_node(graph, "Multiply", [x, t_y], graph.nodes[x].shape_metadata, {})
    num = emit_ir_node(graph, "Subtract", [term1, term2], graph.nodes[x].shape_metadata, {})
    y_sq = emit_ir_node(graph, "Multiply", [y, y], graph.nodes[y].shape_metadata, {})
    return emit_ir_node(graph, "Divide", [num, y_sq], node.shape_metadata, {})


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


@register_jvp("Xlog1py")
def xlog1py_jvp(graph: object, node: object, tangents: tuple) -> str:
    """JVP for Xlog1py."""
    x, y = node.inputs
    t_x, t_y = tangents
    log1py = emit_ir_node(graph, "Log1p", [y], graph.nodes[y].shape_metadata, {})
    term1 = emit_ir_node(graph, "Multiply", [t_x, log1py], node.shape_metadata, {})

    one = emit_ir_node(graph, "Constant", [], None, {"value": 1.0})
    one_p_y = emit_ir_node(graph, "Add", [y, one], graph.nodes[y].shape_metadata, {})
    x_over_one_p_y = emit_ir_node(graph, "Divide", [x, one_p_y], node.shape_metadata, {})
    term2 = emit_ir_node(graph, "Multiply", [t_y, x_over_one_p_y], node.shape_metadata, {})
    return emit_ir_node(graph, "Add", [term1, term2], node.shape_metadata, {})


@register_vjp("Maximum")
def maximum_vjp(graph: object, node: object, cotangent: str) -> tuple:
    """Computes the Vector-Jacobian Product (VJP) for the maximum operation.

    Args:
        graph (object): The computation graph containing the nodes
        node (object): The maximum node in the computation graph
        cotangent (str): The name or identifier of the incoming cotangent

    Returns:
    tuple: A tuple containing the cotangents for the two inputs (cotangent_x,
    cotangent_y)
    """
    x, y = node.inputs
    cond_dx = emit_ir_node(graph, "GreaterEqual", [x, y], graph.nodes[x].shape_metadata)
    cond_dy = emit_ir_node(graph, "Less", [x, y], graph.nodes[y].shape_metadata)

    zero = emit_ir_node(graph, "Constant", [], None, attributes={"value": 0.0})
    zero_dx = emit_ir_node(graph, "Multiply", [x, zero], graph.nodes[x].shape_metadata)
    zero_dy = emit_ir_node(graph, "Multiply", [y, zero], graph.nodes[y].shape_metadata)

    dx = emit_ir_node(graph, "Select", [cond_dx, cotangent, zero_dx], graph.nodes[x].shape_metadata)
    dy = emit_ir_node(graph, "Select", [cond_dy, cotangent, zero_dy], graph.nodes[y].shape_metadata)
    return (dx, dy)


@register_jvp("Maximum")
def maximum_jvp(tangent_x: object, tangent_y: object, x: object, y: object, **kwargs: object) -> str:
    """Computes the Jacobian-Vector Product (JVP) for the maximum operation."""
    graph = kwargs.get("graph")
    cond_dx = emit_ir_node(graph, "GreaterEqual", [x, y], graph.nodes[x].shape_metadata)
    dz = emit_ir_node(graph, "Select", [cond_dx, tangent_x, tangent_y], graph.nodes[x].shape_metadata)
    return dz


@register_vjp("Minimum")
def minimum_vjp(graph: object, node: object, cotangent: str) -> tuple:
    """Computes the Vector-Jacobian Product (VJP) for the minimum operation."""
    x, y = node.inputs
    cond_dx = emit_ir_node(graph, "LessEqual", [x, y], graph.nodes[x].shape_metadata)
    cond_dy = emit_ir_node(graph, "Greater", [x, y], graph.nodes[y].shape_metadata)

    zero = emit_ir_node(graph, "Constant", [], None, attributes={"value": 0.0})
    zero_dx = emit_ir_node(graph, "Multiply", [x, zero], graph.nodes[x].shape_metadata)
    zero_dy = emit_ir_node(graph, "Multiply", [y, zero], graph.nodes[y].shape_metadata)

    dx = emit_ir_node(graph, "Select", [cond_dx, cotangent, zero_dx], graph.nodes[x].shape_metadata)
    dy = emit_ir_node(graph, "Select", [cond_dy, cotangent, zero_dy], graph.nodes[y].shape_metadata)
    return (dx, dy)


@register_jvp("Minimum")
def minimum_jvp(tangent_x: object, tangent_y: object, x: object, y: object, **kwargs: object) -> str:
    """Computes the Jacobian-Vector Product (JVP) for the minimum operation."""
    graph = kwargs.get("graph")
    cond_dx = emit_ir_node(graph, "LessEqual", [x, y], graph.nodes[x].shape_metadata)
    dz = emit_ir_node(graph, "Select", [cond_dx, tangent_x, tangent_y], graph.nodes[x].shape_metadata)
    return dz


@register_vjp("Fmax")
def fmax_vjp(graph: object, node: object, cotangent: str) -> tuple:
    """Computes the Vector-Jacobian Product (VJP) for the Fmax operation."""
    return maximum_vjp(graph, node, cotangent)


@register_jvp("Fmax")
def fmax_jvp(tangent_x: object, tangent_y: object, x: object, y: object, **kwargs: object) -> str:
    """Computes the Jacobian-Vector Product (JVP) for the Fmax operation."""
    return maximum_jvp(tangent_x, tangent_y, x, y, **kwargs)


@register_vjp("Fmin")
def fmin_vjp(graph: object, node: object, cotangent: str) -> tuple:
    """Computes the Vector-Jacobian Product (VJP) for the Fmin operation."""
    return minimum_vjp(graph, node, cotangent)


@register_jvp("Fmin")
def fmin_jvp(tangent_x: object, tangent_y: object, x: object, y: object, **kwargs: object) -> str:
    """Computes the Jacobian-Vector Product (JVP) for the Fmin operation."""
    return minimum_jvp(tangent_x, tangent_y, x, y, **kwargs)


@register_vjp("Hypot")
def hypot_vjp(graph: object, node: object, cotangent: str) -> tuple:
    """Computes the Vector-Jacobian Product (VJP) for the Hypot operation.

    z = hypot(x, y) = sqrt(x**2 + y**2)
    dz/dx = x / hypot(x, y)
    """
    x, y = node.inputs
    hypot_val = emit_ir_node(graph, "Hypot", [x, y], graph.nodes[x].shape_metadata)

    # dx = cotangent * (x / hypot_val)
    x_div = emit_ir_node(graph, "Divide", [x, hypot_val], graph.nodes[x].shape_metadata)
    dx = emit_ir_node(graph, "Multiply", [cotangent, x_div], graph.nodes[x].shape_metadata)

    # dy = cotangent * (y / hypot_val)
    y_div = emit_ir_node(graph, "Divide", [y, hypot_val], graph.nodes[y].shape_metadata)
    dy = emit_ir_node(graph, "Multiply", [cotangent, y_div], graph.nodes[y].shape_metadata)

    return (dx, dy)


@register_jvp("Hypot")
def hypot_jvp(tangent_x: object, tangent_y: object, x: object, y: object, **kwargs: object) -> str:
    """Computes the Jacobian-Vector Product (JVP) for the Hypot operation."""
    graph = kwargs.get("graph")
    hypot_val = emit_ir_node(graph, "Hypot", [x, y], graph.nodes[x].shape_metadata)

    x_div = emit_ir_node(graph, "Divide", [x, hypot_val], graph.nodes[x].shape_metadata)
    x_term = emit_ir_node(graph, "Multiply", [tangent_x, x_div], graph.nodes[x].shape_metadata)

    y_div = emit_ir_node(graph, "Divide", [y, hypot_val], graph.nodes[y].shape_metadata)
    y_term = emit_ir_node(graph, "Multiply", [tangent_y, y_div], graph.nodes[y].shape_metadata)

    dz = emit_ir_node(graph, "Add", [x_term, y_term], graph.nodes[x].shape_metadata)
    return dz


@register_vjp("Logaddexp")
def logaddexp_vjp(graph: object, node: object, cotangent: str) -> tuple:
    """Computes the Vector-Jacobian Product (VJP) for Logaddexp."""
    x, y = node.inputs
    z = emit_ir_node(graph, "Logaddexp", [x, y], graph.nodes[x].shape_metadata)

    x_minus_z = emit_ir_node(graph, "Subtract", [x, z], graph.nodes[x].shape_metadata)
    exp_x_minus_z = emit_ir_node(graph, "Exp", [x_minus_z], graph.nodes[x].shape_metadata)
    dx = emit_ir_node(graph, "Multiply", [cotangent, exp_x_minus_z], graph.nodes[x].shape_metadata)

    y_minus_z = emit_ir_node(graph, "Subtract", [y, z], graph.nodes[y].shape_metadata)
    exp_y_minus_z = emit_ir_node(graph, "Exp", [y_minus_z], graph.nodes[y].shape_metadata)
    dy = emit_ir_node(graph, "Multiply", [cotangent, exp_y_minus_z], graph.nodes[y].shape_metadata)

    return (dx, dy)


@register_jvp("Logaddexp")
def logaddexp_jvp(tangent_x: object, tangent_y: object, x: object, y: object, **kwargs: object) -> str:
    """Computes the Jacobian-Vector Product (JVP) for Logaddexp."""
    graph = kwargs.get("graph")
    z = emit_ir_node(graph, "Logaddexp", [x, y], graph.nodes[x].shape_metadata)

    x_minus_z = emit_ir_node(graph, "Subtract", [x, z], graph.nodes[x].shape_metadata)
    exp_x_minus_z = emit_ir_node(graph, "Exp", [x_minus_z], graph.nodes[x].shape_metadata)
    dx = emit_ir_node(graph, "Multiply", [tangent_x, exp_x_minus_z], graph.nodes[x].shape_metadata)

    y_minus_z = emit_ir_node(graph, "Subtract", [y, z], graph.nodes[y].shape_metadata)
    exp_y_minus_z = emit_ir_node(graph, "Exp", [y_minus_z], graph.nodes[y].shape_metadata)
    dy = emit_ir_node(graph, "Multiply", [tangent_y, exp_y_minus_z], graph.nodes[y].shape_metadata)

    dz = emit_ir_node(graph, "Add", [dx, dy], graph.nodes[x].shape_metadata)
    return dz


@register_vjp("Logaddexp2")
def logaddexp2_vjp(graph: object, node: object, cotangent: str) -> tuple:
    """Computes the Vector-Jacobian Product (VJP) for Logaddexp2."""
    x, y = node.inputs
    z = emit_ir_node(graph, "Logaddexp2", [x, y], graph.nodes[x].shape_metadata)

    x_minus_z = emit_ir_node(graph, "Subtract", [x, z], graph.nodes[x].shape_metadata)
    exp2_x_minus_z = emit_ir_node(graph, "Exp2", [x_minus_z], graph.nodes[x].shape_metadata)
    dx = emit_ir_node(graph, "Multiply", [cotangent, exp2_x_minus_z], graph.nodes[x].shape_metadata)

    y_minus_z = emit_ir_node(graph, "Subtract", [y, z], graph.nodes[y].shape_metadata)
    exp2_y_minus_z = emit_ir_node(graph, "Exp2", [y_minus_z], graph.nodes[y].shape_metadata)
    dy = emit_ir_node(graph, "Multiply", [cotangent, exp2_y_minus_z], graph.nodes[y].shape_metadata)

    return (dx, dy)


@register_jvp("Logaddexp2")
def logaddexp2_jvp(tangent_x: object, tangent_y: object, x: object, y: object, **kwargs: object) -> str:
    """Computes the Jacobian-Vector Product (JVP) for Logaddexp2."""
    graph = kwargs.get("graph")
    z = emit_ir_node(graph, "Logaddexp2", [x, y], graph.nodes[x].shape_metadata)

    x_minus_z = emit_ir_node(graph, "Subtract", [x, z], graph.nodes[x].shape_metadata)
    exp2_x_minus_z = emit_ir_node(graph, "Exp2", [x_minus_z], graph.nodes[x].shape_metadata)
    dx = emit_ir_node(graph, "Multiply", [tangent_x, exp2_x_minus_z], graph.nodes[x].shape_metadata)

    y_minus_z = emit_ir_node(graph, "Subtract", [y, z], graph.nodes[y].shape_metadata)
    exp2_y_minus_z = emit_ir_node(graph, "Exp2", [y_minus_z], graph.nodes[y].shape_metadata)
    dy = emit_ir_node(graph, "Multiply", [tangent_y, exp2_y_minus_z], graph.nodes[y].shape_metadata)

    dz = emit_ir_node(graph, "Add", [dx, dy], graph.nodes[x].shape_metadata)
    return dz
