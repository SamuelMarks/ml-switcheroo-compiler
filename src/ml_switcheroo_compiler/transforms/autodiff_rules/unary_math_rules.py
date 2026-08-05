"""Unary rules for math."""

from ml_switcheroo_compiler.ops.base import emit_ir_node
from ml_switcheroo_compiler.transforms.autodiff_rules.jvp_registry import register_jvp
from ml_switcheroo_compiler.transforms.autodiff_rules.vjp_registry import register_vjp


@register_vjp("Abs")
def abs_vjp(graph: object, node: object, cotangent: str) -> tuple:
    """Docstring.

    Args:
        graph (object): The graph parameter.
        node (object): The node parameter.
        cotangent (str): The cotangent parameter.

    Returns:
        tuple: Result.
    """
    """Docstring."""
    x = node.inputs[0]
    sign_x = emit_ir_node(graph, "Sign", inputs=[x])
    dx = emit_ir_node(graph, "Multiply", inputs=[cotangent, sign_x])
    return (dx,)


@register_jvp("Abs")
def abs_jvp(graph: object, node: object, tangents: tuple) -> str:
    """Docstring.

    Args:
        graph (object): The graph parameter.
        node (object): The node parameter.
        tangents (tuple): The tangents parameter.

    Returns:
        str: Result.
    """
    """Docstring."""
    x = node.inputs[0]
    t_x = tangents[0]
    sign_x = emit_ir_node(graph, "Sign", inputs=[x])
    dy = emit_ir_node(graph, "Multiply", inputs=[t_x, sign_x])
    return dy


@register_vjp("Exp")
def exp_vjp(graph: object, node: object, cotangent: str) -> tuple:
    """Docstring.

    Args:
        graph (object): The graph parameter.
        node (object): The node parameter.
        cotangent (str): The cotangent parameter.

    Returns:
        tuple: Result.
    """
    """Docstring."""
    x = node.inputs[0]
    exp_x = emit_ir_node(graph, "Exp", inputs=[x])
    dx = emit_ir_node(graph, "Multiply", inputs=[cotangent, exp_x])
    return (dx,)


@register_jvp("Exp")
def exp_jvp(graph: object, node: object, tangents: tuple) -> str:
    """Evaluate exp_jvp operation.

    Args:
        graph (object): The graph parameter.
        node (object): The node parameter.
        tangents (tuple): The tangents parameter.

    Returns:
        str: Result.
    """
    x = node.inputs[0]
    t_x = tangents[0]
    exp_x = emit_ir_node(graph, "Exp", inputs=[x])
    dy = emit_ir_node(graph, "Multiply", inputs=[t_x, exp_x])
    return dy


@register_vjp("Exp2")
def exp2_vjp(graph: object, node: object, cotangent: str) -> tuple:
    """Docstring.

    Args:
        graph (object): The graph parameter.
        node (object): The node parameter.
        cotangent (str): The cotangent parameter.

    Returns:
        tuple: Result.
    """
    """Docstring."""
    x = node.inputs[0]
    import math

    ln2 = emit_ir_node(graph, "Constant", inputs=[], attributes={"value": math.log(2.0)})
    exp2_x = emit_ir_node(graph, "Exp2", inputs=[x])
    deriv = emit_ir_node(graph, "Multiply", inputs=[ln2, exp2_x])
    dx = emit_ir_node(graph, "Multiply", inputs=[cotangent, deriv])
    return (dx,)


@register_jvp("Exp2")
def exp2_jvp(graph: object, node: object, tangents: tuple) -> str:
    """Docstring.

    Args:
        graph (object): The graph parameter.
        node (object): The node parameter.
        tangents (tuple): The tangents parameter.

    Returns:
        str: Result.
    """
    """Docstring."""
    x = node.inputs[0]
    t_x = tangents[0]
    import math

    ln2 = emit_ir_node(graph, "Constant", inputs=[], attributes={"value": math.log(2.0)})
    exp2_x = emit_ir_node(graph, "Exp2", inputs=[x])
    deriv = emit_ir_node(graph, "Multiply", inputs=[ln2, exp2_x])
    dy = emit_ir_node(graph, "Multiply", inputs=[t_x, deriv])
    return dy


@register_vjp("Expm1")
def expm1_vjp(graph: object, node: object, cotangent: str) -> tuple:
    """Docstring.

    Args:
        graph (object): The graph parameter.
        node (object): The node parameter.
        cotangent (str): The cotangent parameter.

    Returns:
        tuple: Result.
    """
    """Docstring."""
    x = node.inputs[0]
    exp_x = emit_ir_node(graph, "Exp", inputs=[x])
    dx = emit_ir_node(graph, "Multiply", inputs=[cotangent, exp_x])
    return (dx,)


@register_jvp("Expm1")
def expm1_jvp(graph: object, node: object, tangents: tuple) -> str:
    """Docstring.

    Args:
        graph (object): The graph parameter.
        node (object): The node parameter.
        tangents (tuple): The tangents parameter.

    Returns:
        str: Result.
    """
    """Docstring."""
    x = node.inputs[0]
    t_x = tangents[0]
    exp_x = emit_ir_node(graph, "Exp", inputs=[x])
    dy = emit_ir_node(graph, "Multiply", inputs=[t_x, exp_x])
    return dy


@register_vjp("Log")
def log_vjp(graph: object, node: object, cotangent: str) -> tuple:
    """Docstring.

    Args:
        graph (object): The graph parameter.
        node (object): The node parameter.
        cotangent (str): The cotangent parameter.

    Returns:
        tuple: Result.
    """
    """Docstring."""
    x = node.inputs[0]
    one = emit_ir_node(graph, "Constant", inputs=[], attributes={"value": 1.0})
    deriv = emit_ir_node(graph, "Divide", inputs=[one, x])
    dx = emit_ir_node(graph, "Multiply", inputs=[cotangent, deriv])
    return (dx,)


@register_jvp("Log")
def log_jvp(graph: object, node: object, tangents: tuple) -> str:
    """Docstring.

    Args:
        graph (object): The graph parameter.
        node (object): The node parameter.
        tangents (tuple): The tangents parameter.

    Returns:
        str: Result.
    """
    """Docstring."""
    x = node.inputs[0]
    t_x = tangents[0]
    one = emit_ir_node(graph, "Constant", inputs=[], attributes={"value": 1.0})
    deriv = emit_ir_node(graph, "Divide", inputs=[one, x])
    dy = emit_ir_node(graph, "Multiply", inputs=[t_x, deriv])
    return dy


@register_vjp("Log10")
def log10_vjp(graph: object, node: object, cotangent: str) -> tuple:
    """Docstring.

    Args:
        graph (object): The graph parameter.
        node (object): The node parameter.
        cotangent (str): The cotangent parameter.

    Returns:
        tuple: Result.
    """
    """Docstring."""
    x = node.inputs[0]
    import math

    one_over_ln10 = emit_ir_node(graph, "Constant", inputs=[], attributes={"value": 1.0 / math.log(10.0)})
    deriv = emit_ir_node(graph, "Divide", inputs=[one_over_ln10, x])
    dx = emit_ir_node(graph, "Multiply", inputs=[cotangent, deriv])
    return (dx,)


@register_jvp("Log10")
def log10_jvp(graph: object, node: object, tangents: tuple) -> str:
    """Docstring.

    Args:
        graph (object): The graph parameter.
        node (object): The node parameter.
        tangents (tuple): The tangents parameter.

    Returns:
        str: Result.
    """
    """Docstring."""
    x = node.inputs[0]
    t_x = tangents[0]
    import math

    one_over_ln10 = emit_ir_node(graph, "Constant", inputs=[], attributes={"value": 1.0 / math.log(10.0)})
    deriv = emit_ir_node(graph, "Divide", inputs=[one_over_ln10, x])
    dy = emit_ir_node(graph, "Multiply", inputs=[t_x, deriv])
    return dy


@register_vjp("Log1P")
def log1p_vjp(graph: object, node: object, cotangent: str) -> tuple:
    """Docstring.

    Args:
        graph (object): The graph parameter.
        node (object): The node parameter.
        cotangent (str): The cotangent parameter.

    Returns:
        tuple: Result.
    """
    """Docstring."""
    x = node.inputs[0]
    one = emit_ir_node(graph, "Constant", inputs=[], attributes={"value": 1.0})
    one_plus_x = emit_ir_node(graph, "Add", inputs=[one, x])
    deriv = emit_ir_node(graph, "Divide", inputs=[one, one_plus_x])
    dx = emit_ir_node(graph, "Multiply", inputs=[cotangent, deriv])
    return (dx,)


@register_jvp("Log1P")
def log1p_jvp(graph: object, node: object, tangents: tuple) -> str:
    """Docstring.

    Args:
        graph (object): The graph parameter.
        node (object): The node parameter.
        tangents (tuple): The tangents parameter.

    Returns:
        str: Result.
    """
    """Docstring."""
    x = node.inputs[0]
    t_x = tangents[0]
    one = emit_ir_node(graph, "Constant", inputs=[], attributes={"value": 1.0})
    one_plus_x = emit_ir_node(graph, "Add", inputs=[one, x])
    deriv = emit_ir_node(graph, "Divide", inputs=[one, one_plus_x])
    dy = emit_ir_node(graph, "Multiply", inputs=[t_x, deriv])
    return dy


@register_vjp("Log2")
def log2_vjp(graph: object, node: object, cotangent: str) -> tuple:
    """Docstring.

    Args:
        graph (object): The graph parameter.
        node (object): The node parameter.
        cotangent (str): The cotangent parameter.

    Returns:
        tuple: Result.
    """
    """Docstring."""
    x = node.inputs[0]
    import math

    one_over_ln2 = emit_ir_node(graph, "Constant", inputs=[], attributes={"value": 1.0 / math.log(2.0)})
    deriv = emit_ir_node(graph, "Divide", inputs=[one_over_ln2, x])
    dx = emit_ir_node(graph, "Multiply", inputs=[cotangent, deriv])
    return (dx,)


@register_jvp("Log2")
def log2_jvp(graph: object, node: object, tangents: tuple) -> str:
    """Docstring.

    Args:
        graph (object): The graph parameter.
        node (object): The node parameter.
        tangents (tuple): The tangents parameter.

    Returns:
        str: Result.
    """
    """Docstring."""
    x = node.inputs[0]
    t_x = tangents[0]
    import math

    one_over_ln2 = emit_ir_node(graph, "Constant", inputs=[], attributes={"value": 1.0 / math.log(2.0)})
    deriv = emit_ir_node(graph, "Divide", inputs=[one_over_ln2, x])
    dy = emit_ir_node(graph, "Multiply", inputs=[t_x, deriv])
    return dy


@register_vjp("Sqrt")
def sqrt_vjp(graph: object, node: object, cotangent: str) -> tuple:
    """Docstring.

    Args:
        graph (object): The graph parameter.
        node (object): The node parameter.
        cotangent (str): The cotangent parameter.

    Returns:
        tuple: Result.
    """
    """Docstring."""
    x = node.inputs[0]
    half = emit_ir_node(graph, "Constant", inputs=[], attributes={"value": 0.5})
    sqrt_x = emit_ir_node(graph, "Sqrt", inputs=[x])
    deriv = emit_ir_node(graph, "Divide", inputs=[half, sqrt_x])
    dx = emit_ir_node(graph, "Multiply", inputs=[cotangent, deriv])
    return (dx,)


@register_jvp("Sqrt")
def sqrt_jvp(graph: object, node: object, tangents: tuple) -> str:
    """Docstring.

    Args:
        graph (object): The graph parameter.
        node (object): The node parameter.
        tangents (tuple): The tangents parameter.

    Returns:
        str: Result.
    """
    """Docstring."""
    x = node.inputs[0]
    t_x = tangents[0]
    half = emit_ir_node(graph, "Constant", inputs=[], attributes={"value": 0.5})
    sqrt_x = emit_ir_node(graph, "Sqrt", inputs=[x])
    deriv = emit_ir_node(graph, "Divide", inputs=[half, sqrt_x])
    dy = emit_ir_node(graph, "Multiply", inputs=[t_x, deriv])
    return dy


@register_vjp("Rsqrt")
def rsqrt_vjp(graph: object, node: object, cotangent: str) -> tuple:
    """Docstring.

    Args:
        graph (object): The graph parameter.
        node (object): The node parameter.
        cotangent (str): The cotangent parameter.

    Returns:
        tuple: Result.
    """
    """Docstring."""
    x = node.inputs[0]
    neg_half = emit_ir_node(graph, "Constant", inputs=[], attributes={"value": -0.5})
    rsqrt_x = emit_ir_node(graph, "Rsqrt", inputs=[x])
    rsqrt_cubed = emit_ir_node(graph, "Power", inputs=[rsqrt_x, emit_ir_node(graph, "Constant", inputs=[], attributes={"value": 3.0})])  # noqa: E501
    deriv = emit_ir_node(graph, "Multiply", inputs=[neg_half, rsqrt_cubed])
    dx = emit_ir_node(graph, "Multiply", inputs=[cotangent, deriv])
    return (dx,)


@register_jvp("Rsqrt")
def rsqrt_jvp(graph: object, node: object, tangents: tuple) -> str:
    """Docstring.

    Args:
        graph (object): The graph parameter.
        node (object): The node parameter.
        tangents (tuple): The tangents parameter.

    Returns:
        str: Result.
    """
    """Docstring."""
    x = node.inputs[0]
    t_x = tangents[0]
    neg_half = emit_ir_node(graph, "Constant", inputs=[], attributes={"value": -0.5})
    rsqrt_x = emit_ir_node(graph, "Rsqrt", inputs=[x])
    rsqrt_cubed = emit_ir_node(graph, "Power", inputs=[rsqrt_x, emit_ir_node(graph, "Constant", inputs=[], attributes={"value": 3.0})])  # noqa: E501
    deriv = emit_ir_node(graph, "Multiply", inputs=[neg_half, rsqrt_cubed])
    dy = emit_ir_node(graph, "Multiply", inputs=[t_x, deriv])
    return dy


@register_vjp("Square")
def square_vjp(graph: object, node: object, cotangent: str) -> tuple:
    """Docstring.

    Args:
        graph (object): The graph parameter.
        node (object): The node parameter.
        cotangent (str): The cotangent parameter.

    Returns:
        tuple: Result.
    """
    """Docstring."""
    x = node.inputs[0]
    two = emit_ir_node(graph, "Constant", inputs=[], attributes={"value": 2.0})
    deriv = emit_ir_node(graph, "Multiply", inputs=[two, x])
    dx = emit_ir_node(graph, "Multiply", inputs=[cotangent, deriv])
    return (dx,)


@register_jvp("Square")
def square_jvp(graph: object, node: object, tangents: tuple) -> str:
    """Docstring.

    Args:
        graph (object): The graph parameter.
        node (object): The node parameter.
        tangents (tuple): The tangents parameter.

    Returns:
        str: Result.
    """
    """Docstring."""
    x = node.inputs[0]
    t_x = tangents[0]
    two = emit_ir_node(graph, "Constant", inputs=[], attributes={"value": 2.0})
    deriv = emit_ir_node(graph, "Multiply", inputs=[two, x])
    dy = emit_ir_node(graph, "Multiply", inputs=[t_x, deriv])
    return dy


@register_vjp("Negative")
def negative_vjp(graph: object, node: object, cotangent: str) -> tuple:
    """Docstring.

    Args:
        graph (object): The graph parameter.
        node (object): The node parameter.
        cotangent (str): The cotangent parameter.

    Returns:
        tuple: Result.
    """
    dx = emit_ir_node(graph, "Negative", inputs=[cotangent])
    return (dx,)


@register_jvp("Negative")
def negative_jvp(graph: object, node: object, tangents: tuple) -> str:
    """Docstring.

    Args:
        graph (object): The graph parameter.
        node (object): The node parameter.
        tangents (tuple): The tangents parameter.

    Returns:
        str: Result.
    """
    dy = emit_ir_node(graph, "Negative", inputs=[tangents[0]])
    return dy


@register_vjp("Positive")
def positive_vjp(graph: object, node: object, cotangent: str) -> tuple:
    """Docstring.

    Args:
        graph (object): The graph parameter.
        node (object): The node parameter.
        cotangent (str): The cotangent parameter.

    Returns:
        tuple: Result.
    """
    dx = emit_ir_node(graph, "Positive", inputs=[cotangent])
    return (dx,)


@register_jvp("Positive")
def positive_jvp(graph: object, node: object, tangents: tuple) -> str:
    """Docstring.

    Args:
        graph (object): The graph parameter.
        node (object): The node parameter.
        tangents (tuple): The tangents parameter.

    Returns:
        str: Result.
    """
    dy = emit_ir_node(graph, "Positive", inputs=[tangents[0]])
    return dy


@register_vjp("Reciprocal")
def reciprocal_vjp(graph: object, node: object, cotangent: str) -> tuple:
    """Docstring.

    Args:
        graph (object): The graph parameter.
        node (object): The node parameter.
        cotangent (str): The cotangent parameter.

    Returns:
        tuple: Result.
    """
    x = node.inputs[0]
    neg_one = emit_ir_node(graph, "Constant", inputs=[], attributes={"value": -1.0})
    x_sq = emit_ir_node(graph, "Square", inputs=[x])
    deriv = emit_ir_node(graph, "Divide", inputs=[neg_one, x_sq])
    dx = emit_ir_node(graph, "Multiply", inputs=[cotangent, deriv])
    return (dx,)


@register_jvp("Reciprocal")
def reciprocal_jvp(graph: object, node: object, tangents: tuple) -> str:
    """Docstring.

    Args:
        graph (object): The graph parameter.
        node (object): The node parameter.
        tangents (tuple): The tangents parameter.

    Returns:
        str: Result.
    """
    x = node.inputs[0]
    t_x = tangents[0]
    neg_one = emit_ir_node(graph, "Constant", inputs=[], attributes={"value": -1.0})
    x_sq = emit_ir_node(graph, "Square", inputs=[x])
    deriv = emit_ir_node(graph, "Divide", inputs=[neg_one, x_sq])
    dy = emit_ir_node(graph, "Multiply", inputs=[t_x, deriv])
    return dy


@register_vjp("Cbrt")
def cbrt_vjp(graph: object, node: object, cotangent: str) -> tuple:
    """Docstring.

    Args:
        graph (object): The graph parameter.
        node (object): The node parameter.
        cotangent (str): The cotangent parameter.

    Returns:
        tuple: Result.
    """
    x = node.inputs[0]
    one_third = emit_ir_node(graph, "Constant", inputs=[], attributes={"value": 1.0 / 3.0})
    x_squared = emit_ir_node(graph, "Square", inputs=[x])
    cbrt_x_sq = emit_ir_node(graph, "Power", inputs=[x_squared, one_third])
    three = emit_ir_node(graph, "Constant", inputs=[], attributes={"value": 3.0})
    deriv_denom = emit_ir_node(graph, "Multiply", inputs=[three, cbrt_x_sq])
    one = emit_ir_node(graph, "Constant", inputs=[], attributes={"value": 1.0})
    deriv = emit_ir_node(graph, "Divide", inputs=[one, deriv_denom])
    dx = emit_ir_node(graph, "Multiply", inputs=[cotangent, deriv])
    return (dx,)


@register_jvp("Cbrt")
def cbrt_jvp(graph: object, node: object, tangents: tuple) -> str:
    """Docstring.

    Args:
        graph (object): The graph parameter.
        node (object): The node parameter.
        tangents (tuple): The tangents parameter.

    Returns:
        str: Result.
    """
    x = node.inputs[0]
    t_x = tangents[0]
    one_third = emit_ir_node(graph, "Constant", inputs=[], attributes={"value": 1.0 / 3.0})
    x_squared = emit_ir_node(graph, "Square", inputs=[x])
    cbrt_x_sq = emit_ir_node(graph, "Power", inputs=[x_squared, one_third])
    three = emit_ir_node(graph, "Constant", inputs=[], attributes={"value": 3.0})
    deriv_denom = emit_ir_node(graph, "Multiply", inputs=[three, cbrt_x_sq])
    one = emit_ir_node(graph, "Constant", inputs=[], attributes={"value": 1.0})
    deriv = emit_ir_node(graph, "Divide", inputs=[one, deriv_denom])
    dy = emit_ir_node(graph, "Multiply", inputs=[t_x, deriv])
    return dy


@register_vjp("Cosh")
def cosh_vjp(graph: object, node: object, cotangent: str) -> tuple:
    """VJP for Cosh.

    Args:
        graph (object): The graph parameter.
        node (object): The node parameter.
        cotangent (str): The cotangent parameter.

    Returns:
        tuple: Result.
    """
    from ml_switcheroo_compiler.ops.base import emit_ir_node

    x = node.inputs[0]
    deriv = emit_ir_node(graph, "Sinh", inputs=[x])
    dx = emit_ir_node(graph, "Multiply", inputs=[cotangent, deriv])
    return (dx,)


@register_jvp("Cosh")
def cosh_jvp(graph: object, node: object, tangents: tuple) -> str:
    """JVP for Cosh.

    Args:
        graph (object): The graph parameter.
        node (object): The node parameter.
        tangents (tuple): The tangents parameter.

    Returns:
        str: Result.
    """
    from ml_switcheroo_compiler.ops.base import emit_ir_node

    x = node.inputs[0]
    t_x = tangents[0]
    deriv = emit_ir_node(graph, "Sinh", inputs=[x])
    dy = emit_ir_node(graph, "Multiply", inputs=[t_x, deriv])
    return dy


@register_vjp("Acosh")
def acosh_vjp(graph: object, node: object, cotangent: str) -> tuple:
    """VJP for Acosh.

    Args:
        graph (object): The graph parameter.
        node (object): The node parameter.
        cotangent (str): The cotangent parameter.

    Returns:
        tuple: Result.
    """
    from ml_switcheroo_compiler.ops.base import emit_ir_node

    x = node.inputs[0]
    one = emit_ir_node(graph, "Constant", inputs=[], attributes={"value": 1.0})
    x_sq = emit_ir_node(graph, "Square", inputs=[x])
    x_sq_minus_one = emit_ir_node(graph, "Subtract", inputs=[x_sq, one])
    deriv_denom = emit_ir_node(graph, "Sqrt", inputs=[x_sq_minus_one])
    deriv = emit_ir_node(graph, "Divide", inputs=[one, deriv_denom])
    dx = emit_ir_node(graph, "Multiply", inputs=[cotangent, deriv])
    return (dx,)


@register_jvp("Acosh")
def acosh_jvp(graph: object, node: object, tangents: tuple) -> str:
    """JVP for Acosh.

    Args:
        graph (object): The graph parameter.
        node (object): The node parameter.
        tangents (tuple): The tangents parameter.

    Returns:
        str: Result.
    """
    from ml_switcheroo_compiler.ops.base import emit_ir_node

    x = node.inputs[0]
    t_x = tangents[0]
    one = emit_ir_node(graph, "Constant", inputs=[], attributes={"value": 1.0})
    x_sq = emit_ir_node(graph, "Square", inputs=[x])
    x_sq_minus_one = emit_ir_node(graph, "Subtract", inputs=[x_sq, one])
    deriv_denom = emit_ir_node(graph, "Sqrt", inputs=[x_sq_minus_one])
    deriv = emit_ir_node(graph, "Divide", inputs=[one, deriv_denom])
    dy = emit_ir_node(graph, "Multiply", inputs=[t_x, deriv])
    return dy
