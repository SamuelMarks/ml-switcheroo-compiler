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


@register_vjp("AddN")
def add_n_vjp(graph: object, node: object, cotangent: str) -> tuple:
    """VJP for AddN."""
    return tuple(cotangent for _ in node.inputs)


@register_jvp("AddN")
def add_n_jvp(graph: object, node: object, tangents: list[str]) -> str:
    """JVP for AddN."""
    return emit_ir_node(graph, "AddN", tangents, node.shape_metadata, node.attributes)


@register_vjp("AccumulateN")
def accumulate_n_vjp(graph: object, node: object, cotangent: str) -> tuple:
    """VJP for AccumulateN."""
    return tuple(cotangent for _ in node.inputs)


@register_jvp("AccumulateN")
def accumulate_n_jvp(graph: object, node: object, tangents: list[str]) -> str:
    """JVP for AccumulateN."""
    return emit_ir_node(graph, "AccumulateN", tangents, node.shape_metadata, node.attributes)


@register_vjp("CumulativeLogsumexp")
def cumulative_logsumexp_vjp(graph: object, node: object, cotangent: str) -> tuple:
    """VJP for CumulativeLogsumexp."""
    x = node.inputs[0]
    axis = node.attributes.get("axis", 0)

    # neg_y = -y
    neg_y = emit_ir_node(graph, "Negative", [node.id], node.shape_metadata, {})
    # exp_neg_y = exp(-y)
    exp_neg_y = emit_ir_node(graph, "Exp", [neg_y], node.shape_metadata, {})
    # scaled_cotangent = cotangent * exp(-y)
    scaled_cotangent = emit_ir_node(
        graph, "Multiply", [cotangent, exp_neg_y], node.shape_metadata, {}
    )
    # rev_cumsum = cumsum(scaled_cotangent) reversed... wait. Cumsum doesn't support reverse yet.
    # So we flip, cumsum, flip.

    # flip(scaled_cotangent)
    flipped = emit_ir_node(
        graph, "Reverse", [scaled_cotangent], node.shape_metadata, {"dims": [axis]}
    )
    # cumsum(flipped)
    cumsum_flipped = emit_ir_node(graph, "Cumsum", [flipped], node.shape_metadata, {"axis": axis})
    # flip(cumsum_flipped)
    rev_cumsum = emit_ir_node(
        graph, "Reverse", [cumsum_flipped], node.shape_metadata, {"dims": [axis]}
    )

    # exp_x = exp(x)
    exp_x = emit_ir_node(graph, "Exp", [x], graph.nodes[x].shape_metadata, {})

    # dx = exp_x * rev_cumsum
    dx = emit_ir_node(graph, "Multiply", [exp_x, rev_cumsum], graph.nodes[x].shape_metadata, {})
    return (dx,)


@register_jvp("CumulativeLogsumexp")
def cumulative_logsumexp_jvp(graph: object, node: object, tangent: str) -> str:
    """JVP for CumulativeLogsumexp."""
    return emit_ir_node(
        graph, "CumulativeLogsumexp", [tangent], node.shape_metadata, node.attributes
    )


@register_vjp("ReduceEuclideanNorm")
def reduce_euclidean_norm_vjp(graph: object, node: object, cotangent: str) -> tuple:
    """VJP for ReduceEuclideanNorm."""
    x = node.inputs[0]
    # y = sqrt(sum(x^2)) -> node.id
    y = node.id
    y_bcast = emit_ir_node(
        graph,
        "BroadcastTo",
        [y],
        graph.nodes[x].shape_metadata,
        {"shape": graph.nodes[x].shape_metadata},
    )
    cotangent_bcast = emit_ir_node(
        graph,
        "BroadcastTo",
        [cotangent],
        graph.nodes[x].shape_metadata,
        {"shape": graph.nodes[x].shape_metadata},
    )

    # dx = (cotangent_bcast / y_bcast) * x (using DivideNoNan for safety)
    grad_scale = emit_ir_node(
        graph, "DivideNoNan", [cotangent_bcast, y_bcast], graph.nodes[x].shape_metadata, {}
    )
    dx = emit_ir_node(graph, "Multiply", [grad_scale, x], graph.nodes[x].shape_metadata, {})
    return (dx,)


@register_jvp("ReduceEuclideanNorm")
def reduce_euclidean_norm_jvp(graph: object, node: object, tangent: str) -> str:
    """JVP for ReduceEuclideanNorm."""
    x = node.inputs[0]
    # dy = sum((x / y) * dx)
    y = node.id
    y_bcast = emit_ir_node(
        graph,
        "BroadcastTo",
        [y],
        graph.nodes[x].shape_metadata,
        {"shape": graph.nodes[x].shape_metadata},
    )
    x_over_y = emit_ir_node(graph, "DivideNoNan", [x, y_bcast], graph.nodes[x].shape_metadata, {})
    scaled_tangent = emit_ir_node(
        graph, "Multiply", [x_over_y, tangent], graph.nodes[x].shape_metadata, {}
    )
    return emit_ir_node(graph, "Sum", [scaled_tangent], node.shape_metadata, node.attributes)


@register_vjp("Logsumexp")
def logsumexp_vjp(graph: object, node: object, cotangent: str) -> tuple:
    """VJP for Logsumexp."""
    x = node.inputs[0]
    y = node.id
    y_bcast = emit_ir_node(
        graph,
        "BroadcastTo",
        [y],
        graph.nodes[x].shape_metadata,
        {"shape": graph.nodes[x].shape_metadata},
    )
    cotangent_bcast = emit_ir_node(
        graph,
        "BroadcastTo",
        [cotangent],
        graph.nodes[x].shape_metadata,
        {"shape": graph.nodes[x].shape_metadata},
    )

    # dx = exp(x - y_bcast) * cotangent_bcast
    x_minus_y = emit_ir_node(graph, "Subtract", [x, y_bcast], graph.nodes[x].shape_metadata, {})
    exp_x_minus_y = emit_ir_node(graph, "Exp", [x_minus_y], graph.nodes[x].shape_metadata, {})
    dx = emit_ir_node(
        graph, "Multiply", [exp_x_minus_y, cotangent_bcast], graph.nodes[x].shape_metadata, {}
    )
    return (dx,)


@register_jvp("Logsumexp")
def logsumexp_jvp(graph: object, node: object, tangent: str) -> str:
    """JVP for Logsumexp."""
    x = node.inputs[0]
    y = node.id
    y_bcast = emit_ir_node(
        graph,
        "BroadcastTo",
        [y],
        graph.nodes[x].shape_metadata,
        {"shape": graph.nodes[x].shape_metadata},
    )
    x_minus_y = emit_ir_node(graph, "Subtract", [x, y_bcast], graph.nodes[x].shape_metadata, {})
    softmax_x = emit_ir_node(graph, "Exp", [x_minus_y], graph.nodes[x].shape_metadata, {})
    scaled_tangent = emit_ir_node(
        graph, "Multiply", [softmax_x, tangent], graph.nodes[x].shape_metadata, {}
    )
    return emit_ir_node(graph, "Sum", [scaled_tangent], node.shape_metadata, node.attributes)
