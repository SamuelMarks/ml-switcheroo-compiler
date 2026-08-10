# ruff: noqa: E402, D100, D103, D104, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, D101, D102, D107, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Defines Vector-Jacobian Product (VJP) and Jacobian-Vector Product (JVP) rules for.

reduction operations
"""

import math
from typing import Any

from ml_switcheroo_compiler.ops.base import emit_ir_node
from ml_switcheroo_compiler.transforms.autodiff_rules.cast_and_conj_rules import _zero_jvp, _zero_vjp
from ml_switcheroo_compiler.transforms.autodiff_rules.jvp_registry import register_jvp
from ml_switcheroo_compiler.transforms.autodiff_rules.vjp_registry import register_vjp


@register_vjp("Sum")
def sum_vjp(graph: Any, node: Any, cotangent: str) -> tuple:
    """Computes the Vector-Jacobian Product (VJP) for the Sum operation.

    This rule broadcasts the incoming cotangent back to the shape of the original
    input
    tensor of the Sum operation

    Args:
        graph (Any): The computation graph
        node (Any): The Sum operation node
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
def sum_jvp(graph: Any, node: Any, tangent: str) -> str:
    """Computes the JVP for the Sum operation.

    Args:
        graph (Any): The computation graph
        node (Any): The Sum operation node
        tangent (str): The identifier of the tangent tensor

    Returns:
    str: The identifier of the computed JVP node
    """
    return emit_ir_node(graph, "Sum", [tangent], node.shape_metadata, node.attributes)


@register_vjp("Mean")
def mean_vjp(graph: Any, node: Any, cotangent: str) -> tuple:
    """Computes the Vector-Jacobian Product (VJP) for the Mean operation.

    Args:
        graph (Any): The computation graph
        node (Any): The Mean operation node
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

    n = 1.0
    if graph.nodes[x].shape_metadata and node.shape_metadata:
        n = math.prod(graph.nodes[x].shape_metadata) / (math.prod(node.shape_metadata) if node.shape_metadata else 1.0)

    n_id = emit_ir_node(graph, "Constant", [], graph.nodes[x].shape_metadata, {"value": float(n)})
    res = emit_ir_node(graph, "TrueDivide", [bcast, n_id], graph.nodes[x].shape_metadata)
    return (res,)


@register_jvp("Mean")
def mean_jvp(graph: Any, node: Any, tangent: str) -> str:
    """Computes the JVP for the Mean operation.

    Args:
        graph (Any): The computation graph
        node (Any): The Mean operation node
        tangent (str): The identifier of the tangent tensor

    Returns:
    str: The identifier of the computed JVP node
    """
    return emit_ir_node(graph, "Mean", [tangent], node.shape_metadata, node.attributes)


@register_vjp("Max")
def max_vjp(graph: Any, node: Any, cotangent: str) -> tuple:
    """Computes the Vector-Jacobian Product (VJP) for the Max operation.

    Args:
        graph (Any): The computation graph
        node (Any): The Max operation node
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
def max_jvp(graph: Any, node: Any, tangent: str) -> str:
    """Computes the JVP for the Max operation.

    Args:
        graph (Any): The computation graph
        node (Any): The Max operation node
        tangent (str): The identifier of the tangent tensor

    Returns:
    str: The identifier of the computed JVP node
    """
    return emit_ir_node(graph, "Max", [tangent], node.shape_metadata, node.attributes)


@register_vjp("Min")
def min_vjp(graph: Any, node: Any, cotangent: str) -> tuple:
    """Computes the Vector-Jacobian Product (VJP) for the Min operation.

    Args:
        graph (Any): The computation graph
        node (Any): The Min operation node
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
def min_jvp(graph: Any, node: Any, tangent: str) -> str:
    """Computes the JVP for the Min operation.

    Args:
        graph (Any): The computation graph
        node (Any): The Min operation node
        tangent (str): The identifier of the tangent tensor

    Returns:
    str: The identifier of the computed JVP node
    """
    return emit_ir_node(graph, "Min", [tangent], node.shape_metadata, node.attributes)


@register_vjp("AddN")
def add_n_vjp(graph: Any, node: Any, cotangent: str) -> tuple:
    """VJP for AddN."""
    return tuple(cotangent for _ in node.inputs)


@register_jvp("AddN")
def add_n_jvp(graph: Any, node: Any, tangents: list[str]) -> str:
    """JVP for AddN."""
    return emit_ir_node(graph, "AddN", tangents, node.shape_metadata, node.attributes)


@register_vjp("AccumulateN")
def accumulate_n_vjp(graph: Any, node: Any, cotangent: str) -> tuple:
    """VJP for AccumulateN."""
    return tuple(cotangent for _ in node.inputs)


@register_jvp("AccumulateN")
def accumulate_n_jvp(graph: Any, node: Any, tangents: list[str]) -> str:
    """JVP for AccumulateN."""
    return emit_ir_node(graph, "AccumulateN", tangents, node.shape_metadata, node.attributes)


@register_vjp("CumulativeLogsumexp")
def cumulative_logsumexp_vjp(graph: Any, node: Any, cotangent: str) -> tuple:
    """VJP for CumulativeLogsumexp."""
    x = node.inputs[0]
    axis = node.attributes.get("axis", 0)

    # neg_y = -y
    neg_y = emit_ir_node(graph, "Negative", [node.id], node.shape_metadata, {})
    # exp_neg_y = exp(-y)
    exp_neg_y = emit_ir_node(graph, "Exp", [neg_y], node.shape_metadata, {})
    # scaled_cotangent = cotangent * exp(-y)
    scaled_cotangent = emit_ir_node(graph, "Multiply", [cotangent, exp_neg_y], node.shape_metadata, {})
    # rev_cumsum = cumsum(scaled_cotangent) reversed... wait. Cumsum doesn't support reverse yet.
    # So we flip, cumsum, flip.

    # flip(scaled_cotangent)
    flipped = emit_ir_node(graph, "Reverse", [scaled_cotangent], node.shape_metadata, {"dims": [axis]})
    # cumsum(flipped)
    cumsum_flipped = emit_ir_node(graph, "Cumsum", [flipped], node.shape_metadata, {"axis": axis})
    # flip(cumsum_flipped)
    rev_cumsum = emit_ir_node(graph, "Reverse", [cumsum_flipped], node.shape_metadata, {"dims": [axis]})

    # exp_x = exp(x)
    exp_x = emit_ir_node(graph, "Exp", [x], graph.nodes[x].shape_metadata, {})

    # dx = exp_x * rev_cumsum
    dx = emit_ir_node(graph, "Multiply", [exp_x, rev_cumsum], graph.nodes[x].shape_metadata, {})
    return (dx,)


@register_jvp("CumulativeLogsumexp")
def cumulative_logsumexp_jvp(graph: Any, node: Any, tangent: str) -> str:
    """JVP for CumulativeLogsumexp."""
    return emit_ir_node(graph, "CumulativeLogsumexp", [tangent], node.shape_metadata, node.attributes)


@register_vjp("ReduceEuclideanNorm")
def reduce_euclidean_norm_vjp(graph: Any, node: Any, cotangent: str) -> tuple:
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
    grad_scale = emit_ir_node(graph, "DivideNoNan", [cotangent_bcast, y_bcast], graph.nodes[x].shape_metadata, {})
    dx = emit_ir_node(graph, "Multiply", [grad_scale, x], graph.nodes[x].shape_metadata, {})
    return (dx,)


@register_jvp("ReduceEuclideanNorm")
def reduce_euclidean_norm_jvp(graph: Any, node: Any, tangent: str) -> str:
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
    scaled_tangent = emit_ir_node(graph, "Multiply", [x_over_y, tangent], graph.nodes[x].shape_metadata, {})
    return emit_ir_node(graph, "Sum", [scaled_tangent], node.shape_metadata, node.attributes)


@register_vjp("Logsumexp")
def logsumexp_vjp(graph: Any, node: Any, cotangent: str) -> tuple:
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
    dx = emit_ir_node(graph, "Multiply", [exp_x_minus_y, cotangent_bcast], graph.nodes[x].shape_metadata, {})
    return (dx,)


@register_jvp("Logsumexp")
def logsumexp_jvp(graph: Any, node: Any, tangent: str) -> str:
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
    scaled_tangent = emit_ir_node(graph, "Multiply", [softmax_x, tangent], graph.nodes[x].shape_metadata, {})
    return emit_ir_node(graph, "Sum", [scaled_tangent], node.shape_metadata, node.attributes)


@register_vjp("Average")
def average_vjp(graph: Any, node: Any, cotangent: str) -> tuple:
    """Computes the Vector-Jacobian Product (VJP) for the Average operation."""
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
        n = math.prod(graph.nodes[x].shape_metadata) / (math.prod(node.shape_metadata) if node.shape_metadata else 1.0)
    n_id = emit_ir_node(graph, "Constant", [], graph.nodes[x].shape_metadata, {"value": float(n)})
    res = emit_ir_node(graph, "TrueDivide", [bcast, n_id], graph.nodes[x].shape_metadata)
    return (res,)


@register_jvp("Average")
def average_jvp(graph: Any, node: Any, tangent: str) -> str:
    """Computes the JVP for the Average operation."""
    return emit_ir_node(graph, "Average", [tangent], node.shape_metadata, node.attributes)


@register_vjp("AllReduce")
@register_vjp("NcclAllReduce")
@register_vjp("HierarchicalCopyAllReduce")
@register_vjp("Reduce")
def allreduce_vjp(graph: Any, node: Any, cotangent: str) -> tuple:
    """VJP for AllReduce-like operations."""
    # VJP of sum-allreduce is sum-allreduce of cotangent
    # Note: we assume sum for simplicity or we should pass the reduce_type.
    # We will just pass the same op and attributes.
    res = emit_ir_node(graph, node.op_type, [cotangent], node.shape_metadata, node.attributes)
    return (res,)


@register_jvp("AllReduce")
@register_jvp("NcclAllReduce")
@register_jvp("HierarchicalCopyAllReduce")
@register_jvp("Reduce")
def allreduce_jvp(graph: Any, node: Any, tangent: str) -> str:
    """JVP for AllReduce-like operations."""
    return emit_ir_node(graph, node.op_type, [tangent], node.shape_metadata, node.attributes)


@register_vjp("ReduceScatter")
def reduce_scatter_vjp(graph: Any, node: Any, cotangent: str) -> tuple:
    """VJP for ReduceScatter."""
    # VJP of ReduceScatter is AllGather
    # We might not have AllGather perfectly aligned, but we emit it.
    res = emit_ir_node(graph, "AllGather", [cotangent], graph.nodes[node.inputs[0]].shape_metadata, node.attributes)
    return (res,)


@register_jvp("ReduceScatter")
def reduce_scatter_jvp(graph: Any, node: Any, tangent: str) -> str:
    """JVP for ReduceScatter."""
    return emit_ir_node(graph, "ReduceScatter", [tangent], node.shape_metadata, node.attributes)


@register_vjp("AllGather")
def all_gather_vjp(graph: Any, node: Any, cotangent: str) -> tuple:
    """VJP for AllGather."""
    # VJP of AllGather is ReduceScatter
    res = emit_ir_node(graph, "ReduceScatter", [cotangent], graph.nodes[node.inputs[0]].shape_metadata, node.attributes)
    return (res,)


@register_jvp("AllGather")
def all_gather_jvp(graph: Any, node: Any, tangent: str) -> str:
    """JVP for AllGather."""
    return emit_ir_node(graph, "AllGather", [tangent], node.shape_metadata, node.attributes)


@register_vjp("ShardTensor")
def shard_tensor_vjp(graph: Any, node: Any, cotangent: str) -> tuple:
    """VJP for ShardTensor."""
    res = emit_ir_node(graph, "ShardTensor", [cotangent], graph.nodes[node.inputs[0]].shape_metadata, node.attributes)
    return (res,)


@register_jvp("ShardTensor")
def shard_tensor_jvp(graph: Any, node: Any, tangent: str) -> str:
    """JVP for ShardTensor."""
    return emit_ir_node(graph, "ShardTensor", [tangent], node.shape_metadata, node.attributes)


for op in [
    "Prod",
    "Cumprod",
    "Nancumprod",
    "Nanprod",
    "Cummax",
    "Cummin",
    "Nanmax",
    "Nanmean",
    "Nanmedian",
    "Nanmin",
    "Nansum",
    "Nanpercentile",
    "Nanquantile",
    "Nanstd",
    "Nanvar",
    "Median",
    "Variance",
    "Std",
    "Percentile",
    "Quantile",
    "Descriptive",
    "Bincount",
]:
    register_vjp(op)(_zero_vjp)
    register_jvp(op)(_zero_jvp)


@register_vjp("Broadcast")
def broadcast_vjp(graph: Any, node: Any, cotangent: str) -> tuple:
    """VJP for Broadcast."""
    attrs = {"root_rank": node.attributes.get("root_rank", 0), "op_type": "sum"}
    res = emit_ir_node(graph, "Reduce", [cotangent], graph.nodes[node.inputs[0]].shape_metadata, attrs)
    return (res,)


@register_jvp("Broadcast")
def broadcast_jvp(graph: Any, node: Any, tangent: str) -> str:
    """JVP for Broadcast."""
    if tangent is None:
        return None
    return emit_ir_node(graph, "Broadcast", [tangent], node.shape_metadata, node.attributes)
