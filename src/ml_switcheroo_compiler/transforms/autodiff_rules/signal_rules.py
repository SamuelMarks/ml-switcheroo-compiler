# ruff: noqa: E402, D100, D103, D104, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, D101, D102, D107, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Autodiff rules for signal processing operations."""

from typing import Any

from ml_switcheroo_compiler.ops.base import emit_ir_node
from ml_switcheroo_compiler.transforms.autodiff_rules.common import make_zero_jvp, make_zero_vjp
from ml_switcheroo_compiler.transforms.autodiff_rules.jvp_registry import register_jvp
from ml_switcheroo_compiler.transforms.autodiff_rules.vjp_registry import register_vjp


@register_vjp("Dct")
def dct_vjp(graph: Any, node: Any, cotangent: str) -> tuple:
    """VJP for Dct.

    Args:
        graph (Any): The graph parameter.
        node (Any): The node parameter.
        cotangent (str): The cotangent parameter.

    Returns:
        tuple: Result.
    """
    res = emit_ir_node(graph, "Idct", [cotangent], graph.nodes[node.inputs[0]].shape_metadata, node.attributes)
    return (res,)


@register_jvp("Dct")
def dct_jvp(graph: Any, node: Any, tangents: tuple) -> str:
    """JVP for Dct.

    Args:
        graph (Any): The graph parameter.
        node (Any): The node parameter.
        tangents (tuple): The tangents parameter.

    Returns:
        str: Result.
    """
    t_x = tangents[0]
    if t_x is None:
        return ""
    return emit_ir_node(graph, "Dct", [t_x], node.shape_metadata, node.attributes)


@register_vjp("Idct")
def idct_vjp(graph: Any, node: Any, cotangent: str) -> tuple:
    """VJP for Idct.

    Args:
        graph (Any): The graph parameter.
        node (Any): The node parameter.
        cotangent (str): The cotangent parameter.

    Returns:
        tuple: Result.
    """
    res = emit_ir_node(graph, "Dct", [cotangent], graph.nodes[node.inputs[0]].shape_metadata, node.attributes)
    return (res,)


@register_jvp("Idct")
def idct_jvp(graph: Any, node: Any, tangents: tuple) -> str:
    """JVP for Idct.

    Args:
        graph (Any): The graph parameter.
        node (Any): The node parameter.
        tangents (tuple): The tangents parameter.

    Returns:
        str: Result.
    """
    t_x = tangents[0]
    if t_x is None:
        return ""
    return emit_ir_node(graph, "Idct", [t_x], node.shape_metadata, node.attributes)


@register_vjp("Frame")
def frame_vjp(graph: Any, node: Any, cotangent: str) -> tuple:
    """VJP for Frame.

    Args:
        graph (Any): The graph parameter.
        node (Any): The node parameter.
        cotangent (str): The cotangent parameter.

    Returns:
        tuple: Result.
    """
    res = emit_ir_node(graph, "OverlapAndAdd", [cotangent], graph.nodes[node.inputs[0]].shape_metadata, node.attributes)
    return (res,)


@register_jvp("Frame")
def frame_jvp(graph: Any, node: Any, tangents: tuple) -> str:
    """JVP for Frame.

    Args:
        graph (Any): The graph parameter.
        node (Any): The node parameter.
        tangents (tuple): The tangents parameter.

    Returns:
        str: Result.
    """
    t_x = tangents[0]
    if t_x is None:
        return ""
    return emit_ir_node(graph, "Frame", [t_x], node.shape_metadata, node.attributes)


@register_vjp("OverlapAndAdd")
def overlap_and_add_vjp(graph: Any, node: Any, cotangent: str) -> tuple:
    """VJP for OverlapAndAdd.

    Args:
        graph (Any): The graph parameter.
        node (Any): The node parameter.
        cotangent (str): The cotangent parameter.

    Returns:
        tuple: Result.
    """
    res = emit_ir_node(graph, "Frame", [cotangent], graph.nodes[node.inputs[0]].shape_metadata, node.attributes)
    return (res,)


@register_jvp("OverlapAndAdd")
def overlap_and_add_jvp(graph: Any, node: Any, tangents: tuple) -> str:
    """JVP for OverlapAndAdd.

    Args:
        graph (Any): The graph parameter.
        node (Any): The node parameter.
        tangents (tuple): The tangents parameter.

    Returns:
        str: Result.
    """
    t_x = tangents[0]
    if t_x is None:
        return ""
    return emit_ir_node(graph, "OverlapAndAdd", [t_x], node.shape_metadata, node.attributes)


@register_vjp("Mdct")
def mdct_vjp(graph: Any, node: Any, cotangent: str) -> tuple:
    """VJP for Mdct.

    Args:
        graph (Any): The graph parameter.
        node (Any): The node parameter.
        cotangent (str): The cotangent parameter.

    Returns:
        tuple: Result.
    """
    res = emit_ir_node(graph, "InverseMdct", [cotangent], graph.nodes[node.inputs[0]].shape_metadata, node.attributes)
    return (res,)


@register_jvp("Mdct")
def mdct_jvp(graph: Any, node: Any, tangents: tuple) -> str:
    """JVP for Mdct.

    Args:
        graph (Any): The graph parameter.
        node (Any): The node parameter.
        tangents (tuple): The tangents parameter.

    Returns:
        str: Result.
    """
    t_x = tangents[0]
    if t_x is None:
        return ""
    return emit_ir_node(graph, "Mdct", [t_x], node.shape_metadata, node.attributes)


@register_vjp("InverseMdct")
def inverse_mdct_vjp(graph: Any, node: Any, cotangent: str) -> tuple:
    """VJP for InverseMdct.

    Args:
        graph (Any): The graph parameter.
        node (Any): The node parameter.
        cotangent (str): The cotangent parameter.

    Returns:
        tuple: Result.
    """
    res = emit_ir_node(graph, "Mdct", [cotangent], graph.nodes[node.inputs[0]].shape_metadata, node.attributes)
    return (res,)


@register_jvp("InverseMdct")
def inverse_mdct_jvp(graph: Any, node: Any, tangents: tuple) -> str:
    """JVP for InverseMdct.

    Args:
        graph (Any): The graph parameter.
        node (Any): The node parameter.
        tangents (tuple): The tangents parameter.

    Returns:
        str: Result.
    """
    t_x = tangents[0]
    if t_x is None:
        return ""
    return emit_ir_node(graph, "InverseMdct", [t_x], node.shape_metadata, node.attributes)


for op_name in ["MelFilterbank", "MelSpectrogram", "Mfcc", "MfccsFromLogMelSpectrograms"]:
    register_vjp(op_name)(make_zero_vjp(op_name))
    register_jvp(op_name)(make_zero_jvp(op_name))
