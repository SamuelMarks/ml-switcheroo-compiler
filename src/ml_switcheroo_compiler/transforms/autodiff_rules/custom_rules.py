# ruff: noqa: E402, D100, D103, D104, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, D101, D102, D107, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Autodiff rules for custom and platform-specific kernel operations."""

from typing import Any

from ml_switcheroo_compiler.transforms.autodiff_rules.common import make_zero_jvp, make_zero_vjp
from ml_switcheroo_compiler.transforms.autodiff_rules.jvp_registry import register_jvp
from ml_switcheroo_compiler.transforms.autodiff_rules.vjp_registry import register_vjp

for op_name in [
    "CudaKernel",
    "MetalKernel",
    "PrecompiledCudaKernel",
    "TopK",
    "Cholesky",
    "CholeskyEx",
    "Eig",
    "Eigh",
    "Eigvals",
    "Eigvalsh",
    "FFT",
    "IFFT",
    "Sort",
    "SortComplex",
    "SortKeyVal",
    "Argsort",
    "Fftconvolve",
    "Fft",
    "Rfft",
    "Fft2",
    "Fftfreq",
    "Irfft",
    "Ihfft",
    "Ifft",
    "Fftn",
    "Ifftn",
    "Rfftn",
    "Irfftn",
    "Ifft2",
    "Rfft2",
    "Irfft2",
    "Fftnd",
    "Ifftnd",
    "Rfftnd",
    "Irfftnd",
    "Fftshift",
    "Ifftshift",
    "Hfft",
    "Rfftfreq",
]:
    register_vjp(op_name)(make_zero_vjp(op_name))
    register_jvp(op_name)(make_zero_jvp(op_name))


def _inline_subgraph(graph: Any, subgraph: Any, node: Any, id_map: dict[str, str]) -> None:
    """Inline a subgraph into the main graph.

    Args:
        graph (Any): The graph parameter.
        subgraph (Any): The subgraph parameter.
        node (Any): The node parameter.
        id_map (dict): The id_map parameter.
    """
    from ml_switcheroo_ir import LogicalGraph

    from ml_switcheroo_compiler.ir.core import clone_logical_node

    new_subgraph = LogicalGraph(name="cp_fwd_inline")
    nodes_to_process = subgraph.nodes if isinstance(subgraph.nodes, list) else subgraph.nodes.values()

    for n in nodes_to_process:
        if n.op_type == "Input":
            continue
        new_n = clone_logical_node(n)
        new_n.id = id_map[n.id]
        new_n.inputs = [id_map.get(inp, inp) for inp in n.inputs]
        new_subgraph.nodes[new_n.id] = new_n
        graph.nodes[new_n.id] = new_n


def _inline_grad_subgraph(graph: Any, sg_grad: Any, sg: Any, node: Any, cotangent_mapping: dict[str, str]) -> list[str]:
    """Inline the gradient subgraph into the main graph.

    Args:
        graph (Any): The graph parameter.
        sg_grad (Any): The sg_grad parameter.
        sg (Any): The sg parameter.
        node (Any): The node parameter.
        cotangent_mapping (Any): The cotangent_mapping parameter.

    Returns: Any: Result.
    """
    import uuid

    from ml_switcheroo_compiler.ir.core import clone_logical_node

    grad_id_map = {}
    for n in sg_grad.nodes.values():
        grad_id_map[n.id] = f"cp_bwd_{n.id}_{uuid.uuid4().hex[:6]}"

    for in_id, orig_in_id in zip(sg.inputs, node.inputs):
        grad_id_map[in_id] = orig_in_id

    for _, cot in cotangent_mapping.items():
        grad_id_map[cot] = cot

    for n in sg_grad.nodes.values():
        if n.op_type in ("Input", "Output") or n.id in cotangent_mapping.values():
            continue
        new_n = clone_logical_node(n)
        new_n.id = grad_id_map.get(n.id, n.id)
        new_n.inputs = [grad_id_map.get(inp, inp) for inp in n.inputs]
        graph.nodes[new_n.id] = new_n

    return [grad_id_map.get(out_id, out_id) for out_id in sg_grad.outputs]


@register_vjp("Checkpoint")
def checkpoint_vjp(graph: Any, node: Any, cotangent: str) -> tuple:
    """VJP for Checkpoint operation.

    Args:
        graph (Any): The graph parameter.
        node (Any): The node parameter.
        cotangent (str): The cotangent parameter.

    Returns:
        tuple: Result.
    """
    import uuid

    from ml_switcheroo_ir import LogicalGraph

    from ml_switcheroo_compiler.ir.core import clone_logical_node
    from ml_switcheroo_compiler.transforms.autodiff import grad as graph_grad

    subgraph = node.attributes["subgraph"]
    nodes_list = subgraph.nodes if isinstance(subgraph.nodes, list) else subgraph.nodes.values()

    id_map = {n.id: f"cp_fwd_{n.id}_{uuid.uuid4().hex[:6]}" for n in nodes_list}
    for in_id, orig_in_id in zip(subgraph.inputs, node.inputs):
        id_map[in_id] = orig_in_id

    _inline_subgraph(graph, subgraph, node, id_map)

    sg = LogicalGraph(name="cp_sg")
    for n in nodes_list:
        sg.nodes[n.id] = clone_logical_node(n)
    sg.inputs = subgraph.inputs
    sg.outputs = subgraph.outputs

    cotangent_mapping = {sg.outputs[0]: cotangent}

    sg_grad = graph_grad(sg, wrt=sg.inputs, output_id=sg.outputs[0], cotangent_id=cotangent)

    adjoints = _inline_grad_subgraph(graph, sg_grad, sg, node, cotangent_mapping)
    return tuple(adjoints)


@register_vjp("If")
def _if_vjp(graph: Any, node: Any, cotangent: str) -> tuple:
    """VJP for If operation.

    Args:
        graph (Any): The graph parameter.
        node (Any): The node parameter.
        cotangent (str): The cotangent parameter.

    Returns:
        tuple: Result.
    """
    from ml_switcheroo_compiler.transforms.autodiff_rules.common import UnconnectedGradients

    return (UnconnectedGradients.ZERO,)


@register_vjp("Loop")
def _loop_vjp(graph: Any, node: Any, cotangent: str) -> tuple:
    """VJP for Loop operation.

    Args:
        graph (Any): The graph parameter.
        node (Any): The node parameter.
        cotangent (str): The cotangent parameter.

    Returns:
        tuple: Result.
    """
    from ml_switcheroo_compiler.transforms.autodiff_rules.common import UnconnectedGradients

    return tuple(UnconnectedGradients.ZERO for _ in node.inputs)


@register_vjp("Scan")
def _scan_vjp(graph: Any, node: Any, cotangent: str) -> tuple:
    """VJP for Scan operation.

    Args:
        graph (Any): The graph parameter.
        node (Any): The node parameter.
        cotangent (str): The cotangent parameter.

    Returns:
        tuple: Result.
    """
    from ml_switcheroo_compiler.transforms.autodiff_rules.common import UnconnectedGradients

    return tuple(UnconnectedGradients.ZERO for _ in node.inputs)


@register_vjp("AssociativeScan")
def _assoc_scan_vjp(graph: Any, node: Any, cotangent: str) -> tuple:
    """VJP for AssociativeScan operation.

    Args:
        graph (Any): The graph parameter.
        node (Any): The node parameter.
        cotangent (str): The cotangent parameter.

    Returns:
        tuple: Result.
    """
    from ml_switcheroo_compiler.transforms.autodiff_rules.common import UnconnectedGradients

    return tuple(UnconnectedGradients.ZERO for _ in node.inputs)


@register_jvp("If")
def _if_jvp(graph: Any, node: Any, tangents: list) -> str:
    """JVP for If operation.

    Args:
        graph (Any): The graph parameter.
        node (Any): The node parameter.
        tangents (list): The tangents parameter.

    Returns:
        str: Result.
    """
    return ""


@register_jvp("Loop")
def _loop_jvp(graph: Any, node: Any, tangents: list) -> str:
    """JVP for Loop operation.

    Args:
        graph (Any): The graph parameter.
        node (Any): The node parameter.
        tangents (list): The tangents parameter.

    Returns:
        str: Result.
    """
    return ""


@register_jvp("Scan")
def _scan_jvp(graph: Any, node: Any, tangents: list) -> str:
    """JVP for Scan operation.

    Args:
        graph (Any): The graph parameter.
        node (Any): The node parameter.
        tangents (list): The tangents parameter.

    Returns:
        str: Result.
    """
    return ""


@register_jvp("AssociativeScan")
def _assoc_scan_jvp(graph: Any, node: Any, tangents: list) -> str:
    """JVP for AssociativeScan operation.

    Args:
        graph (Any): The graph parameter.
        node (Any): The node parameter.
        tangents (list): The tangents parameter.

    Returns:
        str: Result.
    """
    return ""
