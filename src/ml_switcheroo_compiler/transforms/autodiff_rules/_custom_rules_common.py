"""Subgraph inlining, metadata extraction, and zero-gradient registrations for custom AD rules."""

from __future__ import annotations

import typing
from collections.abc import Mapping, Sequence

from ml_switcheroo_ir import AttributeValue, IRGraph, IRNode, LogicalGraph

from ml_switcheroo_compiler.core.shape import OpValue
from ml_switcheroo_compiler.transforms.autodiff_rules.common import (
    make_zero_jvp,
    make_zero_vjp,
)
from ml_switcheroo_compiler.transforms.autodiff_rules.jvp_registry import register_jvp
from ml_switcheroo_compiler.transforms.autodiff_rules.vjp_registry import register_vjp


def _make_node_attrs(
    items: Mapping[str, LogicalGraph | AttributeValue],
) -> dict[str, AttributeValue]:
    """Construct a node attributes dictionary containing `LogicalGraph` or `AttributeValue` entries.

    Args:
        items (Mapping[str, LogicalGraph | AttributeValue]): Key-value pairs for node attributes.

    Returns:
        dict[str, AttributeValue]: Populated attributes dictionary.
    """
    return {str(k): v for k, v in items.items()}


def _coerce_node_attrs(raw: AttributeValue | LogicalGraph) -> dict[str, AttributeValue]:
    """Coerce a raw attribute value into a `dict[str, AttributeValue]`.

    Args:
        raw (AttributeValue | LogicalGraph): Raw attributes dictionary or value.

    Returns:
        dict[str, AttributeValue]: Coerced attributes dictionary.
    """
    return {str(k): v for k, v in raw.items()} if isinstance(raw, dict) else {}


def _is_subgraph_branch(val: AttributeValue | LogicalGraph) -> typing.TypeGuard[LogicalGraph]:
    """Return True if `val` is a `LogicalGraph` subgraph branch.

    Args:
        val (AttributeValue | LogicalGraph): Candidate branch attribute value.

    Returns:
        bool: True if `val` is a `LogicalGraph`.
    """
    return isinstance(val, LogicalGraph)


@typing.runtime_checkable
class _StructuralSubgraphProtocol(typing.Protocol):
    """Structural protocol for subgraph containers exposing `inputs`, `outputs`, and `nodes`."""

    inputs: list[str]
    outputs: list[str]
    nodes: Sequence[IRNode] | Mapping[str, IRNode]


def _is_subgraph_with_nodes(
    val: AttributeValue | LogicalGraph,
) -> typing.TypeGuard[LogicalGraph | _StructuralSubgraphProtocol]:
    """Return True if `val` is a `LogicalGraph` or structural subgraph container with nodes.

    Args:
        val (AttributeValue | LogicalGraph): Candidate subgraph value.

    Returns:
        bool: True if `val` is a `LogicalGraph` or `_StructuralSubgraphProtocol`.
    """
    return isinstance(val, (LogicalGraph, _StructuralSubgraphProtocol))


def _extract_graph_nodes(nodes: Mapping[str, IRNode] | Sequence[IRNode]) -> list[IRNode]:
    """Extract a flat list of `IRNode` instances from a graph `nodes` mapping or sequence.

    Args:
        nodes (Mapping[str, IRNode] | Sequence[IRNode]): Graph nodes container.

    Returns:
        list[IRNode]: Flat list of IRNodes.
    """
    if isinstance(nodes, Mapping):
        return list(nodes.values())
    return list(nodes)


def _node_shape_meta(graph: IRGraph, node_id: str) -> tuple[int, ...]:
    """Extract integer shape metadata for `node_id` in `graph` if present.

    Args:
        graph (IRGraph): Computation graph.
        node_id (str): Target node identifier.

    Returns:
        tuple[int, ...]: Shape tuple or empty tuple.
    """
    target = graph.nodes.get(node_id)
    if target is None or target.shape_metadata is None:
        return ()
    return tuple(int(d) for d in target.shape_metadata if isinstance(d, int))


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


def _inline_subgraph(
    graph: IRGraph,
    subgraph: IRGraph | _StructuralSubgraphProtocol,
    node: OpValue,
    id_map: dict[str, str],
) -> None:
    """Inline a subgraph into the main graph.

    Args:
        graph (IRGraph): Input `graph` argument (graph).
        subgraph (IRGraph | _StructuralSubgraphProtocol): Input `subgraph` argument (subgraph).
        node (OpValue): Input `node` argument (node).
        id_map (dict[str, str]): Input `id_map` argument (id map).
    """
    _ = node

    from ml_switcheroo_compiler.ir.core import clone_logical_node

    new_subgraph = LogicalGraph(name="cp_fwd_inline")
    nodes_to_process: list[IRNode] = _extract_graph_nodes(subgraph.nodes)

    for n in nodes_to_process:
        if n.op_type == "Input":
            continue
        new_n = clone_logical_node(n)
        new_n.id = id_map[n.id]
        new_n.inputs = [id_map.get(inp, inp) for inp in n.inputs]
        new_subgraph.nodes[new_n.id] = new_n
        graph.nodes[new_n.id] = new_n


def _inline_grad_subgraph(
    graph: IRGraph,
    sg_grad: IRGraph,
    sg: IRGraph,
    node: IRNode,
    cotangent_mapping: dict[str, str],
) -> list[str]:
    """Inline the gradient subgraph into the main graph.

    Args:
        graph (IRGraph): Input `graph` argument (graph).
        sg_grad (IRGraph): Input `sg_grad` argument (sg grad).
        sg (IRGraph): Input `sg` argument (sg).
        node (IRNode): Input `node` argument (node).
        cotangent_mapping (dict[str, str]): Input `cotangent_mapping` argument (cotangent mapping).

    Returns:
        list[str]: Inlined output gradient node IDs.
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
