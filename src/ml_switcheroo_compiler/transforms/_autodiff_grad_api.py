"""Reverse-mode ``grad`` graph transformation entry point and output extraction."""

from __future__ import annotations

import uuid
from collections.abc import Callable, Sequence  # noqa: F401

from ml_switcheroo_ir import IRGraph, LogicalGraph, LogicalNode, topological_sort

import ml_switcheroo_compiler.transforms.autodiff as _ad_mod
from ml_switcheroo_compiler.transforms._autodiff_vjp_core import (
    _copy_graph,
    _get_reachable_from_output,
)


def _extract_gradients(
    new_graph: LogicalGraph,
    wrt: list[str],
    adjoints: dict[str, str],
) -> list[str]:
    """Extract the required gradients.

    Args:
        new_graph (LogicalGraph): The new_graph parameter for the operation.
        wrt (list[str]): Target nodes.
        adjoints (dict[str, str]): Adjoints map.

    Returns:
        list[str]: Output gradient node IDs.

    Raises:
        ValueError: If a target node is not found.
    """
    grad_outputs = []
    for w in wrt:
        if w not in new_graph.nodes:
            msg = f"Target node '{w}' not found in graph."
            raise ValueError(msg)

        if w in adjoints:
            grad_outputs.append(adjoints[w])
        else:
            zero_id = f"grad_zeros_{uuid.uuid4().hex[:6]}"
            zeros_node = LogicalNode(
                id=zero_id,
                op_type="Constant",
                attributes={"value": 0.0},
                shape_metadata=new_graph.nodes[w].shape_metadata,
            )
            new_graph.nodes[zero_id] = zeros_node
            grad_outputs.append(zero_id)
    return grad_outputs


def _maybe_vectorize_graph(graph: LogicalGraph) -> LogicalGraph:
    """Lower Vmap nodes in the graph before differentiation if any exist.

    Args:
        graph (LogicalGraph): The input graph.

    Returns:
        LogicalGraph: Graph with Vmap nodes vectorized.
    """
    if not any(n.op_type == "Vmap" for n in graph.nodes.values()):
        return graph

    from ml_switcheroo_ir import IRNode

    from ml_switcheroo_compiler.transforms.passes.vectorization import vectorization_pass

    ir_g = IRGraph(name=graph.name)
    ir_g.inputs = list(graph.inputs)
    ir_g.outputs = list(graph.outputs)
    for nid, n in graph.nodes.items():
        ir_g.nodes[nid] = n if type(n) is IRNode else IRNode(**n.model_dump())
    return vectorization_pass(ir_g)


def _init_adjoints(
    new_graph: LogicalGraph,
    output_ids: list[str],
    cotangent_id: str | Sequence[str] | dict[str, str] | None,
) -> dict[str, str]:
    """Initialize output adjoints for reverse-mode autodiff.

    Args:
        new_graph (LogicalGraph): The gradient computation graph.
        output_ids (list[str]): Target output node IDs.
        cotangent_id (str | Sequence[str] | dict[str, str] | None): Optional seed cotangent(s).

    Returns:
        dict[str, str]: Initial mapping of node IDs to adjoint node IDs.
    """
    adjoints: dict[str, str] = {}
    if cotangent_id is not None:
        if isinstance(cotangent_id, dict):
            adjoints.update(cotangent_id)
        elif isinstance(cotangent_id, str):
            for oid in output_ids:
                adjoints[oid] = cotangent_id
        else:
            for oid, cid in zip(output_ids, cotangent_id):
                adjoints[oid] = cid
        return adjoints

    for oid in output_ids:
        one_id = f"grad_ones_{uuid.uuid4().hex[:6]}"
        target_node = new_graph.nodes[oid]
        target_shape = target_node.shape_metadata
        ones_node = LogicalNode(
            id=one_id,
            op_type="Constant",
            attributes={"value": 1.0},
            shape_metadata=target_shape,
        )
        new_graph.nodes[one_id] = ones_node
        adjoints[oid] = one_id
    return adjoints


def grad(
    graph: LogicalGraph,
    wrt: list[str],
    output_id: str | Sequence[str],
    cotangent_id: str | Sequence[str] | dict[str, str] | None = None,
) -> LogicalGraph:
    """Execute grad for compiler processing.

    Args:
        graph (LogicalGraph): Input `graph` argument (graph).
        wrt (list[str]): Input `wrt` argument (wrt).
        output_id (str | Sequence[str]): Input `output_id` argument (output id).
        cotangent_id (str | Sequence[str] | dict[str, str] | None): Input `cotangent_id` argument (cotangent id).

    Returns:
        LogicalGraph: Computed `LogicalGraph` output.

    Raises:
        ValueError: If invalid inputs or unsupported runtime conditions trigger `ValueError`.
    """
    graph = _maybe_vectorize_graph(graph)

    output_ids: list[str] = [output_id] if isinstance(output_id, str) else list(output_id)
    for oid in output_ids:
        if oid not in graph.nodes:
            msg = f"Output node '{oid}' not found in graph."
            raise ValueError(msg)

    new_graph = _copy_graph(graph)
    sorted_nodes = topological_sort(new_graph)

    reachable_from_output: set[str] = set()
    for oid in output_ids:
        reachable_from_output.update(_get_reachable_from_output(sorted_nodes, oid))

    adjoints = _init_adjoints(new_graph, output_ids, cotangent_id)
    _ad_mod._backward_pass(new_graph, sorted_nodes, reachable_from_output, adjoints)

    new_graph.outputs = _extract_gradients(new_graph, wrt, adjoints)
    return new_graph
