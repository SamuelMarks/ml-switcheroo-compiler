"""Forward-mode ``jvp`` and forward-over-reverse ``hvp`` graph transformation APIs."""

from __future__ import annotations

import uuid
from collections.abc import Callable, Sequence  # noqa: F401

from ml_switcheroo_ir import IRGraph, LogicalGraph, LogicalNode, topological_sort

import ml_switcheroo_compiler.transforms.autodiff as _ad_mod
from ml_switcheroo_compiler.transforms._autodiff_jvp_core import _process_jvp_node
from ml_switcheroo_compiler.transforms._autodiff_vjp_core import _copy_graph


def _forward_pass_jvp(
    new_graph: IRGraph,
    sorted_nodes: Sequence[LogicalNode],
    tangents: dict[str, str],
) -> None:
    """Perform the forward pass to compute JVP.

    Args:
        new_graph (IRGraph): The target IR graph.
        sorted_nodes (Sequence[LogicalNode]): Topologically sorted list of IR nodes.
        tangents (dict[str, str]): Mapping of node IDs to tangent node IDs.
    """
    for node in sorted_nodes:
        _process_jvp_node(new_graph, node, tangents)


def jvp(graph: LogicalGraph, primals: list[str], tangents: list[str], outputs: list[str]) -> LogicalGraph:
    """Execute jvp for compiler processing.

    Args:
        graph (LogicalGraph): Input `graph` argument (graph).
        primals (list[str]): Input `primals` argument (primals).
        tangents (list[str]): Input `tangents` argument (tangents).
        outputs (list[str]): Input `outputs` argument (outputs).

    Returns:
        LogicalGraph: Computed `LogicalGraph` output.

    Raises:
        ValueError: If invalid inputs or unsupported runtime conditions trigger `ValueError`.
    """
    if len(primals) != len(tangents):
        raise ValueError("primals and tangents must have the same length")

    for out_id in outputs:
        if out_id not in graph.nodes:
            raise ValueError(f"Output node '{out_id}' not found in graph.")

    new_graph = _copy_graph(graph)
    sorted_nodes = topological_sort(new_graph)

    tangents_map: dict[str, str] = dict(zip(primals, tangents))

    _ad_mod._forward_pass_jvp(new_graph, sorted_nodes, tangents_map)

    # Extract required output tangents
    out_tangents = []
    for out in outputs:
        if out in tangents_map:
            out_tangents.append(tangents_map[out])
        else:
            zero_id = f"jvp_zeros_{uuid.uuid4().hex[:6]}"
            zeros_node = LogicalNode(
                id=zero_id,
                op_type="Constant",
                attributes={"value": 0.0},
                shape_metadata=new_graph.nodes[out].shape_metadata,
            )
            new_graph.nodes[zero_id] = zeros_node
            out_tangents.append(zero_id)

    new_graph.outputs = out_tangents
    return new_graph


def hvp(
    graph: LogicalGraph,
    primals: list[str],
    tangents: list[str],
    outputs: list[str],
    mode: str = "forward-over-reverse",
    projected_tangents: list[str] | None = None,
) -> LogicalGraph:
    """Evaluate Hessian-vector product (HVP) operation.

    Supports both forward-over-reverse and reverse-over-forward modes, and seamlessly
    handles multi-dimensional and non-scalar outputs via projected tangents.

    Args:
        graph (LogicalGraph): Base computation graph.
        primals (list[str]): Input primal variable node IDs.
        tangents (list[str]): Tangent vector node IDs corresponding to primals.
        outputs (list[str]): Output node IDs to differentiate.
        mode (str): HVP computation mode ('forward-over-reverse' or 'reverse-over-forward').
        projected_tangents (list[str] | None): Optional projected tangent node IDs
            for projecting non-scalar or multi-node outputs into a directional scalar/vector.

    Returns:
        LogicalGraph: Graph computing the Hessian-vector product.

    Raises:
        ValueError: If input arguments, shapes, or configuration values in `hvp` are invalid.
    """
    if mode == "forward-over-reverse":
        if projected_tangents is not None:
            cot_dict = {out_id: pt for out_id, pt in zip(outputs, projected_tangents)}
            grad_graph = _ad_mod.grad(graph, primals, outputs, cotangent_id=cot_dict)
        else:
            if len(outputs) != 1:
                raise ValueError(
                    "hvp requires a single output node when using forward-over-reverse mode without projected_tangents"
                )
            grad_graph = _ad_mod.grad(graph, primals, outputs[0])
        grad_outputs = grad_graph.outputs
        return _ad_mod.jvp(grad_graph, primals, tangents, grad_outputs)
    elif mode == "reverse-over-forward":
        jvp_graph = _ad_mod.jvp(graph, primals, tangents, outputs)
        jvp_outputs = jvp_graph.outputs
        if projected_tangents is not None:
            cot_dict = {out_id: pt for out_id, pt in zip(jvp_outputs, projected_tangents)}
            return _ad_mod.grad(jvp_graph, primals, jvp_outputs, cotangent_id=cot_dict)
        else:
            if len(jvp_outputs) != 1:
                raise ValueError(
                    "hvp requires a single output node when using reverse-over-forward mode without projected_tangents"
                )
            return _ad_mod.grad(jvp_graph, primals, jvp_outputs[0])
    else:
        raise ValueError(f"Unknown HVP mode: {mode}")
