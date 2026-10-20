"""Constant Folding pass evaluated via reference eager backend."""

from __future__ import annotations

import typing

from ml_switcheroo_ir import LogicalGraph, LogicalNode

from ml_switcheroo_compiler.backends.registry import BackendRegistry, EagerCallableProtocol, get_active_backend
from ml_switcheroo_compiler.core.shape import TensorOrScalar
from ml_switcheroo_compiler.interpreter import evaluate_graph
from ml_switcheroo_compiler.interpreter.evaluator import EvalValue
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
from ml_switcheroo_compiler.transforms.pass_manager import DAGTopologicalSorter


def _are_all_inputs_constant(canonical_inputs: list[str], graph: IRGraph) -> bool:
    """Check if all inputs to a node are compile-time known constants.

    Args:
        canonical_inputs (list[str]): List of canonical input IDs.
        graph (IRGraph): Computation graph containing nodes.

    Returns:
        bool: True if every input is an existing Constant node.
    """
    if not canonical_inputs:
        return False
    for inp in canonical_inputs:
        if inp not in graph.nodes or graph.nodes[inp].op_type != "Constant":
            return False
    return True


def _evaluate_constant_node(
    node: IRNode,
    canonical_inputs: list[str],
    graph: IRGraph,
    backend: type,
) -> EvalValue:
    """Evaluate a constant node using the reference eager backend.

    Args:
        node (IRNode): Primal IR node to fold.
        canonical_inputs (list[str]): Input node IDs.
        graph (IRGraph): Parent computation graph.
        backend (type): Reference eager backend generator.

    Returns:
        EvalValue: Evaluated concrete value.
    """
    subgraph = LogicalGraph(outputs=[node.id])
    for inp in canonical_inputs:
        subgraph.nodes[inp] = graph.nodes[inp]
    subgraph.nodes[node.id] = LogicalNode(
        id=node.id,
        op_type=node.op_type,
        attributes=dict(node.attributes),
        inputs=list(canonical_inputs),
        shape_metadata=node.shape_metadata,
    )
    outputs = evaluate_graph(subgraph, {}, backend=backend)
    val = outputs[node.id]
    if (hasattr(val, "size") and getattr(val, "size") == 1) or (hasattr(val, "numel") and getattr(val, "numel")() == 1):
        if hasattr(backend, "item"):
            val = getattr(backend, "item")(val)
        elif hasattr(val, "item"):
            val = getattr(val, "item")()
    return val


def _resolve_folding_backend() -> type | None:
    """Resolve the active or fallback NumPy backend for constant evaluation.

    Returns:
        type | None: Backend class or None if unavailable.
    """
    backend: type | None = get_active_backend()
    if backend is None:
        try:
            backend = BackendRegistry.get("numpy")
        except Exception:
            backend = None
    return backend


def _try_fold_single_node(
    node: IRNode,
    canonical_inputs: list[str],
    graph: IRGraph,
    backend: type | None,
) -> bool:
    """Attempt to fold a single node with constant inputs into a Constant node.

    Args:
        node (IRNode): Node to evaluate and fold.
        canonical_inputs (list[str]): Resolved input node IDs.
        graph (IRGraph): Containing computation graph.
        backend (type | None): Reference eager backend.

    Returns:
        bool: True if the node was folded into a Constant, False otherwise.
    """
    import logging

    if node.op_type == "Constant" or not _are_all_inputs_constant(canonical_inputs, graph):
        return False

    try:
        from ml_switcheroo_ir import AttributeValue

        val = typing.cast("EagerCallableProtocol", _evaluate_constant_node)(node, canonical_inputs, graph, backend)
        graph.nodes[node.id].op_type = "Constant"
        graph.nodes[node.id].attributes = typing.cast("dict[str, AttributeValue]", {"value": val})
        graph.nodes[node.id].inputs = []
        return True
    except (ValueError, TypeError, RuntimeError) as e:
        logging.getLogger(__name__).debug(f"Failed to constant fold {node.op_type}: {e}")
        return False
    except Exception as e:
        if type(e).__name__ == "UnimplementedMathError":
            logging.getLogger(__name__).debug(f"Unimplemented math for constant folding {node.op_type}: {e}")
            return False
        raise


def _fold_nested_subgraphs(graph: IRGraph) -> bool:
    """Recursively run constant folding on nested subgraphs in nodes and attributes.

    Args:
        graph (IRGraph): Parent computation graph.

    Returns:
        bool: True if any nested subgraph was modified, False otherwise.
    """
    modified = False
    for node in graph.nodes.values():
        for sub in getattr(node, "subgraphs", {}).values():
            if isinstance(sub, LogicalGraph) and constant_folding_pass(sub):
                modified = True
        for attr_val in typing.cast("dict[str, TensorOrScalar]", node.attributes).values():
            if isinstance(attr_val, LogicalGraph) and constant_folding_pass(attr_val):
                modified = True
    return modified


def constant_folding_pass(graph: IRGraph) -> bool:
    """In-place Constant Folding evaluated via reference eager backend.

    Evaluates compile-time known constant nodes until fixpoint is reached.

    Args:
        graph (IRGraph): The graph parameter.

    Returns:
        bool: True if any nodes were folded into constants.
    """
    total_modified = False
    pass_modified = True

    while pass_modified:
        pass_modified = False
        sorted_nodes = DAGTopologicalSorter.sort(graph)
        id_map: dict[str, str] = {}
        backend = _resolve_folding_backend()

        for node in sorted_nodes:
            canonical_inputs = [id_map.get(inp, inp) for inp in node.inputs]
            if _try_fold_single_node(node, canonical_inputs, graph, backend):
                pass_modified = True
                total_modified = True
            id_map[node.id] = node.id

    if _fold_nested_subgraphs(graph):
        total_modified = True

    return total_modified
