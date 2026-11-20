"""Graph-level mutable state environment tracking and functionalization (State Layer 2).

Provides pure IR graph transformations that rewrite ``Assign`` state mutations
into explicit SSA dataflow inputs and outputs without upward dependencies on
higher-level pass orchestration modules.
"""

from __future__ import annotations

from ml_switcheroo_ir import LogicalGraph, LogicalNode

from ml_switcheroo_compiler.ir.state import ASSIGN_OP


def _process_assign_node(node: LogicalNode, state_env: dict[str, str]) -> None:
    """Process an ``Assign`` node by updating the state environment mapping.

    Args:
        node (LogicalNode): The ``Assign`` node to process.
        state_env (dict[str, str]): Mapping of state variable IDs to their latest value node IDs.
    """
    if len(node.inputs) >= 2 and node.inputs[0] in state_env:
        state_env[node.inputs[0]] = node.inputs[1]


def _rewrite_node(node: LogicalNode, state_env: dict[str, str]) -> LogicalNode:
    """Clone ``node`` with its input references substituted from ``state_env``.

    Args:
        node (LogicalNode): The node to rewrite.
        state_env (dict[str, str]): Active state variable mapping environment.

    Returns:
        LogicalNode: Cloned node referencing current SSA state values.
    """
    new_inputs = [state_env.get(inp, inp) for inp in node.inputs]
    return LogicalNode(
        id=node.id,
        op_type=node.op_type,
        domain=node.domain,
        version=node.version,
        attributes=dict(node.attributes),
        inputs=new_inputs,
        shape_metadata=node.shape_metadata,
        source_ast_ref=node.source_ast_ref,
        sharding=node.sharding,
    )


def _build_functional_outputs(graph_outputs: list[str], state_vars: list[str], state_env: dict[str, str]) -> list[str]:
    """Append final mutated state variable node IDs after ``graph_outputs``.

    Args:
        graph_outputs (list[str]): Original graph output node IDs.
        state_vars (list[str]): Tracked initial state variable node IDs.
        state_env (dict[str, str]): Final state variable mapping environment.

    Returns:
        list[str]: Combined list of original outputs followed by updated state outputs.
    """
    return [*graph_outputs, *(state_env[v] for v in state_vars)]


def lift_state(graph: LogicalGraph, state_vars: list[str]) -> LogicalGraph:
    """Rewrite a graph containing ``Assign`` mutations into a pure functional graph.

    Args:
        graph (LogicalGraph): Input graph containing ``Assign`` nodes.
        state_vars (list[str]): Node IDs representing tracked initial state variables.

    Returns:
        LogicalGraph: Functionalized graph with updated state outputs appended.
    """
    new_graph = LogicalGraph(name=f"{graph.name}_functional", mesh=graph.mesh)
    ordered_nodes = graph.sorted_nodes if graph.sorted_nodes is not None else list(graph.nodes.values())
    state_env: dict[str, str] = {v: v for v in state_vars}

    for node in ordered_nodes:
        if node.op_type == ASSIGN_OP:
            _process_assign_node(node, state_env)
        else:
            new_graph.nodes[node.id] = _rewrite_node(node, state_env)

    new_graph.outputs = _build_functional_outputs(graph.outputs, state_vars, state_env)
    return new_graph


__all__ = [
    "_build_functional_outputs",
    "_process_assign_node",
    "_rewrite_node",
    "lift_state",
]
