# ruff: noqa: E402, D100, D103, D104, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, D101, D102, D107, E701, E722, F403, E711, E712, PLR0913, PLR0915
from typing import Any

"""State functionalization pass."""

from ml_switcheroo_ir import LogicalGraph, LogicalNode, topological_sort


def _process_assign_node(node: LogicalNode, state_env: dict[str, str]) -> None:
    """Process an Assign node by updating the state environment mapping.

    Args:
        node (LogicalNode): The Assign node to process.
        state_env (dict[str, str]): The mapping of state variables to their current node IDs.
    """
    target = node.inputs[0]
    new_val = node.inputs[1]
    if target in state_env:
        state_env[target] = new_val


def _rewrite_node(node: LogicalNode, state_env: dict[str, str]) -> LogicalNode:
    """Rewrite a node by substituting its inputs with their current state mapped IDs.

    Args:
        node (LogicalNode): The node to rewrite.
        state_env (dict[str, str]): The state mapping environment.

    Returns:
        LogicalNode: The new cloned node with updated inputs.
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
    """Build the updated list of outputs containing original outputs and updated states.

    Args:
        graph_outputs (list[str]): The original graph outputs.
        state_vars (list[str]): The initial state variables.
        state_env (dict[str, str]): The final state mapping environment.

    Returns:
        list[str]: The new functional outputs list.
    """
    functional_outputs = list(graph_outputs)
    for v in state_vars:
        functional_outputs.append(state_env[v])
    return functional_outputs


def lift_state(graph: LogicalGraph, state_vars: list[str]) -> LogicalGraph:
    """Lift mutable state into pure functional boundaries.

    In frameworks like Flax or PyTorch, models have mutable state (parameters, batch stats).
    This pass assumes that mutable updates are represented as specific nodes (e.g., 'Assign').
    It rewrites the graph so that state variables are treated as explicit inputs and their updated
    values are explicit outputs.

    Args:
        graph (LogicalGraph): The input graph containing state nodes.
        state_vars (list[str]): The IDs of nodes representing initial state variables.

    Returns:
        LogicalGraph: The functionalized graph.
    """
    new_graph = LogicalGraph(name=f"{graph.name}_functional", mesh=graph.mesh)
    sorted_nodes = topological_sort(graph)
    state_env: dict[str, str] = {v: v for v in state_vars}

    for node in sorted_nodes:
        if node.op_type == "Assign":
            _process_assign_node(node, state_env)
        else:
            new_graph.nodes[node.id] = _rewrite_node(node, state_env)

    new_graph.outputs = _build_functional_outputs(graph.outputs, state_vars, state_env)
    return new_graph
