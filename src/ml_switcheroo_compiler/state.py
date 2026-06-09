"""State functionalization pass."""

from typing import Dict, List
from ml_switcheroo_ir import LogicalGraph, LogicalNode, topological_sort


def lift_state(graph: LogicalGraph, state_vars: List[str]) -> LogicalGraph:
    """Lifts mutable state into pure functional boundaries.

    In frameworks like Flax or PyTorch, models have mutable state (parameters, batch stats).
    This pass assumes that mutable updates are represented as specific nodes
    (e.g., 'Assign' or 'UpdateState' nodes). It rewrites the graph so that
    state variables are treated as explicit inputs and their updated values
    are explicit outputs.

    Args:
        graph (LogicalGraph): The input graph containing state nodes.
        state_vars (List[str]): The IDs of nodes representing initial state variables.

    Returns:
        LogicalGraph: The functionalized graph.
    """
    new_graph = LogicalGraph(name=f"{graph.name}_functional", mesh=graph.mesh)

    sorted_nodes = topological_sort(graph)

    # Map from original node ID to the current latest state node ID
    state_env: Dict[str, str] = {v: v for v in state_vars}

    for node in sorted_nodes:
        if node.op_type == "Assign":
            # Assign expects [target_state, new_value]
            # We don't emit Assign in functional graph, instead we just update the environment
            target = node.inputs[0]
            new_val = node.inputs[1]
            # Trace target back to original state var if it's an alias
            # (assuming simple direct updates for now)
            if target in state_env:
                state_env[target] = new_val
            continue

        # For all other nodes, rewrite inputs using latest state
        new_inputs = []
        for inp in node.inputs:
            # If input is a state variable, use its latest version
            # (Note: standard read of a state var gets mapped here)
            new_inputs.append(state_env.get(inp, inp))

        new_graph.nodes[node.id] = LogicalNode(
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

    # The new outputs are the original outputs + the updated state variables
    functional_outputs = list(graph.outputs)
    for v in state_vars:
        functional_outputs.append(state_env[v])

    new_graph.outputs = functional_outputs
    return new_graph
