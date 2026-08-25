# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Common Subexpression Elimination pass."""

from ml_switcheroo_compiler.ir.core import IRGraph
from ml_switcheroo_compiler.transforms.pass_manager import DAGTopologicalSorter


def _compute_node_signature(node: object, canonical_inputs: list[str]) -> str:
    """Evaluate _compute_node_signature operation.

    Args:
        node (object): The node parameter.
        canonical_inputs (object): The canonical_inputs parameter.

    Returns:
        str: Result.
    """
    if node.op_type == "Input":
        return f"Input_{node.id}"
    attr_list: object = []
    for k, v in node.attributes.items():
        attr_list.append((k, str(v)))
    attr_str: object = str(sorted(attr_list))
    return f"{node.op_type}|{canonical_inputs}|{attr_str}"


def cse_pass(graph: IRGraph) -> bool:
    """In-place Common Subexpression Elimination (CSE).

    Args:
        graph (IRGraph): The graph parameter.

    Returns:
        bool: Result.
    """
    modified: object = False
    seen_expressions: dict[str, str] = {}
    id_map: dict[str, str] = {}

    sorted_nodes: object = DAGTopologicalSorter.sort(graph)

    for node in sorted_nodes:
        # Map inputs to canonical inputs
        canonical_inputs: object = [id_map.get(inp, inp) for inp in node.inputs]
        signature: object = _compute_node_signature(node, canonical_inputs)

        if signature in seen_expressions:
            canonical_id: object = seen_expressions[signature]
            id_map[node.id] = canonical_id
            del graph.nodes[node.id]
            modified: object = True
        else:
            seen_expressions[signature] = node.id
            id_map[node.id] = node.id
            if node.inputs != canonical_inputs:
                node.inputs = canonical_inputs
                modified: object = True

    # Update outputs to point to canonical nodes
    new_outputs: object = []
    for o in graph.outputs:
        new_outputs.append(id_map.get(o, o))

    if graph.outputs != new_outputs:
        graph.outputs = new_outputs
        modified: object = True

    return modified
