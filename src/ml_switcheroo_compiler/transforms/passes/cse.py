"""Common Subexpression Elimination pass."""

from ml_switcheroo_compiler.ir.core import IRGraph
from ml_switcheroo_compiler.transforms.pass_manager import DAGTopologicalSorter


def _compute_node_signature(node: object, canonical_inputs: list[str]) -> str:
    """Function docstring.

    Args:
        node: Arg.
        canonical_inputs: Arg.
    """
    attr_list = []
    for k, v in node.attributes.items():
        attr_list.append((k, str(v)))
    attr_str = str(sorted(attr_list))
    return f"{node.op_type}|{canonical_inputs}|{attr_str}"


def cse_pass(graph: IRGraph) -> bool:
    """In-place Common Subexpression Elimination (CSE).

    Merges nodes that perform the same operation with the same inputs and
    attributes

    graph (IRGraph): The input graph

    Returns:
    bool: True if the graph was modified

    Args:
        graph (IRGraph): Argument graph
    """
    modified = False
    seen_expressions: dict[str, str] = {}
    id_map: dict[str, str] = {}

    sorted_nodes = DAGTopologicalSorter.sort(graph)

    for node in sorted_nodes:
        # Map inputs to canonical inputs
        canonical_inputs = [id_map.get(inp, inp) for inp in node.inputs]
        signature = _compute_node_signature(node, canonical_inputs)

        if signature in seen_expressions:
            canonical_id = seen_expressions[signature]
            id_map[node.id] = canonical_id
            del graph.nodes[node.id]
            modified = True
        else:
            seen_expressions[signature] = node.id
            id_map[node.id] = node.id
            if node.inputs != canonical_inputs:
                node.inputs = canonical_inputs
                modified = True

    # Update outputs to point to canonical nodes
    new_outputs = []
    for o in graph.outputs:
        new_outputs.append(id_map.get(o, o))

    if graph.outputs != new_outputs:
        graph.outputs = new_outputs
        modified = True

    return modified
