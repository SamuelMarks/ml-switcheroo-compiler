# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Common Subexpression Elimination pass using structural hashing."""

from __future__ import annotations

import hashlib
from typing import Any

from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
from ml_switcheroo_compiler.transforms.pass_manager import DAGTopologicalSorter

COMMUTATIVE_OPS: frozenset[str] = frozenset(
    {
        "Add",
        "Mul",
        "Multiply",
        "Maximum",
        "Minimum",
        "Equal",
        "NotEqual",
        "BitwiseAnd",
        "BitwiseOr",
        "BitwiseXor",
        "LogicalAnd",
        "LogicalOr",
    }
)


def _hash_attribute_value(val: Any) -> str:
    """Recursively compute a canonical structural representation of attribute values.

    Args:
        val (Any): Attribute value (scalar, list, dict, IRGraph).

    Returns:
        str: Canonical structural representation.
    """
    if isinstance(val, IRGraph):
        sub_sigs: list[str] = []
        sub_id_map: dict[str, str] = {}
        input_counter = 0
        sub_sorted = DAGTopologicalSorter.sort(val)
        for sn in sub_sorted:
            if sn.op_type == "Input":
                sub_id_map[sn.id] = f"Input_{input_counter}"
                input_counter += 1
                continue
            c_in = [sub_id_map.get(inp, inp) for inp in sn.inputs]
            h = compute_node_structural_hash(sn, c_in)
            sub_id_map[sn.id] = h
            sub_sigs.append(f"{sn.op_type}:{h}")
        sub_out = [sub_id_map.get(o, o) for o in getattr(val, "outputs", [])]
        return f"IRGraph({';'.join(sub_sigs)}->{sub_out})"
    if isinstance(val, dict):
        items = sorted((str(k), _hash_attribute_value(v)) for k, v in val.items())
        return f"dict({items})"
    if isinstance(val, (list, tuple)):
        items_seq = [_hash_attribute_value(v) for v in val]
        return f"list({items_seq})"
    return str(val)


def compute_node_structural_hash(node: IRNode, canonical_inputs: list[str]) -> str:
    """Compute structural hash of an IR node using operator commutativity and nested subgraph canonicalization.

    Args:
        node (IRNode): The IR node to evaluate.
        canonical_inputs (list[str]): Canonical input node identifiers.

    Returns:
        str: Structural hash string.
    """
    if node.op_type == "Input":
        return f"Input_{node.id}"

    # For commutative operations, order of inputs does not affect computed value
    ordered_inputs = list(canonical_inputs)
    if node.op_type in COMMUTATIVE_OPS:
        ordered_inputs.sort()

    attr_list: list[tuple[str, str]] = []
    for k, v in node.attributes.items():
        attr_list.append((k, _hash_attribute_value(v)))
    attr_list.sort(key=lambda x: x[0])

    shape_str = str(getattr(node, "shape_metadata", None))
    dtype_str = str(node.attributes.get("dtype", ""))

    payload = f"{node.op_type}|{ordered_inputs}|{attr_list}|{shape_str}|{dtype_str}"
    # Use deterministic SHA-256 digest for structural fingerprinting
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _compute_node_signature(node: IRNode, canonical_inputs: list[str]) -> str:
    """Legacy signature interface delegating to structural hashing.

    Args:
        node (IRNode): The node parameter.
        canonical_inputs (list[str]): The canonical_inputs parameter.

    Returns:
        str: Result signature.
    """
    return compute_node_structural_hash(node, canonical_inputs)


def cse_pass(graph: IRGraph) -> bool:
    """In-place Common Subexpression Elimination (CSE) via structural hashing.

    Args:
        graph (IRGraph): The graph parameter.

    Returns:
        bool: True if any common subexpressions were eliminated.
    """
    modified = False
    seen_expressions: dict[str, str] = {}
    id_map: dict[str, str] = {}

    sorted_nodes = DAGTopologicalSorter.sort(graph)

    for node in sorted_nodes:
        canonical_inputs = [id_map.get(inp, inp) for inp in node.inputs]
        signature = compute_node_structural_hash(node, canonical_inputs)

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

    new_outputs: list[str] = []
    for o in graph.outputs:
        new_outputs.append(id_map.get(o, o))

    if graph.outputs != new_outputs:
        graph.outputs = new_outputs
        modified = True

    return modified
