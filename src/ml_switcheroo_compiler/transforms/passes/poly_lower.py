"""Pass to lower complex polyfilled operations into simpler IR primitives."""

import os
from typing import Any

import yaml

from ml_switcheroo_compiler.ir.core import IRGraph, IRNode


def _lower_in_top_k(node: IRNode, graph: IRGraph, new_nodes: dict[str, IRNode]) -> IRNode:
    """Lower InTopK.

    Args:
        node (IRNode): node.
        graph (IRGraph): graph.
        new_nodes (dict): new nodes.

    Returns:
        IRNode: node.
    """
    import uuid

    uid = uuid.uuid4().hex[:6]
    k = node.attributes.get("k", 1)

    topk_id = f"topk_{uid}"
    topk = IRNode(id=topk_id, op_type="TopK", inputs=[node.inputs[1]], attributes={"k": k, "axis": -1})
    new_nodes[topk_id] = topk

    # TopK returns tuple (values, indices), we need indices
    idx_id = f"topk_idx_{uid}"
    idx_node = IRNode(id=idx_id, op_type="TupleGetItem", inputs=[topk_id], attributes={"index": 1})
    new_nodes[idx_id] = idx_node

    # Expand targets
    exp_id = f"exp_t_{uid}"
    exp_node = IRNode(id=exp_id, op_type="ExpandDims", inputs=[node.inputs[0]], attributes={"axis": -1})
    new_nodes[exp_id] = exp_node

    # Eq
    eq_id = f"eq_{uid}"
    eq_node = IRNode(id=eq_id, op_type="Equal", inputs=[idx_id, exp_id], attributes={})
    new_nodes[eq_id] = eq_node

    # ReduceAny
    node.op_type = "ReduceAny"
    node.inputs = [eq_id]
    node.attributes = {"axes": [-1], "keepdims": False}
    return node


def _lower_ctc_greedy_decoder(node: IRNode, graph: IRGraph, new_nodes: dict[str, IRNode]) -> IRNode:
    """Lower CtcGreedyDecoder.

    Args:
        node (IRNode): node.
        graph (IRGraph): graph.
        new_nodes (dict): new nodes.

    Returns:
        IRNode: node.
    """
    import uuid

    uid = uuid.uuid4().hex[:6]

    # 1. Argmax over classes
    argmax_id = f"ctc_argmax_{uid}"
    argmax = IRNode(id=argmax_id, op_type="Argmax", inputs=[node.inputs[0]], attributes={"axis": -1, "keepdims": False})
    new_nodes[argmax_id] = argmax

    # 2. Transpose to [batch, seq_len]
    transp_id = f"ctc_transp_{uid}"
    transp = IRNode(id=transp_id, op_type="Transpose", inputs=[argmax_id], attributes={"permutation": [1, 0]})
    new_nodes[transp_id] = transp

    # 3. Shift by 1 to detect duplicates
    # Since we can't easily express Pad+Slice without knowing shapes statically,
    # we emit a Roll or Shift node. Let's use Roll.
    roll_id = f"ctc_roll_{uid}"
    roll = IRNode(id=roll_id, op_type="Roll", inputs=[transp_id], attributes={"shift": 1, "axis": 1})
    new_nodes[roll_id] = roll

    # 4. NotEqual (is_new)
    neq_id = f"ctc_neq_{uid}"
    neq = IRNode(id=neq_id, op_type="NotEqual", inputs=[transp_id, roll_id], attributes={})
    new_nodes[neq_id] = neq

    # 5. NotEqual blank (assuming blank is 0 or -1, typically from attributes)
    blank_idx = node.attributes.get("blank_index", 0)
    blank_const_id = f"ctc_blank_{uid}"
    blank_const = IRNode(id=blank_const_id, op_type="Constant", inputs=[], attributes={"value": blank_idx})
    new_nodes[blank_const_id] = blank_const

    not_blank_id = f"ctc_not_blank_{uid}"
    not_blank = IRNode(id=not_blank_id, op_type="NotEqual", inputs=[transp_id, blank_const_id], attributes={})
    new_nodes[not_blank_id] = not_blank

    # 6. LogicalAnd to get keep mask
    keep_id = f"ctc_keep_{uid}"
    keep = IRNode(id=keep_id, op_type="LogicalAnd", inputs=[neq_id, not_blank_id], attributes={})
    new_nodes[keep_id] = keep

    # Mutate the original node to just return the filtered mask applied to the array.
    # For now, we emit a Select to zero out the non-kept tokens.
    zero_const_id = f"ctc_zero_{uid}"
    zero_const = IRNode(id=zero_const_id, op_type="Constant", inputs=[], attributes={"value": 0})
    new_nodes[zero_const_id] = zero_const

    node.op_type = "Select"
    node.inputs = [keep_id, transp_id, zero_const_id]
    node.attributes = {}
    return node


def _lower_isotonic_regression(node: IRNode, graph: IRGraph, new_nodes: dict[str, IRNode]) -> IRNode:
    """Lower IsotonicRegression into IR.

    Args:
        node (IRNode): node.
        graph (IRGraph): graph.
        new_nodes (dict): new nodes.

    Returns:
        IRNode: node.
    """
    from ml_switcheroo_ir import LogicalGraph

    node.op_type = "WhileLoop"
    cond_graph = LogicalGraph(name="pava_cond")
    body_graph = LogicalGraph(name="pava_body")

    cond_graph.inputs = ["array", "idx", "has_violations"]
    body_graph.inputs = ["array", "idx", "has_violations"]

    cond_graph.nodes["cond_out"] = IRNode(id="cond_out", op_type="Identity", inputs=["has_violations"])
    cond_graph.outputs = ["cond_out"]

    body_graph.nodes["next_idx"] = IRNode(id="next_idx", op_type="Add", inputs=["idx"], attributes={"value": 1})
    body_graph.nodes["false_violations"] = IRNode(id="false_violations", op_type="Constant", inputs=[], attributes={"value": False})
    body_graph.outputs = ["array", "next_idx", "false_violations"]

    node.attributes = {"cond": cond_graph, "body": body_graph}

    return node


def _load_poly_rules() -> dict[str, Any]:
    """Load poly rules.

    Returns:
        dict: loaded rules.
    """
    yaml_path = os.path.join(os.path.dirname(__file__), "poly_lower_rules.yaml")
    if not os.path.exists(yaml_path):
        return {}
    with open(yaml_path) as f:
        return yaml.safe_load(f).get("rules", {})


def polyfill_lowering_pass(graph: IRGraph) -> bool:
    """Lower complex polyfills to IR.

    Args:
        graph (IRGraph): graph.

    Returns:
        bool: modified.
    """
    rules = _load_poly_rules()
    modified = False
    new_nodes: dict[str, IRNode] = {}

    python_handlers = {
        "_lower_in_top_k": _lower_in_top_k,
        "_lower_ctc_greedy_decoder": _lower_ctc_greedy_decoder,
        "_lower_isotonic_regression": _lower_isotonic_regression,
    }

    for _node_id, node in graph.nodes.items():
        if node.op_type in rules:
            rule = rules[node.op_type]
            if rule["type"] == "python":
                func = python_handlers.get(rule["func"])
                if func:
                    func(node, graph, new_nodes)
                    modified = True
            elif rule["type"] == "rewrite":
                node.op_type = str(rule.get("op_type", node.op_type))
                if "attributes" in rule:
                    for k, v in dict(rule["attributes"]).items():
                        node.attributes[k] = v
                modified = True

        new_nodes[node.id] = node

    if modified:
        graph.nodes = new_nodes
    return modified
