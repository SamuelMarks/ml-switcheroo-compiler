"""Unit tests for AST-to-IR parser and visitor syntax construct lowerings."""

from __future__ import annotations

import libcst as cst

from ml_switcheroo_compiler.backends.ast_to_ir import ASTToIRVisitor, parse_ast_to_ir


def test_ifexp_visitor_conditional_expression() -> None:
    """Verify lowering paths in visit_IfExp for test, body, and orelse expressions."""
    visitor = ASTToIRVisitor()

    # Case 1: Names are not in var_table and visitor produces no last_node_id
    ifexp_unbound = cst.IfExp(
        test=cst.Name("unbound_test"),
        body=cst.Name("unbound_body"),
        orelse=cst.Name("unbound_orelse"),
    )
    visitor.visit_IfExp(ifexp_unbound)
    node1 = list(visitor.graph.nodes.values())[-1]
    assert node1.op_type == "Cond"
    assert node1.inputs == []
    assert node1.attributes.get("is_ternary") is True

    # Case 2: Names are bound in var_table
    visitor.var_table["bound_test"] = "n_test"
    visitor.var_table["bound_body"] = "n_body"
    visitor.var_table["bound_orelse"] = "n_orelse"
    ifexp_bound = cst.IfExp(
        test=cst.Name("bound_test"),
        body=cst.Name("bound_body"),
        orelse=cst.Name("bound_orelse"),
    )
    visitor.visit_IfExp(ifexp_bound)
    node2 = list(visitor.graph.nodes.values())[-1]
    assert node2.inputs == ["n_test", "n_body", "n_orelse"]

    # Case 3: Complex expressions setting last_node_id
    code = """
a = 1
b = 2
res = (a + 1) if (a + b) else (b + 2)
"""
    graph = parse_ast_to_ir(code)
    cond_nodes = [n for n in graph.nodes.values() if n.op_type == "Cond"]
    assert len(cond_nodes) == 1
    assert len(cond_nodes[0].inputs) == 3


def test_listcomp_visitor_comprehensions() -> None:
    """Verify list comprehension iteration source lowerings in visit_ListComp."""
    visitor = ASTToIRVisitor()

    # Case 1: Iter expression is an unbound Name (no last_node_id)
    listcomp_unbound = cst.ListComp(
        elt=cst.Name("x"),
        for_in=cst.CompFor(
            target=cst.Name("x"),
            iter=cst.Name("unbound_iter"),
        ),
    )
    visitor.visit_ListComp(listcomp_unbound)
    node1 = list(visitor.graph.nodes.values())[-1]
    assert node1.op_type == "Map"
    assert node1.inputs == []

    # Case 2: Iter expression is a bound Name
    visitor.var_table["bound_iter"] = "n_iter"
    listcomp_bound = cst.ListComp(
        elt=cst.Name("x"),
        for_in=cst.CompFor(
            target=cst.Name("x"),
            iter=cst.Name("bound_iter"),
        ),
    )
    visitor.visit_ListComp(listcomp_bound)
    node2 = list(visitor.graph.nodes.values())[-1]
    assert node2.inputs == ["n_iter"]

    # Case 3: Iter expression is a visited expression setting last_node_id
    code = """
a = [1, 2]
b = [3, 4]
res = [x for x in (a + b)]
"""
    graph = parse_ast_to_ir(code)
    map_nodes = [n for n in graph.nodes.values() if n.op_type == "Map"]
    assert len(map_nodes) == 1
    assert len(map_nodes[0].inputs) == 1


def test_subscript_visitor_indexing_and_slicing() -> None:
    """Verify multidimensional slicing, expression indices, and unbound Name indexing."""
    code = """
x = torch.zeros(10, 10)
s_multi = x[0, 1]
s_expr = x[1 + 2]
s_unbound = x[unbound_var]
"""
    graph = parse_ast_to_ir(code)

    multi_node = next(n for n in graph.nodes.values() if n.id.startswith("node_") and n.op_type == "Slice")
    assert multi_node.op_type == "Slice"

    expr_node = next(n for n in graph.nodes.values() if n.id.startswith("node_") and n.op_type == "Slice" and n != multi_node)
    assert expr_node.op_type == "Slice"

    unbound_node = next(n for n in graph.nodes.values() if n.op_type == "GetItem" and n.attributes.get("index_var") == "unbound_var")
    assert unbound_node is not None
    # unbound_var is not in var_table, so only base tensor x is in inputs
    assert len(unbound_node.inputs) == 1


def test_call_star_args_and_foreign_call_and_dict_get() -> None:
    """Verify varargs, varkwargs, foreign callee attribute, and DictGet subscripting."""
    code = """
args = [1, 2]
kwargs = {"dim": 0}
y = unknown_foreign_fn(*args, **kwargs)
d = {"a": 10}
val = d["a"]
idx = 0
arr = torch.zeros(5)
elem = arr[idx]
"""
    graph = parse_ast_to_ir(code)

    foreign_node = next(n for n in graph.nodes.values() if n.op_type == "ForeignCall")
    assert foreign_node.attributes.get("callee") == "unknown_foreign_fn"
    assert foreign_node.attributes.get("has_varargs") is True
    assert foreign_node.attributes.get("has_varkwargs") is True

    dict_node = next(n for n in graph.nodes.values() if n.op_type == "DictGet")
    assert dict_node.attributes.get("key") == "a"

    getitem_node = next(n for n in graph.nodes.values() if n.op_type == "GetItem" and n.attributes.get("index_var") == "idx")
    assert len(getitem_node.inputs) == 2
