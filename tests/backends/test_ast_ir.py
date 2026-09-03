"""Tests for ast_to_ir and ir_to_ast."""

import libcst as cst

from ml_switcheroo_compiler.backends.ast_to_ir import parse_ast_to_ir
from ml_switcheroo_compiler.backends.ir_to_ast import _build_attribute_chain, emit_ir_to_ast


def test_ast_to_ir() -> None:
    source = "torch.add(x, y)"
    graph = parse_ast_to_ir(source)
    assert len(graph.nodes) == 1
    node = list(graph.nodes.values())[0]
    assert node.op_type == "Add"

    source2 = "add(x, y)"
    graph2 = parse_ast_to_ir(source2)

    source3 = "torch.nn.add(x, y)"
    graph3 = parse_ast_to_ir(source3)


def test_ir_to_ast() -> None:
    source = "torch.add(x, y)"
    graph = parse_ast_to_ir(source)

    ast_module = emit_ir_to_ast(graph, "jax")
    code = ast_module.code
    assert "jax.numpy.add()" in code

    # Test empty list
    expr = _build_attribute_chain([])
    assert isinstance(expr, cst.Name)
    assert expr.value == "empty"

    # Test single
    expr = _build_attribute_chain(["jax"])
    assert isinstance(expr, cst.Name)
    assert expr.value == "jax"


def test_ast_branches() -> None:
    from ml_switcheroo_compiler.backends.ast_to_ir import parse_ast_to_ir
    from ml_switcheroo_compiler.backends.ir_to_ast import emit_ir_to_ast
    from ml_switcheroo_compiler.ir.core import IRNode

    # Not a call
    source = "x = 1\n"
    graph = parse_ast_to_ir(source)
    assert len(graph.nodes) > 0

    # Missing op
    source = "torch.unknown_op(x)\n"
    graph = parse_ast_to_ir(source)
    assert len(graph.nodes) == 0

    # Emit missing op
    graph.nodes["in0"] = IRNode("in0", "Input", [])
    graph.nodes["unknown"] = IRNode("unknown", "UnknownOp", inputs=["in0"])
    ast_module = emit_ir_to_ast(graph, "jax")
    assert len(ast_module.body) == 1  # Only the input node gets emitted


def test_ast_to_ir_edge_cases() -> None:
    # hit line 34 false branch (call a function that is not Name or Attribute)
    source = "funcs[0](x, y)"
    graph = parse_ast_to_ir(source)
    assert len(graph.nodes) == 0

    # hit line 30 (_get_base_name falls through)
    source = "(a + b).add(x, y)"
    graph = parse_ast_to_ir(source)
    assert len(graph.nodes) == 0


def test_ast_to_ir_inline() -> None:
    source = "torch.add(x, torch.add(y, z))"
    graph = parse_ast_to_ir(source)
    assert len(graph.nodes) == 2

    source2 = "x = a + (b * c)"
    graph2 = parse_ast_to_ir(source2)
    assert len(graph2.nodes) == 2


def test_ast_to_ir_extra_nodes() -> None:
    source = """
x = 1
y = 2.0
z = x + y
w = x - y
v = x * y
u = x / y
t = x ** y
s = x % y
if x:
    pass
while x:
    pass
a = x[0]
b = foo(x, non_existent_var)
c = foo(1)
"""
    graph = parse_ast_to_ir(source)
    assert len(graph.nodes) > 0
