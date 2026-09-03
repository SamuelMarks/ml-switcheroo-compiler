from ml_switcheroo_compiler.backends.ast_to_ir import parse_ast_to_ir


def test_ast_control_flow():
    code = """
x = torch.zeros(10)
y = torch.ones(10)
if x[0] > 0:
    z = x + y
while x[1] < 10:
    x = x * y
"""
    graph = parse_ast_to_ir(code)

    op_types = [n.op_type for n in graph.nodes.values()]
    assert "zeros" in op_types
    assert "ones" in op_types
    assert "Slice" in op_types
    assert "Cond" in op_types
    assert "WhileLoop" in op_types
    assert "Add" in op_types
    assert "Mul" in op_types


def test_ast_ssa():
    code = """
a = torch.zeros()
b = torch.ones()
c = torch.add(a, b)
"""
    graph = parse_ast_to_ir(code)

    nodes = list(graph.nodes.values())
    add_node = next(n for n in nodes if n.op_type == "Add")

    assert len(add_node.inputs) == 2
