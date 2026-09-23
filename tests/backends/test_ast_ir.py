"""Tests for ast_to_ir and ir_to_ast."""

import libcst as cst

from ml_switcheroo_compiler.backends.ast_to_ir import parse_ast_to_ir
from ml_switcheroo_compiler.backends.ir_to_ast import (
    _build_attribute_chain,
    emit_ir_to_ast,
    emit_ir_to_class,
)


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
    assert "jax.numpy.add(x, y)" in code

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


def test_semantic_ast_to_ir_features() -> None:
    # 1. Nested calls and keyword arguments
    source_nested = "z = torch.add(torch.relu(x), y, dim=1)"
    graph_nested = parse_ast_to_ir(source_nested)
    assert len(graph_nested.nodes) >= 2
    add_node = next(n for n in graph_nested.nodes.values() if n.op_type == "Add")
    assert "dim" in add_node.attributes
    assert add_node.attributes["dim"] == 1

    # 2. Container unpacking
    source_unpack = "a, b = torch.split(x, 2)"
    graph_unpack = parse_ast_to_ir(source_unpack)
    getitem_nodes = [n for n in graph_unpack.nodes.values() if n.op_type == "GetItem"]
    assert len(getitem_nodes) == 2

    # 3. Lifting stateful class constructs into functional IR
    class_source = """
class MyMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = 1.0
        self.bias = 0.0

    def forward(self, x):
        h = torch.add(x, self.weight)
        return torch.add(h, self.bias)
"""
    graph_class = parse_ast_to_ir(class_source)
    # Check that parameters are registered as explicit inputs
    param_nodes = [n for n in graph_class.nodes.values() if n.attributes.get("is_parameter")]
    assert len(param_nodes) >= 2
    assert any(n.attributes.get("param_name") == "weight" for n in param_nodes)
    assert any(n.attributes.get("param_name") == "bias" for n in param_nodes)

    # 4. emit_ir_to_class reconstruction
    pt_class = emit_ir_to_class(graph_class, "pytorch", class_name="MyMLP")
    assert pt_class.name.value == "MyMLP"
    pt_code = cst.Module(body=[pt_class]).code
    assert "class MyMLP(nn.Module):" in pt_code
    assert "def forward(self" in pt_code

    flax_class = emit_ir_to_class(graph_class, "jax", class_name="MyMLP")
    flax_code = cst.Module(body=[flax_class]).code
    assert "class MyMLP(flax.linen.Module):" in flax_code
    assert "def __call__(self" in flax_code

    # Keras class
    keras_class = emit_ir_to_class(graph_class, "keras", class_name="MyMLP")
    keras_code = cst.Module(body=[keras_class]).code
    assert "class MyMLP(keras.Model):" in keras_code
    assert "def call(self" in keras_code

    # Fallback framework
    unknown_class = emit_ir_to_class(graph_class, "custom_fw", class_name="MyMLP")
    unknown_code = cst.Module(body=[unknown_class]).code
    assert "class MyMLP(object):" in unknown_code


def test_roundtrip_transpilation_mlp() -> None:
    import numpy as np

    from ml_switcheroo_compiler.backends.ast_to_ir import parse_ast_to_ir
    from ml_switcheroo_compiler.backends.cst_transpiler import transpile_source
    from ml_switcheroo_compiler.backends.ir_to_ast import emit_ir_to_ast
    from ml_switcheroo_compiler.interpreter.evaluator import evaluate_graph

    # PyTorch MLP with comments and custom whitespace
    pt_mlp = """# Leading module header comment
class MLP(nn.Module):
    \"\"\"Docstring for MLP model.\"\"\"

    def forward(self, x):
        # Add bias offset
        h = torch.add(x, 1.0)
        # Apply activation
        return torch.relu(h)
"""
    # 1. Transpile source to JAX
    jax_mlp = transpile_source(pt_mlp, target_framework="jax")
    assert "# Leading module header comment" in jax_mlp
    assert '"""Docstring for MLP model."""' in jax_mlp
    assert "# Add bias offset" in jax_mlp
    assert "# Apply activation" in jax_mlp
    assert "class MLP(flax.linen.Module):" in jax_mlp
    assert "def __call__(self, x):" in jax_mlp

    # 2. Roundtrip JAX -> PyTorch
    roundtrip_pt = transpile_source(jax_mlp, target_framework="pytorch")
    assert "# Leading module header comment" in roundtrip_pt
    assert '"""Docstring for MLP model."""' in roundtrip_pt
    assert "class MLP(nn.Module):" in roundtrip_pt
    assert "def forward(self, x):" in roundtrip_pt
    assert "torch.add" in roundtrip_pt
    assert "torch.relu" in roundtrip_pt

    # 3. Roundtrip IR lowering: PyTorch AST -> IR -> JAX AST
    fn_source = "def mlp(x):\n    h = torch.add(x, 1.0)\n    return torch.relu(h)\n"
    ir_graph = parse_ast_to_ir(fn_source)
    jax_ast = emit_ir_to_ast(ir_graph, "jax")
    jax_code = jax_ast.code
    assert "jax.numpy.add" in jax_code
    assert "jax.numpy.relu" in jax_code

    # 4. Numerical execution parity:
    # Evaluate IRGraph with reference NumPy eager tensors
    test_x = np.array([2.0, -5.0, 3.0], dtype=np.float32)
    expected_numerical = np.maximum(0.0, test_x + 1.0)

    input_node = next(n for n in ir_graph.nodes.values() if n.op_type == "Input")
    eval_res = evaluate_graph(ir_graph, {input_node.id: test_x})
    actual_output = list(eval_res.values())[0]
    np.testing.assert_allclose(actual_output, expected_numerical, rtol=1e-5, atol=1e-5)


def test_ir_to_ast_defaults() -> None:
    """Test fallback logic in _get_compute_method_name and _get_class_base_expr."""
    from ml_switcheroo_compiler.backends.ir_to_ast import _get_class_base_expr, _get_compute_method_name

    assert _get_compute_method_name("unknown_fw") == "forward"
    assert _get_class_base_expr("unknown_fw").value == "object"


def test_ir_to_ast_missing_branches() -> None:
    """Test Cond with < 3 inputs and ForeignCall with varargs, varkwargs, and no var_name."""
    from ml_switcheroo_compiler.backends.ir_to_ast import (
        _emit_control_flow_statement,
        _emit_foreign_statement,
    )
    from ml_switcheroo_compiler.ir.core import IRNode

    # 1. Cond with len(inputs) < 3 falls through (branch 84->91)
    cond_node = IRNode("cond1", "Cond", inputs=["cond_flag", "true_val"])
    res_cond = _emit_control_flow_statement(cond_node)
    assert res_cond is None

    # 2. ForeignCall with has_varargs and has_varkwargs (branches 155->157, 157->159)
    foreign_node = IRNode(
        "fc1",
        "ForeignCall",
        inputs=["x"],
        attributes={"callee": "pkg.foo", "has_varargs": True, "has_varkwargs": True},
    )
    stmt = _emit_foreign_statement(foreign_node)
    assert stmt is not None
    code = cst.Module(body=[stmt]).code
    assert "*args" in code
    assert "**kwargs" in code

    # 3. ForeignCall with empty var_name returning cst.Expr (line 163, branch 160->163)
    foreign_no_var = IRNode(
        "",
        "ForeignCall",
        inputs=["x"],
        attributes={"var_name": "", "callee": "pkg.bar"},
    )
    stmt_no_var = _emit_foreign_statement(foreign_no_var)
    assert stmt_no_var is not None
    assert isinstance(stmt_no_var.body[0], cst.Expr)


def test_emit_ir_to_class_outputs_in_var_map_or_missing() -> None:
    """Test emit_ir_to_class when output id is already in var_map or missing from graph nodes."""
    from ml_switcheroo_compiler.backends.ir_to_ast import emit_ir_to_class
    from ml_switcheroo_compiler.ir.core import IRGraph, IRNode

    graph = IRGraph()
    # Node with var_name is included in var_map by _build_var_map
    graph.nodes["n0"] = IRNode(id="n0", op_type="Add", inputs=["a", "b"], attributes={"var_name": "result"})
    # outputs contains "n0" (already in var_map) and "nonexistent" (not in nodes)
    graph.outputs = ["n0", "nonexistent"]
    cls_def = emit_ir_to_class(graph, "pytorch", "OutputModel")
    code = cst.Module(body=[cls_def]).code
    assert "result = torch.add(a, b)" in code
    assert "return nonexistent" in code
