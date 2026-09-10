from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
from ml_switcheroo_compiler.transforms.passes.state_lowering import state_lowering_pass


def test_state_lowering_pass():
    g = IRGraph()
    n1 = IRNode(id="n1", op_type="Input", attributes={"is_state": True, "name": "var_a"})
    n2 = IRNode(id="n2", op_type="Output", attributes={"is_state": True})
    n3 = IRNode(id="n3", op_type="Input")
    g.nodes["n1"] = n1
    g.nodes["n2"] = n2
    g.nodes["n3"] = n3

    modified = state_lowering_pass(g)
    assert modified is True

    assert n1.op_type == "ReadVariable"
    assert n1.attributes.get("variable_name") == "var_a"
    assert "name" not in n1.attributes

    assert n2.op_type == "AssignVariable"
    assert n2.attributes.get("variable_name") == "n2"
    assert "name" not in n2.attributes

    assert n3.op_type == "Input"

    # Run again, should not modify
    modified_again = state_lowering_pass(g)
    assert modified_again is False


def test_state_lowering_empty_graph():
    g = IRGraph()
    assert state_lowering_pass(g) is False


def test_state_lowering_output_with_name():
    g = IRGraph()
    n2 = IRNode(id="n2", op_type="Output", attributes={"is_state": True, "name": "var_b"})
    g.nodes["n2"] = n2

    modified = state_lowering_pass(g)
    assert modified is True

    assert n2.op_type == "AssignVariable"
    assert n2.attributes.get("variable_name") == "var_b"
    assert "name" not in n2.attributes


def test_state_lowering_input_without_name():
    g = IRGraph()
    n1 = IRNode(id="n1", op_type="Input", attributes={"is_state": True})
    g.nodes["n1"] = n1

    modified = state_lowering_pass(g)
    assert modified is True

    assert n1.op_type == "ReadVariable"
    assert n1.attributes.get("variable_name") == "n1"
    assert "name" not in n1.attributes


def test_state_lowering_pass_class() -> None:
    """Test StateLoweringPass class run and emit_stateful_class methods."""
    from ml_switcheroo_compiler.transforms.passes.state_lowering import StateLoweringPass

    pass_pt = StateLoweringPass(target_framework="pytorch")
    g = IRGraph()
    n1 = IRNode(id="weight", op_type="Input", attributes={"is_state": True, "param_name": "linear.weight"}, shape_metadata={"shape": (4, 4)})
    buf = IRNode(id="mean", op_type="Input", attributes={"is_state": True, "is_buffer": True, "param_name": "running_mean"}, shape_metadata={"shape": (4,)})
    x = IRNode(id="x", op_type="Input", attributes={"is_state": False})
    out = IRNode(id="out", op_type="MatMul", inputs=["x", "weight"])
    g.nodes = {"weight": n1, "mean": buf, "x": x, "out": out}
    g.outputs = ["out"]

    # PyTorch emission
    code_pt = pass_pt.emit_stateful_class(g, class_name="MyLinear")
    assert "class MyLinear(nn.Module):" in code_pt
    assert "self.linear_weight = nn.Parameter" in code_pt
    assert "self.register_buffer('running_mean', torch.zeros((4,)))" in code_pt
    assert "def forward(self, x):" in code_pt
    assert "return out" in code_pt

    # JAX emission
    pass_jax = StateLoweringPass(target_framework="jax")
    code_jax = pass_jax.emit_stateful_class(g, class_name="JaxLinear")
    assert "class JaxLinear:" in code_jax
    assert "self.linear_weight = jnp.zeros" in code_jax
    assert "def __call__(self, x):" in code_jax

    # Generic emission
    pass_gen = StateLoweringPass(target_framework="generic")
    code_gen = pass_gen.emit_stateful_class(g, class_name="GenericModel")
    assert "class GenericModel:" in code_gen

    # Empty graph emission
    g_empty = IRGraph()
    code_empty = pass_pt.emit_stateful_class(g_empty)
    assert "pass" in code_empty
    assert "return None" in code_empty

    code_empty_jax = pass_jax.emit_stateful_class(g_empty)
    assert "pass" in code_empty_jax
    assert "return None" in code_empty_jax

    # Run pass
    assert pass_pt.run(g) is True
    assert g.nodes["weight"].op_type == "ReadVariable"
