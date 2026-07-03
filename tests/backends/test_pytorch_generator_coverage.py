"""Module docstring."""

from ml_switcheroo_compiler.backends.pytorch.generator import PyTorchCodeGenerator
from ml_switcheroo_compiler.backends.pytorch.pytorch_mixins import PyTorchDistributedVisitor
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode


def test_pytorch_generator_coverage() -> object:
    """Function docstring."""
    g = IRGraph()
    gen = PyTorchCodeGenerator(g)

    n_conv = IRNode(
        id="n1",
        op_type="ConvTranspose",
        inputs=["x", "w"],
        attributes={"strides": 2, "padding": "SAME"},
        shape_metadata=None,
    )
    assert "pt_conv_transpose" in gen.visit(n_conv, ["x", "w"])

    n_ragged = IRNode(id="n2", op_type="RaggedDot", inputs=["x", "y"], attributes={}, shape_metadata=None)
    assert "pt_ragged_dot" in gen.visit(n_ragged, ["x", "y"])

    n_power = IRNode(
        id="n3",
        op_type="PowerIteration",
        inputs=["x", "u"],
        attributes={"num_iters": 2},
        shape_metadata=None,
    )
    assert "pt_power_iteration" in gen.visit(n_power, ["x", "u"])
    assert "pt_power_iteration" in gen.visit(n_power, ["x"])

    assert gen.get_fallback_prefix() == "torch"
    assert gen.get_fallback_axis_kwarg() == "dim"

    ops_map = gen.get_ops_map({})
    assert "Matmul" in ops_map

    gen._emit_constant_assignment("c", "1")
    assert "c = self.c" in "\n".join(gen.code)

    prefix = gen._get_prefix_code()
    assert len(prefix) > 0
    assert "import torch" in prefix[0]

    n_const = IRNode(id="n4", op_type="Constant", inputs=[], attributes={"value": 1.0}, shape_metadata=None)
    g.nodes["n4"] = n_const
    gen.sorted_nodes = [n_const]
    gen.code = []
    assert gen._emit_init_body() is True
    assert "register_parameter" in "\n".join(gen.code)

    assert gen.get_fallback_keepdims_kwarg() == "keepdim"

    n_not_const = IRNode(id="n5", op_type="Add", inputs=["x", "y"], attributes={}, shape_metadata=None)
    g.nodes["n5"] = n_not_const
    gen.sorted_nodes = [n_not_const]
    gen.code = []
    assert gen._emit_init_body() is False

    mixin = PyTorchDistributedVisitor()
    n_dummy = IRNode(id="nx", op_type="Unknown", inputs=[], attributes={}, shape_metadata=None)
    assert "all_gather" in mixin.visit_all_gather(n_dummy, ["x"])
    assert "reduce_scatter" in mixin.visit_reduce_scatter(n_dummy, ["x"])
    assert "all_reduce" in mixin.visit_all_reduce(n_dummy, ["x"])
