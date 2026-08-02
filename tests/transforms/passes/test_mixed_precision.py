from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
from ml_switcheroo_compiler.transforms.passes.mixed_precision import loss_scaling_pass, mixed_precision_pass


def test_mixed_precision_pass():
    g = IRGraph()
    n1 = IRNode(id="n1", op_type="MatMul", attributes={"dtype": "float32"}, inputs=["in1", "in2"])
    n2 = IRNode(id="n2", op_type="Exp", attributes={"dtype": "float16"}, inputs=["n1"])
    g.nodes["in1"] = IRNode(id="in1", op_type="Input")
    g.nodes["in2"] = IRNode(id="in2", op_type="Input")
    g.nodes["n1"] = n1
    g.nodes["n2"] = n2

    modified = mixed_precision_pass(g)
    assert modified is True

    # Check casts
    assert "in1_cast_float16" in g.nodes
    assert "in2_cast_float16" in g.nodes
    assert g.nodes["n1"].inputs == ["in1_cast_float16", "in2_cast_float16"]

    # n2 is Exp (fp32)
    assert "n1_cast_float32" in g.nodes
    assert g.nodes["n2"].inputs == ["n1_cast_float32"]


def test_mixed_precision_no_change():
    g = IRGraph()
    g.nodes["n1"] = IRNode(id="n1", op_type="Input", attributes={"dtype": "float16"})
    g.nodes["n2"] = IRNode(id="n2", op_type="MatMul", inputs=["n1"], attributes={"dtype": "float16"})
    modified = mixed_precision_pass(g)
    assert modified is False


def test_loss_scaling_pass():
    g = IRGraph()
    g.nodes["grad_in"] = IRNode(id="grad_in", op_type="Input", attributes={"is_grad": True})
    g.nodes["add"] = IRNode(id="add", op_type="Add", inputs=["grad_in", "grad_in"])
    g.outputs = ["add"]

    modified = loss_scaling_pass(g)
    assert modified is True

    assert "loss_scale_factor" in g.nodes
    assert "loss_scale_inv_factor" in g.nodes
    assert "grad_in_scaled" in g.nodes

    # Check that add uses scaled gradient
    assert g.nodes["add"].inputs == ["grad_in_scaled", "grad_in_scaled"]

    # Check that outputs are unscaled
    assert g.outputs == ["add_unscaled"]
    assert g.nodes["add_unscaled"].inputs == ["add", "loss_scale_inv_factor"]


def test_loss_scaling_empty():
    g = IRGraph()
    assert loss_scaling_pass(g) is False


def test_mixed_precision_empty_graph():
    g = IRGraph()
    assert mixed_precision_pass(g) is False


def test_mixed_precision_pass_reuse_cast():
    g = IRGraph()
    # n1 used by two FP16 ops
    n1 = IRNode(id="n1", op_type="Input", attributes={"dtype": "float32"})
    n2 = IRNode(id="n2", op_type="Conv2D", inputs=["n1"])
    n3 = IRNode(id="n3", op_type="MatMul", inputs=["n1"])
    g.nodes["n1"] = n1
    g.nodes["n2"] = n2
    g.nodes["n3"] = n3

    assert mixed_precision_pass(g) is True
    # The cast for n1 should be reused
    assert "n1_cast_float16" in g.nodes


def test_mixed_precision_pass_update_node_dtype():
    g = IRGraph()
    # A node that is an FP16 op but its dtype is float32
    n1 = IRNode(id="n1", op_type="Input", attributes={"dtype": "float16"})
    n2 = IRNode(id="n2", op_type="Conv2D", inputs=["n1"], attributes={"dtype": "float32"})
    g.nodes["n1"] = n1
    g.nodes["n2"] = n2

    assert mixed_precision_pass(g) is True
    assert g.nodes["n2"].attributes["dtype"] == "float16"


def test_loss_scaling_pass_reuse_scale():
    from ml_switcheroo_compiler.transforms.passes.mixed_precision import loss_scaling_pass

    g = IRGraph()
    n1 = IRNode(id="n1", op_type="Input", attributes={"is_grad": True})
    n2 = IRNode(id="n2", op_type="Input", attributes={"is_grad": True})
    n3 = IRNode(id="n3", op_type="Add", inputs=["n1", "n2"])
    g.nodes["n1"] = n1
    g.nodes["n2"] = n2
    g.nodes["n3"] = n3
    g.outputs = ["n3", "n3"]  # duplicate output to trigger unscale reuse

    assert loss_scaling_pass(g, 1024.0) is True
    assert "loss_scale_factor" in g.nodes
    assert "loss_scale_inv_factor" in g.nodes


def test_mixed_precision_pass_other_op():
    g = IRGraph()
    n1 = IRNode(id="n1", op_type="Exp", attributes={"dtype": "float32"})  # In FP32_OPS, already float32, process returns False
    n2 = IRNode(id="n2", op_type="UnknownOp", inputs=["n1"], attributes={"dtype": "float32"})
    g.nodes["n1"] = n1
    g.nodes["n2"] = n2
    assert mixed_precision_pass(g) is False


def test_loss_scaling_pass_nodes_exist():
    from ml_switcheroo_compiler.transforms.passes.mixed_precision import loss_scaling_pass

    g = IRGraph()
    n1 = IRNode(id="n1", op_type="Input", attributes={"is_grad": True})
    n2 = IRNode(id="loss_scale_factor", op_type="Constant")
    n3 = IRNode(id="loss_scale_inv_factor", op_type="Constant")
    n4 = IRNode(id="n1_scaled", op_type="Mul")
    g.nodes["n1"] = n1
    g.nodes["loss_scale_factor"] = n2
    g.nodes["loss_scale_inv_factor"] = n3
    g.nodes["n1_scaled"] = n4

    # It will reuse them
    assert loss_scaling_pass(g, 1024.0) is True
