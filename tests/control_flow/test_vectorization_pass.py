"""Tests for middle-end vectorization pass and vmap transformations."""

import numpy as np

from ml_switcheroo_compiler.core.config import ConfigContext
from ml_switcheroo_compiler.core.device import Device, DeviceType
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.grad.api import grad
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
from ml_switcheroo_compiler.ops.binary import add, multiply
from ml_switcheroo_compiler.ops.vmap import vmap
from ml_switcheroo_compiler.transforms.passes.vectorization import (
    _align_batch_axis,
    _broadcast_unbatched_input,
    _load_vmap_rules,
    vectorization_pass,
    vectorize_graph,
)

device = Device(DeviceType.CPU, 0)


def test_vectorize_elementwise_graph():
    """Test vectorizing a simple elementwise graph."""
    g = IRGraph(name="test_elem")
    g.inputs = ["x", "y"]
    g.nodes["x"] = IRNode(id="x", op_type="Input", inputs=[], shape_metadata=(4,))
    g.nodes["y"] = IRNode(id="y", op_type="Input", inputs=[], shape_metadata=(4,))
    g.nodes["add1"] = IRNode(id="add1", op_type="Add", inputs=["x", "y"], shape_metadata=(4,))
    g.nodes["mul1"] = IRNode(id="mul1", op_type="Multiply", inputs=["add1", "x"], shape_metadata=(4,))
    g.outputs = ["mul1"]

    vec = vectorize_graph(g, in_axes=(0, 0), batch_size=8, out_axes=0)
    assert vec.nodes["add1"].op_type == "Add"
    assert vec.nodes["add1"].shape_metadata == (8, 4)
    assert vec.nodes["mul1"].op_type == "Multiply"
    assert vec.nodes["mul1"].shape_metadata == (8, 4)
    assert vec.outputs == ["mul1"]


def test_vectorize_matmul_lifting():
    """Test lifting MatMul to BatchMatMul when batched."""
    g = IRGraph(name="test_matmul")
    g.inputs = ["a", "b"]
    g.nodes["a"] = IRNode(id="a", op_type="Input", inputs=[], shape_metadata=(3, 4))
    g.nodes["b"] = IRNode(id="b", op_type="Input", inputs=[], shape_metadata=(4, 5))
    g.nodes["mm"] = IRNode(id="mm", op_type="MatMul", inputs=["a", "b"], shape_metadata=(3, 5))
    g.outputs = ["mm"]

    vec = vectorize_graph(g, in_axes=(0, 0), batch_size=2, out_axes=0)
    assert vec.nodes["mm"].op_type == "BatchMatMul"
    assert vec.nodes["mm"].shape_metadata == (2, 3, 5)


def test_vectorize_axis_alignment_and_broadcasting():
    """Test aligning batch axes when in_axes differs from 0 or is None."""
    g = IRGraph(name="test_align")
    g.inputs = ["a", "const_b"]
    g.nodes["a"] = IRNode(id="a", op_type="Input", inputs=[], shape_metadata=(4, 6))
    g.nodes["const_b"] = IRNode(id="const_b", op_type="Input", inputs=[], shape_metadata=(4, 6))
    g.nodes["sub"] = IRNode(id="sub", op_type="Subtract", inputs=["a", "const_b"], shape_metadata=(4, 6))
    g.outputs = ["sub"]

    # a is batched at axis 1, const_b is unbatched (None)
    vec = vectorize_graph(g, in_axes={"a": 1, "const_b": None}, batch_size=6, out_axes=1)
    # Check that alignment and broadcasting nodes are inserted
    assert any(n.op_type == "Transpose" for n in vec.nodes.values())
    assert any(n.op_type == "BroadcastTo" for n in vec.nodes.values())
    assert vec.nodes["sub"].op_type == "Subtract"


def test_vectorize_reduction_shift_axis():
    """Test shifting axis in reduction ops when batched."""
    g = IRGraph(name="test_reduce")
    g.inputs = ["x"]
    g.nodes["x"] = IRNode(id="x", op_type="Input", inputs=[], shape_metadata=(3, 4))
    g.nodes["sum1"] = IRNode(id="sum1", op_type="ReduceSum", inputs=["x"], attributes={"axis": 1}, shape_metadata=(3,))
    g.nodes["sm"] = IRNode(id="sm", op_type="Softmax", inputs=["x"], attributes={"dim": 0}, shape_metadata=(3, 4))
    g.outputs = ["sum1", "sm"]

    vec = vectorize_graph(g, in_axes=0, batch_size=5, out_axes=0)
    assert vec.nodes["sum1"].attributes["axis"] == 2
    assert vec.nodes["sm"].attributes["dim"] == 1


def test_vectorization_pass_inlining():
    """Test vectorization_pass inlining a Vmap node in an outer IRGraph."""
    body = IRGraph(name="body")
    body.inputs = ["bx"]
    body.nodes["bx"] = IRNode(id="bx", op_type="Input", inputs=[], shape_metadata=(4,))
    body.nodes["b_add"] = IRNode(id="b_add", op_type="Add", inputs=["bx", "bx"], shape_metadata=(4,))
    body.outputs = ["b_add"]

    outer = IRGraph(name="outer")
    outer.inputs = ["x_batched"]
    outer.nodes["x_batched"] = IRNode(id="x_batched", op_type="Input", inputs=[], shape_metadata=(3, 4))
    outer.nodes["vmap_node"] = IRNode(
        id="vmap_node",
        op_type="Vmap",
        inputs=["x_batched"],
        attributes={"body": body, "in_axes": 0, "out_axes": 0},
        shape_metadata=(3, 4),
    )
    outer.outputs = ["vmap_node"]

    opt = vectorization_pass(outer)
    assert "vmap_node" in opt.nodes
    assert opt.nodes["vmap_node"].op_type == "Identity"
    assert any(n.op_type == "Add" for n in opt.nodes.values())


def test_eager_vmap_nested():
    """Test nested vmap in eager execution."""
    with ConfigContext(eager_mode=True):
        # 2D batch of matrices: shape (2, 3, 4)
        x_data = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        x = Tensor(x_data, TensorConfig((2, 3, 4), DType.Float32, device))

        def double(t: Tensor) -> Tensor:
            return add(t, t)

        v_inner = vmap(double, in_axes=0, out_axes=0)
        v_outer = vmap(v_inner, in_axes=0, out_axes=0)
        res = v_outer(x)

        expected = x_data * 2
        np.testing.assert_allclose(res.data, expected)


def test_eager_vmap_grad():
    """Test vmap composed with grad."""
    with ConfigContext(eager_mode=True):
        # Batch of 4 scalar inputs
        x_data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        x = Tensor(x_data, TensorConfig((4,), DType.Float32, device))

        def loss_fn(t: Tensor) -> Tensor:
            # f(t) = t * t
            return multiply(t, t)

        grad_fn = grad(loss_fn)
        v_grad = vmap(grad_fn, in_axes=0, out_axes=0)
        res = v_grad(x)

        # d/dt(t^2) = 2*t
        expected = 2.0 * x_data
        np.testing.assert_allclose(res.data, expected, rtol=1e-4)


def test_eager_grad_vmap():
    """Test grad composed with vmap: grad(vmap(f))."""
    from ml_switcheroo_compiler.ops.reductions.frontend import sum as reduce_sum

    with ConfigContext(eager_mode=True):
        x_data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        x = Tensor(x_data, TensorConfig((4,), DType.Float32, device))

        def f(t: Tensor) -> Tensor:
            return multiply(t, t)

        v_f = vmap(f, in_axes=0, out_axes=0)

        def total_loss(t: Tensor) -> Tensor:
            return reduce_sum(v_f(t))

        grad_vmap = grad(total_loss)
        res = grad_vmap(x)
        expected = 2.0 * x_data
        np.testing.assert_allclose(res.data, expected, rtol=1e-4)


def test_vectorization_helpers():
    """Test vectorization helper functions directly."""
    assert isinstance(_load_vmap_rules(), dict)

    g = IRGraph()
    # align when current == target
    assert _align_batch_axis(g, "n1", 0, 0, (2, 3)) == "n1"
    # align when rank <= 1
    assert _align_batch_axis(g, "n1", 1, 0, (2,)) == "n1"

    # broadcast helper
    bid = _broadcast_unbatched_input(g, "n1", 4, (3,), target_axis=0)
    assert bid in g.nodes
    assert g.nodes[bid].shape_metadata == (4, 3)


def test_vectorization_exhaustive_branches():
    """Verify remaining branch edge cases in vectorization pass and helper functions."""
    from unittest.mock import patch

    # 1. _load_vmap_rules when safe_load returns dict without "rules" or non-dict
    with patch("yaml.safe_load", return_value={"no_rules": True}):
        assert _load_vmap_rules() == {}

    # 2. vectorize_graph: inp_id in graph.inputs but not in graph.nodes (branch 141->143)
    #    and Input node in graph.nodes but not in graph.inputs / shape_map (branch 155->157)
    g_inputs = IRGraph(name="inputs_edge")
    g_inputs.inputs = ["missing_inp", "real_inp"]
    g_inputs.nodes["real_inp"] = IRNode(id="real_inp", op_type="Input", inputs=[], shape_metadata=(4,))
    g_inputs.nodes["orphan_inp"] = IRNode(id="orphan_inp", op_type="Input", inputs=[], shape_metadata=(4,))
    g_inputs.nodes["add0"] = IRNode(id="add0", op_type="Add", inputs=["real_inp", "orphan_inp"], shape_metadata=(4,))
    g_inputs.outputs = ["add0"]

    vec_inp = vectorize_graph(g_inputs, in_axes=0, batch_size=2, out_axes=0)
    assert "orphan_inp" in vec_inp.nodes

    # 3. Policy shift_axis with negative axis and dim (branches 201->203, 204->206)
    g_neg = IRGraph(name="neg_axis")
    g_neg.inputs = ["x"]
    g_neg.nodes["x"] = IRNode(id="x", op_type="Input", inputs=[], shape_metadata=(3, 4))
    g_neg.nodes["red_neg"] = IRNode(id="red_neg", op_type="ReduceSum", inputs=["x"], attributes={"axis": -1}, shape_metadata=(3,))
    g_neg.nodes["sm_neg"] = IRNode(id="sm_neg", op_type="Softmax", inputs=["x"], attributes={"dim": -1}, shape_metadata=(3, 4))
    g_neg.outputs = ["red_neg", "sm_neg"]

    vec_neg = vectorize_graph(g_neg, in_axes=0, batch_size=2, out_axes=0)
    assert vec_neg.nodes["red_neg"].attributes["axis"] == -1
    assert vec_neg.nodes["sm_neg"].attributes["dim"] == -1

    # 4. Policy prepend_batch_dim with "shape" (without "newshape") and with "newshape"
    g_prepend = IRGraph(name="prepend")
    g_prepend.inputs = ["x"]
    g_prepend.nodes["x"] = IRNode(id="x", op_type="Input", inputs=[], shape_metadata=(6,))
    g_prepend.nodes["bcast"] = IRNode(
        id="bcast",
        op_type="BroadcastTo",
        inputs=["x"],
        attributes={"shape": (2, 3)},
        shape_metadata=(2, 3),
    )
    g_prepend.nodes["resh"] = IRNode(
        id="resh",
        op_type="Reshape",
        inputs=["x"],
        attributes={"newshape": (2, 3)},
        shape_metadata=(2, 3),
    )
    g_prepend.outputs = ["bcast", "resh"]

    vec_prep = vectorize_graph(g_prepend, in_axes=0, batch_size=5, out_axes=0)
    assert vec_prep.nodes["bcast"].attributes["shape"] == (5, 2, 3)
    assert vec_prep.nodes["resh"].attributes["newshape"] == (5, 2, 3)

    # 5. vectorization_pass edge cases:
    body_v = IRGraph(name="body_v")
    body_v.inputs = ["bx"]
    body_v.nodes["bx"] = IRNode(id="bx", op_type="Input", inputs=[], shape_metadata=(4,))
    body_v.nodes["b_add"] = IRNode(id="b_add", op_type="Add", inputs=["bx", "bx"], shape_metadata=(4,))
    body_v.outputs = ["b_add"]

    # Tuple in_axes
    outer_tuple = IRGraph(name="outer_tuple")
    outer_tuple.inputs = ["x_in"]
    outer_tuple.nodes["x_in"] = IRNode(id="x_in", op_type="Input", inputs=[], shape_metadata=(4,))
    outer_tuple.nodes["vmap_tuple"] = IRNode(
        id="vmap_tuple",
        op_type="Vmap",
        inputs=["x_in"],
        attributes={"body": body_v, "in_axes": (0,), "out_axes": 0},
        shape_metadata=(4,),
    )
    outer_tuple.outputs = ["vmap_tuple"]
    opt_tuple = vectorization_pass(outer_tuple)
    assert "vmap_tuple" in opt_tuple.nodes

    # First input has None shape_metadata
    outer_no_shape = IRGraph(name="outer_no_shape")
    outer_no_shape.inputs = ["x_no_s"]
    outer_no_shape.nodes["x_no_s"] = IRNode(id="x_no_s", op_type="Input", inputs=[], shape_metadata=None)
    outer_no_shape.nodes["vmap_no_s"] = IRNode(
        id="vmap_no_s",
        op_type="Vmap",
        inputs=["x_no_s"],
        attributes={"body": body_v, "in_axes": 0, "out_axes": 0},
    )
    outer_no_shape.outputs = ["vmap_no_s"]
    opt_no_s = vectorization_pass(outer_no_shape)
    assert "vmap_no_s" in opt_no_s.nodes

    # Case where axis >= len(shape) -> axis=5, shape has len 1 -> fallback batch_size=1
    outer_oob = IRGraph(name="outer_oob")
    outer_oob.inputs = ["x_in2"]
    outer_oob.nodes["x_in2"] = IRNode(id="x_in2", op_type="Input", inputs=[], shape_metadata=(4,))
    outer_oob.nodes["vmap_oob"] = IRNode(
        id="vmap_oob",
        op_type="Vmap",
        inputs=["x_in2"],
        attributes={"body": body_v, "in_axes": (5,), "out_axes": 0},
        shape_metadata=(4,),
    )
    outer_oob.outputs = ["vmap_oob"]
    opt_oob = vectorization_pass(outer_oob)
    assert "vmap_oob" in opt_oob.nodes

    # Case where body_graph is NOT IRGraph (e.g., custom object or LogicalGraph) with nodes as list or dict
    class MockGraphNodesList:
        """Mock graph representation with nodes as a list."""

        def __init__(self, inputs: list[str], outputs: list[str], nodes: list[IRNode]) -> None:
            """Initialize MockGraphNodesList.

            Args:
                inputs: List of input node IDs.
                outputs: List of output node IDs.
                nodes: List of nodes.
            """
            self.id = "mock_list_body"
            self.inputs = inputs
            self.outputs = outputs
            self.nodes = nodes

    b_in_node = IRNode(id="b_in_list", op_type="Input", inputs=[], shape_metadata=(1,))
    b_out_node = IRNode(id="b_add_list", op_type="Add", inputs=["b_in_list", "b_in_list"], shape_metadata=(1,))
    mock_body_list = MockGraphNodesList(inputs=["b_in_list"], outputs=["b_add_list"], nodes=[b_in_node, b_out_node])

    outer_list = IRGraph(name="outer_list")
    outer_list.inputs = ["x_list"]
    outer_list.nodes["x_list"] = IRNode(id="x_list", op_type="Input", inputs=[], shape_metadata=(3,))
    outer_list.nodes["vmap_list"] = IRNode(
        id="vmap_list",
        op_type="Vmap",
        inputs=["x_list"],
        attributes={"body": mock_body_list, "in_axes": 0, "out_axes": 0},
    )
    outer_list.outputs = ["vmap_list"]
    opt_list = vectorization_pass(outer_list)
    assert "vmap_list" in opt_list.nodes

    # Mock body graph with nodes as dict (and mixed non-IRNode node)
    class MockGraphNodesDict:
        """Mock graph representation with nodes as a dictionary."""

        def __init__(self, inputs: list[str], outputs: list[str], nodes: dict[str, object]) -> None:
            """Initialize MockGraphNodesDict.

            Args:
                inputs: List of input node IDs.
                outputs: List of output node IDs.
                nodes: Dictionary mapping IDs to node objects.
            """
            self.id = "mock_dict_body"
            self.inputs = inputs
            self.outputs = outputs
            self.nodes = nodes

    from ml_switcheroo_ir import LogicalNode

    pure_logical_node = LogicalNode(id="b_add_dict", op_type="Add", inputs=["b_in_dict", "b_in_dict"], shape_metadata=(1,))
    mock_body_dict = MockGraphNodesDict(
        inputs=["b_in_dict"],
        outputs=["b_add_dict"],
        nodes={
            "b_in_dict": IRNode(id="b_in_dict", op_type="Input", inputs=[], shape_metadata=(1,)),
            "b_add_dict": pure_logical_node,
        },
    )

    outer_dict = IRGraph(name="outer_dict")
    outer_dict.inputs = ["x_dict"]
    outer_dict.nodes["x_dict"] = IRNode(id="x_dict", op_type="Input", inputs=[], shape_metadata=(3,))
    outer_dict.nodes["vmap_dict"] = IRNode(
        id="vmap_dict",
        op_type="Vmap",
        inputs=["x_dict"],
        attributes={"body": mock_body_dict, "in_axes": 0, "out_axes": 0},
    )
    outer_dict.outputs = ["vmap_dict"]
    opt_dict = vectorization_pass(outer_dict)
    assert "vmap_dict" in opt_dict.nodes
