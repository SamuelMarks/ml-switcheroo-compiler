"""Tests for control flow optimization, vectorization, and scan transformations."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np

from ml_switcheroo_compiler.core.config import ConfigContext
from ml_switcheroo_compiler.core.device import Device, DeviceType
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
from ml_switcheroo_compiler.ops.vmap import _eager_vmap, vmap
from ml_switcheroo_compiler.transforms.passes.control_flow_opt import (
    _hoist_common_invariants,
    control_flow_optimization_pass,
    evaluate_static_predicate,
)
from ml_switcheroo_compiler.transforms.passes.parallel_scan import (
    _load_scan_rules,
    detect_associative_reduction,
    parallel_scan_pass,
)
from ml_switcheroo_compiler.transforms.passes.vectorization import (
    _load_vmap_rules,
    vectorization_pass,
    vectorize_graph,
)

device = Device(DeviceType.CPU, 0)


def test_vmap_no_tensor_args():
    """Test calling eager vmap when no arguments are Tensors."""
    with ConfigContext(eager_mode=True):
        f = vmap(lambda: 42)
        assert f() == 42


def test_vmap_backend_without_asarray():
    """Test _eager_vmap with a backend that lacks asarray."""
    x = Tensor(np.array([1, 2]), TensorConfig((2,), DType.Int32, device))

    class BackendNoAsarray:
        def execute_op(self, name, *args, **kwargs):
            return np.array([2, 4])

    with patch("ml_switcheroo_compiler.ops.vmap.get_active_backend", return_value=BackendNoAsarray()):
        res = _eager_vmap(lambda t: t, 0, 0, (x,))
        assert np.array_equal(res.data, np.array([1, 2]))


def test_evaluate_static_predicate_numpy_item():
    """Test evaluate_static_predicate with a numpy scalar having .item()."""
    g = IRGraph()
    np_bool = np.bool_(True)
    g.nodes["c_np"] = IRNode(id="c_np", op_type="Constant", attributes={"value": np_bool})
    assert evaluate_static_predicate(g, "c_np") is True


def test_evaluate_static_predicate_comparisons():
    """Test comparison branches in evaluate_static_predicate."""
    g = IRGraph()
    g.nodes["c1"] = IRNode(id="c1", op_type="Constant", attributes={"value": 1})
    g.nodes["c2"] = IRNode(id="c2", op_type="Constant", attributes={"value": 2})

    # Equal false
    g.nodes["eq_f"] = IRNode(id="eq_f", op_type="Equal", inputs=["c1", "c2"])
    assert evaluate_static_predicate(g, "eq_f") is False

    # Less false
    g.nodes["lt_f"] = IRNode(id="lt_f", op_type="Less", inputs=["c2", "c1"])
    assert evaluate_static_predicate(g, "lt_f") is False

    # LessEqual false
    g.nodes["le_f"] = IRNode(id="le_f", op_type="LessEqual", inputs=["c2", "c1"])
    assert evaluate_static_predicate(g, "le_f") is False

    # GreaterEqual true
    g.nodes["ge_t"] = IRNode(id="ge_t", op_type="GreaterEqual", inputs=["c2", "c1"])
    assert evaluate_static_predicate(g, "ge_t") is True

    # Other non-comparison op
    g.nodes["add"] = IRNode(id="add", op_type="Add", inputs=["c1", "c2"])
    assert evaluate_static_predicate(g, "add") is None

    # Non-constant inputs to Equal
    g.nodes["inp1"] = IRNode(id="inp1", op_type="Input", inputs=[])
    g.nodes["eq_dyn"] = IRNode(id="eq_dyn", op_type="Equal", inputs=["inp1", "c1"])
    assert evaluate_static_predicate(g, "eq_dyn") is None


def test_control_flow_opt_edge_cases():
    """Test edge cases in control_flow_optimization_pass."""
    g = IRGraph(name="test_edge")

    # Cond with non-constant predicate and no branches
    g.nodes["dyn_p"] = IRNode(id="dyn_p", op_type="Input", inputs=[])
    g.nodes["cond_empty"] = IRNode(id="cond_empty", op_type="Cond", inputs=["dyn_p"], attributes={})
    opt = control_flow_optimization_pass(g)
    assert "cond_empty" in opt.nodes

    # Invariant hoisting when input not in outer graph
    g2 = IRGraph(name="test_hoist_missing")
    tb = IRGraph(name="tb")
    tb.nodes["n1"] = IRNode(id="n1", op_type="Sqrt", inputs=["internal_only"])
    fb = IRGraph(name="fb")
    fb.nodes["n2"] = IRNode(id="n2", op_type="Sqrt", inputs=["internal_only"])
    assert _hoist_common_invariants(g2, "c1", tb, fb) == []

    # Non-IRNode in graph
    g3 = IRGraph()
    g3.nodes["not_ir"] = "just_a_string"
    opt3 = control_flow_optimization_pass(g3)
    assert opt3.nodes["not_ir"] == "just_a_string"


def test_parallel_scan_edge_cases():
    """Test parallel scan edge cases."""
    # Non-existent YAML path
    with patch("os.path.exists", return_value=False):
        assert _load_scan_rules() == {}

    # Multi-node or non-associative body
    body_multi = IRGraph()
    body_multi.nodes["n1"] = IRNode(id="n1", op_type="Add", inputs=["a", "b"])
    body_multi.nodes["n2"] = IRNode(id="n2", op_type="Multiply", inputs=["n1", "b"])
    assert detect_associative_reduction(body_multi, {"associative_ops": {"Add": {}}}) is None

    # Single-input scan (no initial carry)
    g = IRGraph()
    body = IRGraph()
    body.nodes["add"] = IRNode(id="add", op_type="Add", inputs=["x", "y"])
    g.nodes["xs"] = IRNode(id="xs", op_type="Input", inputs=[], shape_metadata=(5,))
    g.nodes["scan_single"] = IRNode(id="scan_single", op_type="Scan", inputs=["xs"], attributes={"body": body})
    opt = parallel_scan_pass(g)
    assert opt.nodes["scan_single"].op_type == "Identity"
    assert any("parallel_cumsum" in nid for nid in opt.nodes)


def test_vectorization_edge_cases():
    """Test vectorization pass edge cases and rules branches."""
    # Non-existent YAML path
    with patch("os.path.exists", return_value=False):
        assert _load_vmap_rules() == {}

    g = IRGraph(name="edge_vec")
    g.inputs = ["x"]
    g.nodes["x"] = IRNode(id="x", op_type="Input", inputs=[], shape_metadata=(4,))
    g.nodes["c1"] = IRNode(id="c1", op_type="Constant", inputs=[], attributes={"value": 1.0})
    g.nodes["red_axes"] = IRNode(
        id="red_axes",
        op_type="ReduceSum",
        inputs=["x"],
        attributes={"axes": (0, -1)},
        shape_metadata=(4,),
    )
    g.nodes["resh"] = IRNode(
        id="resh",
        op_type="Reshape",
        inputs=["x"],
        attributes={"shape": (2, 2), "newshape": (2, 2)},
        shape_metadata=(4,),
    )
    g.outputs = ["red_axes", "resh"]

    # Tuple out_axes
    vec = vectorize_graph(g, in_axes=(0,), batch_size=3, out_axes=(0, 0))
    assert vec.nodes["red_axes"].attributes["axes"] == (1, -1)
    assert vec.nodes["resh"].attributes["shape"] == (3, 2, 2)
    assert vec.nodes["resh"].attributes["newshape"] == (3, 2, 2)

    # Vmap node in outer graph without inputs
    outer = IRGraph()
    outer.nodes["vmap_no_inputs"] = IRNode(
        id="vmap_no_inputs",
        op_type="Vmap",
        inputs=[],
        attributes={"body": None},
    )
    outer_opt = vectorization_pass(outer)
    assert "vmap_no_inputs" in outer_opt.nodes


def test_vectorization_unbatched_subgraph_and_transpose():
    """Test vectorizing graph containing unbatched sub-computations and transpose."""
    g = IRGraph(name="unbatched_and_transpose")
    g.inputs = ["batched_x", "c1", "c2"]
    g.nodes["batched_x"] = IRNode(id="batched_x", op_type="Input", inputs=[], shape_metadata=(4, 5))
    g.nodes["c1"] = IRNode(id="c1", op_type="Input", inputs=[], shape_metadata=(2, 2))
    g.nodes["c2"] = IRNode(id="c2", op_type="Input", inputs=[], shape_metadata=(2, 2))

    # Unbatched node: both inputs are unbatched
    g.nodes["unbatched_add"] = IRNode(id="unbatched_add", op_type="Add", inputs=["c1", "c2"], shape_metadata=(2, 2))

    # Batched transpose node
    g.nodes["batched_trans"] = IRNode(
        id="batched_trans",
        op_type="Transpose",
        inputs=["batched_x"],
        attributes={"permutation": (1, 0)},
        shape_metadata=(5, 4),
    )
    g.outputs = ["unbatched_add", "batched_trans"]

    vec = vectorize_graph(g, in_axes={"batched_x": 0, "c1": None, "c2": None}, batch_size=4, out_axes=0)
    assert vec.nodes["unbatched_add"].shape_metadata == (2, 2)
    assert vec.nodes["batched_trans"].attributes["permutation"] == (0, 2, 1)


def test_control_flow_opt_input_output_skipping():
    """Test skipping Input and Output nodes during hoisting and active branch inlining."""
    g = IRGraph(name="test_in_out")
    g.nodes["pred_const"] = IRNode(id="pred_const", op_type="Constant", attributes={"value": True})

    tb = IRGraph(name="tb")
    tb.nodes["inp"] = IRNode(id="inp", op_type="Input", inputs=[])
    tb.nodes["out"] = IRNode(id="out", op_type="Output", inputs=["calc"])
    tb.nodes["calc"] = IRNode(id="calc", op_type="Exp", inputs=["inp"])
    tb.outputs = ["calc"]

    fb = IRGraph(name="fb")
    fb.nodes["inp_f"] = IRNode(id="inp_f", op_type="Input", inputs=[])
    fb.nodes["calc_f"] = IRNode(id="calc_f", op_type="Exp", inputs=["inp_f"])
    fb.outputs = ["calc_f"]

    g.nodes["cond_node"] = IRNode(
        id="cond_node",
        op_type="Cond",
        inputs=["pred_const"],
        attributes={"then_branch": tb, "else_branch": fb},
    )

    opt = control_flow_optimization_pass(g)
    assert opt.nodes["cond_node"].op_type == "Identity"

    # Also test invariant hoisting with Input/Output nodes
    g2 = IRGraph()
    g2.nodes["dyn_p"] = IRNode(id="dyn_p", op_type="Input", inputs=[])
    g2.nodes["ext"] = IRNode(id="ext", op_type="Input", inputs=[])

    tb2 = IRGraph()
    tb2.nodes["inp"] = IRNode(id="inp", op_type="Input", inputs=[])
    tb2.nodes["out"] = IRNode(id="out", op_type="Output", inputs=["h"])
    tb2.nodes["h"] = IRNode(id="h", op_type="Tanh", inputs=["ext"])

    fb2 = IRGraph()
    fb2.nodes["inp"] = IRNode(id="inp", op_type="Input", inputs=[])
    fb2.nodes["out"] = IRNode(id="out", op_type="Output", inputs=["h"])
    fb2.nodes["h"] = IRNode(id="h", op_type="Tanh", inputs=["ext"])

    hoisted = _hoist_common_invariants(g2, "c_node", tb2, fb2)
    assert len(hoisted) == 1
