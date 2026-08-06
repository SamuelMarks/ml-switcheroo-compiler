# ruff: noqa: E501
from ml_switcheroo_compiler.ir.core import LogicalGraph, LogicalNode
from ml_switcheroo_compiler.transforms.autodiff_rules.common import make_zero_jvp, make_zero_vjp
from ml_switcheroo_compiler.transforms.autodiff_rules.jvp_registry import get_jvp
from ml_switcheroo_compiler.transforms.autodiff_rules.signal_rules import dct_jvp, frame_jvp, idct_jvp, inverse_mdct_jvp, mdct_jvp, overlap_and_add_jvp
from ml_switcheroo_compiler.transforms.autodiff_rules.time_distributed_rules import time_distributed_jvp
from ml_switcheroo_compiler.transforms.autodiff_rules.vjp_registry import get_vjp

"Extra tests for extra rules."


def test_make_zero_vjp_and_jvp() -> None:
    """Test the zero vjp and jvp generators."""
    vjp_func = make_zero_vjp("MyOp")

    class DummyNode:
        inputs = ["in1", "in2"]

    res = vjp_func(None, DummyNode(), "cotangent")
    from ml_switcheroo_compiler.transforms.autodiff_rules.common import UnconnectedGradients

    assert res == (UnconnectedGradients.ZERO, UnconnectedGradients.ZERO)
    jvp_func = make_zero_jvp("MyOp")
    res_jvp = jvp_func(None, DummyNode(), ("t1", "t2"))
    assert res_jvp is None


"Test extra rules coverage."


def test_extra_rules_jvps_none() -> None:
    """Test JVPs returning None when tangent is None."""
    graph = LogicalGraph(name="TestGraph")
    in_node = LogicalNode(id="in_a", op_type="Op", inputs=[], shape_metadata=None)
    graph.nodes["in_a"] = in_node
    node = LogicalNode(id="n", op_type="Op", inputs=["in_a"], shape_metadata=None)
    assert dct_jvp(graph, node, (None,)) is None
    assert idct_jvp(graph, node, (None,)) is None
    assert time_distributed_jvp(graph, node, (None,)) is None
    assert frame_jvp(graph, node, (None,)) is None
    assert overlap_and_add_jvp(graph, node, (None,)) is None
    assert mdct_jvp(graph, node, (None,)) is None
    assert inverse_mdct_jvp(graph, node, (None,)) is None
    assert dct_jvp(graph, node, ("t",)) is not None
    assert idct_jvp(graph, node, ("t",)) is not None
    assert time_distributed_jvp(graph, node, ("t",)) is not None
    assert frame_jvp(graph, node, ("t",)) is not None
    assert overlap_and_add_jvp(graph, node, ("t",)) is not None
    assert mdct_jvp(graph, node, ("t",)) is not None
    assert inverse_mdct_jvp(graph, node, ("t",)) is not None
    from ml_switcheroo_compiler.transforms.autodiff_rules.signal_rules import dct_vjp, frame_vjp, idct_vjp, inverse_mdct_vjp, mdct_vjp, overlap_and_add_vjp
    from ml_switcheroo_compiler.transforms.autodiff_rules.time_distributed_rules import time_distributed_vjp

    assert dct_vjp(graph, node, "t") is not None
    assert idct_vjp(graph, node, "t") is not None
    assert time_distributed_vjp(graph, node, "t") is not None
    assert frame_vjp(graph, node, "t") is not None
    assert overlap_and_add_vjp(graph, node, "t") is not None
    assert mdct_vjp(graph, node, "t") is not None
    assert inverse_mdct_vjp(graph, node, "t") is not None


def test_extra_rules_unimplemented() -> None:
    """Test unimplemented ops."""
    graph = LogicalGraph(name="TestGraph")
    node = LogicalNode(id="n", op_type="Op", inputs=["a"], shape_metadata=None)
    vjp = get_vjp("CudaKernel")
    assert vjp(graph, node, "t_a") is not None
    jvp = get_jvp("CudaKernel")
    assert jvp(graph, node, ("t_a",)) in (None, 0.0)
    vjp = get_vjp("MelFilterbank")
    assert vjp(graph, node, "t_a") is not None
    jvp = get_jvp("MelFilterbank")
    assert jvp(graph, node, ("t_a",)) in (None, 0.0)
    for op in ["GroupMean", "GroupNorm", "GroupVariance", "Rope", "ScaledDotProductAttention"]:
        assert get_vjp(op) is not None
        assert get_jvp(op) is not None
