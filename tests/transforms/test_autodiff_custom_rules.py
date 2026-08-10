"""Coverage tests for custom rules autodiff."""

from ml_switcheroo_ir import LogicalGraph, LogicalNode

from ml_switcheroo_compiler.transforms.autodiff_rules.common import UnconnectedGradients
from ml_switcheroo_compiler.transforms.autodiff_rules.custom_rules import (
    _assoc_scan_jvp,
    _assoc_scan_vjp,
    _if_jvp,
    _if_vjp,
    _inline_grad_subgraph,
    _inline_subgraph,
    _loop_jvp,
    _loop_vjp,
    _scan_jvp,
    _scan_vjp,
    checkpoint_vjp,
)


def test_if_vjp():
    assert _if_vjp(None, None, None) == (UnconnectedGradients.ZERO,)


def test_loop_vjp():
    node = LogicalNode(id="n1", op_type="Loop", inputs=["a", "b"])
    assert _loop_vjp(None, node, None) == (UnconnectedGradients.ZERO, UnconnectedGradients.ZERO)


def test_scan_vjp():
    node = LogicalNode(id="n1", op_type="Scan", inputs=["a"])
    assert _scan_vjp(None, node, None) == (UnconnectedGradients.ZERO,)


def test_assoc_scan_vjp():
    node = LogicalNode(id="n1", op_type="AssociativeScan", inputs=["a", "b", "c"])
    assert _assoc_scan_vjp(None, node, None) == (UnconnectedGradients.ZERO, UnconnectedGradients.ZERO, UnconnectedGradients.ZERO)


def test_jvp_nulls():
    assert _if_jvp(None, None, None) == ""
    assert _loop_jvp(None, None, None) == ""
    assert _scan_jvp(None, None, None) == ""
    assert _assoc_scan_jvp(None, None, None) == ""


def test_zero_vjps_custom():
    from ml_switcheroo_compiler.transforms.autodiff_rules.jvp_registry import _JVP_REGISTRY
    from ml_switcheroo_compiler.transforms.autodiff_rules.vjp_registry import _VJP_REGISTRY

    for op_name in [
        "CudaKernel",
        "MetalKernel",
        "PrecompiledCudaKernel",
        "TopK",
        "Cholesky",
        "CholeskyEx",
        "Eig",
        "Eigh",
        "Eigvals",
        "Eigvalsh",
        "FFT",
        "IFFT",
        "Sort",
        "SortComplex",
        "SortKeyVal",
        "Argsort",
        "Fftconvolve",
        "Fft",
        "Rfft",
        "Fft2",
        "Fftfreq",
        "Irfft",
        "Ihfft",
        "Ifft",
        "Fftn",
        "Ifftn",
        "Rfftn",
        "Irfftn",
        "Ifft2",
        "Rfft2",
        "Irfft2",
        "Fftnd",
        "Ifftnd",
        "Rfftnd",
        "Irfftnd",
        "Fftshift",
        "Ifftshift",
        "Hfft",
        "Rfftfreq",
    ]:
        assert op_name in _VJP_REGISTRY
        assert op_name in _JVP_REGISTRY

        node = LogicalNode(id="n1", op_type=op_name, inputs=["a", "b"])
        vjp_func = _VJP_REGISTRY[op_name]
        res = vjp_func(None, node, "cot")
        assert res == (UnconnectedGradients.ZERO, UnconnectedGradients.ZERO)

        jvp_func = _JVP_REGISTRY[op_name]
        assert jvp_func(None, node, ("tan1", "tan2")) == ""


def test_inline_subgraph():
    graph = LogicalGraph()

    subgraph = LogicalGraph()
    # Create some nodes
    subgraph.nodes["in1"] = LogicalNode(id="in1", op_type="Input")
    subgraph.nodes["add1"] = LogicalNode(id="add1", op_type="Add", inputs=["in1", "nonexistent"])

    node = LogicalNode(id="cp_node", op_type="dummy", inputs=["real_in1", "real_in2"])

    id_map = {"in1": "real_in1", "add1": "new_add1", "nonexistent": "new_nonexistent"}

    _inline_subgraph(graph, subgraph, node, id_map)

    assert "new_add1" in graph.nodes
    assert graph.nodes["new_add1"].inputs == ["real_in1", "new_nonexistent"]


def test_inline_grad_subgraph():
    graph = LogicalGraph()

    sg = LogicalGraph()
    sg.inputs = ["sg_in1"]

    sg_grad = LogicalGraph()
    sg_grad.nodes["in_node"] = LogicalNode(id="in_node", op_type="Input")
    sg_grad.nodes["out_node"] = LogicalNode(id="out_node", op_type="Output")
    sg_grad.nodes["cot"] = LogicalNode(id="cot", op_type="Add")
    sg_grad.nodes["add_grad"] = LogicalNode(id="add_grad", op_type="Add", inputs=["sg_in1", "cot"])
    sg_grad.outputs = ["add_grad"]

    node = LogicalNode(id="cp_node", op_type="dummy", inputs=["real_in1"])
    cotangent_mapping = {"sg_out1": "cot"}

    adjoints = _inline_grad_subgraph(graph, sg_grad, sg, node, cotangent_mapping)

    # We should have an adjoint corresponding to the output
    assert len(adjoints) == 1

    # Check that add_grad was inlined with new inputs
    inlined_node_id = adjoints[0]
    assert inlined_node_id in graph.nodes

    inlined_node = graph.nodes[inlined_node_id]
    assert "real_in1" in inlined_node.inputs  # mapped from sg_in1
    assert "cot" in inlined_node.inputs  # mapped from cot


def test_checkpoint_vjp():
    graph = LogicalGraph()
    graph.nodes["real_in1"] = LogicalNode(id="real_in1", op_type="Input")

    subgraph = LogicalGraph()
    subgraph.nodes["in1"] = LogicalNode(id="in1", op_type="Input")
    subgraph.nodes["mul"] = LogicalNode(id="mul", op_type="Multiply", inputs=["in1", "in1"])
    subgraph.nodes["out"] = LogicalNode(id="out", op_type="Output", inputs=["mul"])
    subgraph.inputs = ["in1"]
    subgraph.outputs = ["mul"]

    node = LogicalNode(id="cp_node", op_type="Checkpoint", inputs=["real_in1"], attributes={"subgraph": subgraph})

    cotangent = "my_cotangent"

    from unittest.mock import patch

    # The import in custom_rules.py is: from ml_switcheroo_compiler.transforms.autodiff import grad as graph_grad
    # We patch ml_switcheroo_compiler.transforms.autodiff.grad directly
    with patch("ml_switcheroo_compiler.transforms.autodiff.grad") as mock_grad:
        sg_grad = LogicalGraph()
        sg_grad.nodes["add_grad"] = LogicalNode(id="add_grad", op_type="Add", inputs=["in1", "cot"])
        sg_grad.outputs = ["add_grad"]
        mock_grad.return_value = sg_grad

        adjoints = checkpoint_vjp(graph, node, cotangent)
        assert len(adjoints) == 1
