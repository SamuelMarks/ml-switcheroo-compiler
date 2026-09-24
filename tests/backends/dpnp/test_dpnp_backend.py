"""Unit tests for Data Parallel NumPy (DPNP) backend."""

from __future__ import annotations

import sys
from unittest import mock

import numpy as np
import pytest

import ml_switcheroo_compiler.backends.dpnp as dpnp_pkg
from ml_switcheroo_compiler.backends.dpnp.generator import DPNPGenerator
from ml_switcheroo_compiler.backends.registry import BackendRegistry
from ml_switcheroo_compiler.core.errors import BackendNotSupportedError
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode


def _create_sample_graph(multi_output: bool = False, no_output: bool = False) -> IRGraph:
    """Create a sample computation graph for testing.

    Args:
        multi_output: Whether to create multiple graph outputs.
        no_output: Whether to create zero graph outputs.

    Returns:
        IRGraph: Test graph.
    """
    g = IRGraph()
    n_in1 = IRNode(id="in1", op_type="Input", inputs=[], shape_metadata=[2, 2])
    n_in2 = IRNode(id="in2", op_type="Input", inputs=[], shape_metadata=[2, 2])
    n_add = IRNode(id="add1", op_type="Add", inputs=["in1", "in2"], shape_metadata=[2, 2])
    n_sub = IRNode(id="sub1", op_type="Sub", inputs=["in1", "in2"], shape_metadata=[2, 2])
    g.nodes = {"in1": n_in1, "in2": n_in2, "add1": n_add, "sub1": n_sub}
    g.inputs = ["in1", "in2"]
    if no_output:
        g.outputs = []
    elif multi_output:
        g.outputs = ["add1", "sub1"]
    else:
        g.outputs = ["add1"]
    return g


def test_dpnp_registered() -> None:
    """Verify DPNP backend is registered in BackendRegistry."""
    assert BackendRegistry.get("dpnp") is DPNPGenerator


def test_dpnp_generator_and_aot() -> None:
    """Verify DPNP generator, SYCL queue configuration, and AOT compilation paths."""
    g = _create_sample_graph()
    gen = DPNPGenerator(g, device="gpu", sycl_queue="fake_queue")
    assert gen.device == "gpu"
    assert gen.sycl_queue == "fake_queue"
    assert gen.get_fallback_prefix() == "dpnp"
    assert gen.get_helper_functions() == []

    code = gen.generate()
    assert "import dpnp" in code
    assert "def evaluate(args):" in code

    node = IRNode(id="add_node", op_type="Add", inputs=["a", "b"])
    assert gen.generic_visit(node, ["a", "b"]) == "dpnp.add(a, b)"

    # AOT compile single output with Tensor unwrapping and kwargs
    runner = gen._compile_aot_impl(g)
    cfg = TensorConfig((2, 2), "float32", "cpu")
    t_in1 = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), cfg)
    t_in2 = Tensor(np.array([[5.0, 6.0], [7.0, 8.0]]), cfg)
    res = runner(t_in1, in2=np.array([[5.0, 6.0], [7.0, 8.0]]))
    assert res is not None

    # AOT compile multi-output
    g_multi = _create_sample_graph(multi_output=True)
    runner_multi = gen._compile_aot_impl(g_multi)
    res_multi = runner_multi(t_in1, t_in2)
    assert isinstance(res_multi, tuple)
    assert len(res_multi) == 2

    # AOT compile no outputs
    g_no_out = _create_sample_graph(no_output=True)
    runner_no_out = gen._compile_aot_impl(g_no_out)
    res_no_out = runner_no_out(t_in1, t_in2)
    assert isinstance(res_no_out, dict)


def test_dpnp_types() -> None:
    """Verify DPNP types creation functions and fallback handling."""
    # Direct calls with tuples / sequences
    z = dpnp_pkg.zeros((2, 3))
    assert z is not None
    z_cls = dpnp_pkg.DPNPGenerator.zeros((2, 3))
    assert z_cls is not None

    arr = dpnp_pkg.array([1.0, 2.0])
    assert arr is not None
    arr_cls = dpnp_pkg.DPNPGenerator.array([1.0, 2.0])
    assert arr_cls is not None

    # Custom dtype branch
    arr_dtype = dpnp_pkg.array([1.0, 2.0], dtype="float64")
    assert arr_dtype is not None
    arr_dtype_cls = dpnp_pkg.DPNPGenerator.array([1.0, 2.0], dtype="float32")
    assert arr_dtype_cls is not None

    as_arr = dpnp_pkg.asarray([3.0, 4.0])
    assert as_arr is not None
    as_arr_cls = dpnp_pkg.DPNPGenerator.asarray([3.0, 4.0])
    assert as_arr_cls is not None

    # item extraction
    val = dpnp_pkg.item(np.array([4.2]))
    assert np.isclose(val, 4.2)
    val_cls = dpnp_pkg.DPNPGenerator.item(np.array([4.2]))
    assert np.isclose(val_cls, 4.2)

    # item fallback when object lacks .item()
    class NoItemObj:
        """Object without .item() method implementing __array__."""

        def __array__(self) -> np.ndarray:
            """Convert to array.

            Returns:
                np.ndarray: Array representation.
            """
            return np.array(7.5)

    val_no_item = dpnp_pkg.item(NoItemObj())
    assert np.isclose(val_no_item, 7.5)

    # When dpnp is mock-imported
    mock_dpnp = mock.MagicMock()
    mock_dpnp.zeros.return_value = "mock_zeros"
    mock_dpnp.array.return_value = "mock_array"
    mock_dpnp.asarray.return_value = "mock_asarray"
    with mock.patch.dict(sys.modules, {"dpnp": mock_dpnp}):
        assert dpnp_pkg.zeros((1,)) == "mock_zeros"
        assert dpnp_pkg.array([1]) == "mock_array"
        assert dpnp_pkg.asarray([1]) == "mock_asarray"


def test_dpnp_eager_dispatch() -> None:
    """Verify DPNP eager op dispatching, class context, aliases, and errors."""
    # Normal dispatch
    res = dpnp_pkg.execute_op("Add", np.array([1.0, 2.0]), np.array([3.0, 4.0]))
    np.testing.assert_allclose(res, [4.0, 6.0])

    # Classmethod invocation: cls_or_op is class
    res_cls = dpnp_pkg.DPNPGenerator.execute_op("Add", np.array([1.0, 2.0]), np.array([3.0, 4.0]))
    np.testing.assert_allclose(res_cls, [4.0, 6.0])

    # Alias mapping branch: Sub -> subtract, Mul -> multiply, Div -> divide, Neg -> negative
    sub_res = dpnp_pkg.execute_op("Sub", np.array([5.0]), np.array([2.0]))
    np.testing.assert_allclose(sub_res, [3.0])
    mul_res = dpnp_pkg.execute_op("Mul", np.array([5.0]), np.array([2.0]))
    np.testing.assert_allclose(mul_res, [10.0])
    div_res = dpnp_pkg.execute_op("Div", np.array([6.0]), np.array([2.0]))
    np.testing.assert_allclose(div_res, [3.0])
    neg_res = dpnp_pkg.execute_op("Neg", np.array([5.0]))
    np.testing.assert_allclose(neg_res, [-5.0])
    truediv_res = dpnp_pkg.execute_op("TrueDivide", np.array([7.0]), np.array([2.0]))
    np.testing.assert_allclose(truediv_res, [3.5])

    # Unsupported op
    with pytest.raises(BackendNotSupportedError, match="not supported"):
        dpnp_pkg.execute_op("UnknownCustomOp12345")

    # With mocked dpnp present
    mock_dpnp = mock.MagicMock()
    mock_dpnp.add.return_value = "dpnp_added"
    with mock.patch.dict(sys.modules, {"dpnp": mock_dpnp}):
        assert dpnp_pkg.execute_op("Add", 1, 2) == "dpnp_added"
