"""Comprehensive tests for roadmap backends (dpnp, awkward, pyarrow_compute, bohrium)."""

from __future__ import annotations

import sys
from unittest import mock

import numpy as np
import pytest

import ml_switcheroo_compiler.backends.awkward as ak_pkg
import ml_switcheroo_compiler.backends.bohrium as bh_pkg
import ml_switcheroo_compiler.backends.dpnp as dpnp_pkg
import ml_switcheroo_compiler.backends.pyarrow_compute as pa_pkg
from ml_switcheroo_compiler.backends.awkward.generator import AwkwardGenerator
from ml_switcheroo_compiler.backends.bohrium.generator import BohriumGenerator
from ml_switcheroo_compiler.backends.dpnp.generator import DPNPGenerator
from ml_switcheroo_compiler.backends.pyarrow_compute.generator import PyArrowComputeGenerator
from ml_switcheroo_compiler.backends.registry import BackendRegistry
from ml_switcheroo_compiler.core.errors import BackendNotSupportedError
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode


def _create_sample_graph(multi_output: bool = False, no_output: bool = False) -> IRGraph:
    """Create a sample computation graph for testing.

    Args:
        multi_output (bool): Whether to create multiple graph outputs.
        no_output (bool): Whether to create zero graph outputs.

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


def test_roadmap_backends_registered() -> None:
    """Verify all 4 roadmap backends are correctly registered in BackendRegistry."""
    assert BackendRegistry.get("dpnp") is DPNPGenerator
    assert BackendRegistry.get("awkward") is AwkwardGenerator
    assert BackendRegistry.get("pyarrow_compute") is PyArrowComputeGenerator
    assert BackendRegistry.get("bohrium") is BohriumGenerator


# ==============================================================================
# DPNP Backend Tests
# ==============================================================================


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


# ==============================================================================
# Awkward Backend Tests
# ==============================================================================


def test_awkward_generator_and_aot() -> None:
    """Verify Awkward Array generator, ragged mode, and AOT execution."""
    g = _create_sample_graph()
    gen = AwkwardGenerator(g, ragged_mode=True)
    assert gen.ragged_mode is True
    assert gen.get_fallback_prefix() == "ak"
    assert gen.get_helper_functions() == []

    gen_non_ragged = AwkwardGenerator(g, ragged_mode=False)
    assert gen_non_ragged.ragged_mode is False

    code = gen.generate()
    assert "import awkward as ak" in code
    assert "def evaluate(args):" in code

    node = IRNode(id="sum_node", op_type="Sum", inputs=["x"])
    assert gen.generic_visit(node, ["x"]) == "ak.sum(x)"

    # AOT single output with Tensor unwrapping
    runner = gen._compile_aot_impl(g)
    cfg = TensorConfig((2, 2), "float32", "cpu")
    t_in1 = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), cfg)
    t_in2 = Tensor(np.array([[5.0, 6.0], [7.0, 8.0]]), cfg)
    res = runner(t_in1, in2=np.array([[5.0, 6.0], [7.0, 8.0]]))
    assert res is not None

    # AOT multi-output
    g_multi = _create_sample_graph(multi_output=True)
    runner_multi = gen._compile_aot_impl(g_multi)
    res_multi = runner_multi(t_in1, t_in2)
    assert isinstance(res_multi, tuple)
    assert len(res_multi) == 2

    # AOT no outputs
    g_no_out = _create_sample_graph(no_output=True)
    runner_no_out = gen._compile_aot_impl(g_no_out)
    res_no_out = runner_no_out(t_in1, t_in2)
    assert isinstance(res_no_out, dict)


def test_awkward_types() -> None:
    """Verify Awkward types conversion, ragged array handling, and fallbacks."""
    z = ak_pkg.zeros((2, 2))
    assert z is not None
    z_cls = ak_pkg.AwkwardGenerator.zeros((2, 2))
    assert z_cls is not None

    arr = ak_pkg.array([1.0, 2.0])
    assert arr is not None
    arr_cls = ak_pkg.AwkwardGenerator.array([1.0, 2.0])
    assert arr_cls is not None

    as_arr = ak_pkg.asarray([1.0, 2.0])
    assert as_arr is not None
    as_arr_cls = ak_pkg.AwkwardGenerator.asarray([1.0, 2.0])
    assert as_arr_cls is not None

    # Ragged array data triggering ValueError in regular np.array
    ragged_data = [[1.0, 2.0], [3.0]]
    arr_ragged = ak_pkg.array(ragged_data)
    assert arr_ragged is not None

    # item extraction via .item()
    val = ak_pkg.item(np.array([9.5]))
    assert np.isclose(val, 9.5)
    val_cls = ak_pkg.AwkwardGenerator.item(np.array([9.5]))
    assert np.isclose(val_cls, 9.5)

    # item extraction via .to_list() (scalar and list)
    class FakeAkArrayScalar:
        """Mock awkward array with scalar to_list."""

        def to_list(self) -> float:
            """Return scalar list representation.

            Returns:
                float: Scalar representation.
            """
            return 8.25

    class FakeAkArrayList:
        """Mock awkward array with list to_list."""

        def to_list(self) -> list[float]:
            """Return list representation.

            Returns:
                list[float]: Single-element list representation.
            """
            return [14.5]

    assert np.isclose(ak_pkg.item(FakeAkArrayScalar()), 8.25)
    assert np.isclose(ak_pkg.item(FakeAkArrayList()), 14.5)

    # item fallback when no .to_list() or .item()
    class NoItemAkObj:
        """Object without .to_list() or .item()."""

        def __array__(self) -> np.ndarray:
            """Convert to array.

            Returns:
                np.ndarray: Array representation.
            """
            return np.array(3.14)

    assert np.isclose(ak_pkg.item(NoItemAkObj()), 3.14)

    # When awkward module is present with Array class
    mock_ak = mock.MagicMock()
    mock_ak.Array = lambda x: f"ak_array_{type(x)}"
    with mock.patch.dict(sys.modules, {"awkward": mock_ak}):
        assert "ak_array_" in str(ak_pkg.zeros((2,)))
        assert "ak_array_" in str(ak_pkg.array([1, 2]))
        assert "ak_array_" in str(ak_pkg.asarray([3, 4]))


def test_awkward_eager_dispatch() -> None:
    """Verify Awkward eager op execution, aliases, mock awkward, and errors."""
    res = ak_pkg.execute_op("Add", np.array([2.0]), np.array([3.0]))
    np.testing.assert_allclose(res, [5.0])

    # Classmethod execution
    res_cls = ak_pkg.AwkwardGenerator.execute_op("Add", np.array([2.0]), np.array([3.0]))
    np.testing.assert_allclose(res_cls, [5.0])

    # Alias mapping branch: Sub, Mul, Div, Neg, TrueDivide
    sub_res = ak_pkg.execute_op("Sub", np.array([10.0]), np.array([4.0]))
    np.testing.assert_allclose(sub_res, [6.0])
    mul_res = ak_pkg.execute_op("Mul", np.array([3.0]), np.array([4.0]))
    np.testing.assert_allclose(mul_res, [12.0])
    div_res = ak_pkg.execute_op("Div", np.array([15.0]), np.array([3.0]))
    np.testing.assert_allclose(div_res, [5.0])
    neg_res = ak_pkg.execute_op("Neg", np.array([7.0]))
    np.testing.assert_allclose(neg_res, [-7.0])
    truediv_res = ak_pkg.execute_op("TrueDivide", np.array([7.0]), np.array([2.0]))
    np.testing.assert_allclose(truediv_res, [3.5])

    # Unsupported op
    with pytest.raises(BackendNotSupportedError, match="not supported"):
        ak_pkg.execute_op("UnknownCustomOp12345")

    # When awkward module is present
    mock_ak = mock.MagicMock()
    mock_ak.add.return_value = "ak_added"
    with mock.patch.dict(sys.modules, {"awkward": mock_ak}):
        assert ak_pkg.execute_op("Add", 1, 2) == "ak_added"


# ==============================================================================
# PyArrow Compute Backend Tests
# ==============================================================================


def test_pyarrow_compute_generator_and_aot() -> None:
    """Verify PyArrow Compute generator, zero_copy option, and AOT compilation."""
    g = _create_sample_graph()
    gen = PyArrowComputeGenerator(g, zero_copy=True)
    assert gen.zero_copy is True
    assert gen.get_fallback_prefix() == "pc"
    assert gen.get_helper_functions() == []

    gen_no_copy = PyArrowComputeGenerator(g, zero_copy=False)
    assert gen_no_copy.zero_copy is False

    code = gen.generate()
    assert "import pyarrow.compute as pc" in code
    assert "def evaluate(args):" in code

    node = IRNode(id="mul_node", op_type="Mul", inputs=["a", "b"])
    assert gen.generic_visit(node, ["a", "b"]) == "pc.mul(a, b)"

    # AOT single output with Tensor unwrapping
    runner = gen._compile_aot_impl(g)
    cfg = TensorConfig((2, 2), "float32", "cpu")
    t_in1 = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), cfg)
    t_in2 = Tensor(np.array([[5.0, 6.0], [7.0, 8.0]]), cfg)
    res = runner(t_in1, in2=np.array([[5.0, 6.0], [7.0, 8.0]]))
    assert res is not None

    # AOT multi-output
    g_multi = _create_sample_graph(multi_output=True)
    runner_multi = gen._compile_aot_impl(g_multi)
    res_multi = runner_multi(t_in1, t_in2)
    assert isinstance(res_multi, tuple)
    assert len(res_multi) == 2

    # AOT no outputs
    g_no_out = _create_sample_graph(no_output=True)
    runner_no_out = gen._compile_aot_impl(g_no_out)
    res_no_out = runner_no_out(t_in1, t_in2)
    assert isinstance(res_no_out, dict)


def test_pyarrow_compute_types() -> None:
    """Verify PyArrow Compute array creation, scalar extraction, and fallbacks."""
    z = pa_pkg.zeros((4,))
    assert z is not None
    z_cls = pa_pkg.PyArrowComputeGenerator.zeros((4,))
    assert z_cls is not None

    arr = pa_pkg.array([10.0, 20.0])
    assert arr is not None
    arr_cls = pa_pkg.PyArrowComputeGenerator.array([10.0, 20.0])
    assert arr_cls is not None

    as_arr = pa_pkg.asarray([30.0, 40.0])
    assert as_arr is not None
    as_arr_cls = pa_pkg.PyArrowComputeGenerator.asarray([30.0, 40.0])
    assert as_arr_cls is not None

    # item extraction via as_py
    import pyarrow as pa

    scalar = pa.scalar(12.5)
    assert np.isclose(pa_pkg.item(scalar), 12.5)

    # item extraction via to_pylist
    pa_arr = pa.array([7.25])
    assert np.isclose(pa_pkg.item(pa_arr), 7.25)
    assert np.isclose(pa_pkg.PyArrowComputeGenerator.item(pa_arr), 7.25)

    # item extraction via item()
    val_np = pa_pkg.item(np.array([42.0]))
    assert np.isclose(val_np, 42.0)

    # item extraction fallback when no as_py, to_pylist, or item
    class NoItemPaObj:
        """Object without specialized item methods."""

        def __array__(self) -> np.ndarray:
            """Convert to array.

            Returns:
                np.ndarray: Array representation.
            """
            return np.array(99.0)

    assert np.isclose(pa_pkg.item(NoItemPaObj()), 99.0)

    # item extraction when to_pylist returns empty list
    class EmptyPylistPaObj:
        """Object with to_pylist returning empty list."""

        def to_pylist(self) -> list[object]:
            """Return empty list.

            Returns:
                list[object]: Empty list.
            """
            return []

        def item(self) -> float:
            """Return scalar.

            Returns:
                float: Extracted value.
            """
            return 55.0

    assert np.isclose(pa_pkg.item(EmptyPylistPaObj()), 55.0)

    # Test array and asarray exception fallback branch to numpy
    class BadObjPa:
        """Object that fails during pyarrow.array conversion."""

        def __iter__(self) -> object:
            """Fail on iteration.

            Raises:
                TypeError: Explicit iteration failure.
            """
            raise TypeError("cannot iterate")

    bad_inst = BadObjPa()
    assert isinstance(pa_pkg.array(bad_inst), np.ndarray)
    assert isinstance(pa_pkg.asarray(bad_inst), np.ndarray)

    # When pa_mod lacks 'array' attribute
    mock_pa_no_array = mock.MagicMock(spec=[])
    with mock.patch("ml_switcheroo_compiler.backends.pyarrow_compute.types._get_pa_module", return_value=mock_pa_no_array):
        assert isinstance(pa_pkg.zeros((2, 2)), np.ndarray)
        assert isinstance(pa_pkg.array([1, 2]), np.ndarray)
        assert isinstance(pa_pkg.asarray([1, 2]), np.ndarray)

    # Fallbacks when pyarrow.array raises exception in zeros
    mock_pa_fail = mock.MagicMock()
    mock_pa_fail.array.side_effect = TypeError("Conversion failed")
    with mock.patch("ml_switcheroo_compiler.backends.pyarrow_compute.types._get_pa_module", return_value=mock_pa_fail):
        assert isinstance(pa_pkg.zeros((3,)), np.ndarray)

    # When pyarrow module import fails
    with mock.patch.dict(sys.modules, {"pyarrow": None}):
        with mock.patch(
            "importlib.import_module",
            side_effect=lambda name: np if name == "numpy" else (_ for _ in ()).throw(ImportError(name)),
        ):
            pa_mod = pa_pkg.types._get_pa_module()
            assert pa_mod is np


def test_pyarrow_compute_eager_dispatch() -> None:
    """Verify PyArrow Compute eager op execution, mappings, aliases, and errors."""
    # Direct function call in pyarrow.compute (e.g., Abs -> pc.abs)
    import pyarrow as pa

    pa_val = pa.array([-5.0, 3.0])
    res_abs = pa_pkg.execute_op("Abs", pa_val)
    assert res_abs is not None

    res = pa_pkg.execute_op("Mul", np.array([2.0, 3.0]), np.array([4.0, 5.0]))
    assert res is not None

    # Classmethod execution
    res_cls = pa_pkg.PyArrowComputeGenerator.execute_op("Mul", np.array([2.0, 3.0]), np.array([4.0, 5.0]))
    assert res_cls is not None

    # Operations mapped in pc_map (sum, mean, equal, etc.)
    import pyarrow as pa

    pa_1 = pa.array([1.0, 2.0])
    pa_2 = pa.array([3.0, 4.0])
    res_add = pa_pkg.execute_op("Add", pa_1, pa_2)
    assert res_add is not None
    res_sub = pa_pkg.execute_op("Sub", pa_2, pa_1)
    assert res_sub is not None
    res_div = pa_pkg.execute_op("Div", pa_2, pa_1)
    assert res_div is not None
    res_sum = pa_pkg.execute_op("Sum", pa_1)
    assert res_sum is not None
    res_mean = pa_pkg.execute_op("Mean", pa_1)
    assert res_mean is not None
    res_min = pa_pkg.execute_op("Min", pa_1)
    assert res_min is not None
    res_max = pa_pkg.execute_op("Max", pa_1)
    assert res_max is not None
    res_eq = pa_pkg.execute_op("Equal", pa_1, pa_2)
    assert res_eq is not None
    res_gt = pa_pkg.execute_op("Greater", pa_2, pa_1)
    assert res_gt is not None
    res_lt = pa_pkg.execute_op("Less", pa_1, pa_2)
    assert res_lt is not None

    # Fallback to NumPy when pyarrow.compute cannot be imported
    with mock.patch(
        "importlib.import_module",
        side_effect=lambda name: np if name == "numpy" else (_ for _ in ()).throw(ImportError(name)),
    ):
        res_np_fallback = pa_pkg.execute_op("Add", np.array([1.0]), np.array([2.0]))
        np.testing.assert_allclose(res_np_fallback, [3.0])

    # Fallback to NumPy when pyarrow.compute is None
    with mock.patch("importlib.import_module", side_effect=lambda name: np if name == "numpy" else None):
        res_np_add = pa_pkg.execute_op("Add", np.array([1.0]), np.array([2.0]))
        np.testing.assert_allclose(res_np_add, [3.0])
        res_np_neg = pa_pkg.execute_op("Neg", np.array([5.0]))
        np.testing.assert_allclose(res_np_neg, [-5.0])
        res_np_truediv = pa_pkg.execute_op("TrueDivide", np.array([7.0]), np.array([2.0]))
        np.testing.assert_allclose(res_np_truediv, [3.5])

    # Unsupported op
    with pytest.raises(BackendNotSupportedError, match="not supported"):
        pa_pkg.execute_op("UnknownCustomOp12345")


# ==============================================================================
# Bohrium Backend Tests
# ==============================================================================


def test_bohrium_generator_and_aot() -> None:
    """Verify Bohrium generator, stack options, and AOT compilation."""
    g = _create_sample_graph()
    gen = BohriumGenerator(g, stack="opencl")
    assert gen.stack == "opencl"
    assert gen.get_fallback_prefix() == "bh"
    assert gen.get_helper_functions() == []

    code = gen.generate()
    assert "import bohrium as bh" in code
    assert "def evaluate(args):" in code

    node = IRNode(id="div_node", op_type="Div", inputs=["a", "b"])
    assert gen.generic_visit(node, ["a", "b"]) == "bh.div(a, b)"

    # AOT single output with Tensor unwrapping
    runner = gen._compile_aot_impl(g)
    cfg = TensorConfig((2, 2), "float32", "cpu")
    t_in1 = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), cfg)
    t_in2 = Tensor(np.array([[5.0, 6.0], [7.0, 8.0]]), cfg)
    res = runner(t_in1, in2=np.array([[5.0, 6.0], [7.0, 8.0]]))
    assert res is not None

    # AOT multi-output
    g_multi = _create_sample_graph(multi_output=True)
    runner_multi = gen._compile_aot_impl(g_multi)
    res_multi = runner_multi(t_in1, t_in2)
    assert isinstance(res_multi, tuple)
    assert len(res_multi) == 2

    # AOT no outputs
    g_no_out = _create_sample_graph(no_output=True)
    runner_no_out = gen._compile_aot_impl(g_no_out)
    res_no_out = runner_no_out(t_in1, t_in2)
    assert isinstance(res_no_out, dict)


def test_bohrium_types() -> None:
    """Verify Bohrium types conversion, dtype specification, and fallbacks."""
    z = bh_pkg.zeros((3, 3))
    assert z is not None
    z_cls = bh_pkg.BohriumGenerator.zeros((3, 3))
    assert z_cls is not None

    arr = bh_pkg.array([1.0, 2.0, 3.0])
    assert arr is not None
    arr_cls = bh_pkg.BohriumGenerator.array([1.0, 2.0, 3.0])
    assert arr_cls is not None

    # Custom dtype branch
    arr_dtype = bh_pkg.array([1.0, 2.0], dtype="float64")
    assert arr_dtype is not None
    arr_dtype_cls = bh_pkg.BohriumGenerator.array([1.0, 2.0], dtype="float32")
    assert arr_dtype_cls is not None

    as_arr = bh_pkg.asarray([4.0, 5.0, 6.0])
    assert as_arr is not None
    as_arr_cls = bh_pkg.BohriumGenerator.asarray([4.0, 5.0, 6.0])
    assert as_arr_cls is not None

    val = bh_pkg.item(np.array([12.0]))
    assert np.isclose(val, 12.0)
    val_cls = bh_pkg.BohriumGenerator.item(np.array([12.0]))
    assert np.isclose(val_cls, 12.0)

    # item fallback when object lacks .item()
    class NoItemBhObj:
        """Object without .item() method."""

        def __array__(self) -> np.ndarray:
            """Convert to array.

            Returns:
                np.ndarray: Array representation.
            """
            return np.array(24.0)

    assert np.isclose(bh_pkg.item(NoItemBhObj()), 24.0)

    # When bohrium is mock-imported
    mock_bh = mock.MagicMock()
    mock_bh.zeros.return_value = "mock_bh_zeros"
    mock_bh.array.return_value = "mock_bh_array"
    mock_bh.asarray.return_value = "mock_bh_asarray"
    with mock.patch.dict(sys.modules, {"bohrium": mock_bh}):
        assert bh_pkg.zeros((1,)) == "mock_bh_zeros"
        assert bh_pkg.array([1]) == "mock_bh_array"
        assert bh_pkg.asarray([1]) == "mock_bh_asarray"


def test_bohrium_eager_dispatch() -> None:
    """Verify Bohrium eager op execution, aliases, mock bohrium, and errors."""
    res = bh_pkg.execute_op("Add", np.array([10.0]), np.array([20.0]))
    np.testing.assert_allclose(res, [30.0])

    # Classmethod execution
    res_cls = bh_pkg.BohriumGenerator.execute_op("Add", np.array([10.0]), np.array([20.0]))
    np.testing.assert_allclose(res_cls, [30.0])

    # Alias mapping branch: Sub, Mul, Div, Neg, TrueDivide
    sub_res = bh_pkg.execute_op("Sub", np.array([20.0]), np.array([5.0]))
    np.testing.assert_allclose(sub_res, [15.0])
    mul_res = bh_pkg.execute_op("Mul", np.array([4.0]), np.array([5.0]))
    np.testing.assert_allclose(mul_res, [20.0])
    div_res = bh_pkg.execute_op("Div", np.array([40.0]), np.array([8.0]))
    np.testing.assert_allclose(div_res, [5.0])
    neg_res = bh_pkg.execute_op("Neg", np.array([13.0]))
    np.testing.assert_allclose(neg_res, [-13.0])
    truediv_res = bh_pkg.execute_op("TrueDivide", np.array([9.0]), np.array([2.0]))
    np.testing.assert_allclose(truediv_res, [4.5])

    # Unsupported op
    with pytest.raises(BackendNotSupportedError, match="not supported"):
        bh_pkg.execute_op("UnknownCustomOp12345")

    # When bohrium module is present
    mock_bh = mock.MagicMock()
    mock_bh.add.return_value = "bh_added"
    with mock.patch.dict(sys.modules, {"bohrium": mock_bh}):
        assert bh_pkg.execute_op("Add", 1, 2) == "bh_added"
