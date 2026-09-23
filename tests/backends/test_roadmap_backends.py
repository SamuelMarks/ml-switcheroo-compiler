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
import ml_switcheroo_compiler.backends.sparse as sp_pkg
from ml_switcheroo_compiler.backends.awkward.generator import AwkwardGenerator
from ml_switcheroo_compiler.backends.bohrium.generator import BohriumGenerator
from ml_switcheroo_compiler.backends.dpnp.generator import DPNPGenerator
from ml_switcheroo_compiler.backends.numba.generator import NumbaGenerator
from ml_switcheroo_compiler.backends.pyarrow_compute.generator import PyArrowComputeGenerator
from ml_switcheroo_compiler.backends.registry import BackendRegistry
from ml_switcheroo_compiler.backends.sparse.generator import SparseGenerator
from ml_switcheroo_compiler.backends.sparse.types import COOTensor, CSCTensor, CSRTensor
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
    """Verify all roadmap backends are correctly registered in BackendRegistry."""
    assert BackendRegistry.get("dpnp") is DPNPGenerator
    assert BackendRegistry.get("awkward") is AwkwardGenerator
    assert BackendRegistry.get("pyarrow_compute") is PyArrowComputeGenerator
    assert BackendRegistry.get("bohrium") is BohriumGenerator
    assert BackendRegistry.get("sparse") is SparseGenerator
    assert BackendRegistry.get("numba") is NumbaGenerator


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
    assert gen.generic_visit(node, ["a", "b"]) == "pc.multiply(a, b)"

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

    # Error when pyarrow.compute cannot be imported
    with mock.patch(
        "importlib.import_module",
        side_effect=lambda name: np if name == "numpy" else (_ for _ in ()).throw(ImportError(name)),
    ):
        with pytest.raises(BackendNotSupportedError, match="pyarrow.compute is required"):
            pa_pkg.execute_op("Add", np.array([1.0]), np.array([2.0]))

    # Error when pyarrow.compute is None
    with mock.patch("importlib.import_module", side_effect=lambda name: np if name == "numpy" else None):
        with pytest.raises(BackendNotSupportedError):
            pa_pkg.execute_op("Add", np.array([1.0]), np.array([2.0]))

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


# ==============================================================================
# PyArrow ChunkedArray, RecordBatch & Table Tests
# ==============================================================================


def test_pyarrow_compute_chunked_and_tables() -> None:
    """Verify PyArrow ChunkedArray, zero-copy RecordBatch slicing, and Table conversion."""
    import pyarrow as pa

    # ChunkedArray creation and computation
    ca = pa_pkg.chunked_array([[1.0, 2.0], [3.0, 4.0]])
    assert ca is not None
    ca_cls = pa_pkg.PyArrowComputeGenerator.chunked_array([[1.0, 2.0], [3.0, 4.0]])
    assert ca_cls is not None

    res_add = pa_pkg.execute_op("Add", ca, ca)
    np.testing.assert_allclose(np.array(res_add), [2.0, 4.0, 6.0, 8.0])

    res_sum = pa_pkg.execute_op("Sum", ca)
    assert np.isclose(pa_pkg.item(res_sum), 10.0)

    # RecordBatch creation and zero-copy slicing
    c1 = pa.array([10, 20, 30, 40])
    c2 = pa.array([100, 200, 300, 400])
    batch = pa_pkg.record_batch([c1, c2], names=["c1", "c2"])
    assert batch is not None
    batch_cls = pa_pkg.PyArrowComputeGenerator.record_batch([c1, c2], names=["c1", "c2"])
    assert batch_cls is not None

    sliced = pa_pkg.slice_record_batch(batch, offset=1, length=2)
    assert sliced.num_rows == 2
    assert sliced.column(0).to_pylist() == [20, 30]

    sliced_cls = pa_pkg.PyArrowComputeGenerator.slice_record_batch(batch, offset=2, length=1)
    assert sliced_cls.num_rows == 1
    assert sliced_cls.column(1).to_pylist() == [300]

    # Table conversions
    tbl_from_batch = pa_pkg.to_table(batch)
    assert isinstance(tbl_from_batch, pa.Table)
    assert tbl_from_batch.num_rows == 4

    tbl_from_batches = pa_pkg.to_table([batch, batch])
    assert isinstance(tbl_from_batches, pa.Table)
    assert tbl_from_batches.num_rows == 8

    tbl_from_dict = pa_pkg.to_table({"x": [1, 2], "y": [3, 4]})
    assert isinstance(tbl_from_dict, pa.Table)

    tbl_from_arrays = pa_pkg.to_table([c1, c2], names=["col1", "col2"])
    assert isinstance(tbl_from_arrays, pa.Table)

    # Item extraction from Table and RecordBatch
    assert np.isclose(pa_pkg.item(tbl_from_dict), 1.0)
    assert np.isclose(pa_pkg.item(batch), 10.0)

    # Empty table item
    empty_tbl = pa.table({})
    assert np.isclose(pa_pkg.item(empty_tbl), 0.0)

    # Item on numpy scalar fallback
    assert np.isclose(pa_pkg.item(np.array([42.0])), 42.0)

    # Tensor unwrapping in eager execute_op
    cfg = TensorConfig((2,), "float32", "cpu")
    t_inp = Tensor(np.array([1.0, 2.0]), cfg)
    res_tensor_add = pa_pkg.execute_op("Add", t_inp, t_inp)
    assert res_tensor_add is not None

    # Error branches in execute_op
    with pytest.raises(BackendNotSupportedError):
        pa_pkg.execute_op("Add", "invalid_type", 123)

    # Exception branches in types
    mock_pa_err = mock.MagicMock()
    mock_pa_err.chunked_array.side_effect = RuntimeError("err")
    mock_pa_err.RecordBatch.from_arrays.side_effect = RuntimeError("err")
    with mock.patch("ml_switcheroo_compiler.backends.pyarrow_compute.types._get_pa_module", return_value=mock_pa_err):
        assert isinstance(pa_pkg.chunked_array([[1, 2]]), list)
        assert isinstance(pa_pkg.record_batch([[1, 2]]), list)

    # Fallback paths when pyarrow is mocked out
    mock_pa_empty = mock.MagicMock(spec=[])
    with mock.patch("ml_switcheroo_compiler.backends.pyarrow_compute.types._get_pa_module", return_value=mock_pa_empty):
        ca_fallback = pa_pkg.chunked_array([[1, 2]])
        assert isinstance(ca_fallback, list)
        rb_fallback = pa_pkg.record_batch([[1, 2], [3, 4]], names=["a", "b"])
        assert isinstance(rb_fallback, dict)
        slice_fallback = pa_pkg.slice_record_batch({"a": [1, 2, 3]}, offset=1, length=1)
        assert slice_fallback["a"] == [2]
        list_slice = pa_pkg.slice_record_batch([10, 20, 30], offset=1, length=1)
        assert list_slice == [20]
        tbl_fallback = pa_pkg.to_table({"a": [1, 2]})
        assert isinstance(tbl_fallback, dict)


# ==============================================================================
# Awkward Ragged Operations & Records Tests
# ==============================================================================


def test_awkward_ragged_operations_and_records() -> None:
    """Verify Awkward ragged array creation, jagged reductions, and record transformations."""
    import awkward as ak

    # Ragged array creation
    ragged = ak_pkg.ragged_array([[1.0, 2.0, 3.0], [4.0], [5.0, 6.0]])
    assert isinstance(ragged, ak.Array)
    ragged_cls = ak_pkg.AwkwardGenerator.ragged_array([[10.0], [20.0, 30.0]])
    assert isinstance(ragged_cls, ak.Array)

    # Jagged reductions
    sum_axis1 = ak_pkg.execute_op("Sum", ragged, axis=1)
    np.testing.assert_allclose(ak.to_list(sum_axis1), [6.0, 4.0, 11.0])

    mean_axis1 = ak_pkg.execute_op("Mean", ragged, axis=1)
    np.testing.assert_allclose(ak.to_list(mean_axis1), [2.0, 4.0, 5.5])

    min_axis1 = ak_pkg.execute_op("Min", ragged, axis=1)
    np.testing.assert_allclose(ak.to_list(min_axis1), [1.0, 4.0, 5.0])

    max_axis1 = ak_pkg.execute_op("Max", ragged, axis=1)
    np.testing.assert_allclose(ak.to_list(max_axis1), [3.0, 4.0, 6.0])

    prod_axis1 = ak_pkg.execute_op("Prod", ragged, axis=1)
    np.testing.assert_allclose(ak.to_list(prod_axis1), [6.0, 4.0, 30.0])

    # Variable-length flattening and unflattening
    flat = ak_pkg.flatten(ragged)
    np.testing.assert_allclose(ak.to_list(flat), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    flat_cls = ak_pkg.AwkwardGenerator.flatten(ragged)
    np.testing.assert_allclose(ak.to_list(flat_cls), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    unflat = ak_pkg.unflatten([10, 20, 30, 40], counts=[1, 3])
    assert ak.to_list(unflat) == [[10], [20, 30, 40]]
    unflat_cls = ak_pkg.AwkwardGenerator.unflatten([10, 20, 30, 40], counts=[2, 2])
    assert ak.to_list(unflat_cls) == [[10, 20], [30, 40]]

    # Nested record transformations
    rec = ak_pkg.record_array({"f1": [1, 2], "f2": [3, 4]})
    assert rec is not None
    rec_cls = ak_pkg.AwkwardGenerator.record_array({"a": [5]})
    assert rec_cls is not None

    with_f = ak_pkg.with_field(rec, [5, 6], where="f3")
    assert "f3" in ak.fields(with_f)
    with_f_cls = ak_pkg.AwkwardGenerator.with_field(rec, [7, 8], where="f4")
    assert "f4" in ak.fields(with_f_cls)

    unzipped = ak_pkg.unzip(rec)
    assert len(unzipped) == 2
    unzipped_cls = ak_pkg.AwkwardGenerator.unzip(rec)
    assert len(unzipped_cls) == 2

    # Arithmetic without rectangularization
    add_res = ak_pkg.execute_op("Add", ragged, ragged)
    assert ak.to_list(add_res) == [[2.0, 4.0, 6.0], [8.0], [10.0, 12.0]]

    exp_res = ak_pkg.execute_op("Exp", ragged)
    assert isinstance(exp_res, ak.Array)

    # Classmethod execution and other overloaded ops
    sub_res = ak_pkg.execute_op(AwkwardGenerator, "Sub", ragged, ragged)
    assert ak.to_list(sub_res) == [[0.0, 0.0, 0.0], [0.0], [0.0, 0.0]]
    mul_res = ak_pkg.execute_op("Mul", ragged, ragged)
    assert isinstance(mul_res, ak.Array)
    div_res = ak_pkg.execute_op("Div", ragged, ragged)
    assert isinstance(div_res, ak.Array)
    pow_res = ak_pkg.execute_op("Pow", ragged, 2)
    assert isinstance(pow_res, ak.Array)
    neg_res = ak_pkg.execute_op("Neg", ragged)
    assert isinstance(neg_res, ak.Array)
    abs_res = ak_pkg.execute_op("Abs", ragged)
    assert isinstance(abs_res, ak.Array)
    sin_res = ak_pkg.execute_op("Sin", ragged)
    assert isinstance(sin_res, ak.Array)

    # Error branches in awkward execute_op
    with pytest.raises(BackendNotSupportedError):
        ak_pkg.execute_op("Flatten", 12345)
    with pytest.raises(BackendNotSupportedError):
        ak_pkg.execute_op("Exp", "invalid_type")

    # Awkward types branch checks
    z_int = ak_pkg.zeros(5)
    assert z_int is not None
    arr_obj = ak_pkg.array([object()])
    assert arr_obj is not None
    arr_rag = ak_pkg.ragged_array([object()])
    assert arr_rag is not None
    assert np.isclose(ak_pkg.item(np.array([77.0])), 77.0)

    # Fallback paths when awkward is mocked out
    mock_ak_empty = mock.MagicMock(spec=[])
    with mock.patch("ml_switcheroo_compiler.backends.awkward.types._get_ak_module", return_value=mock_ak_empty):
        ragged_fallback = ak_pkg.ragged_array([1, 2])
        assert isinstance(ragged_fallback, np.ndarray)
        rec_fallback = ak_pkg.record_array({"a": [1]})
        assert isinstance(rec_fallback, dict)
        flat_fallback = ak_pkg.flatten(np.array([[1, 2], [3, 4]]))
        assert flat_fallback.shape == (4,)
        unflat_fallback = ak_pkg.unflatten([1, 2], counts=[1, 1])
        assert unflat_fallback == [1, 2]
        with_f_fallback = ak_pkg.with_field({"a": 1}, 2, where="b")
        assert with_f_fallback["b"] == 2
        unzip_fallback = ak_pkg.unzip({"k1": 10, "k2": 20})
        assert unzip_fallback == (10, 20)


# ==============================================================================
# Sparse Conversions & Kernels Tests
# ==============================================================================


def test_sparse_conversions_and_kernels() -> None:
    """Verify Sparse format inter-conversions, SpMM, SpGEMM, addition, and convolution masks."""
    dense_mat = np.array([[1.0, 0.0, 2.0], [0.0, 3.0, 0.0], [4.0, 0.0, 5.0]], dtype=np.float32)

    # Direct format inter-conversions
    coo = COOTensor.from_dense(dense_mat)
    csr = sp_pkg.coo_to_csr(coo)
    csc = sp_pkg.coo_to_csc(coo)

    np.testing.assert_allclose(csr.to_dense(), dense_mat)
    np.testing.assert_allclose(csc.to_dense(), dense_mat)

    # CSR <-> CSC and COO roundtrips
    coo_from_csr = sp_pkg.csr_to_coo(csr)
    np.testing.assert_allclose(coo_from_csr.to_dense(), dense_mat)

    csc_from_csr = sp_pkg.csr_to_csc(csr)
    np.testing.assert_allclose(csc_from_csr.to_dense(), dense_mat)

    coo_from_csc = sp_pkg.csc_to_coo(csc)
    np.testing.assert_allclose(coo_from_csc.to_dense(), dense_mat)

    csr_from_csc = sp_pkg.csc_to_csr(csc)
    np.testing.assert_allclose(csr_from_csc.to_dense(), dense_mat)

    # Non-2D error checking
    coo_1d = COOTensor.from_dense(np.array([1.0, 0.0, 2.0]))
    with pytest.raises(ValueError, match="CSR conversion requires 2D tensor"):
        sp_pkg.coo_to_csr(coo_1d)
    with pytest.raises(ValueError, match="CSC conversion requires 2D tensor"):
        sp_pkg.coo_to_csc(coo_1d)

    # Empty tensor conversions
    empty_coo = COOTensor(indices=np.empty((2, 0), dtype=np.int64), values=np.empty((0,), dtype=np.float32), shape=(3, 3))
    empty_csr = sp_pkg.coo_to_csr(empty_coo)
    assert empty_csr.nnz == 0
    empty_csc = sp_pkg.coo_to_csc(empty_coo)
    assert empty_csc.nnz == 0

    # SpMM and dense_spmm
    dense_rhs = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    expected_prod = dense_mat @ dense_rhs

    spmm_res = sp_pkg.spmm(csr, dense_rhs)
    np.testing.assert_allclose(spmm_res, expected_prod)

    spmm_csc_res = sp_pkg.spmm(csc, dense_rhs)
    np.testing.assert_allclose(spmm_csc_res, expected_prod)

    dense_lhs = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    expected_dense_spmm = dense_lhs @ dense_mat
    dense_spmm_res = sp_pkg.dense_spmm(dense_lhs, csr)
    np.testing.assert_allclose(dense_spmm_res, expected_dense_spmm)

    # SpGEMM
    dense_b = np.array([[0.0, 2.0, 0.0], [1.0, 0.0, 3.0], [0.0, 4.0, 0.0]], dtype=np.float32)
    csr_b = CSRTensor.from_dense(dense_b)
    expected_spgemm = dense_mat @ dense_b
    spgemm_res = sp_pkg.spgemm(csr, csr_b)
    np.testing.assert_allclose(spgemm_res.to_dense(), expected_spgemm)

    bad_b = CSRTensor.from_dense(np.ones((4, 4), dtype=np.float32))
    with pytest.raises(ValueError, match="Shape mismatch for SpGEMM"):
        sp_pkg.spgemm(csr, bad_b)

    # Elementwise addition (CSR and CSC)
    expected_add = dense_mat + dense_b
    csr_add_res = sp_pkg.csr_add(csr, csr_b)
    np.testing.assert_allclose(csr_add_res.to_dense(), expected_add)

    csc_b = CSCTensor.from_dense(dense_b)
    csc_add_res = sp_pkg.csc_add(csc, csc_b)
    np.testing.assert_allclose(csc_add_res.to_dense(), expected_add)

    # Empty operand branches for addition
    empty_csr_zero = CSRTensor.from_dense(np.zeros((3, 3), dtype=np.float32))
    assert sp_pkg.csr_add(csr, empty_csr_zero) is csr
    assert sp_pkg.csr_add(empty_csr_zero, csr) is csr

    empty_csc_zero = CSCTensor.from_dense(np.zeros((3, 3), dtype=np.float32))
    assert sp_pkg.csc_add(csc, empty_csc_zero) is csc
    assert sp_pkg.csc_add(empty_csc_zero, csc) is csc

    with pytest.raises(ValueError, match="Shape mismatch for CSR addition"):
        sp_pkg.csr_add(csr, bad_b)
    with pytest.raises(ValueError, match="Shape mismatch for CSC addition"):
        sp_pkg.csc_add(csc, CSCTensor.from_dense(np.ones((4, 4), dtype=np.float32)))

    # Sparse convolution masks with 2D, 3D and padding
    x_2d = np.ones((4, 4), dtype=np.float32)
    w_2d = np.ones((3, 3), dtype=np.float32)
    mask_pad = np.ones((4, 4), dtype=np.int64)
    conv_2d_res = sp_pkg.sparse_conv2d_mask(x_2d, w_2d, mask_pad, padding=(1, 1))
    assert conv_2d_res is not None

    x_3d = np.ones((1, 4, 4), dtype=np.float32)
    w_3d = np.ones((1, 3, 3), dtype=np.float32)
    mask_3d = np.ones((2, 2), dtype=np.int64)
    conv_3d_res = sp_pkg.sparse_conv2d_mask(x_3d, w_3d, mask_3d)
    assert conv_3d_res is not None

    # Sparse convolution masks and sparse_mask
    mask = np.array([[1, 0], [0, 1]], dtype=np.int64)
    tensor_to_mask = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
    s_mask_res = sp_pkg.sparse_mask(tensor_to_mask, mask)
    assert s_mask_res.nnz == 2
    np.testing.assert_allclose(s_mask_res.to_dense(), [[10.0, 0.0], [0.0, 40.0]])

    x = np.ones((1, 1, 4, 4), dtype=np.float32)
    weight = np.ones((1, 1, 3, 3), dtype=np.float32)
    conv_mask_res = sp_pkg.sparse_conv2d_mask(x, weight, mask)
    assert conv_mask_res.nnz == 2
    np.testing.assert_allclose(conv_mask_res.values, [9.0, 9.0])

    # Eager dispatch
    res_eager_spmm = sp_pkg.execute_op(SparseGenerator, "spmm", csr, dense_rhs)
    np.testing.assert_allclose(res_eager_spmm, expected_prod)
    res_eager_spgemm = sp_pkg.execute_op(SparseGenerator, "spgemm", csr, csr_b)
    np.testing.assert_allclose(res_eager_spgemm.to_dense(), expected_spgemm)


# ==============================================================================
# Numba Parallel Loop Emission Tests
# ==============================================================================


def test_numba_parallel_loop_emission() -> None:
    """Verify Numba automatic parallel loop emission with @nb.njit(fastmath=True, parallel=True)."""
    # Graph for MatMul
    g_matmul = IRGraph()
    g_matmul.nodes["a"] = IRNode(id="a", op_type="Input", inputs=[])
    g_matmul.nodes["b"] = IRNode(id="b", op_type="Input", inputs=[])
    g_matmul.nodes["mm"] = IRNode(id="mm", op_type="MatMul", inputs=["a", "b"])
    g_matmul.inputs = ["a", "b"]
    g_matmul.outputs = ["mm"]

    gen_mm = NumbaGenerator(g_matmul, fastmath=True, parallel=True)
    code_mm = gen_mm.generate()
    assert "nb.njit(fastmath=True, parallel=True)" in code_mm
    assert "for _i in nb.prange(" in code_mm

    gen_mm_strict = NumbaGenerator(g_matmul, fastmath=True, parallel=True, emit_safe_fallback=False)
    assert "@nb.njit(fastmath=True, parallel=True)" in gen_mm_strict.generate()

    fn_mm = gen_mm.compile_fn()
    a_mat = np.ones((4, 4), dtype=np.float32)
    b_mat = np.ones((4, 4), dtype=np.float32)
    res_mm = fn_mm([a_mat, b_mat])
    np.testing.assert_allclose(res_mm, a_mat @ b_mat)

    # BatchMatMul
    g_bmm = IRGraph()
    g_bmm.nodes["a"] = IRNode(id="a", op_type="Input", inputs=[])
    g_bmm.nodes["b"] = IRNode(id="b", op_type="Input", inputs=[])
    g_bmm.nodes["bmm"] = IRNode(id="bmm", op_type="BatchMatMul", inputs=["a", "b"])
    g_bmm.inputs = ["a", "b"]
    g_bmm.outputs = ["bmm"]

    gen_bmm = NumbaGenerator(g_bmm, fastmath=True, parallel=True)
    code_bmm = gen_bmm.generate()
    assert "for _b in nb.prange(" in code_bmm

    fn_bmm = gen_bmm.compile_fn()
    a_batch = np.ones((2, 3, 4), dtype=np.float32)
    b_batch = np.ones((2, 4, 5), dtype=np.float32)
    res_bmm = fn_bmm([a_batch, b_batch])
    np.testing.assert_allclose(res_bmm, a_batch @ b_batch)

    # Conv2D
    g_conv = IRGraph()
    g_conv.nodes["x"] = IRNode(id="x", op_type="Input", inputs=[])
    g_conv.nodes["w"] = IRNode(id="w", op_type="Input", inputs=[])
    g_conv.nodes["conv"] = IRNode(id="conv", op_type="Conv2D", inputs=["x", "w"])
    g_conv.inputs = ["x", "w"]
    g_conv.outputs = ["conv"]

    gen_conv = NumbaGenerator(g_conv, fastmath=True, parallel=True)
    code_conv = gen_conv.generate()
    assert "for _b in nb.prange(" in code_conv
    assert "for _h in nb.prange(_h_out):" in code_conv

    fn_conv = gen_conv.compile_fn()
    x_conv = np.ones((2, 1, 4, 4), dtype=np.float32)
    w_conv = np.ones((1, 1, 3, 3), dtype=np.float32)
    res_conv = fn_conv([x_conv, w_conv])
    assert res_conv.shape == (2, 1, 2, 2)
    np.testing.assert_allclose(res_conv, 9.0)

    # MaxPool2D and AvgPool2D
    g_pool = IRGraph()
    g_pool.nodes["x"] = IRNode(id="x", op_type="Input", inputs=[])
    g_pool.nodes["maxp"] = IRNode(id="maxp", op_type="MaxPool2D", inputs=["x"])
    g_pool.nodes["avgp"] = IRNode(id="avgp", op_type="AvgPool2D", inputs=["x"])
    g_pool.inputs = ["x"]
    g_pool.outputs = ["maxp", "avgp"]

    gen_pool = NumbaGenerator(g_pool, fastmath=True, parallel=True)
    code_pool = gen_pool.generate()
    assert "for _b in nb.prange(" in code_pool
    assert "for _h in nb.prange(_h_out):" in code_pool

    fn_pool = gen_pool.compile_fn()
    x_pool = np.arange(32, dtype=np.float32).reshape(2, 1, 4, 4)
    res_maxp, res_avgp = fn_pool([x_pool])
    assert res_maxp.shape == (2, 1, 2, 2)
    assert res_avgp.shape == (2, 1, 2, 2)

    # Non-parallel fallback branch coverage
    gen_nopar = NumbaGenerator(g_matmul, fastmath=False, parallel=False)
    code_nopar = gen_nopar.generate()
    assert "np.matmul" in code_nopar

    gen_nopar.visit_Prod(IRNode(id="p", op_type="Prod", inputs=["a"]), ["a"])
    assert any("np.prod" in line for line in gen_nopar.code)
    gen_nopar.visit_Mean(IRNode(id="m", op_type="Mean", inputs=["a"]), ["a"])
    assert any("np.mean" in line for line in gen_nopar.code)
    gen_nopar.visit_Max(IRNode(id="mx", op_type="Max", inputs=["a"]), ["a"])
    assert any("np.max" in line for line in gen_nopar.code)
    gen_nopar.visit_Min(IRNode(id="mn", op_type="Min", inputs=["a"]), ["a"])
    assert any("np.min" in line for line in gen_nopar.code)
    gen_nopar.visit_BatchMatMul(IRNode(id="bmm", op_type="BatchMatMul", inputs=["a", "b"]), ["a", "b"])
    assert any("np.matmul" in line for line in gen_nopar.code)
    gen_nopar.visit_Conv2D(IRNode(id="cv", op_type="Conv2D", inputs=["x", "w"]), ["x", "w"])
    assert any("np.zeros" in line for line in gen_nopar.code)
    gen_nopar.visit_MaxPool2D(IRNode(id="mp", op_type="MaxPool2D", inputs=["x"]), ["x"])
    assert any("np.zeros" in line for line in gen_nopar.code)
    gen_nopar.visit_AvgPool2D(IRNode(id="ap", op_type="AvgPool2D", inputs=["x"]), ["x"])
    assert any("np.zeros" in line for line in gen_nopar.code)
