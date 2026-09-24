"""Unit tests for Apache Arrow Compute backend."""

from __future__ import annotations

import sys
from unittest import mock

import numpy as np
import pytest

import ml_switcheroo_compiler.backends.pyarrow_compute as pa_pkg
from ml_switcheroo_compiler.backends.pyarrow_compute.generator import PyArrowComputeGenerator
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


def test_pyarrow_compute_registered() -> None:
    """Verify PyArrow Compute backend is registered in BackendRegistry."""
    assert BackendRegistry.get("pyarrow_compute") is PyArrowComputeGenerator


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
