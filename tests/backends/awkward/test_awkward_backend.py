"""Unit tests for Awkward Array backend."""

from __future__ import annotations

import sys
from unittest import mock

import numpy as np
import pytest

import ml_switcheroo_compiler.backends.awkward as ak_pkg
from ml_switcheroo_compiler.backends.awkward.generator import AwkwardGenerator
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


def test_awkward_registered() -> None:
    """Verify Awkward backend is registered in BackendRegistry."""
    assert BackendRegistry.get("awkward") is AwkwardGenerator


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


def test_awkward_types_coverage_branches() -> None:
    """Test full branch coverage for ml_switcheroo_compiler.backends.awkward.types."""
    from ml_switcheroo_compiler.backends.awkward import types as ak_types

    # 1. _get_ak_module import exception fallback (lines 17-18)
    with mock.patch("importlib.import_module") as mock_import:

        def side_effect(name: str) -> object:
            """Side effect to simulate missing awkward package."""
            if name == "awkward":
                raise ImportError("No awkward")
            return np

        mock_import.side_effect = side_effect
        assert ak_types._get_ak_module() is np

    # 2. zeros when ak_mod has no Array attribute (line 37)
    with mock.patch(
        "ml_switcheroo_compiler.backends.awkward.types._get_ak_module",
        return_value=mock.MagicMock(spec=[]),
    ):
        z = ak_types.zeros((3,))
        assert isinstance(z, np.ndarray)

    # 3. array when ak_mod.Array raises Exception (lines 62-63)
    mock_ak_throwing = mock.MagicMock()
    mock_ak_throwing.Array.side_effect = TypeError("Boom")
    with mock.patch(
        "ml_switcheroo_compiler.backends.awkward.types._get_ak_module",
        return_value=mock_ak_throwing,
    ):
        arr = ak_types.array([1, 2, 3])
        assert isinstance(arr, np.ndarray)

    # 4. asarray when ak_mod.Array raises Exception (lines 81-84) and when ak_mod has no Array attribute (branch 78->83)
    with mock.patch(
        "ml_switcheroo_compiler.backends.awkward.types._get_ak_module",
        return_value=mock_ak_throwing,
    ):
        as_arr = ak_types.asarray([1, 2, 3])
        assert isinstance(as_arr, np.ndarray)

    with mock.patch(
        "ml_switcheroo_compiler.backends.awkward.types._get_ak_module",
        return_value=mock.MagicMock(spec=[]),
    ):
        as_arr_no_ak = ak_types.asarray([1, 2, 3])
        assert isinstance(as_arr_no_ak, np.ndarray)

    # 5. ragged_array when ak_mod.Array raises Exception (lines 113-114)
    with mock.patch(
        "ml_switcheroo_compiler.backends.awkward.types._get_ak_module",
        return_value=mock_ak_throwing,
    ):
        rag = ak_types.ragged_array([[1], [2, 3]])
        assert isinstance(rag, np.ndarray)

    # 6. record_array when ak_mod has neither zip nor Array (line 137)
    mock_ak_no_zip = mock.MagicMock(spec=["Array"])
    mock_ak_no_zip.Array.return_value = "array_record"
    with mock.patch(
        "ml_switcheroo_compiler.backends.awkward.types._get_ak_module",
        return_value=mock_ak_no_zip,
    ):
        rec_arr = ak_types.record_array([1, 2, 3])
        assert rec_arr == "array_record"

    with mock.patch(
        "ml_switcheroo_compiler.backends.awkward.types._get_ak_module",
        return_value=mock.MagicMock(spec=[]),
    ):
        rec = ak_types.record_array("non_mapping_data")
        assert rec == "non_mapping_data"

    # 7. with_field fallback when base is not dict and ak has no with_field (line 223)
    with mock.patch(
        "ml_switcheroo_compiler.backends.awkward.types._get_ak_module",
        return_value=mock.MagicMock(spec=[]),
    ):
        res = ak_types.with_field("not_a_dict", "val", where="fld")
        assert res == "not_a_dict"

    # 8. unzip fallback when base is not dict and ak has no unzip (line 246)
    with mock.patch(
        "ml_switcheroo_compiler.backends.awkward.types._get_ak_module",
        return_value=mock.MagicMock(spec=[]),
    ):
        unzipped = ak_types.unzip("not_a_dict")
        assert unzipped == ("not_a_dict",)

    # 9. item fallback with nested list/tuple where first is list/tuple (line 265)
    class NestedToListItem:
        """Dummy object with nested to_list structure."""

        def to_list(self) -> list[list[float]]:
            """Return nested 2D list."""
            return [[123.45]]

    assert ak_types.item(NestedToListItem()) == 123.45

    # 10. array when numpy_mod.array raises ValueError (lines 62-63)
    with mock.patch(
        "ml_switcheroo_compiler.backends.awkward.types._get_ak_module",
        return_value=mock.MagicMock(spec=[]),
    ):
        with mock.patch("numpy.array", side_effect=[ValueError("Ragged"), np.array(["mock"])]):
            res_val_err = ak_types.array([[1], [2, 3]])
            assert res_val_err is not None

    # 11. item fallback with object having .item() (line 267)
    class ItemHolder:
        """Dummy object implementing item method."""

        def item(self) -> float:
            """Return scalar float."""
            return 42.5

    assert ak_types.item(ItemHolder()) == 42.5
