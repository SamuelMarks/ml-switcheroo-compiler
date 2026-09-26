"""Exhaustive unit and parity tests for Apache Arrow Compute backend supporting zero-copy columnar execution."""

from __future__ import annotations

from unittest import mock

import numpy as np
import pyarrow as pa
import pytest

import ml_switcheroo_compiler.backends.pyarrow_compute as pa_pkg
from ml_switcheroo_compiler.backends.pyarrow_compute.eager import execute_op
from ml_switcheroo_compiler.backends.pyarrow_compute.generator import PyArrowComputeGenerator
from ml_switcheroo_compiler.backends.pyarrow_compute.types import (
    arrow_tensor_to_tensor,
    from_arrow,
    is_arrow_object,
    record_batch_to_tensor,
    shares_buffer,
    tensor_to_arrow_tensor,
    tensor_to_record_batch,
    to_arrow,
)
from ml_switcheroo_compiler.core.errors import BackendNotSupportedError
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode


def test_zero_copy_buffer_views_between_arrow_and_tensor() -> None:
    """Verify zero-copy buffer views between Arrow Tensor, RecordBatch, and Unified IR Tensor."""
    # 1. NumPy array -> Arrow Tensor -> Unified IR Tensor (zero-copy memory sharing)
    raw_np = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    cfg = TensorConfig(raw_np.shape, "float32", "cpu")
    ir_tensor = Tensor(raw_np, cfg)

    arrow_tensor = tensor_to_arrow_tensor(ir_tensor)
    assert arrow_tensor is not None
    assert arrow_tensor.shape == (2, 3)

    back_tensor = arrow_tensor_to_tensor(arrow_tensor)
    assert isinstance(back_tensor, Tensor)
    np.testing.assert_allclose(back_tensor.data, raw_np)

    # Verify zero-copy buffer sharing
    assert shares_buffer(ir_tensor, arrow_tensor) is True
    assert shares_buffer(arrow_tensor, back_tensor) is True
    assert shares_buffer(ir_tensor, back_tensor) is True

    # Mutating raw_np mutates back_tensor view
    raw_np[0, 0] = 99.0
    assert back_tensor.data[0, 0] == 99.0

    # 2. Classmethod variants
    arrow_t_cls = PyArrowComputeGenerator.tensor_to_arrow_tensor(ir_tensor)
    back_t_cls = PyArrowComputeGenerator.arrow_tensor_to_tensor(arrow_t_cls)
    assert shares_buffer(ir_tensor, back_t_cls) is True


def test_record_batch_and_tensor_conversions() -> None:
    """Verify conversions between RecordBatch and Unified IR Tensor."""
    # 2D Tensor to RecordBatch
    mat = np.array([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]], dtype=np.float32)
    cfg = TensorConfig(mat.shape, "float32", "cpu")
    t2d = Tensor(mat, cfg)

    rb = tensor_to_record_batch(t2d, names=["col_x", "col_y"])
    assert rb.num_rows == 3
    assert rb.num_columns == 2
    assert rb.schema.field(0).name == "col_x"
    assert rb.schema.field(1).name == "col_y"

    # RecordBatch back to Tensor
    t_reconstructed = record_batch_to_tensor(rb)
    assert isinstance(t_reconstructed, Tensor)
    np.testing.assert_allclose(t_reconstructed.data, mat)

    # 1D Tensor to RecordBatch
    vec = np.array([1.5, 2.5, 3.5], dtype=np.float32)
    t1d = Tensor(vec, TensorConfig(vec.shape, "float32", "cpu"))
    rb1d = tensor_to_record_batch(t1d, names=["single"])
    assert rb1d.num_columns == 1
    t1d_rec = record_batch_to_tensor(rb1d)
    np.testing.assert_allclose(t1d_rec.data, vec)

    # Subset of columns extraction
    t_subset = record_batch_to_tensor(rb, columns=["col_y"])
    np.testing.assert_allclose(t_subset.data, mat[:, 1])

    # Classmethod invocation
    rb_cls = PyArrowComputeGenerator.tensor_to_record_batch(t2d)
    t_cls = PyArrowComputeGenerator.record_batch_to_tensor(rb_cls)
    np.testing.assert_allclose(t_cls.data, mat)


def test_from_arrow_and_to_arrow_universal_dispatch() -> None:
    """Verify from_arrow and to_arrow across Table, RecordBatch, Tensor, and ChunkedArray."""
    raw = np.array([1, 2, 3, 4], dtype=np.int32)
    t = Tensor(raw, TensorConfig(raw.shape, "int32", "cpu"))

    # To arrow types
    arrow_t = to_arrow(t, target_type="tensor")
    assert type(arrow_t).__name__ == "Tensor"

    arrow_rb = to_arrow(t, target_type="record_batch")
    assert isinstance(arrow_rb, pa.RecordBatch)

    arrow_tbl = to_arrow(t, target_type="table")
    assert isinstance(arrow_tbl, pa.Table)

    arrow_arr = to_arrow(t, target_type="array")
    assert isinstance(arrow_arr, pa.Array)

    arrow_ca = to_arrow(t, target_type="chunked_array")
    assert isinstance(arrow_ca, pa.ChunkedArray)

    # Convert back via from_arrow
    assert np.all(from_arrow(arrow_t).data == raw)
    assert np.all(from_arrow(arrow_rb).data == raw)
    assert np.all(from_arrow(arrow_tbl).data == raw)
    assert np.all(from_arrow(arrow_arr).data == raw)

    # Multi-batch Table
    b1 = pa.RecordBatch.from_arrays([pa.array([10, 20])], names=["x"])
    b2 = pa.RecordBatch.from_arrays([pa.array([30, 40])], names=["x"])
    multi_tbl = pa.Table.from_batches([b1, b2])
    t_multi = from_arrow(multi_tbl)
    assert t_multi.data.shape[0] == 4

    # Arrow Scalar
    scalar = pa.scalar(42)
    assert from_arrow(scalar) == 42

    # Unsupported target_type
    with pytest.raises(ValueError, match="Unsupported target Arrow type"):
        to_arrow(t, target_type="invalid_type")

    # Classmethods
    assert PyArrowComputeGenerator.is_arrow_object(arrow_tbl) is True
    assert PyArrowComputeGenerator.is_arrow_object(raw) is False
    assert is_arrow_object(arrow_arr) is True


def test_tabular_and_columnar_array_kernels() -> None:
    """Verify tabular operations (filter, take, drop_null, fill_null, unique, sort_indices, rank)."""
    # Filter
    data_arr = pa.array([10, 20, 30, 40])
    mask = pa.array([True, False, True, False])
    filtered = execute_op("filter", data_arr, mask)
    assert filtered.to_pylist() == [10, 30]

    # Take
    indices = pa.array([3, 0, 1])
    taken = execute_op("take", data_arr, indices)
    assert taken.to_pylist() == [40, 10, 20]

    # Drop null
    with_nulls = pa.array([1, None, 2, None, 3])
    dropped = execute_op("drop_null", with_nulls)
    assert dropped.to_pylist() == [1, 2, 3]

    # Fill null
    filled = execute_op("fill_null", with_nulls, 0)
    assert filled.to_pylist() == [1, 0, 2, 0, 3]

    # Fill null forward and backward
    fwd = execute_op("fill_null_forward", with_nulls)
    assert fwd.to_pylist() == [1, 1, 2, 2, 3]
    bwd = execute_op("fill_null_backward", with_nulls)
    assert bwd.to_pylist() == [1, 2, 2, 3, 3]

    # Unique & Value Counts
    dup_arr = pa.array([5, 5, 2, 8, 2])
    uniq = execute_op("unique", dup_arr)
    assert set(uniq.to_pylist()) == {5, 2, 8}

    vc = execute_op("value_counts", dup_arr)
    assert vc is not None

    # Sort indices & rank
    unordered = pa.array([30, 10, 20])
    s_idx = execute_op("sort_indices", unordered)
    assert s_idx.to_pylist() == [1, 2, 0]

    ranked = execute_op("rank", unordered)
    assert ranked.to_pylist() == [3, 1, 2]

    # Partition nth indices
    part = execute_op("partition_nth_indices", unordered, pivot=1)
    assert part is not None

    # Indices nonzero
    nz_arr = pa.array([0, 5, 0, 9])
    nz = execute_op("indices_nonzero", nz_arr)
    assert nz.to_pylist() == [1, 3]

    # Dictionary encode
    cat_arr = pa.array(["a", "b", "a"])
    dict_encoded = execute_op("dictionary_encode", cat_arr)
    assert dict_encoded is not None


def test_structural_and_nested_columnar_kernels() -> None:
    """Verify nested list and struct operations."""
    # List flatten, slice, element
    list_arr = pa.array([[1, 2], [3], [4, 5, 6]])
    flat = execute_op("list_flatten", list_arr)
    assert flat.to_pylist() == [1, 2, 3, 4, 5, 6]

    elem = execute_op("list_element", list_arr, 0)
    assert elem.to_pylist() == [1, 3, 4]

    sliced_list = execute_op("list_slice", list_arr, 0, 1)
    assert sliced_list.to_pylist() == [[1], [3], [4]]

    # Make struct & struct field
    f1 = pa.array([10, 20])
    f2 = pa.array(["a", "b"])
    struct_arr = execute_op("make_struct", f1, f2, field_names=["num", "text"])
    assert struct_arr is not None

    extracted_field = execute_op("struct_field", struct_arr, [0])
    assert extracted_field.to_pylist() == [10, 20]


def test_columnar_string_kernels() -> None:
    """Verify string operations (lower, upper, length, replace, match)."""
    s_arr = pa.array(["Hello World", "Apache Arrow"])

    # Lower & Upper
    low = execute_op("lower", s_arr)
    assert low.to_pylist() == ["hello world", "apache arrow"]

    up = execute_op("upper", s_arr)
    assert up.to_pylist() == ["HELLO WORLD", "APACHE ARROW"]

    # String length
    length = execute_op("string_length", s_arr)
    assert length.to_pylist() == [11, 12]

    # Replace substring & match substring
    replaced = execute_op("replace_substring", s_arr, pattern="World", replacement="Compiler")
    assert replaced.to_pylist() == ["Hello Compiler", "Apache Arrow"]

    matched = execute_op("match_substring", s_arr, pattern="Arrow")
    assert matched.to_pylist() == [False, True]

    # Capitalize & Title
    cap = execute_op("capitalize", pa.array(["hello", "world"]))
    assert cap.to_pylist() == ["Hello", "World"]


def test_reductions_stats_and_extra_math() -> None:
    """Verify extra math, cumulative ops, and stats."""
    arr = pa.array([2, 3, 4])

    # Product
    prod = execute_op("product", arr)
    assert pa_pkg.item(prod) == 24.0

    # Extra math: cbrt, shift_left, shift_right
    cbrt_res = execute_op("cbrt", pa.array([8.0, 27.0]))
    np.testing.assert_allclose(np.array(cbrt_res), [2.0, 3.0])

    shl = execute_op("shift_left", pa.array([1, 2]), 2)
    assert shl.to_pylist() == [4, 8]

    shr = execute_op("shift_right", pa.array([8, 16]), 2)
    assert shr.to_pylist() == [2, 4]

    # Cumulative ops
    c_sum = execute_op("cumsum", arr)
    assert c_sum.to_pylist() == [2, 5, 9]

    c_prod = execute_op("cumprod", arr)
    assert c_prod.to_pylist() == [2, 6, 24]

    c_max = execute_op("cummax", pa.array([3, 1, 5, 2]))
    assert c_max.to_pylist() == [3, 3, 5, 5]

    c_min = execute_op("cummin", pa.array([3, 1, 5, 2]))
    assert c_min.to_pylist() == [3, 1, 1, 1]

    # Approximate median & mode
    med = execute_op("median", pa.array([1.0, 3.0, 5.0]))
    assert pa_pkg.item(med) == 3.0

    modes = execute_op("mode", pa.array([1, 2, 2, 3]))
    assert modes is not None


def test_table_groupby_and_hash_aggregations() -> None:
    """Verify group_by and hash aggregations on PyArrow Table."""
    tbl = pa.table(
        {
            "dept": ["eng", "hr", "eng", "hr", "eng"],
            "salary": [100, 70, 120, 80, 110],
        }
    )

    # Group by with multiple aggregations
    grouped = execute_op("group_by", tbl, keys=["dept"], aggregations=[("salary", "sum"), ("salary", "mean")])
    assert isinstance(grouped, pa.Table)
    res_dict = grouped.to_pydict()
    assert "dept" in res_dict
    assert "salary_sum" in res_dict

    # Hash sum and hash mean
    h_sum = execute_op("hash_sum", tbl, "dept", "salary")
    assert isinstance(h_sum, pa.Table)

    h_mean = execute_op("hash_mean", tbl, "dept", "salary")
    assert isinstance(h_mean, pa.Table)

    # Error handling in groupby
    mock_bad_tbl = mock.MagicMock()
    mock_bad_tbl.group_by.side_effect = RuntimeError("Group failure")
    with pytest.raises(BackendNotSupportedError, match="Failed executing group_by"):
        execute_op("group_by", mock_bad_tbl, keys=["a"], aggregations=[("b", "sum")])

    with pytest.raises(BackendNotSupportedError, match="Failed executing hash_sum"):
        execute_op("hash_sum", mock_bad_tbl, "a", "b")


def test_aot_execution_with_arrow_inputs_and_zero_copy() -> None:
    """Verify AOT compilation and execution with zero_copy and Arrow input objects."""
    g = IRGraph()
    n_in1 = IRNode(id="x", op_type="Input", inputs=[], shape_metadata=[3])
    n_in2 = IRNode(id="y", op_type="Input", inputs=[], shape_metadata=[3])
    n_add = IRNode(id="out", op_type="Add", inputs=["x", "y"], shape_metadata=[3])
    g.nodes = {"x": n_in1, "y": n_in2, "out": n_add}
    g.inputs = ["x", "y"]
    g.outputs = ["out"]

    gen_zc = PyArrowComputeGenerator(g, zero_copy=True)
    runner_zc = gen_zc._compile_aot_impl(g)

    # Execute with Arrow arrays
    pa_x = pa.array([1.0, 2.0, 3.0], type=pa.float32())
    pa_y = pa.array([10.0, 20.0, 30.0], type=pa.float32())
    out_zc = runner_zc(pa_x, y=pa_y)
    assert out_zc is not None
    assert type(out_zc).__name__ == "Tensor"

    # Execute without zero_copy
    gen_no_zc = PyArrowComputeGenerator(g, zero_copy=False)
    runner_no_zc = gen_no_zc._compile_aot_impl(g)
    out_no_zc = runner_no_zc(pa_x, pa_y)
    assert out_no_zc is not None


def test_generator_ops_map_coverage() -> None:
    """Verify PyArrowComputeGenerator get_ops_map contains all expected operations."""
    g = IRGraph()
    gen = PyArrowComputeGenerator(g)
    ops_map = gen.get_ops_map({})

    expected_ops = [
        "Filter",
        "Take",
        "DropNull",
        "FillNull",
        "Unique",
        "ValueCounts",
        "SortIndices",
        "Rank",
        "ListFlatten",
        "ListSlice",
        "ListElement",
        "MakeStruct",
        "StructField",
        "Lower",
        "Upper",
        "StringLength",
        "ReplaceSubstring",
        "MatchSubstring",
        "Cbrt",
        "ShiftLeft",
        "ShiftRight",
        "Prod",
        "CumSum",
        "CumProd",
        "CumMax",
        "CumMin",
    ]
    for op in expected_ops:
        assert op in ops_map, f"Operation '{op}' missing in PyArrowComputeGenerator.get_ops_map"


def test_fallback_and_edge_branches() -> None:
    """Verify fallback branches and edge cases in types and eager."""
    # record_batch_to_tensor with dictionary fallback
    d = {"c1": [1.0, 2.0], "c2": [3.0, 4.0]}
    t_from_dict = record_batch_to_tensor(d)
    assert t_from_dict.data.shape == (2, 2)

    # Empty dictionary
    t_empty = record_batch_to_tensor({})
    assert t_empty.data.shape == (0,)

    # Empty columns on RecordBatch
    empty_rb = pa.RecordBatch.from_arrays([], names=[])
    t_empty_rb = record_batch_to_tensor(empty_rb)
    assert t_empty_rb.data.shape == (0,)

    # tensor_to_record_batch with 0D scalar
    t_scalar = Tensor(np.array(42), TensorConfig((), "int64", "cpu"))
    rb_scalar = tensor_to_record_batch(t_scalar)
    assert rb_scalar is not None

    # tensor_to_record_batch fallback when RecordBatch fails
    mock_pa_fail = mock.MagicMock()
    mock_pa_fail.RecordBatch.from_arrays.side_effect = RuntimeError("fail")
    with mock.patch("ml_switcheroo_compiler.backends.pyarrow_compute.types._get_pa_module", return_value=mock_pa_fail):
        rb_dict = tensor_to_record_batch(np.array([1, 2]))
        assert isinstance(rb_dict, dict)
        rb_dict2d = tensor_to_record_batch(np.array([[1, 2], [3, 4]]))
        assert isinstance(rb_dict2d, dict)

    # tensor_to_arrow_tensor fallback when Tensor.from_numpy fails
    mock_pa_t_fail = mock.MagicMock()
    mock_pa_t_fail.Tensor.from_numpy.side_effect = RuntimeError("tensor fail")
    with mock.patch("ml_switcheroo_compiler.backends.pyarrow_compute.types._get_pa_module", return_value=mock_pa_t_fail):
        t_fallback = tensor_to_arrow_tensor(np.array([1, 2]))
        assert isinstance(t_fallback, np.ndarray)

    # arrow_tensor_to_tensor fallback when no to_numpy
    t_mock_no_np = mock.MagicMock(spec=[])
    t_conv = arrow_tensor_to_tensor(t_mock_no_np)
    assert isinstance(t_conv, Tensor)

    # from_arrow with single-batch Table
    single_tbl = pa.Table.from_batches([pa.RecordBatch.from_arrays([pa.array([1, 2])], names=["a"])])
    assert from_arrow(single_tbl).data.shape == (2,)

    # from_arrow with Table having empty to_batches
    mock_empty_batches_tbl = mock.MagicMock()
    mock_empty_batches_tbl.to_batches.return_value = []
    assert from_arrow(mock_empty_batches_tbl) is not None

    # from_arrow with ChunkedArray where to_numpy fails
    mock_ca_err = mock.MagicMock(spec=["to_numpy", "to_pylist"])
    mock_ca_err.to_numpy.side_effect = RuntimeError("no numpy")
    mock_ca_err.to_pylist.return_value = [10, 20]
    t_ca_err = from_arrow(mock_ca_err)
    assert t_ca_err.data.shape == (2,)

    # from_arrow with fallback python list
    t_list_fallback = from_arrow([100, 200])
    assert isinstance(t_list_fallback, Tensor)

    # record_batch_to_tensor with explicit dtype and single column
    t_single_col = record_batch_to_tensor(pa.RecordBatch.from_arrays([pa.array([1, 2])], names=["a"]), dtype="float64")
    assert t_single_col.data.dtype == np.float64

    # record_batch_to_tensor with integer column indexing and col.to_numpy failure
    mock_col = mock.MagicMock()
    mock_col.to_numpy.side_effect = RuntimeError("to_numpy fail")
    mock_col.to_pylist.return_value = [5, 6]
    mock_rb_int = mock.MagicMock()
    mock_rb_int.num_columns = 1
    mock_rb_int.schema.field.return_value.name = "0"
    mock_rb_int.column.side_effect = lambda idx: mock_col
    t_int_col = record_batch_to_tensor(mock_rb_int, columns=["0"], dtype="float32")
    assert t_int_col.data.shape == (2,)

    # record_batch_to_tensor dict fallback with explicit dtype
    d_typed = record_batch_to_tensor({"a": [1, 2]}, dtype="float64")
    assert d_typed.data.dtype == np.float64

    # tensor_to_record_batch with mismatched names length on 2D array
    rb_mismatched = tensor_to_record_batch(np.array([[1, 2], [3, 4]]), names=["only_one"])
    assert rb_mismatched.num_columns == 2

    # item with zero-column table
    zero_col_tbl = mock.MagicMock()
    zero_col_tbl.num_columns = 0
    assert pa_pkg.item(zero_col_tbl) == 0.0

    # shares_buffer buffer address comparison branch
    mock_buf1 = mock.MagicMock()
    mock_buf1.address = 88888
    mock_buf2 = mock.MagicMock()
    mock_buf2.address = 88888
    obj_with_buf1 = mock.MagicMock(spec=["buffer"])
    obj_with_buf1.buffer = mock_buf1
    obj_with_buf2 = mock.MagicMock(spec=["buffer"])
    obj_with_buf2.buffer = mock_buf2
    with mock.patch("numpy.shares_memory", side_effect=TypeError("no shares memory")):
        assert shares_buffer(obj_with_buf1, obj_with_buf2) is True
        mock_buf2.address = 0
        assert shares_buffer(obj_with_buf1, obj_with_buf2) is False

    # record_batch_to_tensor on raw non-dict non-RecordBatch data with explicit dtype
    raw_typed_t = record_batch_to_tensor([1.0, 2.0, 3.0], dtype="float64")
    assert raw_typed_t.data.dtype == np.float64
    raw_t = record_batch_to_tensor([4.0, 5.0])
    assert raw_t.data.shape == (2,)

    # record_batch_to_tensor when column lacks to_numpy
    col_no_to_numpy = mock.MagicMock(spec=[])
    col_no_to_numpy.__array__ = lambda: np.array([7.0, 8.0])
    rb_col_no_to_numpy = mock.MagicMock()
    rb_col_no_to_numpy.num_columns = 1
    rb_col_no_to_numpy.schema.field.return_value.name = "col"
    rb_col_no_to_numpy.column.return_value = col_no_to_numpy
    t_no_to_np = record_batch_to_tensor(rb_col_no_to_numpy)
    assert t_no_to_np.data.shape == (2,)

    # item with object having num_columns but no column
    obj_num_cols_only = mock.MagicMock(spec=["num_columns", "item"])
    obj_num_cols_only.num_columns = 5
    obj_num_cols_only.item.return_value = 17.0
    assert pa_pkg.item(obj_num_cols_only) == 17.0

    # tensor_to_record_batch and tensor_to_arrow_tensor when pa lacks attributes
    mock_pa_no_rb = mock.MagicMock(spec=[])
    with mock.patch("ml_switcheroo_compiler.backends.pyarrow_compute.types._get_pa_module", return_value=mock_pa_no_rb):
        assert isinstance(tensor_to_record_batch(np.array([1, 2])), dict)
        assert isinstance(tensor_to_record_batch(np.array([[1, 2], [3, 4]])), dict)
        assert isinstance(tensor_to_arrow_tensor(np.array([1, 2])), np.ndarray)

    # shares_buffer exception in shares_memory
    class FailArrayObj:
        def __array__(self) -> np.ndarray:
            raise TypeError("fail")

    assert shares_buffer(FailArrayObj(), np.array([1])) is False

    # shares_buffer fallback with non-array objects
    class DummyObj:
        pass

    assert shares_buffer(DummyObj(), DummyObj()) is False


def test_eager_and_generator_coverage_branches() -> None:
    """Verify error and branching coverage in eager execution and generator runner."""
    import pyarrow.compute as pc_lib

    # hash_sum with fewer than 3 arguments
    with pytest.raises(BackendNotSupportedError, match="not supported"):
        execute_op("hash_sum", pa.table({"a": [1]}))

    # hash_sum with non-table object without group_by
    with pytest.raises(BackendNotSupportedError, match="not supported"):
        execute_op("hash_sum", object(), "dept", "salary")

    # cbrt with no arguments
    with pytest.raises(BackendNotSupportedError, match="not supported"):
        execute_op("cbrt")

    # cbrt with invalid argument to trigger cbrt exception branch
    with pytest.raises(BackendNotSupportedError, match="Failed executing cbrt"):
        execute_op("cbrt", "not_a_valid_number")

    # item directly on Table with positive num_columns
    t_item_tbl = pa.table({"col": [99.5]})
    assert pa_pkg.item(t_item_tbl) == 99.5

    # Direct function exception branch in execute_op
    with mock.patch.object(pc_lib, "abs", side_effect=RuntimeError("direct abs fail")):
        with pytest.raises(BackendNotSupportedError, match="Failed executing pyarrow.compute.abs"):
            execute_op("abs", pa.array([1, 2]))

    # Target mapped function exception branch in execute_op (e.g. BitwiseAnd with invalid types)
    with pytest.raises(BackendNotSupportedError, match="Failed executing pyarrow.compute.bit_wise_and"):
        execute_op("BitwiseAnd", "invalid", "arg")

    # to_table with sequence of record batches
    b_seq1 = pa.RecordBatch.from_arrays([pa.array([1, 2])], names=["f"])
    b_seq2 = pa.RecordBatch.from_arrays([pa.array([3, 4])], names=["f"])
    tbl_from_seq = pa_pkg.to_table([b_seq1, b_seq2])
    assert isinstance(tbl_from_seq, pa.Table)

    # to_table with dict
    tbl_from_d = pa_pkg.to_table({"col": [1, 2]})
    assert isinstance(tbl_from_d, pa.Table)

    # to_table with empty list
    tbl_empty_list = pa_pkg.to_table([])
    assert isinstance(tbl_empty_list, pa.Table) and tbl_empty_list.num_columns == 0

    # to_table when Table lacks from_batches
    mock_table_no_methods = mock.MagicMock(spec=[])
    mock_pa_no_table_methods = mock.MagicMock()
    mock_pa_no_table_methods.Table = mock_table_no_methods
    with mock.patch("ml_switcheroo_compiler.backends.pyarrow_compute.types._get_pa_module", return_value=mock_pa_no_table_methods):
        assert pa_pkg.to_table([1, 2]) == [1, 2]

    # group_by on object without group_by method
    with pytest.raises(BackendNotSupportedError, match="not supported"):
        execute_op("group_by", object())

    # AOT runner with raw scalar inputs and keyword Tensor inputs
    g = IRGraph()
    n_in1 = IRNode(id="a", op_type="Input", inputs=[], shape_metadata=[2])
    n_in2 = IRNode(id="b", op_type="Input", inputs=[], shape_metadata=[2])
    n_add = IRNode(id="sum_ab", op_type="Add", inputs=["a", "b"], shape_metadata=[2])
    g.nodes = {"a": n_in1, "b": n_in2, "sum_ab": n_add}
    g.inputs = ["a", "b"]
    g.outputs = ["sum_ab"]

    gen = PyArrowComputeGenerator(g, zero_copy=False)
    runner = gen._compile_aot_impl(g)

    # Raw list/ndarray as positional, Tensor as keyword
    t_kw = Tensor(np.array([10.0, 20.0]), TensorConfig((2,), "float32", "cpu"))
    res_kw = runner(np.array([1.0, 2.0]), b=t_kw)
    assert res_kw is not None
