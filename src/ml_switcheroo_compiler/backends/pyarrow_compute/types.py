"""Apache Arrow Compute backend type conversions and columnar array creation utilities."""

from __future__ import annotations

import importlib
import logging
import math
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, cast

from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue

if TYPE_CHECKING:
    from ml_switcheroo_compiler.core.tensor import Tensor


def _get_pa_module() -> EagerValue:
    """Retrieve pyarrow module dynamically with fallback.

    Returns:
        EagerValue: The pyarrow module or None when unavailable.
    """
    try:
        return importlib.import_module("pyarrow")
    except Exception:
        return None


def _unwrap_tensor(val: EagerValue) -> EagerValue:
    """Unwrap a Unified IR Tensor to its underlying data buffer.

    Args:
        val (EagerValue): Input Tensor or raw value.

    Returns:
        EagerValue: Underlying buffer or original value.
    """
    if hasattr(val, "config") and hasattr(val, "data"):
        return cast(EagerValue, getattr(val, "data"))
    return val


def _pa_to_ndarray(raw_data: EagerValue, dtype: str | None = None) -> EagerValue:
    """Convert raw data or sequences to an Arrow-backed buffer array without importing numpy.

    Args:
        raw_data (EagerValue): Input sequence, Arrow array, or buffer object.
        dtype (str | None): Optional target datatype string.

    Returns:
        EagerValue: Buffer array with shape and dtype metadata.
    """
    if hasattr(raw_data, "shape") and hasattr(raw_data, "dtype") and hasattr(raw_data, "ndim"):
        if dtype is not None and hasattr(raw_data, "astype"):
            return cast(EagerValue, getattr(raw_data, "astype")(dtype))
        return raw_data

    if hasattr(raw_data, "__array__"):
        arr = getattr(raw_data, "__array__")()
        return cast(EagerValue, arr)

    import pyarrow as pa

    pa_type = pa.from_numpy_dtype(dtype) if dtype is not None else None
    if isinstance(raw_data, (list, tuple)):
        if not raw_data:
            fill_type = pa_type if pa_type is not None else pa.float32()
            return cast(EagerValue, pa.nulls(0, type=fill_type).to_numpy(zero_copy_only=False))
        return cast(EagerValue, pa.array(list(raw_data), type=pa_type).to_numpy(zero_copy_only=False))

    fill_type = pa_type if pa_type is not None else pa.float32()
    return cast(EagerValue, pa.nulls(0, type=fill_type).to_numpy(zero_copy_only=False))


def zeros(shape: tuple[int, ...]) -> EagerValue:
    """Create a zero-filled array compatible with Arrow Compute.

    Args:
        shape (tuple[int, ...]): Shape tuple.

    Returns:
        EagerValue: Allocated zero-filled array.
    """
    total = int(math.prod(shape)) if shape else 1
    pa_mod = _get_pa_module()
    if hasattr(pa_mod, "array"):
        try:
            return cast(EagerValue, getattr(pa_mod, "array")([0.0] * total))
        except Exception:
            import pyarrow as pa

            base = pa.repeat(pa.scalar(0.0, type=pa.float32()), total).to_numpy(zero_copy_only=False)
            return cast(EagerValue, getattr(base, "reshape")(shape))
    import pyarrow as pa

    base = pa.repeat(pa.scalar(0.0, type=pa.float32()), total).to_numpy(zero_copy_only=False)
    return cast(EagerValue, getattr(base, "reshape")(shape))


def ones(shape: tuple[int, ...]) -> EagerValue:
    """Create a one-filled array compatible with Arrow Compute.

    Args:
        shape (tuple[int, ...]): Shape tuple.

    Returns:
        EagerValue: Allocated one-filled Arrow array or sequence fallback.
    """
    total = int(math.prod(shape)) if shape else 1
    pa_mod = _get_pa_module()
    if hasattr(pa_mod, "repeat"):
        return cast(EagerValue, getattr(pa_mod, "repeat")(1.0, total))
    return [1.0] * total


def empty(shape: tuple[int, ...]) -> EagerValue:
    """Create a null-initialized array compatible with Arrow Compute.

    Args:
        shape (tuple[int, ...]): Shape tuple.

    Returns:
        EagerValue: Allocated null-filled Arrow array or sequence fallback.
    """
    total = int(math.prod(shape)) if shape else 1
    pa_mod = _get_pa_module()
    if hasattr(pa_mod, "nulls"):
        import pyarrow as pa

        return cast(EagerValue, getattr(pa_mod, "nulls")(total, type=pa.float32()))
    return [None] * total


def array(data: EagerValue, dtype: str | None = None) -> EagerValue:
    """Create a columnar Arrow array from data.

    Args:
        data (EagerValue): Data sequence or buffer.
        dtype (str | None): Optional datatype name.

    Returns:
        EagerValue: Constructed Arrow array.
    """
    raw_data = _unwrap_tensor(data)
    pa_mod = _get_pa_module()
    if hasattr(pa_mod, "array"):
        try:
            return cast(EagerValue, getattr(pa_mod, "array")(raw_data))
        except (ImportError, AttributeError, TypeError, ValueError, KeyError, RuntimeError, OSError):
            logging.getLogger(__name__).debug("Ignored exception in %s", __name__)
    return _pa_to_ndarray(raw_data, dtype=dtype)


def asarray(data: EagerValue) -> EagerValue:
    """Convert input to an Arrow columnar array.

    Args:
        data (EagerValue): Data sequence or buffer.

    Returns:
        EagerValue: Columnar array representation.
    """
    raw_data = _unwrap_tensor(data)
    pa_mod = _get_pa_module()
    if hasattr(pa_mod, "array"):
        try:
            return cast(EagerValue, getattr(pa_mod, "array")(raw_data))
        except (ImportError, AttributeError, TypeError, ValueError, KeyError, RuntimeError, OSError):
            logging.getLogger(__name__).debug("Ignored exception in %s", __name__)
    return _pa_to_ndarray(raw_data)


def chunked_array(
    chunks: EagerValue,
    dtype: EagerValue | None = None,
) -> EagerValue:
    """Construct a columnar ChunkedArray from a sequence of chunks.

    Args:
        chunks (EagerValue): Sequence of array chunks.
        dtype (EagerValue | None): Optional Arrow type for the chunked array.

    Returns:
        EagerValue: Constructed pyarrow ChunkedArray or list of chunks fallback.
    """
    pa_mod = _get_pa_module()
    if hasattr(pa_mod, "chunked_array"):
        try:
            chunk_list = chunks if isinstance(chunks, (list, tuple)) else [chunks]
            conv_chunks = [array(c) if not hasattr(c, "type") else c for c in chunk_list]
            return cast(EagerValue, getattr(pa_mod, "chunked_array")(conv_chunks, type=dtype))
        except (ImportError, AttributeError, TypeError, ValueError, KeyError, RuntimeError, OSError):
            logging.getLogger(__name__).debug("Ignored exception in %s", __name__)
    return list(chunks) if isinstance(chunks, (list, tuple)) else [chunks]


def record_batch(
    data: EagerValue,
    names: Sequence[str] | None = None,
    schema: EagerValue | None = None,
) -> EagerValue:
    """Create a columnar RecordBatch from arrays and field names or schema.

    Args:
        data (EagerValue): Column arrays.
        names (Sequence[str] | None): Field names for the record batch columns.
        schema (EagerValue | None): Optional pyarrow schema.

    Returns:
        EagerValue: Constructed RecordBatch or dict fallback.
    """
    pa_mod = _get_pa_module()
    if hasattr(pa_mod, "RecordBatch") and hasattr(getattr(pa_mod, "RecordBatch"), "from_arrays"):
        try:
            data_list = list(data) if isinstance(data, Iterable) else [data]
            name_list = list(names) if names is not None else None
            return cast(EagerValue, getattr(pa_mod, "RecordBatch").from_arrays(data_list, names=name_list, schema=schema))
        except (ImportError, AttributeError, TypeError, ValueError, KeyError, RuntimeError, OSError):
            logging.getLogger(__name__).debug("Ignored exception in %s", __name__)
    if names is not None and isinstance(data, (list, tuple)):
        return dict(zip(names, data))
    return data


def slice_record_batch(
    batch: EagerValue,
    offset: int = 0,
    length: int | None = None,
) -> EagerValue:
    """Perform zero-copy slicing of a RecordBatch.

    Args:
        batch (EagerValue): Target RecordBatch.
        offset (int): Starting index for slicing. Defaults to 0.
        length (int | None): Number of records to include in the slice.

    Returns:
        EagerValue: Sliced RecordBatch.
    """
    if hasattr(batch, "slice"):
        return cast(EagerValue, getattr(batch, "slice")(offset=offset, length=length))
    if isinstance(batch, dict):
        end = offset + length if length is not None else None
        return {k: cast(EagerTensorProtocol, v)[offset:end] for k, v in batch.items()}
    end = offset + length if length is not None else None
    return cast(EagerTensorProtocol, batch)[offset:end]


def to_table(
    data: EagerValue,
    names: Sequence[str] | None = None,
    schema: EagerValue | None = None,
) -> EagerValue:
    """Convert RecordBatches, columnar arrays, or dict to an Arrow Table.

    Args:
        data (EagerValue): Data sequence, dict, or record batch.
        names (Sequence[str] | None): Column names if converting from arrays.
        schema (EagerValue | None): Optional pyarrow schema.

    Returns:
        EagerValue: Constructed pyarrow Table.
    """
    pa_mod = _get_pa_module()
    if hasattr(pa_mod, "Table"):
        table_cls = getattr(pa_mod, "Table")
        if hasattr(table_cls, "from_batches"):
            if hasattr(data, "schema"):
                return cast(EagerValue, table_cls.from_batches([data], schema=schema))
            if isinstance(data, (list, tuple)) and data:
                if hasattr(data[0], "schema"):
                    return cast(EagerValue, table_cls.from_batches(list(data), schema=schema))
        if hasattr(table_cls, "from_pydict") and isinstance(data, dict):
            return cast(EagerValue, table_cls.from_pydict(data, schema=schema))
        if hasattr(table_cls, "from_arrays") and isinstance(data, (list, tuple)):
            col_names = list(names) if names else [f"f{i}" for i in range(len(data))]
            return cast(EagerValue, table_cls.from_arrays(list(data), names=col_names, schema=schema))
    return data


def table(
    data: EagerValue,
    names: Sequence[str] | None = None,
    schema: EagerValue | None = None,
) -> EagerValue:
    """Construct an Arrow Table from columnar arrays, batches, or a dictionary.

    Args:
        data (EagerValue): Data sequence, dict, or record batch.
        names (Sequence[str] | None): Column names if converting from arrays.
        schema (EagerValue | None): Optional pyarrow schema.

    Returns:
        EagerValue: Constructed pyarrow Table.
    """
    return to_table(data, names=names, schema=schema)


def item(data: EagerValue) -> float:
    """Extract scalar item value from a columnar Arrow array, chunked array, or record batch.

    Args:
        data (EagerValue): Target data.

    Returns:
        float: Extracted scalar value.
    """
    if hasattr(data, "num_columns") and hasattr(data, "column"):
        if getattr(data, "num_columns") > 0:
            return item(getattr(data, "column")(0))
        return 0.0
    if hasattr(data, "as_py"):
        return float(getattr(data, "as_py")())
    if hasattr(data, "to_pylist"):
        lst = getattr(data, "to_pylist")()
        if lst:
            first = lst[0]
            val = next(iter(first.values())) if isinstance(first, dict) else (first[0] if isinstance(first, (list, tuple)) else first)
            return float(val)
    if hasattr(data, "item"):
        return float(getattr(data, "item")())
    arr = _pa_to_ndarray(data)
    return float(getattr(arr, "item")())


def tensor_to_record_batch(
    tensor: EagerValue,
    names: Sequence[str] | None = None,
) -> EagerValue:
    """Convert a Unified IR Tensor or array to an Arrow RecordBatch.

    Args:
        tensor (EagerValue): Tensor instance or array.
        names (Sequence[str] | None): Optional field names for columns.

    Returns:
        EagerValue: Constructed Arrow RecordBatch or dict fallback.
    """
    raw_data = _unwrap_tensor(tensor)
    arr = _pa_to_ndarray(raw_data)
    pa_mod = _get_pa_module()

    ndim = int(getattr(arr, "ndim", 1))
    if ndim <= 1:
        col_names = list(names) if names else ["f0"]
        if hasattr(pa_mod, "RecordBatch") and hasattr(getattr(pa_mod, "RecordBatch"), "from_arrays") and hasattr(pa_mod, "array"):
            try:
                pa_col = getattr(pa_mod, "array")(arr if ndim == 1 else [getattr(arr, "item")()])
                return cast(EagerValue, getattr(pa_mod, "RecordBatch").from_arrays([pa_col], names=col_names))
            except (ImportError, AttributeError, TypeError, ValueError, KeyError, RuntimeError, OSError):
                logging.getLogger(__name__).debug("Ignored exception in %s", __name__)
        return {col_names[0]: arr}

    shape = cast(tuple[int, ...], getattr(arr, "shape"))
    num_cols = shape[1]
    col_names = list(names) if names and len(names) == num_cols else [f"f{i}" for i in range(num_cols)]
    arr_proto = cast(EagerTensorProtocol, arr)
    if hasattr(pa_mod, "RecordBatch") and hasattr(getattr(pa_mod, "RecordBatch"), "from_arrays") and hasattr(pa_mod, "array"):
        try:
            col_arrays = [getattr(pa_mod, "array")(arr_proto[:, i]) for i in range(num_cols)]
            return cast(EagerValue, getattr(pa_mod, "RecordBatch").from_arrays(col_arrays, names=col_names))
        except (ImportError, AttributeError, TypeError, ValueError, KeyError, RuntimeError, OSError):
            logging.getLogger(__name__).debug("Ignored exception in %s", __name__)
    return {col_names[i]: arr_proto[:, i] for i in range(num_cols)}


def _stack_columns_1d(columns: list[EagerValue], dtype: str | None = None) -> EagerValue:
    """Stack multiple 1D column arrays into a 2D array using PyArrow concatenation.

    Args:
        columns (list[EagerValue]): List of 1D arrays.
        dtype (str | None): Optional target datatype string.

    Returns:
        EagerValue: 2D array of shape (num_rows, num_cols).
    """
    if len(columns) == 1:
        return _pa_to_ndarray(columns[0], dtype=dtype)
    import pyarrow as pa

    pa_cols = [pa.array(c) for c in columns]
    num_rows = len(pa_cols[0])
    num_cols = len(pa_cols)
    concat_buf = pa.concat_arrays(pa_cols).to_numpy(zero_copy_only=False)
    reshaped = getattr(getattr(concat_buf, "reshape")((num_cols, num_rows)), "T")
    return cast(EagerValue, getattr(reshaped, "copy")())


def _dict_to_tensor(
    actual_batch: dict[str, EagerValue],
    columns: Sequence[str] | None,
    dtype: str | None,
) -> Tensor:
    """Convert dictionary of columns to Unified IR Tensor.

    Args:
        actual_batch (dict[str, EagerValue]): Column dictionary.
        columns (Sequence[str] | None): Optional column names.
        dtype (str | None): Optional target dtype.

    Returns:
        Tensor: Constructed Tensor.
    """
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    col_keys = list(columns) if columns else list(actual_batch.keys())
    if not col_keys:
        empty_arr = _pa_to_ndarray([], dtype=dtype or "float32")
        return Tensor(empty_arr, TensorConfig(shape=getattr(empty_arr, "shape"), dtype=str(getattr(empty_arr, "dtype")), device="cpu"))
    arrays = [_pa_to_ndarray(actual_batch[k]) for k in col_keys]
    arr = _stack_columns_1d(arrays, dtype=dtype)
    return Tensor(arr, TensorConfig(shape=getattr(arr, "shape"), dtype=str(getattr(arr, "dtype")), device="cpu"))


def _rb_to_tensor(
    actual_batch: EagerValue,
    columns: Sequence[str] | None,
    dtype: str | None,
) -> Tensor:
    """Convert PyArrow RecordBatch to Unified IR Tensor.

    Args:
        actual_batch (EagerValue): Arrow RecordBatch.
        columns (Sequence[str] | None): Optional column names.
        dtype (str | None): Optional target dtype.

    Returns:
        Tensor: Constructed Tensor.
    """
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    col_names = list(columns) if columns else [getattr(actual_batch, "schema").field(i).name for i in range(getattr(actual_batch, "num_columns"))]
    if not col_names:
        empty_arr = _pa_to_ndarray([], dtype=dtype or "float32")
        return Tensor(empty_arr, TensorConfig(shape=getattr(empty_arr, "shape"), dtype=str(getattr(empty_arr, "dtype")), device="cpu"))

    extracted: list[EagerValue] = []
    for name in col_names:
        col = getattr(actual_batch, "column")(name) if hasattr(actual_batch, "column") and isinstance(name, str) else getattr(actual_batch, "column")(int(name))
        if hasattr(col, "to_numpy"):
            try:
                extracted.append(col.to_numpy(zero_copy_only=False))
            except Exception:
                extracted.append(_pa_to_ndarray(col.to_pylist()))
        else:
            extracted.append(_pa_to_ndarray(col))

    arr = _stack_columns_1d(extracted, dtype=dtype)
    return Tensor(arr, TensorConfig(shape=getattr(arr, "shape"), dtype=str(getattr(arr, "dtype")), device="cpu"))


def record_batch_to_tensor(
    batch: EagerValue,
    columns: Sequence[str] | None = None,
    dtype: str | None = None,
) -> EagerValue:
    """Convert an Arrow RecordBatch to a Unified IR Tensor.

    Args:
        batch (EagerValue): RecordBatch or column dictionary.
        columns (Sequence[str] | None): Optional subset of column names to extract.
        dtype (str | None): Target dtype for the resulting Tensor.

    Returns:
        EagerValue: Resulting Unified IR Tensor.
    """
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    if isinstance(batch, dict):
        return _dict_to_tensor(batch, columns, dtype)

    if hasattr(batch, "num_columns") and hasattr(batch, "column"):
        return _rb_to_tensor(batch, columns, dtype)

    arr = _pa_to_ndarray(batch, dtype=dtype)
    return Tensor(arr, TensorConfig(shape=getattr(arr, "shape"), dtype=str(getattr(arr, "dtype")), device="cpu"))


def tensor_to_arrow_tensor(
    tensor: EagerValue,
) -> EagerValue:
    """Convert a Unified IR Tensor or array to a pyarrow.Tensor zero-copy.

    Args:
        tensor (EagerValue): Tensor or array.

    Returns:
        EagerValue: pyarrow.Tensor or underlying buffer fallback.
    """
    raw_data = _unwrap_tensor(tensor)
    arr = _pa_to_ndarray(raw_data)
    pa_mod = _get_pa_module()
    if hasattr(pa_mod, "Tensor") and hasattr(getattr(pa_mod, "Tensor"), "from_numpy"):
        try:
            return cast(EagerValue, getattr(pa_mod, "Tensor").from_numpy(arr))
        except (ImportError, AttributeError, TypeError, ValueError, KeyError, RuntimeError, OSError):
            logging.getLogger(__name__).debug("Ignored exception in %s", __name__)
    return arr


def tensor(
    data: EagerValue,
) -> EagerValue:
    """Construct a pyarrow.Tensor from a Unified IR Tensor or sequence.

    Args:
        data (EagerValue): Tensor or sequence.

    Returns:
        EagerValue: Constructed pyarrow.Tensor or buffer fallback.
    """
    return tensor_to_arrow_tensor(data)


def arrow_tensor_to_tensor(
    arrow_tensor: EagerValue,
) -> EagerValue:
    """Convert a pyarrow.Tensor to a Unified IR Tensor zero-copy.

    Args:
        arrow_tensor (EagerValue): pyarrow.Tensor.

    Returns:
        EagerValue: Unified IR Tensor wrapping the zero-copy buffer view.
    """
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    if hasattr(arrow_tensor, "to_numpy"):
        arr = getattr(arrow_tensor, "to_numpy")()
    else:
        arr = _pa_to_ndarray(arrow_tensor)
    return Tensor(arr, TensorConfig(shape=getattr(arr, "shape"), dtype=str(getattr(arr, "dtype")), device="cpu"))


def _table_to_tensor(actual_data: EagerValue) -> EagerValue:
    """Convert an Arrow Table to a Unified IR Tensor.

    Args:
        actual_data (EagerValue): Arrow Table instance.

    Returns:
        EagerValue: Resulting Unified IR Tensor.
    """
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    batches = getattr(actual_data, "to_batches")() if hasattr(actual_data, "to_batches") else []
    if batches:
        if len(batches) == 1:
            return record_batch_to_tensor(batches[0])
        import pyarrow as pa

        sub_tensors = [getattr(record_batch_to_tensor(b), "data", record_batch_to_tensor(b)) for b in batches]
        combined = pa.concat_arrays([pa.array(s) for s in sub_tensors]).to_numpy(zero_copy_only=False)
        return Tensor(combined, TensorConfig(shape=getattr(combined, "shape"), dtype=str(getattr(combined, "dtype")), device="cpu"))
    empty_arr = _pa_to_ndarray([], dtype="float32")
    return Tensor(empty_arr, TensorConfig(shape=(0,), dtype="float32", device="cpu"))


def _chunked_or_array_to_tensor(actual_data: EagerValue) -> EagerValue:
    """Convert an Arrow ChunkedArray or Array to a Unified IR Tensor.

    Args:
        actual_data (EagerValue): Arrow Array or ChunkedArray instance.

    Returns:
        EagerValue: Resulting Unified IR Tensor.
    """
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    try:
        arr = getattr(actual_data, "to_numpy")(zero_copy_only=False)
    except Exception:
        arr = _pa_to_ndarray(getattr(actual_data, "to_pylist")())
    return Tensor(arr, TensorConfig(shape=getattr(arr, "shape"), dtype=str(getattr(arr, "dtype")), device="cpu"))


def from_arrow(
    data: EagerValue,
) -> EagerValue:
    """Convert an arbitrary Arrow structure to a Unified IR Tensor.

    Args:
        data (EagerValue): Arrow object.

    Returns:
        EagerValue: Unified IR Tensor or Python scalar.
    """
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    # pyarrow.Tensor
    if type(data).__name__ == "Tensor" and hasattr(data, "to_numpy") and not hasattr(data, "config"):
        return arrow_tensor_to_tensor(data)

    # RecordBatch
    if hasattr(data, "num_columns") and hasattr(data, "column") and hasattr(data, "schema") and not hasattr(data, "combine_chunks"):
        return record_batch_to_tensor(data)

    # Table
    if hasattr(data, "to_batches"):
        return _table_to_tensor(data)

    # ChunkedArray / Array
    if hasattr(data, "to_numpy"):
        return _chunked_or_array_to_tensor(data)

    # Scalar
    if hasattr(data, "as_py"):
        return cast(EagerValue, getattr(data, "as_py")())

    # Fallback
    arr = _pa_to_ndarray(data)
    return Tensor(arr, TensorConfig(shape=getattr(arr, "shape"), dtype=str(getattr(arr, "dtype")), device="cpu"))


def to_arrow(
    data: EagerValue,
    target_type: str = "tensor",
) -> EagerValue:
    """Convert a Unified IR Tensor or array to the specified Arrow type.

    Args:
        data (EagerValue): Tensor or array.
        target_type (str): Target Arrow type ('tensor', 'record_batch', 'table', 'array', 'chunked_array').

    Returns:
        EagerValue: Converted Arrow object.

    Raises:
        ValueError: If target_type is unrecognized.
    """
    target = target_type.lower()
    if target == "tensor":
        return tensor_to_arrow_tensor(data)
    if target in ("record_batch", "recordbatch", "batch"):
        return tensor_to_record_batch(data)
    if target in ("table",):
        rb = tensor_to_record_batch(data)
        return to_table(rb)
    if target in ("array", "asarray"):
        return asarray(data)
    if target in ("chunked_array", "chunked"):
        return chunked_array([data])
    msg = f"Unsupported target Arrow type: '{target_type}'"
    raise ValueError(msg)


def is_arrow_object(obj: EagerValue) -> bool:
    """Determine whether an object is a PyArrow construct.

    Args:
        obj (EagerValue): Candidate object.

    Returns:
        bool: True if obj is an Arrow array, table, batch, tensor, scalar, or chunked array.
    """
    mod_name = getattr(type(obj), "__module__", "")
    return mod_name.startswith("pyarrow")


def shares_buffer(obj1: EagerValue, obj2: EagerValue) -> bool:
    """Determine whether two objects share the underlying memory buffer.

    Args:
        obj1 (EagerValue): First tensor or array object.
        obj2 (EagerValue): Second tensor or array object.

    Returns:
        bool: True if both objects share memory.
    """
    data1 = _unwrap_tensor(obj1)
    data2 = _unwrap_tensor(obj2)

    buf1 = getattr(data1, "buffer", None)
    buf2 = getattr(data2, "buffer", None)
    if buf1 is not None and buf2 is not None and hasattr(buf1, "address") and hasattr(buf2, "address"):
        return bool(buf1.address == buf2.address and buf1.address != 0)

    arr1 = getattr(data1, "to_numpy")() if hasattr(data1, "to_numpy") else data1
    arr2 = getattr(data2, "to_numpy")() if hasattr(data2, "to_numpy") else data2
    iface1 = getattr(arr1, "__array_interface__", None)
    iface2 = getattr(arr2, "__array_interface__", None)
    if isinstance(iface1, dict) and isinstance(iface2, dict):
        addr1 = iface1.get("data", (0,))[0]
        addr2 = iface2.get("data", (0,))[0]
        return bool(addr1 == addr2 and addr1 != 0)
    return False


def check_parity(
    lhs: EagerValue,
    rhs: EagerValue,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> bool:
    """Check numerical parity between two Arrow or sequence structures.

    Args:
        lhs (EagerValue): First operand.
        rhs (EagerValue): Second operand.
        rtol (float): Relative tolerance. Defaults to 1e-5.
        atol (float): Absolute tolerance. Defaults to 1e-8.

    Returns:
        bool: True if all elements match within tolerance.
    """
    arr1 = getattr(_pa_to_ndarray(_unwrap_tensor(lhs)), "reshape")(-1)
    arr2 = getattr(_pa_to_ndarray(_unwrap_tensor(rhs)), "reshape")(-1)
    list1 = getattr(arr1, "tolist")()
    list2 = getattr(arr2, "tolist")()
    if len(list1) != len(list2):
        return False
    return all(math.isclose(float(a), float(b), rel_tol=rtol, abs_tol=atol) for a, b in zip(list1, list2))
