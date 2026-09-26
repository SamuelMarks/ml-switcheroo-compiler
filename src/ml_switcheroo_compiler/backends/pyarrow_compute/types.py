"""Apache Arrow Compute backend type conversions and columnar array creation utilities."""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ml_switcheroo_compiler.core.tensor import Tensor


def _get_pa_module() -> object:
    """Retrieve pyarrow module dynamically with fallback.

    Returns:
        object: The pyarrow or numpy module.
    """
    try:
        return importlib.import_module("pyarrow")
    except Exception:
        return importlib.import_module("numpy")


def zeros(shape_or_cls: object, shape: tuple[int, ...] | None = None) -> object:
    """Create a zero-filled array compatible with Arrow Compute.

    Args:
        shape_or_cls (object): Shape tuple or class context.
        shape (tuple[int, ...] | None): Optional shape when called as classmethod.

    Returns:
        object: Allocated zero-filled array.
    """
    actual_shape = shape_or_cls if shape is None and isinstance(shape_or_cls, tuple) else (shape or (0,))
    numpy_mod = importlib.import_module("numpy")
    arr = numpy_mod.zeros(actual_shape, dtype=numpy_mod.float32)
    pa_mod = _get_pa_module()
    if hasattr(pa_mod, "array"):
        try:
            return pa_mod.array(arr.flatten())
        except Exception:
            return arr
    return arr


def array(data_or_cls: object, data: Sequence[object] | None = None, dtype: str | None = None) -> object:
    """Create a columnar Arrow array from data.

    Args:
        data_or_cls (object): Data sequence or class context.
        data (Sequence[object] | None): Optional data sequence when called as classmethod.
        dtype (str | None): Optional datatype name.

    Returns:
        object: Constructed Arrow array.
    """
    del dtype
    actual_data = data_or_cls if data is None else data
    raw_data = getattr(actual_data, "data", actual_data)
    pa_mod = _get_pa_module()
    if hasattr(pa_mod, "array"):
        try:
            return pa_mod.array(raw_data)
        except Exception:
            pass
    numpy_mod = importlib.import_module("numpy")
    return numpy_mod.array(raw_data)


def asarray(data_or_cls: object, data: Sequence[object] | None = None) -> object:
    """Convert input to an Arrow or NumPy array.

    Args:
        data_or_cls (object): Data sequence or class context.
        data (Sequence[object] | None): Optional data sequence when called as classmethod.

    Returns:
        object: Columnar array representation.
    """
    actual_data = data_or_cls if data is None else data
    raw_data = getattr(actual_data, "data", actual_data)
    pa_mod = _get_pa_module()
    if hasattr(pa_mod, "array"):
        try:
            return pa_mod.array(raw_data)
        except Exception:
            pass
    numpy_mod = importlib.import_module("numpy")
    return numpy_mod.asarray(raw_data)


def chunked_array(
    chunks_or_cls: object,
    chunks: Sequence[object] | None = None,
    dtype: object | None = None,
) -> object:
    """Construct a columnar ChunkedArray from a sequence of chunks.

    Args:
        chunks_or_cls (object): Chunks sequence or class context.
        chunks (Sequence[object] | None): Sequence of array chunks when called as classmethod.
        dtype (object | None): Optional Arrow type for the chunked array.

    Returns:
        object: Constructed pyarrow ChunkedArray or list of chunks fallback.
    """
    actual_chunks = chunks_or_cls if chunks is None else chunks
    pa_mod = _get_pa_module()
    if hasattr(pa_mod, "chunked_array"):
        try:
            chunk_list = actual_chunks if isinstance(actual_chunks, (list, tuple)) else [actual_chunks]
            conv_chunks = [array(c) if not hasattr(c, "type") else c for c in chunk_list]
            return pa_mod.chunked_array(conv_chunks, type=dtype)
        except Exception:
            pass
    return list(actual_chunks) if isinstance(actual_chunks, (list, tuple)) else [actual_chunks]


def record_batch(
    data_or_cls: object,
    data: Sequence[object] | None = None,
    names: Sequence[str] | None = None,
    schema: object | None = None,
) -> object:
    """Create a columnar RecordBatch from arrays and field names or schema.

    Args:
        data_or_cls (object): Data array sequence or class context.
        data (Sequence[object] | None): Column arrays when called as classmethod.
        names (Sequence[str] | None): Field names for the record batch columns.
        schema (object | None): Optional pyarrow schema.

    Returns:
        object: Constructed RecordBatch or dict fallback.
    """
    actual_data = data_or_cls if data is None else data
    pa_mod = _get_pa_module()
    if hasattr(pa_mod, "RecordBatch") and hasattr(pa_mod.RecordBatch, "from_arrays"):
        try:
            return pa_mod.RecordBatch.from_arrays(list(actual_data), names=list(names) if names else None, schema=schema)
        except Exception:
            pass
    if names is not None and isinstance(actual_data, (list, tuple)):
        return dict(zip(names, actual_data))
    return actual_data


def slice_record_batch(
    batch_or_cls: object,
    batch: object = None,
    offset: int = 0,
    length: int | None = None,
) -> object:
    """Perform zero-copy slicing of a RecordBatch.

    Args:
        batch_or_cls (object): Target RecordBatch or class context.
        batch (object): Target RecordBatch when called as classmethod.
        offset (int): Starting index for slicing. Defaults to 0.
        length (int | None): Number of records to include in the slice.

    Returns:
        object: Sliced RecordBatch.
    """
    if isinstance(batch_or_cls, type):
        actual_batch = batch
        actual_offset = offset
        actual_length = length
    else:
        actual_batch = batch_or_cls
        actual_offset = int(batch) if isinstance(batch, (int, float)) else offset
        actual_length = length

    if hasattr(actual_batch, "slice"):
        return actual_batch.slice(offset=actual_offset, length=actual_length)
    if isinstance(actual_batch, dict):
        end = actual_offset + actual_length if actual_length is not None else None
        return {k: v[actual_offset:end] for k, v in actual_batch.items()}
    end = actual_offset + actual_length if actual_length is not None else None
    return actual_batch[actual_offset:end]


def to_table(
    data_or_cls: object,
    data: object = None,
    names: Sequence[str] | None = None,
    schema: object | None = None,
) -> object:
    """Convert RecordBatches, columnar arrays, or dict to an Arrow Table.

    Args:
        data_or_cls (object): Data sequence or class context.
        data (object): Data sequence or record batch when called as classmethod.
        names (Sequence[str] | None): Column names if converting from arrays.
        schema (object | None): Optional pyarrow schema.

    Returns:
        object: Constructed pyarrow Table.
    """
    actual_data = data_or_cls if data is None else data
    pa_mod = _get_pa_module()
    if hasattr(pa_mod, "Table"):
        table_cls = pa_mod.Table
        if hasattr(table_cls, "from_batches"):
            if hasattr(actual_data, "schema"):
                return table_cls.from_batches([actual_data], schema=schema)
            if isinstance(actual_data, (list, tuple)) and actual_data:
                if hasattr(actual_data[0], "schema"):
                    return table_cls.from_batches(list(actual_data), schema=schema)
        if hasattr(table_cls, "from_pydict") and isinstance(actual_data, dict):
            return table_cls.from_pydict(actual_data, schema=schema)
        if hasattr(table_cls, "from_arrays") and isinstance(actual_data, (list, tuple)):
            col_names = list(names) if names else [f"f{i}" for i in range(len(actual_data))]
            return table_cls.from_arrays(list(actual_data), names=col_names, schema=schema)
    return actual_data


def item(data_or_cls: object, data: object = None) -> float:
    """Extract scalar item value from a columnar Arrow array, chunked array, or record batch.

    Args:
        data_or_cls (object): Target data or class context.
        data (object): Target data when called as classmethod.

    Returns:
        float: Extracted scalar value.
    """
    actual_data = data_or_cls if data is None else data
    if hasattr(actual_data, "num_columns") and hasattr(actual_data, "column"):
        if actual_data.num_columns > 0:
            return item(actual_data.column(0))
        return 0.0
    if hasattr(actual_data, "as_py"):
        return float(actual_data.as_py())
    if hasattr(actual_data, "to_pylist"):
        lst = actual_data.to_pylist()
        if lst:
            first = lst[0]
            val = next(iter(first.values())) if isinstance(first, dict) else (first[0] if isinstance(first, (list, tuple)) else first)
            return float(val)
    if hasattr(actual_data, "item"):
        return float(actual_data.item())
    numpy_mod = importlib.import_module("numpy")
    return float(numpy_mod.asarray(actual_data).item())


def tensor_to_record_batch(
    tensor_or_cls: object,
    tensor: object = None,
    names: Sequence[str] | None = None,
) -> object:
    """Convert a Unified IR Tensor or array to an Arrow RecordBatch.

    Args:
        tensor_or_cls (object): Tensor instance or class context.
        tensor (object): Tensor instance when called as classmethod.
        names (Sequence[str] | None): Optional field names for columns.

    Returns:
        object: Constructed Arrow RecordBatch or dict fallback.
    """
    actual_tensor = tensor_or_cls if tensor is None else tensor
    raw_data = getattr(actual_tensor, "data", actual_tensor)
    numpy_mod = importlib.import_module("numpy")
    arr = numpy_mod.asarray(raw_data)
    pa_mod = _get_pa_module()

    if arr.ndim <= 1:
        col_names = list(names) if names else ["f0"]
        if hasattr(pa_mod, "RecordBatch") and hasattr(pa_mod.RecordBatch, "from_arrays") and hasattr(pa_mod, "array"):
            try:
                pa_col = pa_mod.array(arr)
                return pa_mod.RecordBatch.from_arrays([pa_col], names=col_names)
            except Exception:
                pass
        return {col_names[0]: arr}

    num_cols = arr.shape[1]
    col_names = list(names) if names and len(names) == num_cols else [f"f{i}" for i in range(num_cols)]
    if hasattr(pa_mod, "RecordBatch") and hasattr(pa_mod.RecordBatch, "from_arrays") and hasattr(pa_mod, "array"):
        try:
            col_arrays = [pa_mod.array(arr[:, i]) for i in range(num_cols)]
            return pa_mod.RecordBatch.from_arrays(col_arrays, names=col_names)
        except Exception:
            pass
    return {col_names[i]: arr[:, i] for i in range(num_cols)}


def _dict_to_tensor(
    actual_batch: dict[str, object],
    columns: Sequence[str] | None,
    dtype: str | None,
) -> Tensor:
    """Convert dictionary of columns to Unified IR Tensor.

    Args:
        actual_batch (dict[str, object]): Column dictionary.
        columns (Sequence[str] | None): Optional column names.
        dtype (str | None): Optional target dtype.

    Returns:
        Tensor: Constructed Tensor.
    """
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    numpy_mod = importlib.import_module("numpy")
    col_keys = list(columns) if columns else list(actual_batch.keys())
    if not col_keys:
        empty_arr = numpy_mod.empty((0,), dtype=dtype or numpy_mod.float32)
        return Tensor(empty_arr, TensorConfig(shape=empty_arr.shape, dtype=str(empty_arr.dtype), device="cpu"))
    arrays = [numpy_mod.asarray(actual_batch[k]) for k in col_keys]
    arr = arrays[0] if len(arrays) == 1 else numpy_mod.column_stack(arrays)
    if dtype is not None:
        arr = arr.astype(dtype)
    return Tensor(arr, TensorConfig(shape=arr.shape, dtype=str(arr.dtype), device="cpu"))


def _rb_to_tensor(
    actual_batch: object,
    columns: Sequence[str] | None,
    dtype: str | None,
) -> Tensor:
    """Convert PyArrow RecordBatch to Unified IR Tensor.

    Args:
        actual_batch (object): Arrow RecordBatch.
        columns (Sequence[str] | None): Optional column names.
        dtype (str | None): Optional target dtype.

    Returns:
        Tensor: Constructed Tensor.
    """
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    numpy_mod = importlib.import_module("numpy")
    col_names = list(columns) if columns else [actual_batch.schema.field(i).name for i in range(actual_batch.num_columns)]
    if not col_names:
        empty_arr = numpy_mod.empty((0,), dtype=dtype or numpy_mod.float32)
        return Tensor(empty_arr, TensorConfig(shape=empty_arr.shape, dtype=str(empty_arr.dtype), device="cpu"))

    extracted: list[object] = []
    for name in col_names:
        col = actual_batch.column(name) if hasattr(actual_batch, "column") and isinstance(name, str) else actual_batch.column(int(name))
        if hasattr(col, "to_numpy"):
            try:
                extracted.append(col.to_numpy(zero_copy_only=False))
            except Exception:
                extracted.append(numpy_mod.asarray(col.to_pylist()))
        else:
            extracted.append(numpy_mod.asarray(col))

    arr = extracted[0] if len(extracted) == 1 else numpy_mod.column_stack(extracted)
    if dtype is not None:
        arr = arr.astype(dtype)
    return Tensor(arr, TensorConfig(shape=arr.shape, dtype=str(arr.dtype), device="cpu"))


def record_batch_to_tensor(
    batch_or_cls: object,
    batch: object = None,
    columns: Sequence[str] | None = None,
    dtype: str | None = None,
) -> object:
    """Convert an Arrow RecordBatch to a Unified IR Tensor.

    Args:
        batch_or_cls (object): RecordBatch or class context.
        batch (object): RecordBatch when called as classmethod.
        columns (Sequence[str] | None): Optional subset of column names to extract.
        dtype (str | None): Target dtype for the resulting Tensor.

    Returns:
        object: Resulting Unified IR Tensor.
    """
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    actual_batch = batch_or_cls if batch is None else batch

    if isinstance(actual_batch, dict):
        return _dict_to_tensor(actual_batch, columns, dtype)

    if hasattr(actual_batch, "num_columns") and hasattr(actual_batch, "column"):
        return _rb_to_tensor(actual_batch, columns, dtype)

    numpy_mod = importlib.import_module("numpy")
    arr = numpy_mod.asarray(actual_batch)
    if dtype is not None:
        arr = arr.astype(dtype)
    return Tensor(arr, TensorConfig(shape=arr.shape, dtype=str(arr.dtype), device="cpu"))


def tensor_to_arrow_tensor(
    tensor_or_cls: object,
    tensor: object = None,
) -> object:
    """Convert a Unified IR Tensor or array to a pyarrow.Tensor zero-copy.

    Args:
        tensor_or_cls (object): Tensor or class context.
        tensor (object): Tensor when called as classmethod.

    Returns:
        object: pyarrow.Tensor or numpy ndarray fallback.
    """
    actual_tensor = tensor_or_cls if tensor is None else tensor
    raw_data = getattr(actual_tensor, "data", actual_tensor)
    numpy_mod = importlib.import_module("numpy")
    arr = numpy_mod.ascontiguousarray(numpy_mod.asarray(raw_data))
    pa_mod = _get_pa_module()
    if hasattr(pa_mod, "Tensor") and hasattr(pa_mod.Tensor, "from_numpy"):
        try:
            return pa_mod.Tensor.from_numpy(arr)
        except Exception:
            pass
    return arr


def arrow_tensor_to_tensor(
    tensor_or_cls: object,
    arrow_tensor: object = None,
) -> object:
    """Convert a pyarrow.Tensor to a Unified IR Tensor zero-copy.

    Args:
        tensor_or_cls (object): pyarrow.Tensor or class context.
        arrow_tensor (object): pyarrow.Tensor when called as classmethod.

    Returns:
        object: Unified IR Tensor wrapping the zero-copy buffer view.
    """
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    actual_pa_tensor = tensor_or_cls if arrow_tensor is None else arrow_tensor
    if hasattr(actual_pa_tensor, "to_numpy"):
        arr = actual_pa_tensor.to_numpy()
    else:
        numpy_mod = importlib.import_module("numpy")
        arr = numpy_mod.asarray(actual_pa_tensor)
    return Tensor(arr, TensorConfig(shape=arr.shape, dtype=str(arr.dtype), device="cpu"))


def _table_to_tensor(actual_data: object) -> object:
    """Convert an Arrow Table to a Unified IR Tensor.

    Args:
        actual_data (object): Arrow Table instance.

    Returns:
        object: Resulting Unified IR Tensor.
    """
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    batches = actual_data.to_batches() if hasattr(actual_data, "to_batches") else []
    if batches:
        if len(batches) == 1:
            return record_batch_to_tensor(batches[0])
        numpy_mod = importlib.import_module("numpy")
        sub_tensors = [getattr(record_batch_to_tensor(b), "data", record_batch_to_tensor(b)) for b in batches]
        combined = numpy_mod.concatenate(sub_tensors, axis=0)
        return Tensor(combined, TensorConfig(shape=combined.shape, dtype=str(combined.dtype), device="cpu"))
    numpy_mod = importlib.import_module("numpy")
    empty_arr = numpy_mod.empty((0,), dtype=numpy_mod.float32)
    return Tensor(empty_arr, TensorConfig(shape=(0,), dtype="float32", device="cpu"))


def _chunked_or_array_to_tensor(actual_data: object) -> object:
    """Convert an Arrow ChunkedArray or Array to a Unified IR Tensor.

    Args:
        actual_data (object): Arrow Array or ChunkedArray instance.

    Returns:
        object: Resulting Unified IR Tensor.
    """
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    numpy_mod = importlib.import_module("numpy")
    try:
        arr = actual_data.to_numpy(zero_copy_only=False)
    except Exception:
        arr = numpy_mod.asarray(actual_data.to_pylist())
    return Tensor(arr, TensorConfig(shape=arr.shape, dtype=str(arr.dtype), device="cpu"))


def from_arrow(
    data_or_cls: object,
    data: object = None,
) -> object:
    """Convert an arbitrary Arrow structure to a Unified IR Tensor.

    Args:
        data_or_cls (object): Arrow object or class context.
        data (object): Arrow object when called as classmethod.

    Returns:
        object: Unified IR Tensor or Python scalar.
    """
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    actual_data = data_or_cls if data is None else data

    # pyarrow.Tensor
    if type(actual_data).__name__ == "Tensor" and hasattr(actual_data, "to_numpy") and not hasattr(actual_data, "config"):
        return arrow_tensor_to_tensor(actual_data)

    # RecordBatch
    if hasattr(actual_data, "num_columns") and hasattr(actual_data, "column") and hasattr(actual_data, "schema") and not hasattr(actual_data, "combine_chunks"):
        return record_batch_to_tensor(actual_data)

    # Table
    if hasattr(actual_data, "to_batches"):
        return _table_to_tensor(actual_data)

    # ChunkedArray / Array
    if hasattr(actual_data, "to_numpy"):
        return _chunked_or_array_to_tensor(actual_data)

    # Scalar
    if hasattr(actual_data, "as_py"):
        return actual_data.as_py()

    # Fallback
    numpy_mod = importlib.import_module("numpy")
    arr = numpy_mod.asarray(actual_data)
    return Tensor(arr, TensorConfig(shape=arr.shape, dtype=str(arr.dtype), device="cpu"))


def to_arrow(
    data_or_cls: object,
    data: object = None,
    target_type: str = "tensor",
) -> object:
    """Convert a Unified IR Tensor or array to the specified Arrow type.

    Args:
        data_or_cls (object): Tensor or class context.
        data (object): Tensor when called as classmethod.
        target_type (str): Target Arrow type ('tensor', 'record_batch', 'table', 'array', 'chunked_array').

    Returns:
        object: Converted Arrow object.

    Raises:
        ValueError: If target_type is unrecognized.
    """
    actual_data = data_or_cls if data is None else data
    target = target_type.lower()
    if target == "tensor":
        return tensor_to_arrow_tensor(actual_data)
    if target in ("record_batch", "recordbatch", "batch"):
        return tensor_to_record_batch(actual_data)
    if target in ("table",):
        rb = tensor_to_record_batch(actual_data)
        return to_table(rb)
    if target in ("array", "asarray"):
        return asarray(actual_data)
    if target in ("chunked_array", "chunked"):
        return chunked_array([actual_data])
    msg = f"Unsupported target Arrow type: '{target_type}'"
    raise ValueError(msg)


def is_arrow_object(cls_or_obj: object, obj: object = None) -> bool:
    """Determine whether an object is a PyArrow construct.

    Args:
        cls_or_obj (object): Candidate object or class context when called as classmethod.
        obj (object): Candidate object when called via classmethod.

    Returns:
        bool: True if obj is an Arrow array, table, batch, tensor, scalar, or chunked array.
    """
    target = cls_or_obj if obj is None else obj
    mod_name = getattr(type(target), "__module__", "")
    return mod_name.startswith("pyarrow")


def shares_buffer(obj1: object, obj2: object) -> bool:
    """Determine whether two objects share the underlying memory buffer.

    Args:
        obj1 (object): First tensor or array object.
        obj2 (object): Second tensor or array object.

    Returns:
        bool: True if both objects share memory.
    """
    data1 = getattr(obj1, "data", obj1)
    data2 = getattr(obj2, "data", obj2)

    buf1 = getattr(data1, "buffer", None)
    buf2 = getattr(data2, "buffer", None)
    if buf1 is not None and buf2 is not None and hasattr(buf1, "address") and hasattr(buf2, "address"):
        return bool(buf1.address == buf2.address and buf1.address != 0)

    numpy_mod = importlib.import_module("numpy")
    try:
        arr1 = data1.to_numpy() if hasattr(data1, "to_numpy") else numpy_mod.asarray(data1)
        arr2 = data2.to_numpy() if hasattr(data2, "to_numpy") else numpy_mod.asarray(data2)
        return bool(numpy_mod.shares_memory(arr1, arr2))
    except Exception:
        return False
