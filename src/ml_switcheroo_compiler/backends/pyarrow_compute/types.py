"""Apache Arrow Compute backend type conversions and columnar array creation utilities."""

from __future__ import annotations

import importlib
from collections.abc import Sequence


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
    pa_mod = _get_pa_module()
    if hasattr(pa_mod, "array"):
        try:
            return pa_mod.array(actual_data)
        except Exception:
            pass
    numpy_mod = importlib.import_module("numpy")
    return numpy_mod.array(actual_data)


def asarray(data_or_cls: object, data: Sequence[object] | None = None) -> object:
    """Convert input to an Arrow or NumPy array.

    Args:
        data_or_cls (object): Data sequence or class context.
        data (Sequence[object] | None): Optional data sequence when called as classmethod.

    Returns:
        object: Columnar array representation.
    """
    actual_data = data_or_cls if data is None else data
    pa_mod = _get_pa_module()
    if hasattr(pa_mod, "array"):
        try:
            return pa_mod.array(actual_data)
        except Exception:
            pass
    numpy_mod = importlib.import_module("numpy")
    return numpy_mod.asarray(actual_data)


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
            return pa_mod.chunked_array(actual_chunks, type=dtype)
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
        if hasattr(table_cls, "from_batches") and hasattr(actual_data, "schema"):
            return table_cls.from_batches([actual_data], schema=schema)
        if hasattr(table_cls, "from_batches") and isinstance(actual_data, (list, tuple)) and actual_data and hasattr(actual_data[0], "schema"):
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
        return item(actual_data.column(0)) if actual_data.num_columns > 0 else 0.0
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
