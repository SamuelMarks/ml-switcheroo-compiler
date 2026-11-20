"""Apache Arrow Compute backend type conversions and columnar array creation utilities."""

from __future__ import annotations

import importlib
import logging
import math
import typing
from collections.abc import Iterable, Sequence

from ml_switcheroo_compiler.backends.eager_registry import (
    BackendArg,
    BackendArrayProtocol,
    EagerTensorProtocol,
    coerce_array,
    coerce_float,
    coerce_int,
    coerce_operand_list,
    coerce_shape,
    require_callable,
)
from ml_switcheroo_compiler.backends.protocols import adapt_backend_module

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.core.tensor import Tensor


@typing.runtime_checkable
class _ArrowArrayInterfaceProto(typing.Protocol):
    """Protocol for host arrays exposing the NumPy `__array_interface__` descriptor."""

    @property
    def __array_interface__(self) -> dict[str, BackendArg]:
        """Return the host buffer interface dictionary.

        Returns:
            dict[str, BackendArg]: Buffer interface metadata.
        """
        ...


@typing.runtime_checkable
class ArrowArrayProtocol(typing.Protocol):
    """Protocol for PyArrow Array and ChunkedArray objects."""

    def to_numpy(self, zero_copy_only: bool = False) -> BackendArg:
        """Convert Arrow array to a zero-copy or copied buffer.

        Args:
            zero_copy_only (bool): Whether to require zero-copy.

        Returns:
            BackendArg: Buffer view.
        """
        ...

    def to_pylist(self) -> list[BackendArg]:
        """Convert Arrow array to a Python list.

        Returns:
            list[BackendArg]: Python list.
        """
        ...


@typing.runtime_checkable
class ArrowRecordBatchProtocol(typing.Protocol):
    """Protocol for PyArrow RecordBatch objects."""

    @property
    def num_columns(self) -> int:
        """Return number of columns.

        Returns:
            int: Column count.
        """
        ...

    @property
    def schema(self) -> BackendArg:
        """Return record batch schema.

        Returns:
            BackendArg: Schema object.
        """
        ...

    def column(self, i: int | str) -> BackendArg:
        """Retrieve column by index or name.

        Args:
            i (int | str): Column index or name.

        Returns:
            BackendArg: Column array.
        """
        ...

    def slice(self, offset: int = 0, length: int | None = None) -> BackendArg:
        """Slice record batch zero-copy.

        Args:
            offset (int): Starting index.
            length (int | None): Slice length.

        Returns:
            BackendArg: Sliced RecordBatch.
        """
        ...


@typing.runtime_checkable
class ArrowTableProtocol(typing.Protocol):
    """Protocol for PyArrow Table objects."""

    def to_batches(self) -> list[BackendArg]:
        """Convert table to a list of RecordBatches.

        Returns:
            list[BackendArg]: RecordBatch list.
        """
        ...


@typing.runtime_checkable
class ArrowTensorProtocol(typing.Protocol):
    """Protocol for PyArrow Tensor objects."""

    def to_numpy(self) -> BackendArg:
        """Convert Arrow Tensor to buffer view.

        Returns:
            BackendArg: Buffer view.
        """
        ...


@typing.runtime_checkable
class _ArrowScalarProtocol(typing.Protocol):
    """Protocol for PyArrow Scalar objects exposing `.as_py()`."""

    def as_py(self) -> float | int | bool:
        """Convert scalar to Python value.

        Returns:
            float | int | bool: Python scalar.
        """
        ...


@typing.runtime_checkable
class _DunderArrayProtocol(typing.Protocol):
    """Protocol for objects exposing `__array__()`."""

    def __array__(self, dtype: str | None = None) -> BackendArg:
        """Convert to array buffer.

        Args:
            dtype (str | None): Optional target datatype string.

        Returns:
            BackendArg: Array buffer.
        """
        ...


@typing.runtime_checkable
class _ArrayInterfaceProtocol(typing.Protocol):
    """Protocol for array buffers exposing `__array_interface__`."""

    __array_interface__: dict[str, BackendArg]


def _get_pa_module() -> BackendArg:
    """Retrieve pyarrow module dynamically with fallback.

    Returns:
        BackendArg: The pyarrow module or None when unavailable.
    """
    try:
        return importlib.import_module("pyarrow")
    except (ImportError, ModuleNotFoundError, OSError, AttributeError, RuntimeError, TypeError, ValueError):
        return None


def _unwrap_tensor(val: BackendArg) -> BackendArg:
    """Unwrap a Unified IR Tensor to its underlying data buffer.

    Args:
        val (BackendArg): Input Tensor or raw value.

    Returns:
        BackendArg: Underlying buffer or original value.
    """
    if adapt_backend_module(val).has_symbol("config") and adapt_backend_module(val).has_symbol("data"):
        return adapt_backend_module(val).require_symbol("data")
    return val


def _pa_to_ndarray(raw_data: BackendArg, dtype: str | None = None) -> BackendArg:
    """Convert raw data or sequences to an Arrow-backed buffer array without importing numpy.

    Args:
        raw_data (BackendArg): Input sequence, Arrow array, or buffer object.
        dtype (str | None): Optional target datatype string.

    Returns:
        BackendArg: Buffer array with shape and dtype metadata.
    """
    if (
        adapt_backend_module(raw_data).has_symbol("shape")
        and adapt_backend_module(raw_data).has_symbol("dtype")
        and adapt_backend_module(raw_data).has_symbol("ndim")
    ):
        if dtype is not None and adapt_backend_module(raw_data).has_symbol("astype"):
            return adapt_backend_module(raw_data).require_op("astype")(dtype)
        return raw_data

    import pyarrow as pa

    pa_type = pa.from_numpy_dtype(dtype) if dtype is not None else None
    cand_dunder: BackendArg | _DunderArrayProtocol = raw_data
    if isinstance(cand_dunder, _DunderArrayProtocol):
        arr_fn = cand_dunder.__array__
        return arr_fn(dtype) if dtype is not None else arr_fn()
    if isinstance(raw_data, (list, tuple)):
        if not raw_data:
            fill_type = pa_type if pa_type is not None else pa.float32()
            null_arr = require_callable(pa.nulls)(0, type=fill_type)
            return adapt_backend_module(null_arr).require_op("to_numpy")(zero_copy_only=False)
        pa_arr = require_callable(pa.array)(list(raw_data), type=pa_type)
        return adapt_backend_module(pa_arr).require_op("to_numpy")(zero_copy_only=False)

    fill_type = pa_type if pa_type is not None else pa.float32()
    null_arr_fb = require_callable(pa.nulls)(0, type=fill_type)
    return adapt_backend_module(null_arr_fb).require_op("to_numpy")(zero_copy_only=False)


def zeros(shape: tuple[int, ...]) -> BackendArg:
    """Create a zero-filled array compatible with Arrow Compute.

    Args:
        shape (tuple[int, ...]): Shape tuple.

    Returns:
        BackendArg: Allocated zero-filled array.
    """
    total = int(math.prod(shape)) if shape else 1
    pa_mod = _get_pa_module()
    if adapt_backend_module(pa_mod).has_symbol("array"):
        try:
            return adapt_backend_module(pa_mod).require_op("array")([0.0] * total)
        except (ImportError, ModuleNotFoundError, OSError, AttributeError, RuntimeError, TypeError, ValueError):
            import pyarrow as pa

            rep = require_callable(pa.repeat)(pa.scalar(0.0, type=pa.float32()), total)
            base: EagerTensorProtocol = adapt_backend_module(rep).require_op("to_numpy")(zero_copy_only=False)
            return adapt_backend_module(base).require_op("reshape")(shape)
    import pyarrow as pa

    rep_fb = require_callable(pa.repeat)(pa.scalar(0.0, type=pa.float32()), total)
    base_fb: EagerTensorProtocol = adapt_backend_module(rep_fb).require_op("to_numpy")(zero_copy_only=False)
    return adapt_backend_module(base_fb).require_op("reshape")(shape)


def ones(shape: tuple[int, ...]) -> BackendArg:
    """Create a one-filled array compatible with Arrow Compute.

    Args:
        shape (tuple[int, ...]): Shape tuple.

    Returns:
        BackendArg: Allocated one-filled Arrow array or sequence fallback.
    """
    total = int(math.prod(shape)) if shape else 1
    pa_mod = _get_pa_module()
    if adapt_backend_module(pa_mod).has_symbol("repeat"):
        return adapt_backend_module(pa_mod).require_op("repeat")(1.0, total)
    return [1.0] * total


def empty(shape: tuple[int, ...]) -> BackendArg:
    """Create a null-initialized array compatible with Arrow Compute.

    Args:
        shape (tuple[int, ...]): Shape tuple.

    Returns:
        BackendArg: Allocated null-filled Arrow array or sequence fallback.
    """
    total = int(math.prod(shape)) if shape else 1
    pa_mod = _get_pa_module()
    if adapt_backend_module(pa_mod).has_symbol("nulls"):
        import pyarrow as pa

        return adapt_backend_module(pa_mod).require_op("nulls")(total, type=pa.float32())
    return [None] * total


def array(data: BackendArg, dtype: str | None = None) -> BackendArg:
    """Create a columnar Arrow array from data.

    Args:
        data (BackendArg): Data sequence or buffer.
        dtype (str | None): Optional datatype name.

    Returns:
        BackendArg: Constructed Arrow array.
    """
    raw_data = _unwrap_tensor(data)
    pa_mod = _get_pa_module()
    if adapt_backend_module(pa_mod).has_symbol("array"):
        try:
            return adapt_backend_module(pa_mod).require_op("array")(raw_data)
        except (ImportError, AttributeError, TypeError, ValueError, KeyError, RuntimeError, OSError):
            logging.getLogger(__name__).debug("Ignored exception in %s", __name__)
    return _pa_to_ndarray(raw_data, dtype=dtype)


def asarray(data: BackendArg) -> BackendArg:
    """Convert input to an Arrow columnar array.

    Args:
        data (BackendArg): Data sequence or buffer.

    Returns:
        BackendArg: Columnar array representation.
    """
    raw_data = _unwrap_tensor(data)
    pa_mod = _get_pa_module()
    if adapt_backend_module(pa_mod).has_symbol("array"):
        try:
            return adapt_backend_module(pa_mod).require_op("array")(raw_data)
        except (ImportError, AttributeError, TypeError, ValueError, KeyError, RuntimeError, OSError):
            logging.getLogger(__name__).debug("Ignored exception in %s", __name__)
    return _pa_to_ndarray(raw_data)


def chunked_array(
    chunks: BackendArg,
    dtype: BackendArg | None = None,
) -> BackendArg:
    """Construct a columnar ChunkedArray from a sequence of chunks.

    Args:
        chunks (BackendArg): Sequence of array chunks.
        dtype (BackendArg | None): Optional Arrow type for the chunked array.

    Returns:
        BackendArg: Constructed pyarrow ChunkedArray or list of chunks fallback.
    """
    pa_mod = _get_pa_module()
    if adapt_backend_module(pa_mod).has_symbol("chunked_array"):
        try:
            chunk_list = chunks if isinstance(chunks, (list, tuple)) else [chunks]
            conv_chunks = [array(c) if not adapt_backend_module(c).has_symbol("type") else c for c in chunk_list]
            return adapt_backend_module(pa_mod).require_op("chunked_array")(conv_chunks, type=dtype)
        except (ImportError, AttributeError, TypeError, ValueError, KeyError, RuntimeError, OSError):
            logging.getLogger(__name__).debug("Ignored exception in %s", __name__)
    return list(chunks) if isinstance(chunks, (list, tuple)) else [chunks]


def record_batch(
    data: BackendArg,
    names: Sequence[str] | None = None,
    schema: BackendArg | None = None,
) -> BackendArg:
    """Create a columnar RecordBatch from arrays and field names or schema.

    Args:
        data (BackendArg): Column arrays.
        names (Sequence[str] | None): Field names for the record batch columns.
        schema (BackendArg | None): Optional pyarrow schema.

    Returns:
        BackendArg: Constructed RecordBatch or dict fallback.
    """
    pa_mod = _get_pa_module()
    rb_cls = adapt_backend_module(pa_mod).get_symbol("RecordBatch", None)
    if adapt_backend_module(rb_cls).has_symbol("from_arrays"):
        try:
            data_list = list(data) if isinstance(data, Iterable) else [data]
            name_list = list(names) if names is not None else None
            return adapt_backend_module(rb_cls).require_op("from_arrays")(data_list, names=name_list, schema=schema)
        except (ImportError, AttributeError, TypeError, ValueError, KeyError, RuntimeError, OSError):
            logging.getLogger(__name__).debug("Ignored exception in %s", __name__)
    if names is not None and isinstance(data, (list, tuple)):
        return dict(zip(names, data))
    return data


def slice_record_batch(
    batch: BackendArg,
    offset: int = 0,
    length: int | None = None,
) -> BackendArg:
    """Perform zero-copy slicing of a RecordBatch.

    Args:
        batch (BackendArg): Target RecordBatch.
        offset (int): Starting index for slicing. Defaults to 0.
        length (int | None): Number of records to include in the slice.

    Returns:
        BackendArg: Sliced RecordBatch.
    """
    raw_batch: BackendArg | ArrowRecordBatchProtocol = batch
    if isinstance(raw_batch, ArrowRecordBatchProtocol) or adapt_backend_module(batch).has_symbol("slice"):
        return adapt_backend_module(batch).require_op("slice")(offset=offset, length=length)
    if isinstance(batch, dict):
        end = offset + length if length is not None else None
        return {k: coerce_operand_list(v)[offset:end] for k, v in batch.items()}
    end = offset + length if length is not None else None
    return coerce_operand_list(batch)[offset:end]


def to_table(
    data: BackendArg,
    names: Sequence[str] | None = None,
    schema: BackendArg | None = None,
) -> BackendArg:
    """Convert RecordBatches, columnar arrays, or dict to an Arrow Table.

    Args:
        data (BackendArg): Data sequence, dict, or record batch.
        names (Sequence[str] | None): Column names if converting from arrays.
        schema (BackendArg | None): Optional pyarrow schema.

    Returns:
        BackendArg: Constructed pyarrow Table.
    """
    pa_mod = _get_pa_module()
    if adapt_backend_module(pa_mod).has_symbol("Table"):
        table_cls = adapt_backend_module(pa_mod).require_symbol("Table")
        if adapt_backend_module(table_cls).has_symbol("from_batches"):
            if adapt_backend_module(data).has_symbol("schema"):
                return adapt_backend_module(table_cls).require_op("from_batches")([data], schema=schema)
            if isinstance(data, (list, tuple)) and data:
                if adapt_backend_module(data[0]).has_symbol("schema"):
                    return adapt_backend_module(table_cls).require_op("from_batches")(list(data), schema=schema)
        if adapt_backend_module(table_cls).has_symbol("from_pydict") and isinstance(data, dict):
            return adapt_backend_module(table_cls).require_op("from_pydict")(data, schema=schema)
        if adapt_backend_module(table_cls).has_symbol("from_arrays") and isinstance(data, (list, tuple)):
            col_names = list(names) if names else [f"f{i}" for i in range(len(data))]
            return adapt_backend_module(table_cls).require_op("from_arrays")(list(data), names=col_names, schema=schema)
    return data


def table(
    data: BackendArg,
    names: Sequence[str] | None = None,
    schema: BackendArg | None = None,
) -> BackendArg:
    """Construct an Arrow Table from columnar arrays, batches, or a dictionary.

    Args:
        data (BackendArg): Data sequence, dict, or record batch.
        names (Sequence[str] | None): Column names if converting from arrays.
        schema (BackendArg | None): Optional pyarrow schema.

    Returns:
        BackendArg: Constructed pyarrow Table.
    """
    return to_table(data, names=names, schema=schema)


def item(data: BackendArg) -> float:
    """Extract scalar item value from a columnar Arrow array, chunked array, or record batch.

    Args:
        data (BackendArg): Target data.

    Returns:
        float: Extracted scalar value.
    """
    raw_op: BackendArg | ArrowRecordBatchProtocol = data
    raw_scalar: BackendArg | _ArrowScalarProtocol = data
    if isinstance(raw_op, ArrowRecordBatchProtocol) or (
        adapt_backend_module(data).has_symbol("num_columns") and adapt_backend_module(data).has_symbol("column")
    ):
        if coerce_int(adapt_backend_module(data).get_symbol("num_columns", 0)) > 0:
            return item(adapt_backend_module(data).require_op("column")(0))
        return 0.0
    if isinstance(raw_scalar, _ArrowScalarProtocol) or adapt_backend_module(data).has_symbol("as_py"):
        return float(coerce_float(adapt_backend_module(data).require_op("as_py")()))
    if adapt_backend_module(data).has_symbol("to_pylist"):
        lst: BackendArg = adapt_backend_module(data).require_op("to_pylist")()
        if isinstance(lst, (list, tuple)) and lst:
            first = lst[0]
            val = (
                next(iter(first.values()))
                if isinstance(first, dict)
                else (first[0] if isinstance(first, (list, tuple)) else first)
            )
            return float(coerce_float(val))
    if adapt_backend_module(data).has_symbol("item"):
        return float(coerce_float(adapt_backend_module(data).require_op("item")()))
    arr = _pa_to_ndarray(data)
    return float(coerce_float(adapt_backend_module(arr).require_op("item")()))


def tensor_to_record_batch(
    tensor: BackendArg,
    names: Sequence[str] | None = None,
) -> BackendArg:
    """Convert a Unified IR Tensor or array to an Arrow RecordBatch.

    Args:
        tensor (BackendArg): Tensor instance or array.
        names (Sequence[str] | None): Optional field names for columns.

    Returns:
        BackendArg: Constructed Arrow RecordBatch or dict fallback.
    """
    raw_data = _unwrap_tensor(tensor)
    arr = _pa_to_ndarray(raw_data)
    pa_mod = _get_pa_module()
    rb_cls = adapt_backend_module(pa_mod).get_symbol("RecordBatch", None)

    ndim = coerce_int(adapt_backend_module(arr).get_symbol("ndim", 1))
    if ndim <= 1:
        col_names = list(names) if names else ["f0"]
        if adapt_backend_module(rb_cls).has_symbol("from_arrays") and adapt_backend_module(pa_mod).has_symbol("array"):
            try:
                pa_col = adapt_backend_module(pa_mod).require_op("array")(
                    arr if ndim == 1 else [adapt_backend_module(arr).require_op("item")()]
                )
                return adapt_backend_module(rb_cls).require_op("from_arrays")([pa_col], names=col_names)
            except (ImportError, AttributeError, TypeError, ValueError, KeyError, RuntimeError, OSError):
                logging.getLogger(__name__).debug("Ignored exception in %s", __name__)
        return {col_names[0]: arr}

    shape = coerce_shape(adapt_backend_module(arr).require_symbol("shape"))
    num_cols = int(shape[1])
    col_names = list(names) if names and len(names) == num_cols else [f"f{i}" for i in range(num_cols)]
    arr_proto: BackendArrayProtocol = coerce_array(pa_mod, arr)
    if adapt_backend_module(rb_cls).has_symbol("from_arrays") and adapt_backend_module(pa_mod).has_symbol("array"):
        try:
            col_arrays = [adapt_backend_module(pa_mod).require_op("array")(arr_proto[:, i]) for i in range(num_cols)]
            return adapt_backend_module(rb_cls).require_op("from_arrays")(col_arrays, names=col_names)
        except (ImportError, AttributeError, TypeError, ValueError, KeyError, RuntimeError, OSError):
            logging.getLogger(__name__).debug("Ignored exception in %s", __name__)
    return {col_names[i]: arr_proto[:, i] for i in range(num_cols)}


def _stack_columns_1d(columns: list[BackendArg], dtype: str | None = None) -> BackendArg:
    """Stack multiple 1D column arrays into a 2D array using PyArrow concatenation.

    Args:
        columns (list[BackendArg]): List of 1D arrays.
        dtype (str | None): Optional target datatype string.

    Returns:
        BackendArg: 2D array of shape (num_rows, num_cols).
    """
    if len(columns) == 1:
        return _pa_to_ndarray(columns[0], dtype=dtype)
    import pyarrow as pa

    pa_cols = [pa.array(c) for c in columns]
    num_rows = len(pa_cols[0])
    num_cols = len(pa_cols)
    concat_arr = require_callable(pa.concat_arrays)(pa_cols)
    concat_buf = adapt_backend_module(concat_arr).require_op("to_numpy")(zero_copy_only=False)
    reshaped = adapt_backend_module(concat_buf).require_op("reshape")((num_cols, num_rows))
    transposed = adapt_backend_module(reshaped).require_symbol("T")
    return adapt_backend_module(transposed).require_op("copy")()


def _dict_to_tensor(
    actual_batch: dict[str, BackendArg],
    columns: Sequence[str] | None,
    dtype: str | None,
) -> Tensor:
    """Convert dictionary of columns to Unified IR Tensor.

    Args:
        actual_batch (dict[str, BackendArg]): Column dictionary.
        columns (Sequence[str] | None): Optional column names.
        dtype (str | None): Optional target dtype.

    Returns:
        Tensor: Constructed Tensor.
    """
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    col_keys = list(columns) if columns else list(actual_batch.keys())
    if not col_keys:
        empty_arr = _pa_to_ndarray([], dtype=dtype or "float32")
        return Tensor(
            empty_arr,
            TensorConfig(
                shape=coerce_shape(adapt_backend_module(empty_arr).require_symbol("shape")),
                dtype=str(adapt_backend_module(empty_arr).require_symbol("dtype")),
                device="cpu",
            ),
        )
    arrays = [_pa_to_ndarray(actual_batch[k]) for k in col_keys]
    arr = _stack_columns_1d(arrays, dtype=dtype)
    return Tensor(
        arr,
        TensorConfig(
            shape=coerce_shape(adapt_backend_module(arr).require_symbol("shape")),
            dtype=str(adapt_backend_module(arr).require_symbol("dtype")),
            device="cpu",
        ),
    )


def _rb_to_tensor(
    actual_batch: BackendArg,
    columns: Sequence[str] | None,
    dtype: str | None,
) -> Tensor:
    """Convert PyArrow RecordBatch to Unified IR Tensor.

    Args:
        actual_batch (BackendArg): Arrow RecordBatch.
        columns (Sequence[str] | None): Optional column names.
        dtype (str | None): Optional target dtype.

    Returns:
        Tensor: Constructed Tensor.
    """
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    schema_obj = adapt_backend_module(actual_batch).require_symbol("schema")
    num_cols = coerce_int(adapt_backend_module(actual_batch).get_symbol("num_columns", 0))
    col_names = (
        list(columns)
        if columns
        else [
            str(adapt_backend_module(adapt_backend_module(schema_obj).require_op("field")(i)).require_symbol("name"))
            for i in range(num_cols)
        ]
    )
    if not col_names:
        empty_arr = _pa_to_ndarray([], dtype=dtype or "float32")
        return Tensor(
            empty_arr,
            TensorConfig(
                shape=coerce_shape(adapt_backend_module(empty_arr).require_symbol("shape")),
                dtype=str(adapt_backend_module(empty_arr).require_symbol("dtype")),
                device="cpu",
            ),
        )

    extracted: list[BackendArg] = []
    col_fn = adapt_backend_module(actual_batch).require_op("column")
    for name in col_names:
        col = (
            col_fn(name)
            if adapt_backend_module(actual_batch).has_symbol("column") and isinstance(name, str)
            else col_fn(int(name))
        )
        raw_col: BackendArg | ArrowArrayProtocol = col
        if isinstance(raw_col, ArrowArrayProtocol) or adapt_backend_module(col).has_symbol("to_numpy"):
            try:
                extracted.append(adapt_backend_module(col).require_op("to_numpy")(zero_copy_only=False))
            except (
                AttributeError,
                ImportError,
                LookupError,
                OSError,
                ArithmeticError,
                RuntimeError,
                SyntaxError,
                TypeError,
                ValueError,
            ):
                extracted.append(_pa_to_ndarray(adapt_backend_module(col).require_op("to_pylist")()))
        else:
            extracted.append(_pa_to_ndarray(col))

    arr = _stack_columns_1d(extracted, dtype=dtype)
    return Tensor(
        arr,
        TensorConfig(
            shape=coerce_shape(adapt_backend_module(arr).require_symbol("shape")),
            dtype=str(adapt_backend_module(arr).require_symbol("dtype")),
            device="cpu",
        ),
    )


def record_batch_to_tensor(
    batch: BackendArg,
    columns: Sequence[str] | None = None,
    dtype: str | None = None,
) -> BackendArg:
    """Convert an Arrow RecordBatch to a Unified IR Tensor.

    Args:
        batch (BackendArg): RecordBatch or column dictionary.
        columns (Sequence[str] | None): Optional subset of column names to extract.
        dtype (str | None): Target dtype for the resulting Tensor.

    Returns:
        BackendArg: Resulting Unified IR Tensor.
    """
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    if isinstance(batch, dict):
        return _dict_to_tensor(batch, columns, dtype)

    raw_batch: BackendArg | ArrowRecordBatchProtocol = batch
    if isinstance(raw_batch, ArrowRecordBatchProtocol) or (
        adapt_backend_module(batch).has_symbol("num_columns") and adapt_backend_module(batch).has_symbol("column")
    ):
        return _rb_to_tensor(batch, columns, dtype)

    arr = _pa_to_ndarray(batch, dtype=dtype)
    return Tensor(
        arr,
        TensorConfig(
            shape=coerce_shape(adapt_backend_module(arr).require_symbol("shape")),
            dtype=str(adapt_backend_module(arr).require_symbol("dtype")),
            device="cpu",
        ),
    )


def tensor_to_arrow_tensor(
    tensor: BackendArg,
) -> BackendArg:
    """Convert a Unified IR Tensor or array to a pyarrow.Tensor zero-copy.

    Args:
        tensor (BackendArg): Tensor or array.

    Returns:
        BackendArg: pyarrow.Tensor or underlying buffer fallback.
    """
    raw_data = _unwrap_tensor(tensor)
    arr = _pa_to_ndarray(raw_data)
    pa_mod = _get_pa_module()
    pa_tensor_cls = adapt_backend_module(pa_mod).get_symbol("Tensor", None)
    if adapt_backend_module(pa_tensor_cls).has_symbol("from_numpy"):
        try:
            return adapt_backend_module(pa_tensor_cls).require_op("from_numpy")(arr)
        except (ImportError, AttributeError, TypeError, ValueError, KeyError, RuntimeError, OSError):
            logging.getLogger(__name__).debug("Ignored exception in %s", __name__)
    return arr


def tensor(
    data: BackendArg,
) -> BackendArg:
    """Construct a pyarrow.Tensor from a Unified IR Tensor or sequence.

    Args:
        data (BackendArg): Tensor or sequence.

    Returns:
        BackendArg: Constructed pyarrow.Tensor or buffer fallback.
    """
    return tensor_to_arrow_tensor(data)


def arrow_tensor_to_tensor(
    arrow_tensor: BackendArg,
) -> BackendArg:
    """Convert a pyarrow.Tensor to a Unified IR Tensor zero-copy.

    Args:
        arrow_tensor (BackendArg): pyarrow.Tensor.

    Returns:
        BackendArg: Unified IR Tensor wrapping the zero-copy buffer view.
    """
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    raw_at: BackendArg | ArrowTensorProtocol = arrow_tensor
    arr: BackendArg
    if isinstance(raw_at, ArrowTensorProtocol) or adapt_backend_module(arrow_tensor).has_symbol("to_numpy"):
        arr = adapt_backend_module(arrow_tensor).require_op("to_numpy")()
    else:
        arr = _pa_to_ndarray(arrow_tensor)
    return Tensor(
        arr,
        TensorConfig(
            shape=coerce_shape(adapt_backend_module(arr).require_symbol("shape")),
            dtype=str(adapt_backend_module(arr).require_symbol("dtype")),
            device="cpu",
        ),
    )


def _table_to_tensor(actual_data: BackendArg) -> BackendArg:
    """Convert an Arrow Table to a Unified IR Tensor.

    Args:
        actual_data (BackendArg): Arrow Table instance.

    Returns:
        BackendArg: Resulting Unified IR Tensor.
    """
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    raw_data: BackendArg | ArrowTableProtocol = actual_data
    batches_raw: BackendArg = (
        adapt_backend_module(actual_data).require_op("to_batches")()
        if (isinstance(raw_data, ArrowTableProtocol) or adapt_backend_module(actual_data).has_symbol("to_batches"))
        else []
    )
    batches = list(batches_raw) if isinstance(batches_raw, (list, tuple)) else []
    if batches:
        if len(batches) == 1:
            return record_batch_to_tensor(batches[0])
        import pyarrow as pa

        sub_tensors = []
        for b in batches:
            rb_t = record_batch_to_tensor(b)
            sub_tensors.append(
                rb_t.data if isinstance(rb_t, Tensor) else adapt_backend_module(rb_t).get_symbol("data", rb_t)
            )
        combined = pa.concat_arrays([pa.array(s) for s in sub_tensors]).to_numpy(zero_copy_only=False)
        return Tensor(
            combined,
            TensorConfig(
                shape=coerce_shape(adapt_backend_module(combined).require_symbol("shape")),
                dtype=str(adapt_backend_module(combined).require_symbol("dtype")),
                device="cpu",
            ),
        )
    empty_arr = _pa_to_ndarray([], dtype="float32")
    return Tensor(empty_arr, TensorConfig(shape=(0,), dtype="float32", device="cpu"))


def _chunked_or_array_to_tensor(actual_data: BackendArg) -> BackendArg:
    """Convert an Arrow ChunkedArray or Array to a Unified IR Tensor.

    Args:
        actual_data (BackendArg): Arrow Array or ChunkedArray instance.

    Returns:
        BackendArg: Resulting Unified IR Tensor.
    """
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    arr: BackendArg
    try:
        arr = adapt_backend_module(actual_data).require_op("to_numpy")(zero_copy_only=False)
    except (
        AttributeError,
        ImportError,
        LookupError,
        OSError,
        ArithmeticError,
        RuntimeError,
        SyntaxError,
        TypeError,
        ValueError,
    ):
        arr = _pa_to_ndarray(adapt_backend_module(actual_data).require_op("to_pylist")())
    return Tensor(
        arr,
        TensorConfig(
            shape=coerce_shape(adapt_backend_module(arr).require_symbol("shape")),
            dtype=str(adapt_backend_module(arr).require_symbol("dtype")),
            device="cpu",
        ),
    )


def from_arrow(
    data: BackendArg,
) -> BackendArg:
    """Convert an arbitrary Arrow structure to a Unified IR Tensor.

    Args:
        data (BackendArg): Arrow object.

    Returns:
        BackendArg: Unified IR Tensor or Python scalar.
    """
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    # pyarrow.Tensor
    if (
        type(data).__name__ == "Tensor"
        and adapt_backend_module(data).has_symbol("to_numpy")
        and not adapt_backend_module(data).has_symbol("config")
    ):
        return arrow_tensor_to_tensor(data)

    # RecordBatch
    if (
        adapt_backend_module(data).has_symbol("num_columns")
        and adapt_backend_module(data).has_symbol("column")
        and adapt_backend_module(data).has_symbol("schema")
        and not adapt_backend_module(data).has_symbol("combine_chunks")
    ):
        return record_batch_to_tensor(data)

    # Table
    raw_table: BackendArg | ArrowTableProtocol = data
    if isinstance(raw_table, ArrowTableProtocol) or adapt_backend_module(data).has_symbol("to_batches"):
        return _table_to_tensor(data)

    # ChunkedArray / Array
    raw_arr: BackendArg | ArrowArrayProtocol = data
    if isinstance(raw_arr, ArrowArrayProtocol) or adapt_backend_module(data).has_symbol("to_numpy"):
        return _chunked_or_array_to_tensor(data)

    # Scalar
    raw_scalar: BackendArg | _ArrowScalarProtocol = data
    if isinstance(raw_scalar, _ArrowScalarProtocol) or adapt_backend_module(data).has_symbol("as_py"):
        return adapt_backend_module(data).require_op("as_py")()

    # Fallback
    arr = _pa_to_ndarray(data)
    return Tensor(
        arr,
        TensorConfig(
            shape=coerce_shape(adapt_backend_module(arr).require_symbol("shape")),
            dtype=str(adapt_backend_module(arr).require_symbol("dtype")),
            device="cpu",
        ),
    )


def to_arrow(
    data: BackendArg,
    target_type: str = "tensor",
) -> BackendArg:
    """Convert a Unified IR Tensor or array to the specified Arrow type.

    Args:
        data (BackendArg): Tensor or array.
        target_type (str): Target Arrow type ('tensor', 'record_batch', 'table', 'array', 'chunked_array').

    Returns:
        BackendArg: Converted Arrow object.

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


def is_arrow_object(obj: BackendArg) -> bool:
    """Determine whether an object is a PyArrow construct.

    Args:
        obj (BackendArg): Candidate object.

    Returns:
        bool: True if obj is an Arrow array, table, batch, tensor, scalar, or chunked array.
    """
    mod_name = str(type(obj).__module__)
    return mod_name.startswith("pyarrow")


def shares_buffer(obj1: BackendArg, obj2: BackendArg) -> bool:
    """Determine whether two objects share the underlying memory buffer.

    Args:
        obj1 (BackendArg): First tensor or array object.
        obj2 (BackendArg): Second tensor or array object.

    Returns:
        bool: True if both objects share memory.
    """
    data1 = _unwrap_tensor(obj1)
    data2 = _unwrap_tensor(obj2)

    buf1: BackendArg = adapt_backend_module(data1).get_symbol("buffer", None)
    buf2: BackendArg = adapt_backend_module(data2).get_symbol("buffer", None)
    if (
        buf1 is not None
        and buf2 is not None
        and adapt_backend_module(buf1).has_symbol("address")
        and adapt_backend_module(buf2).has_symbol("address")
    ):
        addr1: int = int(coerce_int(adapt_backend_module(buf1).get_symbol("address", 0)))
        addr2: int = int(coerce_int(adapt_backend_module(buf2).get_symbol("address", 0)))
        return bool(addr1 == addr2 and addr1 != 0)

    probe1: BackendArg | _ArrowArrayInterfaceProto = (
        adapt_backend_module(data1).require_op("to_numpy")()
        if adapt_backend_module(data1).has_symbol("to_numpy")
        else data1
    )
    probe2: BackendArg | _ArrowArrayInterfaceProto = (
        adapt_backend_module(data2).require_op("to_numpy")()
        if adapt_backend_module(data2).has_symbol("to_numpy")
        else data2
    )
    iface1 = probe1.__array_interface__ if isinstance(probe1, _ArrowArrayInterfaceProto) else None
    iface2 = probe2.__array_interface__ if isinstance(probe2, _ArrowArrayInterfaceProto) else None
    if isinstance(iface1, dict) and isinstance(iface2, dict):
        data_tup1 = iface1.get("data", (0,))
        data_tup2 = iface2.get("data", (0,))
        addr1 = int(coerce_int(data_tup1[0])) if isinstance(data_tup1, tuple) and data_tup1 else 0
        addr2 = int(coerce_int(data_tup2[0])) if isinstance(data_tup2, tuple) and data_tup2 else 0
        return bool(addr1 == addr2 and addr1 != 0)
    return False


def check_parity(
    lhs: BackendArg,
    rhs: BackendArg,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> bool:
    """Check numerical parity between two Arrow or sequence structures.

    Args:
        lhs (BackendArg): First operand.
        rhs (BackendArg): Second operand.
        rtol (float): Relative tolerance. Defaults to 1e-5.
        atol (float): Absolute tolerance. Defaults to 1e-8.

    Returns:
        bool: True if all elements match within tolerance.
    """
    arr1 = adapt_backend_module(_pa_to_ndarray(_unwrap_tensor(lhs))).require_op("reshape")(-1)
    arr2 = adapt_backend_module(_pa_to_ndarray(_unwrap_tensor(rhs))).require_op("reshape")(-1)
    list1 = coerce_operand_list(adapt_backend_module(arr1).require_op("tolist")())
    list2 = coerce_operand_list(adapt_backend_module(arr2).require_op("tolist")())
    if len(list1) != len(list2):
        return False
    return all(
        math.isclose(float(coerce_float(a)), float(coerce_float(b)), rel_tol=rtol, abs_tol=atol)
        for a, b in zip(list1, list2)
    )
