"""Serialization utilities."""

from __future__ import annotations

import io
import typing
from collections.abc import Mapping, Sequence
from types import ModuleType

from ml_switcheroo_ir import IRGraph

from ml_switcheroo_compiler.backends.eager_registry import ShapeProtocol
from ml_switcheroo_compiler.core.shape import OpValue, ShapeDim, is_variadic_callable

TensorLike = OpValue


def _extract_util_shape(seq: Sequence[ShapeDim]) -> tuple[int, ...]:
    """Convert a sequence of dimensions into a concrete integer shape tuple.

    Args:
        seq (Sequence[ShapeDim]): Shape sequence.

    Returns:
        tuple[int, ...]: Concrete integer shape tuple.
    """
    return tuple(int(d) for d in seq if isinstance(d, int))


def _ensure_util_bytes(val: OpValue | bytes | bytearray | memoryview) -> bytes:
    """Convert a bytes-like value to `bytes`.

    Args:
        val (OpValue | bytes | bytearray | memoryview): Bytes-like value.

    Returns:
        bytes: Raw bytes.
    """
    assert isinstance(val, bytes)
    return val


@typing.runtime_checkable
class _NumpyMethodProtocol(typing.Protocol):
    """Protocol for tensors supporting `.numpy()`."""

    def numpy(self) -> TensorLike:
        """Convert to NumPy array.

        Returns:
            TensorLike: NumPy array.
        """
        ...


@typing.runtime_checkable
class _DataFieldProtocol(typing.Protocol):
    """Protocol for tensor wrappers exposing `.data`."""

    data: OpValue


@typing.runtime_checkable
class _ToListMethodProtocol(typing.Protocol):
    """Protocol for tensors supporting `.tolist()`."""

    def tolist(self) -> OpValue:
        """Convert to nested Python list.

        Returns:
            OpValue: Nested Python list.
        """
        ...


@typing.runtime_checkable
class _EvalMethodProtocol(typing.Protocol):
    """Protocol for lazy tensors supporting `.eval()`."""

    def eval(self) -> TensorLike:
        """Evaluate lazy tensor.

        Returns:
            TensorLike: Evaluated array.
        """
        ...


@typing.runtime_checkable
class _ToBytesMethodProtocol(typing.Protocol):
    """Protocol for arrays supporting `.tobytes()`."""

    def tobytes(self) -> bytes:
        """Serialize array buffer to bytes.

        Returns:
            bytes: Raw buffer bytes.
        """
        ...


@typing.runtime_checkable
class _ArrayInterfaceProtocol(typing.Protocol):
    """Protocol for objects implementing `__array__`."""

    def __array__(self) -> OpValue:
        """Return NumPy array representation.

        Returns:
            OpValue: NumPy array.
        """
        ...


@typing.runtime_checkable
class _DTypeNameProtocol(typing.Protocol):
    """Protocol for dtype objects exposing `.name`."""

    name: str


@typing.runtime_checkable
class _DTypeAttrProtocol(typing.Protocol):
    """Protocol for objects exposing `.dtype`."""

    dtype: OpValue


@typing.runtime_checkable
class _GetNpzBytesBackendProtocol(typing.Protocol):
    """Protocol for backends implementing `get_npz_bytes`."""

    def get_npz_bytes(self, weights: OpValue) -> bytes:
        """Serialize weights mapping to `.npz` archive bytes.

        Args:
            weights (OpValue): Weights dictionary.

        Returns:
            bytes: Serialized `.npz` bytes.
        """
        ...


@typing.runtime_checkable
class _LoadNpzListBackendProtocol(typing.Protocol):
    """Protocol for backends implementing `load_npz`."""

    def load_npz(self, file_obj: str | bytes | TensorLike) -> Sequence[TensorLike]:
        """Load weights sequence from `.npz` archive.

        Args:
            file_obj (str | bytes | TensorLike): File path or bytes.

        Returns:
            Sequence[TensorLike]: Loaded weight arrays.
        """
        ...


@typing.runtime_checkable
class _NpzFilesProtocol(typing.Protocol):
    """Protocol for loaded NumPy `.npz` archives exposing `.files`."""

    files: Sequence[str]

    def __getitem__(self, key: str) -> TensorLike:
        """Retrieve array for `key`.

        Args:
            key (str): Archive member name.

        Returns:
            TensorLike: Loaded array.
        """
        ...


@typing.runtime_checkable
class _NpzLoadContextProtocol(typing.Protocol):
    """Protocol for context managers returned by `module.load(target)`."""

    def __enter__(self) -> _NpzFilesProtocol | int | None:
        """Enter archive context.

        Returns:
            _NpzFilesProtocol | int | None: Loaded archive object.
        """
        ...

    def __exit__(self, exc_type: OpValue, exc_val: OpValue, exc_tb: OpValue) -> OpValue:
        """Exit archive context.

        Args:
            exc_type (OpValue): Exception type.
            exc_val (OpValue): Exception instance.
            exc_tb (OpValue): Traceback.

        Returns:
            OpValue: Suppression indicator.
        """
        ...


@typing.runtime_checkable
class _LoadableModuleProtocol(typing.Protocol):
    """Protocol for backend modules exposing `.load(target)`."""

    def load(self, target: io.BytesIO | str | bytes | TensorLike) -> _NpzLoadContextProtocol:
        """Load archive from `target`.

        Args:
            target (io.BytesIO | str | bytes | TensorLike): Path or BytesIO stream.

        Returns:
            _NpzLoadContextProtocol: Context manager yielding archive mapping.
        """
        ...


@typing.runtime_checkable
class _GetModuleBackendProtocol(typing.Protocol):
    """Protocol for backends exposing `get_module()`."""

    def get_module(self) -> _LoadableModuleProtocol | int | None:
        """Return underlying backend module.

        Returns:
            _LoadableModuleProtocol | int | None: Backend module.
        """
        ...


def _extract_numpy_weights(weights: typing.Mapping[str, TensorLike]) -> dict[str, TensorLike]:
    """Extract numpy weights.

    Args:
        weights (typing.Mapping[str, TensorLike]): Input `weights` argument (weights).

    Returns:
        dict[str, TensorLike]: Extracted weights dictionary.
    """
    return {k: to_numpy(w) for k, w in weights.items()}


def to_numpy(tensor: TensorLike) -> TensorLike:
    """Convert tensor to numpy array.

    Args:
        tensor (TensorLike): Input `tensor` argument (tensor).

    Returns:
        TensorLike: Converted array.
    """
    probe: TensorLike | _NumpyMethodProtocol | _DataFieldProtocol | _ToListMethodProtocol = tensor
    if type(probe) is not ModuleType and isinstance(probe, _NumpyMethodProtocol):
        return probe.numpy()
    if isinstance(probe, _DataFieldProtocol):
        inner: TensorLike | _NumpyMethodProtocol = probe.data
        if type(inner) is not ModuleType and isinstance(inner, _NumpyMethodProtocol):
            return inner.numpy()

    try:
        if type(probe) is not ModuleType and isinstance(probe, _ToListMethodProtocol):
            return probe.tolist()
    except (ValueError, TypeError):
        pass
    return tensor


def concatenate_arrays(arrays: list[TensorLike]) -> TensorLike:
    """Concatenate numpy arrays.

    Args:
        arrays (list[TensorLike]): Input `arrays` argument (arrays).

    Returns:
        TensorLike: Concatenated array.
    """
    from ml_switcheroo_compiler import ops

    _callable_fn_1: OpValue = ops.concatenate
    assert is_variadic_callable(_callable_fn_1)
    return _callable_fn_1(arrays, axis=0)


def is_numpy_array(array: OpValue) -> bool:
    """Check if array is a numpy array.

    Args:
        array (OpValue): Input `array` argument (array).

    Returns:
        bool: True if array is a numpy array or array-like.
    """
    cand_arr: OpValue | _NumpyMethodProtocol | _ArrayInterfaceProtocol = array
    return (
        type(cand_arr).__module__ == "numpy"
        or isinstance(cand_arr, _NumpyMethodProtocol)
        or isinstance(cand_arr, _ArrayInterfaceProtocol)
    )


def _dtype_to_descr(dtype: OpValue) -> str:
    """Map DType to numpy descr string.

    Args:
        dtype (OpValue): Input `dtype` argument (dtype).

    Returns:
        str: NumPy descriptor string.
    """
    from ml_switcheroo_compiler.core.dtype import DType

    mapping: dict[DType, str] = {
        DType.Float32: "<f4",
        DType.Float64: "<f8",
        DType.Float16: "<f2",
        DType.Int64: "<i8",
        DType.Int32: "<i4",
        DType.Int16: "<i2",
        DType.Int8: "|i1",
        DType.UInt64: "<u8",
        DType.UInt32: "<u4",
        DType.UInt16: "<u2",
        DType.UInt8: "|u1",
        DType.Bool: "|b1",
    }
    if isinstance(dtype, str):
        return next((v for k, v in mapping.items() if k.value == dtype), "<f4")
    return mapping.get(dtype, "<f4") if isinstance(dtype, DType) else "<f4"


def _is_shape_obj(val: OpValue) -> typing.TypeGuard[ShapeProtocol]:
    """Check whether `val` exposes a `shape` attribute.

    Args:
        val (OpValue): Candidate array object.

    Returns:
        bool: True if `val` has `shape`.
    """
    return isinstance(val, ShapeProtocol)


def _extract_arr_shape_dtype(arr: TensorLike) -> tuple[tuple[int, ...], str]:
    """Extract shape and dtype string from an array object.

    Args:
        arr (TensorLike): The array object.

    Returns:
        tuple[tuple[int, ...], str]: (shape, dtype_str).
    """
    raw_arr: OpValue = arr
    raw_dt: OpValue = raw_arr.dtype if isinstance(raw_arr, _DTypeAttrProtocol) else None
    if isinstance(raw_dt, _DTypeNameProtocol):
        dtype_str = str(raw_dt.name)
    else:
        dtype_str = str(raw_dt if raw_dt is not None else "<f4")
    shape_tup = _extract_util_shape(raw_arr.shape) if _is_shape_obj(raw_arr) else ()
    return shape_tup, dtype_str


def _get_shape_and_dtype(
    tensor: OpValue | _EvalMethodProtocol | _NumpyMethodProtocol | _DataFieldProtocol,
) -> tuple[tuple[int, ...], str]:
    """Get shape and dtype from a generic tensor object.

    Args:
        tensor (OpValue | _EvalMethodProtocol | _NumpyMethodProtocol | _DataFieldProtocol): The tensor object.

    Returns:
        tuple[tuple[int, ...], str]: (shape, dtype).
    """
    if type(tensor) is not ModuleType and isinstance(tensor, _EvalMethodProtocol):
        return _extract_arr_shape_dtype(tensor.eval())
    if type(tensor) is not ModuleType and isinstance(tensor, _NumpyMethodProtocol):
        return _extract_arr_shape_dtype(tensor.numpy())
    if isinstance(tensor, _DataFieldProtocol):
        inner: OpValue | _NumpyMethodProtocol = tensor.data
        if type(inner) is not ModuleType and isinstance(inner, _NumpyMethodProtocol):
            return _extract_arr_shape_dtype(inner.numpy())
    shape_tup = _extract_util_shape(tensor.shape) if isinstance(tensor, ShapeProtocol) else ()
    dtype_val = str(tensor.dtype) if isinstance(tensor, _DTypeAttrProtocol) else "<f4"
    return shape_tup, dtype_val


def _extract_arr_bytes(arr: OpValue | _ToBytesMethodProtocol | _DataFieldProtocol) -> bytes:
    """Extract raw bytes from an array object.

    Args:
        arr (OpValue | _ToBytesMethodProtocol | _DataFieldProtocol): The array object.

    Returns:
        bytes: The raw data bytes.
    """
    if type(arr) is not ModuleType and isinstance(arr, _ToBytesMethodProtocol):
        return _ensure_util_bytes(arr.tobytes())
    if isinstance(arr, _DataFieldProtocol):
        inner: OpValue | _ToBytesMethodProtocol = arr.data
        if type(inner) is not ModuleType and isinstance(inner, _ToBytesMethodProtocol):
            return _ensure_util_bytes(inner.tobytes())
    return b""


def _get_data_bytes(
    tensor: OpValue | _EvalMethodProtocol | _NumpyMethodProtocol | _DataFieldProtocol | _ToBytesMethodProtocol,
) -> bytes:
    """Get raw data bytes from a generic tensor object.

    Args:
        tensor: The tensor object to extract bytes from.

    Returns:
        bytes: The raw data bytes.
    """
    if type(tensor) is not ModuleType and isinstance(tensor, _EvalMethodProtocol):
        return _extract_arr_bytes(tensor.eval())
    if type(tensor) is not ModuleType and isinstance(tensor, _NumpyMethodProtocol):
        return _extract_arr_bytes(tensor.numpy())
    if isinstance(tensor, _DataFieldProtocol):
        inner: OpValue | _NumpyMethodProtocol = tensor.data
        if type(inner) is not ModuleType and isinstance(inner, _NumpyMethodProtocol):
            return _extract_arr_bytes(inner.numpy())
    if type(tensor) is not ModuleType and isinstance(tensor, _ToBytesMethodProtocol):
        return _ensure_util_bytes(tensor.tobytes())
    return b""


def _tensor_to_npy_bytes(tensor: OpValue) -> bytes:
    """Convert a tensor to npy bytes.

    Args:
        tensor (OpValue): Input `tensor` argument (tensor).

    Returns:
        bytes: Serialized `.npy` byte stream.
    """
    import struct

    shape, dtype = _get_shape_and_dtype(tensor)
    data_bytes = _get_data_bytes(tensor)
    dtype_str = _dtype_to_descr(dtype)

    magic = b"\x93NUMPY"
    major, minor = 1, 0
    shape_str = f"({shape[0]},)" if len(shape) == 1 else str(shape)

    dict_str = "{'descr': '" + dtype_str + "', 'fortran_order': False, 'shape': " + shape_str + ", }"
    dict_str += " " * (16 - (len(dict_str) + 10) % 16) + "\n"

    header = magic + struct.pack("<BBH", major, minor, len(dict_str)) + dict_str.encode("ascii")
    return header + data_bytes


def _is_get_npz_bytes_backend(
    val: OpValue | _GetNpzBytesBackendProtocol,
) -> typing.TypeGuard[_GetNpzBytesBackendProtocol]:
    """Check whether `val` exposes `get_npz_bytes`.

    Args:
        val (OpValue | _GetNpzBytesBackendProtocol): Active backend.

    Returns:
        bool: True if `val` has `get_npz_bytes`.
    """
    from ml_switcheroo_compiler.core._mock_protocol import is_mock_with_attrs

    found: bool = isinstance(val, _GetNpzBytesBackendProtocol)
    return found or is_mock_with_attrs(val, "get_npz_bytes")


def get_npz_bytes(weights: OpValue) -> bytes:
    """Get npz bytes.

    Args:
        weights (OpValue): Input `weights` argument (weights).

    Returns:
        bytes: Serialized `.npz` archive bytes.
    """
    import io
    import zipfile

    from ml_switcheroo_compiler.backends.registry import get_active_backend

    backend: OpValue | _GetNpzBytesBackendProtocol = get_active_backend()
    if _is_get_npz_bytes_backend(backend):
        return _ensure_util_bytes(backend.get_npz_bytes(weights))

    assert isinstance(weights, Mapping)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_STORED) as zf:
        for k, v in weights.items():
            k_str = str(k)
            name = k_str if k_str.endswith(".npy") else f"{k_str}.npy"
            zf.writestr(name, _tensor_to_npy_bytes(v))
        zf_out = buf
    return zf_out.getvalue()


def _is_npz_files(val: OpValue | _NpzFilesProtocol) -> typing.TypeGuard[_NpzFilesProtocol]:
    """Check whether `val` exposes a `files` attribute.

    Args:
        val (OpValue | _NpzFilesProtocol): Loaded npz archive.

    Returns:
        bool: True if `val` has `files`.
    """
    from ml_switcheroo_compiler.core._mock_protocol import is_mock_with_attrs

    found: bool = isinstance(val, _NpzFilesProtocol)
    return found or is_mock_with_attrs(val, "files")


def _decode_npy_payload(raw_npy: bytes) -> TensorLike:
    """Decode a `.npy` v1.0 byte payload into a flat or nested float/int list using pure Python.

    Args:
        raw_npy (bytes): Raw `.npy` file bytes starting with the NUMPY magic header.

    Returns:
        TensorLike: Decoded numeric scalar or list.
    """
    import ast
    import struct

    hdr_len = struct.unpack_from("<H", raw_npy, 8)[0]
    hdr_dict = ast.literal_eval(raw_npy[10 : 10 + hdr_len].decode("latin1").strip())
    descr = str(hdr_dict.get("descr", "<f4"))
    payload = raw_npy[10 + hdr_len :]
    fmt_map = {"<f8": ("d", 8), "<f4": ("f", 4), "<i8": ("q", 8), "<i4": ("i", 4), "|b1": ("?", 1)}
    code, item_sz = fmt_map.get(descr, ("f", 4))
    count = len(payload) // item_sz if item_sz > 0 else 0
    return list(struct.unpack(f"<{count}{code}", payload[: count * item_sz])) if count > 0 else []


def _is_get_module_backend(val: OpValue | _GetModuleBackendProtocol) -> typing.TypeGuard[_GetModuleBackendProtocol]:
    """Check whether `val` exposes `get_module`.

    Args:
        val (OpValue | _GetModuleBackendProtocol): Active backend.

    Returns:
        bool: True if `val` has `get_module`.
    """
    from ml_switcheroo_compiler.core._mock_protocol import is_mock_with_attrs

    found: bool = isinstance(val, _GetModuleBackendProtocol)
    return found or is_mock_with_attrs(val, "get_module")


def _is_loadable_mod(val: OpValue | _LoadableModuleProtocol) -> typing.TypeGuard[_LoadableModuleProtocol]:
    """Check whether `val` exposes `load`.

    Args:
        val (OpValue | _LoadableModuleProtocol): Backend module.

    Returns:
        bool: True if `val` has `load`.
    """
    from ml_switcheroo_compiler.core._mock_protocol import is_mock_with_attrs

    found: bool = isinstance(val, _LoadableModuleProtocol)
    return found or is_mock_with_attrs(val, "load")


def parse_npz(file_obj: str | bytes | io.IOBase | TensorLike) -> dict[str, TensorLike]:
    """Parse npz file.

    Args:
        file_obj (str | bytes | io.IOBase | TensorLike): Input `file_obj` argument (file obj).

    Returns:
        dict[str, TensorLike]: Computed `dict[str, TensorLike]` output.
    """
    import io
    import os
    import zipfile

    from ml_switcheroo_compiler.backends.registry import get_active_backend

    try:
        active_be: OpValue | _GetModuleBackendProtocol = get_active_backend()
        if isinstance(file_obj, (bytes, bytearray)):
            target: io.BytesIO | str | bytes | TensorLike = io.BytesIO(file_obj)
        elif isinstance(file_obj, io.IOBase):
            raw_bytes = file_obj.read()
            target = io.BytesIO(raw_bytes if isinstance(raw_bytes, (bytes, bytearray)) else b"")
        else:
            target = file_obj
        if _is_get_module_backend(active_be):
            active_mod = active_be.get_module()
            assert _is_loadable_mod(active_mod)
            with active_mod.load(target) as data:
                assert _is_npz_files(data)
                return {k: data[k] for k in data.files}
        assert isinstance(target, (str, os.PathLike, io.BytesIO))
        with zipfile.ZipFile(target, "r") as zf:
            return {
                name[:-4] if name.endswith(".npy") else name: _decode_npy_payload(zf.read(name))
                for name in zf.namelist()
            }
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
        return {}


def _is_load_npz_backend(val: OpValue) -> typing.TypeGuard[_LoadNpzListBackendProtocol]:
    """Check whether `val` exposes a `load_npz` method.

    Args:
        val (OpValue): Active backend.

    Returns:
        bool: True if `val` has `load_npz`.
    """
    from ml_switcheroo_compiler.core._mock_protocol import is_mock_with_attrs

    found: bool = isinstance(val, _LoadNpzListBackendProtocol)
    return found or is_mock_with_attrs(val, "load_npz")


def load_npz(file_obj: str | bytes | TensorLike) -> list[TensorLike]:
    """Load weights from a .npz file object.

    Args:
        file_obj (str | bytes | TensorLike): Input `file_obj` argument (file obj).

    Returns:
        list[TensorLike]: Computed `list[TensorLike]` output.
    """
    from ml_switcheroo_compiler.backends.registry import get_active_backend

    backend: OpValue = get_active_backend()
    if _is_load_npz_backend(backend):
        try:
            return list(backend.load_npz(file_obj))
        except (
            EOFError,
            ImportError,
            ModuleNotFoundError,
            OSError,
            AttributeError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as e:
            import logging

            logging.getLogger(__name__).warning(f"Backend load_npz failed: {e}. Falling back.")

    parsed = parse_npz(file_obj)
    return list(parsed.values())


def save_ir_graph(graph: IRGraph, filepath: str) -> None:
    """Save an IR graph configuration to a file reliably.

    Args:
        graph: The IRGraph to serialize.
        filepath: The path to write out the configuration.
    """
    from ml_switcheroo_compiler.serialization.ir_format import graph_to_json

    with open(filepath, "w") as f:
        f.write(graph_to_json(graph))


def load_ir_graph(filepath: str) -> IRGraph:
    """Load an IR graph configuration from a file reliably.

    Args:
        filepath: The path to read the configuration from.

    Returns:
        The deserialized IRGraph.
    """
    from ml_switcheroo_compiler.serialization.ir_format import json_to_graph

    with open(filepath) as f:
        return json_to_graph(f.read())
