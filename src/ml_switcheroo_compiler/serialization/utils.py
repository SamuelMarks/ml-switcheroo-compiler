"""Serialization utilities."""

from __future__ import annotations

import typing
from collections.abc import Mapping, Sequence

from ml_switcheroo_compiler.backends.eager_registry import (
    ShapeProtocol,
    ensure_callable,
)
from ml_switcheroo_compiler.backends.protocols import get_backend_member, has_backend_member
from ml_switcheroo_compiler.core.shape import TensorOrScalar
from ml_switcheroo_compiler.interpreter.evaluator import TensorLike
from ml_switcheroo_compiler.ir.core import IRGraph


def _is_util_tensor(val: TensorOrScalar) -> typing.TypeGuard[TensorLike]:
    """Type guard narrowing `val` to `TensorLike`.

    Args:
        val (TensorOrScalar): Candidate array or list value.

    Returns:
        bool: True if non-None.
    """
    return val is not None


def _ensure_util_tensor(val: TensorOrScalar) -> TensorLike:
    """Narrow `val` to `TensorLike`.

    Args:
        val (TensorOrScalar): Candidate array or list value.

    Returns:
        TensorLike: Narrowed tensor.
    """
    assert _is_util_tensor(val)
    return val


def _extract_util_shape(seq: Sequence[int | str | None]) -> tuple[int, ...]:
    """Convert a sequence of dimensions into a concrete integer shape tuple.

    Args:
        seq (Sequence[int | str | None]): Shape sequence.

    Returns:
        tuple[int, ...]: Concrete integer shape tuple.
    """
    return tuple(int(d) for d in seq if isinstance(d, int))


def _ensure_util_bytes(val: TensorOrScalar | bytes | bytearray | memoryview) -> bytes:
    """Convert a bytes-like value to `bytes`.

    Args:
        val (TensorOrScalar | bytes | bytearray | memoryview): Bytes-like value.

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

    data: TensorOrScalar | _NumpyMethodProtocol | _ToBytesMethodProtocol


@typing.runtime_checkable
class _ConcatCallableProtocol(typing.Protocol):
    """Protocol for `ops.concatenate` accepting a sequence of arrays."""

    def __call__(self, tensors: Sequence[TensorLike], axis: int = 0) -> TensorOrScalar:
        """Concatenate arrays along `axis`.

        Args:
            tensors (Sequence[TensorLike]): Sequence of arrays.
            axis (int): Concatenation axis.

        Returns:
            TensorOrScalar: Concatenated array.
        """
        ...


def _is_concat_callable(val: TensorOrScalar) -> typing.TypeGuard[_ConcatCallableProtocol]:
    """Type guard narrowing `val` to `_ConcatCallableProtocol`.

    Args:
        val (TensorOrScalar): Concatenate callable.

    Returns:
        bool: True if callable.
    """
    return callable(val)


@typing.runtime_checkable
class _ToListMethodProtocol(typing.Protocol):
    """Protocol for tensors supporting `.tolist()`."""

    def tolist(self) -> TensorOrScalar:
        """Convert to nested Python list.

        Returns:
            TensorOrScalar: Nested Python list.
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

    def __array__(self) -> TensorOrScalar:
        """Return NumPy array representation.

        Returns:
            TensorOrScalar: NumPy array.
        """
        ...


@typing.runtime_checkable
class _DTypeNameProtocol(typing.Protocol):
    """Protocol for dtype objects exposing `.name`."""

    name: str


@typing.runtime_checkable
class _DTypeAttrProtocol(typing.Protocol):
    """Protocol for objects exposing `.dtype`."""

    dtype: TensorOrScalar


@typing.runtime_checkable
class _GetNpzBytesBackendProtocol(typing.Protocol):
    """Protocol for backends implementing `get_npz_bytes`."""

    def get_npz_bytes(self, weights: TensorOrScalar) -> bytes:
        """Serialize weights mapping to `.npz` archive bytes.

        Args:
            weights (TensorOrScalar): Weights dictionary.

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


def _extract_numpy_weights(weights: dict[str, TensorLike]) -> dict[str, TensorLike]:
    """Extract numpy weights.

    Args:
        weights (dict[str, TensorLike]): Input `weights` argument (weights).

    Returns:
        dict[str, TensorLike]: Extracted weights dictionary.
    """
    weights_np: dict[str, TensorLike] = {}
    for k, w in weights.items():
        if has_backend_member(w, "numpy"):
            weights_np[k] = _ensure_util_tensor(ensure_callable(get_backend_member(w, "numpy"))())
            continue

        if has_backend_member(w, "data") and has_backend_member(get_backend_member(w, "data"), "numpy"):
            weights_np[k] = _ensure_util_tensor(ensure_callable(get_backend_member(get_backend_member(w, "data"), "numpy"))())
            continue

        try:
            weights_np[k] = _ensure_util_tensor(ensure_callable(get_backend_member(w, "tolist"))()) if has_backend_member(w, "tolist") else w
        except (ValueError, TypeError):
            weights_np[k] = w
    return weights_np


def to_numpy(tensor: TensorLike) -> TensorLike:
    """Convert tensor to numpy array.

    Args:
        tensor (TensorLike): Input `tensor` argument (tensor).

    Returns:
        TensorLike: Converted array.
    """
    if has_backend_member(tensor, "numpy"):
        return _ensure_util_tensor(ensure_callable(get_backend_member(tensor, "numpy"))())
    if has_backend_member(tensor, "data") and has_backend_member(get_backend_member(tensor, "data"), "numpy"):
        return _ensure_util_tensor(ensure_callable(get_backend_member(get_backend_member(tensor, "data"), "numpy"))())

    try:
        return _ensure_util_tensor(ensure_callable(get_backend_member(tensor, "tolist"))()) if has_backend_member(tensor, "tolist") else tensor
    except (ValueError, TypeError):
        return tensor


def concatenate_arrays(arrays: list[TensorLike]) -> TensorLike:
    """Concatenate numpy arrays.

    Args:
        arrays (list[TensorLike]): Input `arrays` argument (arrays).

    Returns:
        TensorLike: Concatenated array.
    """
    from ml_switcheroo_compiler import ops

    concat_fn: TensorOrScalar = ops.concatenate
    assert _is_concat_callable(concat_fn)
    return _ensure_util_tensor(concat_fn(arrays, axis=0))


def is_numpy_array(array: TensorOrScalar) -> bool:
    """Check if array is a numpy array.

    Args:
        array (TensorOrScalar): Input `array` argument (array).

    Returns:
        bool: True if array is a numpy array or array-like.
    """
    return type(array).__module__ == "numpy" or has_backend_member(array, "numpy") or isinstance(array, _ArrayInterfaceProtocol)


def _dtype_to_descr(dtype: TensorOrScalar) -> str:
    """Map DType to numpy descr string.

    Args:
        dtype (TensorOrScalar): Input `dtype` argument (dtype).

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


def _is_shape_obj(val: TensorOrScalar) -> typing.TypeGuard[ShapeProtocol]:
    """Check whether `val` exposes a `shape` attribute.

    Args:
        val (TensorOrScalar): Candidate array object.

    Returns:
        bool: True if `val` has `shape`.
    """
    return has_backend_member(val, "shape")


def _extract_arr_shape_dtype(arr: TensorLike) -> tuple[tuple[int, ...], str]:
    """Extract shape and dtype string from an array object.

    Args:
        arr (TensorLike): The array object.

    Returns:
        tuple[tuple[int, ...], str]: (shape, dtype_str).
    """
    raw_arr: TensorOrScalar = arr
    raw_dt: TensorOrScalar = get_backend_member(raw_arr, "dtype") if has_backend_member(raw_arr, "dtype") else None
    if has_backend_member(raw_dt, "name"):
        dtype_str = str(get_backend_member(raw_dt, "name"))
    else:
        dtype_str = str(raw_dt if raw_dt is not None else "<f4")
    shape_tup = _extract_util_shape(raw_arr.shape) if _is_shape_obj(raw_arr) else ()
    return shape_tup, dtype_str


def _get_shape_and_dtype(tensor: TensorOrScalar) -> tuple[tuple[int, ...], str]:
    """Get shape and dtype from a generic tensor object.

    Args:
        tensor (TensorOrScalar): The tensor object.

    Returns:
        tuple[tuple[int, ...], str]: (shape, dtype).
    """
    if has_backend_member(tensor, "eval"):
        return _extract_arr_shape_dtype(ensure_callable(get_backend_member(tensor, "eval"))())
    if has_backend_member(tensor, "numpy"):
        return _extract_arr_shape_dtype(_ensure_util_tensor(ensure_callable(get_backend_member(tensor, "numpy"))()))
    if has_backend_member(tensor, "data") and has_backend_member(get_backend_member(tensor, "data"), "numpy"):
        return _extract_arr_shape_dtype(_ensure_util_tensor(ensure_callable(get_backend_member(get_backend_member(tensor, "data"), "numpy"))()))
    shape_tup = _extract_util_shape(tensor.shape) if _is_shape_obj(tensor) else ()
    dtype_val = str(get_backend_member(tensor, "dtype")) if has_backend_member(tensor, "dtype") else "<f4"
    return shape_tup, dtype_val


def _extract_arr_bytes(arr: TensorOrScalar | _ToBytesMethodProtocol | _DataFieldProtocol) -> bytes:
    """Extract raw bytes from an array object.

    Args:
        arr (TensorOrScalar | _ToBytesMethodProtocol | _DataFieldProtocol): The array object.

    Returns:
        bytes: The raw data bytes.
    """
    if has_backend_member(arr, "tobytes"):
        return _ensure_util_bytes(ensure_callable(get_backend_member(arr, "tobytes"))())
    if has_backend_member(arr, "data") and has_backend_member(get_backend_member(arr, "data"), "tobytes"):
        return _ensure_util_bytes(ensure_callable(get_backend_member(get_backend_member(arr, "data"), "tobytes"))())
    return b""


def _get_data_bytes(tensor: TensorOrScalar | _EvalMethodProtocol | _NumpyMethodProtocol | _DataFieldProtocol) -> bytes:
    """Get raw data bytes from a generic tensor object.

    Args:
        tensor (TensorOrScalar | _EvalMethodProtocol | _NumpyMethodProtocol | _DataFieldProtocol): The tensor object.

    Returns:
        bytes: The raw data bytes.
    """
    if has_backend_member(tensor, "eval"):
        return _extract_arr_bytes(ensure_callable(get_backend_member(tensor, "eval"))())
    if has_backend_member(tensor, "numpy"):
        return _extract_arr_bytes(ensure_callable(get_backend_member(tensor, "numpy"))())
    if has_backend_member(tensor, "data") and has_backend_member(get_backend_member(tensor, "data"), "numpy"):
        return _extract_arr_bytes(ensure_callable(get_backend_member(get_backend_member(tensor, "data"), "numpy"))())
    if has_backend_member(tensor, "tobytes"):
        return _ensure_util_bytes(ensure_callable(get_backend_member(tensor, "tobytes"))())
    return b""


def _tensor_to_npy_bytes(tensor: TensorOrScalar) -> bytes:
    """Convert a tensor to npy bytes.

    Args:
        tensor (TensorOrScalar): Input `tensor` argument (tensor).

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


def get_npz_bytes(weights: TensorOrScalar) -> bytes:
    """Get npz bytes.

    Args:
        weights (TensorOrScalar): Input `weights` argument (weights).

    Returns:
        bytes: Serialized `.npz` archive bytes.
    """
    import io
    import zipfile

    from ml_switcheroo_compiler.backends.registry import get_active_backend

    backend: TensorOrScalar = get_active_backend()
    if has_backend_member(backend, "get_npz_bytes"):
        return _ensure_util_bytes(ensure_callable(get_backend_member(backend, "get_npz_bytes"))(weights))

    assert isinstance(weights, Mapping)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_STORED) as zf:
        for k, v in weights.items():
            k_str = str(k)
            name = k_str if k_str.endswith(".npy") else f"{k_str}.npy"
            zf.writestr(name, _tensor_to_npy_bytes(v))
        zf_out = buf
    return zf_out.getvalue()


def _is_npz_files(val: TensorOrScalar) -> typing.TypeGuard[_NpzFilesProtocol]:
    """Check whether `val` exposes a `files` attribute.

    Args:
        val (TensorOrScalar): Loaded npz archive.

    Returns:
        bool: True if `val` has `files`.
    """
    return has_backend_member(val, "files")


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
    vals = list(struct.unpack(f"<{count}{code}", payload[: count * item_sz])) if count > 0 else []
    return _ensure_util_tensor(vals)


def parse_npz(file_obj: str | bytes | TensorLike) -> dict[str, TensorLike]:
    """Parse npz file.

    Args:
        file_obj (str | bytes | TensorLike): Input `file_obj` argument (file obj).

    Returns:
        dict[str, TensorLike]: Computed `dict[str, TensorLike]` output.
    """
    import io
    import os
    import zipfile

    from ml_switcheroo_compiler.backends.registry import get_active_backend

    try:
        active_be: TensorOrScalar = get_active_backend()
        target = io.BytesIO(file_obj) if isinstance(file_obj, (bytes, bytearray)) else file_obj
        if has_backend_member(active_be, "get_module"):
            active_mod = ensure_callable(get_backend_member(active_be, "get_module"))()
            with ensure_callable(get_backend_member(active_mod, "load"))(target) as data:
                raw_data: TensorOrScalar = data
                files_list = raw_data.files if _is_npz_files(raw_data) else ()
                return {k: _ensure_util_tensor(data)[k] for k in files_list}
        assert isinstance(target, (str, os.PathLike, io.BytesIO))
        with zipfile.ZipFile(target, "r") as zf:
            return {name[:-4] if name.endswith(".npy") else name: _decode_npy_payload(zf.read(name)) for name in zf.namelist()}
    except Exception:
        return {}


def _is_load_npz_backend(val: TensorOrScalar) -> typing.TypeGuard[_LoadNpzListBackendProtocol]:
    """Check whether `val` exposes a `load_npz` method.

    Args:
        val (TensorOrScalar): Active backend.

    Returns:
        bool: True if `val` has `load_npz`.
    """
    return has_backend_member(val, "load_npz")


def load_npz(file_obj: str | bytes | TensorLike) -> list[TensorLike]:
    """Load weights from a .npz file object.

    Args:
        file_obj (str | bytes | TensorLike): Input `file_obj` argument (file obj).

    Returns:
        list[TensorLike]: Computed `list[TensorLike]` output.
    """
    from ml_switcheroo_compiler.backends.registry import get_active_backend

    backend: TensorOrScalar = get_active_backend()
    if _is_load_npz_backend(backend):
        try:
            return list(backend.load_npz(file_obj))
        except Exception as e:
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
