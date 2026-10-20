"""Serialization utilities."""

from __future__ import annotations

import typing

from ml_switcheroo_compiler.core.shape import TensorOrScalar, TypedVariadicCallable
from ml_switcheroo_compiler.interpreter.evaluator import TensorLike
from ml_switcheroo_compiler.ir.core import IRGraph


def _extract_numpy_weights(weights: dict[str, TensorLike]) -> dict[str, TensorLike]:
    """Extract numpy weights.

    Args:
        weights (dict[str, TensorLike]): The weights parameter.

    Returns:
        dict[str, TensorLike]: Extracted weights dictionary.
    """
    weights_np = {}
    for k, w in weights.items():
        if hasattr(w, "numpy"):
            weights_np[k] = w.numpy()
            continue

        if hasattr(w, "data") and hasattr(w.data, "numpy"):
            weights_np[k] = w.data.numpy()
            continue

        try:
            weights_np[k] = w.tolist() if hasattr(w, "tolist") else w
        except (ValueError, TypeError):
            weights_np[k] = w
    return weights_np


def to_numpy(tensor: TensorLike) -> TensorLike:
    """Convert tensor to numpy array.

    Args:
        tensor (TensorLike): The tensor parameter.

    Returns:
        TensorLike: Converted array.
    """
    if hasattr(tensor, "numpy"):
        return typing.cast(TensorLike, tensor.numpy())
    if hasattr(tensor, "data") and hasattr(tensor.data, "numpy"):
        return typing.cast(TensorLike, tensor.data.numpy())

    try:
        return tensor.tolist() if hasattr(tensor, "tolist") else tensor
    except (ValueError, TypeError):
        return tensor


def concatenate_arrays(arrays: list[TensorLike]) -> TensorLike:
    """Concatenate numpy arrays.

    Args:
        arrays (list[TensorLike]): The arrays parameter.

    Returns:
        TensorLike: Concatenated array.
    """
    from ml_switcheroo_compiler import ops

    return typing.cast(TensorLike, getattr(ops, "concatenate")(arrays, axis=0))


def is_numpy_array(array: TensorOrScalar) -> bool:
    """Check if array is a numpy array.

    Args:
        array (TensorOrScalar): The array parameter.

    Returns:
        bool: True if array is a numpy array or array-like.
    """
    return type(array).__module__ == "numpy" or hasattr(array, "numpy") or hasattr(array, "__array__")


def _dtype_to_descr(dtype: TensorOrScalar) -> str:
    """Map DType to numpy descr string.

    Args:
        dtype (TensorOrScalar): The dtype parameter.

    Returns:
        str: NumPy descriptor string.
    """
    from ml_switcheroo_compiler.core.dtype import DType

    mapping = {
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
        for k, v in mapping.items():
            if k.value == dtype:
                return v
    elif hasattr(dtype, "value"):
        return typing.cast(str, getattr(mapping, "get")(dtype, "<f4"))
    return "<f4"


def _extract_arr_shape_dtype(arr: TensorLike) -> tuple[tuple[int, ...], str]:
    """Extract shape and dtype string from an array object.

    Args:
        arr (TensorLike): The array object.

    Returns:
        tuple[tuple[int, ...], str]: (shape, dtype_str).
    """
    dtype = getattr(arr.dtype, "name") if hasattr(getattr(arr, "dtype", None), "name") else str(getattr(arr, "dtype", "<f4"))
    return getattr(arr, "shape", ()), dtype


def _get_shape_and_dtype(tensor: TensorLike) -> tuple[tuple[int, ...], str]:
    """Get shape and dtype from a generic tensor object.

    Args:
        tensor (TensorLike): The tensor object.

    Returns:
        tuple[tuple[int, ...], str]: (shape, dtype).
    """
    if hasattr(tensor, "eval"):
        return _extract_arr_shape_dtype(tensor.eval())
    if hasattr(tensor, "numpy"):
        return _extract_arr_shape_dtype(tensor.numpy())
    if hasattr(tensor, "data") and hasattr(tensor.data, "numpy"):
        return _extract_arr_shape_dtype(tensor.data.numpy())
    return getattr(tensor, "shape", ()), getattr(tensor, "dtype", "<f4")


def _extract_arr_bytes(arr: TensorOrScalar) -> bytes:
    """Extract raw bytes from an array object.

    Args:
        arr (TensorOrScalar): The array object.

    Returns:
        bytes: The raw data bytes.
    """
    if hasattr(arr, "tobytes"):
        return typing.cast(bytes, getattr(arr, "tobytes")())
    if hasattr(arr, "data") and hasattr(getattr(arr, "data"), "tobytes"):
        return typing.cast(bytes, getattr(arr, "data").tobytes())
    return b""


def _get_data_bytes(tensor: TensorOrScalar) -> bytes:
    """Get raw data bytes from a generic tensor object.

    Args:
        tensor (TensorOrScalar): The tensor object.

    Returns:
        bytes: The raw data bytes.
    """
    if hasattr(tensor, "eval"):
        return _extract_arr_bytes(getattr(tensor, "eval")())
    if hasattr(tensor, "numpy"):
        return typing.cast(bytes, getattr(tensor, "numpy")().tobytes())
    if hasattr(tensor, "data") and hasattr(getattr(tensor, "data"), "numpy"):
        return typing.cast(bytes, getattr(tensor, "data").numpy().tobytes())
    return b""


def _tensor_to_npy_bytes(tensor: TensorOrScalar) -> bytes:
    """Convert a tensor to npy bytes.

    Args:
        tensor (TensorOrScalar): The tensor parameter.

    Returns:
        bytes: Serialized `.npy` byte stream.
    """
    import struct

    shape, dtype = typing.cast("TypedVariadicCallable[tuple[tuple[int, ...], str]]", _get_shape_and_dtype)(tensor)
    data_bytes = _get_data_bytes(tensor)
    dtype_str = _dtype_to_descr(dtype)

    magic = b"\x93NUMPY"
    major, minor = 1, 0
    shape_str = f"({shape[0]},)" if len(shape) == 1 else str(shape)

    dict_str = "{'descr': '" + dtype_str + "', 'fortran_order': False, 'shape': " + shape_str + "}, "
    dict_str += " " * (16 - (len(dict_str) + 10) % 16) + "\n"

    header = magic + struct.pack("<BBH", major, minor, len(dict_str)) + dict_str.encode("ascii")
    return header + data_bytes


def get_npz_bytes(weights: TensorOrScalar) -> bytes:
    """Get npz bytes.

    Args:
        weights (TensorOrScalar): The weights parameter.

    Returns:
        bytes: Serialized `.npz` archive bytes.
    """
    import io
    import zipfile

    from ml_switcheroo_compiler.backends.registry import get_active_backend

    backend = get_active_backend()
    if hasattr(backend, "get_npz_bytes"):
        return typing.cast(bytes, backend.get_npz_bytes(weights))

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_STORED) as zf:
        for k, v in getattr(weights, "items")():
            name = k if getattr(k, "endswith")(".npy") else f"{k}.npy"
            zf.writestr(name, _tensor_to_npy_bytes(v))
    return buf.getvalue()


def parse_npz(file_obj: str | bytes | TensorLike) -> dict[str, TensorLike]:
    """Parse npz file.

    Args:
        file_obj (str | bytes | TensorLike): The file_obj parameter.

    Returns:
        dict[str, TensorLike]: Result.
    """
    import io

    from ml_switcheroo_compiler.backends.registry import BackendRegistry

    try:
        backend_cls = BackendRegistry.get("numpy")
        target = io.BytesIO(file_obj) if isinstance(file_obj, (bytes, bytearray)) else file_obj
        with getattr(backend_cls, "load")(target) as data:
            return {k: data[k] for k in getattr(data, "files", [])}
    except Exception:
        return {}


def load_npz(file_obj: str | bytes | TensorLike) -> list[TensorLike]:
    """Load weights from a .npz file object.

    Args:
        file_obj (str | bytes | TensorLike): The file_obj parameter.

    Returns:
        list[TensorLike]: Result.
    """
    from ml_switcheroo_compiler.backends.registry import get_active_backend

    backend = get_active_backend()
    if hasattr(backend, "load_npz"):
        try:
            return typing.cast(list[TensorLike], backend.load_npz(file_obj))
        except Exception as e:
            import logging

            logging.getLogger(__name__).warning(f"Backend load_npz failed: {e}. Falling back.")

    parsed = parse_npz(file_obj)
    # the original load_npz returned a list of weights (assuming ordered arrays or dict?)
    # Wait, the signature says -> list[object]. Let's return list of values
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
