# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Serialization utilities."""

from typing import Any


def _extract_numpy_weights(weights: dict[str, Any]) -> dict[str, Any]:
    """Extract numpy weights.

    Args:
        weights (dict): The weights parameter.

    Returns:
        dict: Result.
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


def to_numpy(tensor: Any) -> Any:
    """Convert tensor to numpy array.

    Args:
        tensor (object): The tensor parameter.

    Returns: Any: Result.
    """
    if hasattr(tensor, "numpy"):
        return tensor.numpy()
    if hasattr(tensor, "data") and hasattr(tensor.data, "numpy"):
        return tensor.data.numpy()

    try:
        return tensor.tolist() if hasattr(tensor, "tolist") else tensor
    except (ValueError, TypeError):
        return tensor


def concatenate_arrays(arrays: list[Any]) -> Any:
    """Concatenate numpy arrays.

    Args:
        arrays (list): The arrays parameter.

    Returns: Any: Result.
    """
    from ml_switcheroo_compiler import ops

    return ops.concatenate(arrays, axis=0)


def is_numpy_array(array: Any) -> bool:
    """Check if array is a numpy array.

    Args:
        array (object): The array parameter.

    Returns:
        bool: Result.
    """
    return type(array).__module__ == "numpy" or hasattr(array, "numpy") or hasattr(array, "__array__")


def _dtype_to_descr(dtype: Any) -> str:
    """Map DType to numpy descr string.

    Args:
        dtype (object): The dtype parameter.

    Returns:
        str: Result.
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
        return mapping.get(dtype, "<f4")
    return "<f4"


def _extract_arr_shape_dtype(arr: Any) -> tuple[Any, ...]:
    """Extract shape and dtype string from an array object.

    Args:
        arr (object): The array object.

    Returns:
        tuple: (shape, dtype_str)
    """
    dtype = arr.dtype.name if hasattr(getattr(arr, "dtype", None), "name") else str(getattr(arr, "dtype", "<f4"))
    return getattr(arr, "shape", ()), dtype


def _get_shape_and_dtype(tensor: Any) -> tuple[Any, ...]:
    """Get shape and dtype from a generic tensor object.

    Args:
        tensor (object): The tensor object.

    Returns:
        tuple: (shape, dtype)
    """
    if hasattr(tensor, "eval"):
        return _extract_arr_shape_dtype(tensor.eval())
    if hasattr(tensor, "numpy"):
        return _extract_arr_shape_dtype(tensor.numpy())
    if hasattr(tensor, "data") and hasattr(tensor.data, "numpy"):
        return _extract_arr_shape_dtype(tensor.data.numpy())
    return getattr(tensor, "shape", ()), getattr(tensor, "dtype", "<f4")


def _extract_arr_bytes(arr: Any) -> bytes:
    """Extract raw bytes from an array object.

    Args:
        arr (object): The array object.

    Returns:
        bytes: The raw data bytes.
    """
    if hasattr(arr, "tobytes"):
        return arr.tobytes()  # type: ignore
    if hasattr(arr, "data") and hasattr(arr.data, "tobytes"):
        return arr.data.tobytes()  # type: ignore
    return b""


def _get_data_bytes(tensor: Any) -> bytes:
    """Get raw data bytes from a generic tensor object.

    Args:
        tensor (object): The tensor object.

    Returns:
        bytes: The raw data bytes.
    """
    if hasattr(tensor, "eval"):
        return _extract_arr_bytes(tensor.eval())
    if hasattr(tensor, "numpy"):
        return tensor.numpy().tobytes()  # type: ignore
    if hasattr(tensor, "data") and hasattr(tensor.data, "numpy"):
        return tensor.data.numpy().tobytes()  # type: ignore
    return b""


def _tensor_to_npy_bytes(tensor: Any) -> bytes:
    """Convert a tensor to npy bytes.

    Args:
        tensor (object): The tensor parameter.

    Returns:
        bytes: Result.
    """
    import struct

    shape, dtype = _get_shape_and_dtype(tensor)
    data_bytes = _get_data_bytes(tensor)
    dtype_str = _dtype_to_descr(dtype)

    magic = b"\x93NUMPY"
    major, minor = 1, 0
    shape_str = f"({shape[0]},)" if len(shape) == 1 else str(shape)

    dict_str = "{'descr': '" + dtype_str + "', 'fortran_order': False, 'shape': " + shape_str + "}, "
    dict_str += " " * (16 - (len(dict_str) + 10) % 16) + "\n"

    header = magic + struct.pack("<BBH", major, minor, len(dict_str)) + dict_str.encode("ascii")
    return header + data_bytes


def get_npz_bytes(weights: dict[str, Any]) -> bytes:
    """Get npz bytes.

    Args:
        weights (dict): The weights parameter.

    Returns:
        bytes: Result.
    """
    import io
    import zipfile

    from ml_switcheroo_compiler.backends.registry import get_active_backend

    backend = get_active_backend()
    if hasattr(backend, "get_npz_bytes"):
        return backend.get_npz_bytes(weights)  # type: ignore

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_STORED) as zf:
        for k, v in weights.items():
            name = k if k.endswith(".npy") else f"{k}.npy"
            zf.writestr(name, _tensor_to_npy_bytes(v))
    return buf.getvalue()


def parse_npz(file_obj: Any) -> dict[str, Any]:
    """Parse npz file.

    Args:
        file_obj (object): The file_obj parameter.

    Returns:
        dict: Result.
    """
    from ml_switcheroo_compiler.backends.registry import BackendRegistry

    try:
        backend_cls = BackendRegistry.get("numpy")
        with backend_cls.load(file_obj) as data:  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism
            return {k: data[k] for k in getattr(data, "files", [])}
    except Exception:
        return {}


def load_npz(file_obj: Any) -> list[Any]:
    """Load weights from a .npz file object.

    Args:
        file_obj (object): The file_obj parameter.

    Returns:
        list: Result.
    """
    from ml_switcheroo_compiler.backends.registry import get_active_backend

    backend = get_active_backend()
    if hasattr(backend, "load_npz"):
        try:
            return backend.load_npz(file_obj)  # type: ignore
        except Exception as e:
            import logging

            logging.getLogger(__name__).warning(f"Backend load_npz failed: {e}. Falling back.")

    parsed = parse_npz(file_obj)
    # the original load_npz returned a list of weights (assuming ordered arrays or dict?)
    # Wait, the signature says -> list[Any]. Let's return list of values
    return list(parsed.values())


def save_ir_graph(graph: Any, filepath: str) -> None:
    """Save an IR graph configuration to a file reliably.

    Args:
        graph: The IRGraph to serialize.
        filepath: The path to write out the configuration.
    """
    from ml_switcheroo_compiler.serialization.ir_format import graph_to_json

    with open(filepath, "w") as f:
        f.write(graph_to_json(graph))


def load_ir_graph(filepath: str) -> Any:
    """Load an IR graph configuration from a file reliably.

    Args:
        filepath: The path to read the configuration from.

    Returns:
        The deserialized IRGraph.
    """
    from ml_switcheroo_compiler.serialization.ir_format import json_to_graph

    with open(filepath) as f:
        return json_to_graph(f.read())
