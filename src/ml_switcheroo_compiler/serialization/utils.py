"""Serialization utilities."""


def _extract_numpy_weights(weights: dict) -> dict:
    """Extract numpy weights."""
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
            pass
    return weights_np


def to_numpy(tensor: object) -> object:
    """Convert tensor to numpy array."""
    if hasattr(tensor, "numpy"):
        return tensor.numpy()
    if hasattr(tensor, "data") and hasattr(tensor.data, "numpy"):
        return tensor.data.numpy()

    try:
        return tensor.tolist() if hasattr(tensor, "tolist") else tensor
    except (ValueError, TypeError):
        return tensor


def concatenate_arrays(arrays: list) -> object:
    """Concatenate numpy arrays."""
    from ml_switcheroo_compiler import ops

    return ops.concatenate(arrays, dim=0)


def is_numpy_array(array: object) -> bool:
    """Check if array is a numpy array."""
    return type(array).__module__ == "numpy" or hasattr(array, "numpy") or hasattr(array, "__array__")


def _dtype_to_descr(dtype: object) -> str:
    """Map DType to numpy descr string."""
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


def _extract_arr_shape_dtype(arr: object) -> tuple:
    dtype = arr.dtype.name if hasattr(getattr(arr, "dtype", None), "name") else str(getattr(arr, "dtype", "<f4"))
    return getattr(arr, "shape", ()), dtype


def _get_shape_and_dtype(tensor: object) -> tuple:
    if hasattr(tensor, "eval"):
        return _extract_arr_shape_dtype(tensor.eval())
    if hasattr(tensor, "numpy"):
        return _extract_arr_shape_dtype(tensor.numpy())
    if hasattr(tensor, "data") and hasattr(tensor.data, "numpy"):
        return _extract_arr_shape_dtype(tensor.data.numpy())
    return getattr(tensor, "shape", ()), getattr(tensor, "dtype", "<f4")


def _extract_arr_bytes(arr: object) -> bytes:
    if hasattr(arr, "tobytes"):
        return arr.tobytes()
    if hasattr(arr, "data") and hasattr(arr.data, "tobytes"):
        return arr.data.tobytes()
    return b""


def _get_data_bytes(tensor: object) -> bytes:
    if hasattr(tensor, "eval"):
        return _extract_arr_bytes(tensor.eval())
    if hasattr(tensor, "numpy"):
        return tensor.numpy().tobytes()
    if hasattr(tensor, "data") and hasattr(tensor.data, "numpy"):
        return tensor.data.numpy().tobytes()
    return b""


def _tensor_to_npy_bytes(tensor: object) -> bytes:
    """Convert a tensor to npy bytes."""
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


def get_npz_bytes(weights: dict) -> bytes:
    """Get npz bytes."""
    import io
    import zipfile

    from ml_switcheroo_compiler.backends.registry import get_active_backend

    backend = get_active_backend()
    if hasattr(backend, "get_npz_bytes"):
        return backend.get_npz_bytes(weights)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_STORED) as zf:
        for k, v in weights.items():
            name = k if k.endswith(".npy") else f"{k}.npy"
            zf.writestr(name, _tensor_to_npy_bytes(v))
    return buf.getvalue()


def parse_npz(file_obj: object) -> dict:
    """Parse npz file."""
    import numpy as np

    try:
        with np.load(file_obj) as data:
            return {k: data[k] for k in data.files}
    except Exception:
        return {}


def load_npz(file_obj: object) -> list:
    """Load weights from a .npz file object."""
    from ml_switcheroo_compiler.backends.registry import get_active_backend

    backend = get_active_backend()
    if hasattr(backend, "load_npz"):
        try:
            return backend.load_npz(file_obj)
        except NotImplementedError:
            pass

    parsed = parse_npz(file_obj)
    # the original load_npz returned a list of weights (assuming ordered arrays or dict?)
    # Wait, the signature says -> list. Let's return list of values
    return list(parsed.values())


def save_ir_graph(graph: object, filepath: str) -> None:
    """Saves an IR graph configuration to a file reliably.

    Args:
        graph: The IRGraph to serialize.
        filepath: The path to write out the configuration.
    """
    from ml_switcheroo_compiler.serialization.ir_format import graph_to_json

    with open(filepath, "w") as f:
        f.write(graph_to_json(graph))


def load_ir_graph(filepath: str) -> object:
    """Loads an IR graph configuration from a file reliably.

    Args:
        filepath: The path to read the configuration from.

    Returns:
        The deserialized IRGraph.
    """
    from ml_switcheroo_compiler.serialization.ir_format import json_to_graph

    with open(filepath) as f:
        return json_to_graph(f.read())
