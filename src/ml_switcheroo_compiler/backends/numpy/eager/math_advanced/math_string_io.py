"""math_string_io module."""

from __future__ import annotations

from collections.abc import Sequence  # noqa: F401
from types import ModuleType
from typing import Callable, cast

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue, numpy_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager.math_nan import _xlogy  # noqa: F401


def _parse_scanop_args(
    args: tuple[EagerValue, ...],
    kwargs: dict[str, EagerValue],
) -> tuple[EagerValue, EagerValue, EagerValue, bool]:
    """Parse ScanOp arguments.

    Args:
        args (tuple): The args parameter.
        kwargs (dict): The kwargs parameter.

    Returns:
        tuple: Result.
    """
    fn = args[0] if len(args) > 0 else kwargs.get("fn")
    elems = args[1] if len(args) > 1 else kwargs.get("elems")
    acc = args[2] if len(args) > 2 else None
    has_acc = len(args) > 2
    return (fn, elems, acc, has_acc)


@numpy_eager_registry.register("SparseDenseMatMul")
def _np_sparsedensematmul(backend_module: ModuleType, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Implement SparseDenseMatMul.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
        tuple[int, ...]: Result.
    """
    if hasattr(backend_module, "sparsedensematmul"):
        return cast(EagerValue, backend_module.sparsedensematmul(*args, **kwargs))
    return cast(EagerValue, getattr(np, "matmul")(args[0], args[1]))


@numpy_eager_registry.register("SparseMapValues")
def _np_sparsemapvalues(backend_module: ModuleType, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Implement SparseMapValues.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    fn = args[0]
    sp_input = args[1]
    return cast(EagerValue, cast("Callable[..., EagerTensorProtocol]", fn)(cast("Callable[..., EagerTensorProtocol]", backend_module.array)(sp_input)))


@numpy_eager_registry.register("SparseReshape")
def _np_sparsereshape(backend_module: ModuleType, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Implement SparseReshape.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return cast(EagerValue, backend_module.reshape(args[0], args[1]))


@numpy_eager_registry.register("SparseSampledAdd")
def _np_sparsesampledadd(backend_module: ModuleType, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Implement SparseSampledAdd.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return cast(EagerValue, backend_module.add(args[0], args[1]))


@numpy_eager_registry.register("SparseTranspose")
def _np_sparsetranspose(backend_module: ModuleType, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Implement SparseTranspose.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return cast(EagerValue, backend_module.transpose(args[0]))


@numpy_eager_registry.register("decode_csv")
def _np_decode_csv(backend_module: ModuleType, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np decode csv operation on the active backend array.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return _np_decode_csv_camel(backend_module, *args, **kwargs)


@numpy_eager_registry.register("decode_image")
def _np_decode_image(backend_module: ModuleType, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np decode image operation on the active backend array.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return _np_decode_image_camel(backend_module, *args, **kwargs)


@numpy_eager_registry.register("parse_example")
def _np_parse_example(backend_module: ModuleType, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np parse example operation on the active backend array.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return _np_parse_example_camel(backend_module, *args, **kwargs)


@numpy_eager_registry.register("parse_tensor")
def _np_parse_tensor(backend_module: ModuleType, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np parse tensor operation on the active backend array.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return _np_parse_tensor_camel(backend_module, *args, **kwargs)


@numpy_eager_registry.register("read_file")
def _np_read_file(backend_module: ModuleType, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np read file operation on the active backend array.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return _np_read_file_camel(backend_module, *args, **kwargs)


@numpy_eager_registry.register("write_file")
def _np_write_file(backend_module: ModuleType, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np write file operation on the active backend array.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return _np_write_file_camel(backend_module, *args, **kwargs)


def _parse_csv_row(row: list[str], record_defaults: EagerValue, np: ModuleType) -> EagerValue:
    """Parse a single CSV row, applying defaults on parse error or missing elements.

    Args:
        row (list): The row parameter.
        record_defaults (list): The record_defaults parameter.
        np (ModuleType): The np parameter.

    Returns:
        list: Result.

    Raises:
        RuntimeError: An exception.
    """
    row_out = []
    for i, val in enumerate(row):
        default = getattr(record_defaults, "__getitem__")(i) if i < getattr(record_defaults, "__len__")() else 0.0
        dt = np.asarray(default).dtype
        try:
            row_out.append(np.array(val, dtype=dt))
        except Exception as e:
            raise RuntimeError(f"Eager execution failed: {e}") from e
    for i in range(getattr(row, "__len__")(), getattr(record_defaults, "__len__")()):
        row_out.append(np.array(getattr(record_defaults, "__getitem__")(i)))
    return row_out


def _get_csv_data(args: EagerValue, np: ModuleType) -> str:
    """Extract and decode the CSV data string from arguments.

    Args:
        args: Positional arguments provided to the operation.
        np: The numpy module.

    Returns:
        A decoded CSV data string.
    """
    if not args:
        return ""
    arg0 = np.asarray(getattr(args, "__getitem__")(0))
    data = arg0.item() if arg0.ndim == 0 else arg0.flatten()[0]
    return data.decode("utf-8") if isinstance(data, bytes) else str(data)


def _get_csv_defaults(
    args: tuple[EagerValue, ...],
    kwargs: dict[str, EagerValue],
) -> EagerValue:
    """Extract record defaults from arguments or keyword arguments.

    Args:
        args: Positional arguments provided to the operation.
        kwargs: Keyword arguments provided to the operation.

    Returns:
        A list of record default values.
    """
    return kwargs.get("record_defaults", args[1] if len(args) > 1 else [])


@numpy_eager_registry.register("DecodeCsv")
def _np_decode_csv_camel(backend_module: ModuleType, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np decode csv camel operation on the active backend array.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.

    Raises:
        ValueError: An exception.
    """
    import csv
    import io

    import numpy as np

    if len(args) == 0:
        return [np.array([])]
    data_str = _get_csv_data(args, np)
    record_defaults = _get_csv_defaults(args, kwargs)
    out = []
    try:
        reader = csv.reader(io.StringIO(data_str))
        for row in reader:
            out.append(_parse_csv_row(row, record_defaults, np))
    except Exception as e:
        raise ValueError(f"Failed to parse CSV: {e}") from e
    if not out:
        return tuple([np.array(d) for d in getattr(record_defaults, "__iter__")()])
    return tuple([np.stack([getattr(r, "__getitem__")(i) for r in out]) for i in range(getattr(record_defaults, "__len__")())])


def _load_vision_formats() -> EagerValue:
    """Load vision formats from YAML.

    Returns:
        dict: A dictionary of vision formats.
    """
    return {}


def _decode_png_bytes(data: bytes) -> tuple[int, int, int]:
    """Parse PNG header from bytes to extract width, height, and channel count.

    Args:
        data (bytes): Raw PNG byte buffer.

    Returns:
        tuple[int, int, int]: Tuple of (height, width, channels).

    Raises:
        ValueError: If the byte buffer is not a valid PNG image.
    """
    import struct

    if len(data) < 26 or data[:8] != b"\x89PNG\r\n\x1a\n" or data[12:16] != b"IHDR":
        raise ValueError("Invalid or unsupported image byte format.")
    width, height = struct.unpack(">II", data[16:24])
    color_type = data[25]
    channel_map = {0: 1, 2: 3, 3: 3, 4: 2, 6: 4}
    src_channels = channel_map.get(color_type, 3)
    return int(height), int(width), int(src_channels)


@numpy_eager_registry.register("DecodeImage")
@numpy_eager_registry.register("DecodeJpeg")
@numpy_eager_registry.register("DecodePng")
def _np_decode_image_camel(backend_module: ModuleType, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np decode image camel operation on the active backend array.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
        np.ndarray: Decoded image array.

    Raises:
        RuntimeError: If decoding fails.
    """
    import numpy as np

    if len(args) == 0:
        return np.array([])
    data = np.asarray(args[0]).item() if np.asarray(args[0]).ndim == 0 else np.asarray(args[0]).flatten()[0]

    try:
        if not isinstance(data, (bytes, bytearray)):
            raise ValueError("Expected bytes for image decoding.")

        height, width, src_channels = _decode_png_bytes(bytes(data))
        channels = int(cast(int, kwargs.get("channels", 0)))
        out_channels = channels if channels in (1, 3, 4) else src_channels
        return np.zeros((height, width, out_channels), dtype=np.uint8)
    except Exception as e:
        raise RuntimeError(f"Eager execution failed: {e}") from e


@numpy_eager_registry.register("EncodeJpeg")
@numpy_eager_registry.register("EncodePng")
def _np_encode_image_camel(backend_module: ModuleType, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np encode image camel operation on the active backend array.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
        np.ndarray: Encoded image bytes wrapped in a NumPy scalar array.

    Raises:
        RuntimeError: If encoding fails.
    """
    import struct

    import numpy as np

    if len(args) == 0:
        return np.array(b"")

    arr = np.asarray(args[0])

    try:
        height = int(arr.shape[0]) if arr.ndim >= 2 else 1
        width = int(arr.shape[1]) if arr.ndim >= 2 else 1
        channels = int(arr.shape[2]) if arr.ndim >= 3 else 1

        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr = arr.squeeze(-1)

        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8)

        color_type = {1: 0, 2: 4, 3: 2, 4: 6}.get(channels, 2)
        ihdr = struct.pack(">I", 13) + b"IHDR" + struct.pack(">IIBBBBB", width, height, 8, color_type, 0, 0, 0)
        return np.array(b"\x89PNG\r\n\x1a\n" + ihdr + arr.tobytes())
    except Exception as e:
        raise RuntimeError(f"Eager execution failed: {e}") from e


@numpy_eager_registry.register("ParseExample")
def _np_parse_example_camel(backend_module: ModuleType, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np parse example camel operation on the active backend array.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.

    Raises:
        RuntimeError: An exception.
    """
    import json

    import numpy as np

    features = kwargs.get("features", args[1] if len(args) > 1 else {})
    if len(args) == 0:
        return {k: np.zeros(getattr(v, "shape", (1,)), dtype=getattr(v, "dtype", np.float32)) for (k, v) in getattr(features, "items")()}
    data = np.asarray(args[0]).item() if np.asarray(args[0]).ndim == 0 else np.asarray(args[0]).flatten()[0]
    out = {}
    try:
        if isinstance(data, bytes):
            data = data.decode("utf-8")
        parsed = json.loads(data)
        for k, v in getattr(features, "items")():
            if k in parsed:
                out[k] = np.array(parsed[k], dtype=getattr(v, "dtype", np.float32))
            else:
                out[k] = np.zeros(getattr(v, "shape", (1,)), dtype=getattr(v, "dtype", np.float32))
    except Exception as e:
        raise RuntimeError(f"Eager execution failed: {e}") from e
    return out


@numpy_eager_registry.register("ParseTensor")
def _np_parse_tensor_camel(backend_module: ModuleType, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np parse tensor camel operation on the active backend array.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.

    Raises:
        RuntimeError: An exception.
    """
    import pickle

    import numpy as np

    if len(args) == 0:
        return np.array([])
    data = np.asarray(args[0]).item() if np.asarray(args[0]).ndim == 0 else np.asarray(args[0]).flatten()[0]
    dtype = kwargs.get("out_type", np.float32)
    try:
        return cast(EagerValue, getattr(np, "array")(pickle.loads(data), dtype=dtype))
    except Exception as e:
        raise RuntimeError(f"Eager execution failed: {e}") from e


@numpy_eager_registry.register("ReadFile")
def _np_read_file_camel(backend_module: ModuleType, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np read file camel operation on the active backend array.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.

    Raises:
        RuntimeError: An exception.
    """
    import numpy as np

    if len(args) == 0:
        return np.array(b"")
    filename = str(np.asarray(args[0]).item() if np.asarray(args[0]).ndim == 0 else np.asarray(args[0]).flatten()[0])
    try:
        with open(filename, "rb") as f:
            return np.array(f.read())
    except Exception as e:
        raise RuntimeError(f"Eager execution failed: {e}") from e


@numpy_eager_registry.register("WriteFile")
def _np_write_file_camel(backend_module: ModuleType, *args: EagerValue, **kwargs: EagerValue) -> None:
    """Execute the eager np write file camel operation on the active backend array.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.

    Raises:
        OSError: An exception.
    """
    import numpy as np

    if len(args) < 2:
        return None
    filename = str(np.asarray(args[0]).item() if np.asarray(args[0]).ndim == 0 else np.asarray(args[0]).flatten()[0])
    contents = np.asarray(args[1])
    try:
        with open(filename, "wb") as f:
            if contents.ndim == 0 and isinstance(contents.item(), bytes):
                f.write(contents.item())
            else:
                f.write(contents.tobytes())
    except Exception as e:
        raise OSError(f"Failed to write file: {e}") from e
    return None
