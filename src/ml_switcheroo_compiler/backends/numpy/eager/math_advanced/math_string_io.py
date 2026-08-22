# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""math_string_io module."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager.math_nan import _xlogy


def _parse_scanop_args(args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[Any, ...]:
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
def _np_sparsedensematmul(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement SparseDenseMatMul.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.

    Raises:
        RuntimeError: An exception.
    """
    try:
        import ml_switcheroo_compiler.ops as _ops

        if hasattr(_ops, "SparseDenseMatMul"):
            cls_or_func = _ops.SparseDenseMatMul
            if isinstance(cls_or_func, type) and (not issubclass(cls_or_func, _ops.OpDef)):
                return cls_or_func(*args, **kwargs)
    except Exception as e:
        if not isinstance(e, (ImportError, AttributeError)):
            raise RuntimeError(f"Eager execution failed: {e}") from e
    if hasattr(backend_module, "sparsedensematmul"):
        return backend_module.sparsedensematmul(*args, **kwargs)
    return np.matmul(args[0], args[1])


@numpy_eager_registry.register("SparseMapValues")
def _np_sparsemapvalues(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement SparseMapValues.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    fn = args[0]
    sp_input = args[1]
    return fn(backend_module.array(sp_input))


@numpy_eager_registry.register("SparseReshape")
def _np_sparsereshape(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement SparseReshape.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return backend_module.reshape(args[0], args[1])


@numpy_eager_registry.register("SparseSampledAdd")
def _np_sparsesampledadd(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement SparseSampledAdd.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return backend_module.add(args[0], args[1])


@numpy_eager_registry.register("SparseTranspose")
def _np_sparsetranspose(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement SparseTranspose.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return backend_module.transpose(args[0])


@numpy_eager_registry.register("decode_csv")
def _np_decode_csv(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_decode_csv operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return _np_decode_csv_camel(backend_module, *args, **kwargs)


@numpy_eager_registry.register("decode_image")
def _np_decode_image(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_decode_image operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return _np_decode_image_camel(backend_module, *args, **kwargs)


@numpy_eager_registry.register("parse_example")
def _np_parse_example(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_parse_example operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return _np_parse_example_camel(backend_module, *args, **kwargs)


@numpy_eager_registry.register("parse_tensor")
def _np_parse_tensor(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_parse_tensor operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return _np_parse_tensor_camel(backend_module, *args, **kwargs)


@numpy_eager_registry.register("read_file")
def _np_read_file(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_read_file operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return _np_read_file_camel(backend_module, *args, **kwargs)


@numpy_eager_registry.register("write_file")
def _np_write_file(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_write_file operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return _np_write_file_camel(backend_module, *args, **kwargs)


def _parse_csv_row(row: list[str], record_defaults: list[Any], np: Any) -> list[Any]:
    """Parse a single CSV row, applying defaults on parse error or missing elements.

    Args:
        row (list): The row parameter.
        record_defaults (list): The record_defaults parameter.
        np (object): The np parameter.

    Returns:
        list: Result.

    Raises:
        RuntimeError: An exception.
    """
    row_out = []
    for i, val in enumerate(row):
        default = record_defaults[i] if i < len(record_defaults) else 0.0
        dt = np.asarray(default).dtype
        try:
            row_out.append(np.array(val, dtype=dt))
        except Exception as e:
            raise RuntimeError(f"Eager execution failed: {e}") from e
    for i in range(len(row), len(record_defaults)):
        row_out.append(np.array(record_defaults[i]))
    return row_out


def _get_csv_data(args: tuple[Any, ...], np: Any) -> str:
    """Extract and decode the CSV data string from arguments.

    Args:
        args: Positional arguments provided to the operation.
        np: The numpy module.

    Returns:
        A decoded CSV data string.
    """
    if not args:
        return ""
    arg0 = np.asarray(args[0])
    data = arg0.item() if arg0.ndim == 0 else arg0.flatten()[0]
    return data.decode("utf-8") if isinstance(data, bytes) else str(data)


def _get_csv_defaults(args: tuple[Any, ...], kwargs: dict[str, Any]) -> list[Any]:
    """Extract record defaults from arguments or keyword arguments.

    Args:
        args: Positional arguments provided to the operation.
        kwargs: Keyword arguments provided to the operation.

    Returns:
        A list of record default values.
    """
    return kwargs.get("record_defaults", args[1] if len(args) > 1 else [])  # type: ignore


@numpy_eager_registry.register("DecodeCsv")
def _np_decode_csv_camel(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_decode_csv_camel operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

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
        return tuple([np.array(d) for d in record_defaults])
    return tuple([np.stack([r[i] for r in out]) for i in range(len(record_defaults))])


def _load_vision_formats() -> dict[str, Any]:
    """Load vision formats from YAML.

    Returns:
        dict: A dictionary of vision formats.
    """
    import os

    import yaml

    yaml_path = os.path.join(os.path.dirname(__file__), "..", "vision_formats.yaml")
    if os.path.exists(yaml_path):
        with open(yaml_path) as f:
            from typing import cast

            return cast(dict[str, Any], yaml.safe_load(f).get("formats", {}))
    return {}


@numpy_eager_registry.register("DecodeImage")
@numpy_eager_registry.register("DecodeJpeg")
@numpy_eager_registry.register("DecodePng")
def _np_decode_image_camel(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_decode_image_camel operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.

    Raises:
        RuntimeError: An exception.
    """
    import io

    import numpy as np
    from PIL import Image

    if len(args) == 0:
        return np.array([])
    data = np.asarray(args[0]).item() if np.asarray(args[0]).ndim == 0 else np.asarray(args[0]).flatten()[0]

    try:
        if not isinstance(data, (bytes, bytearray)):
            raise ValueError("Expected bytes for image decoding.")

        image = Image.open(io.BytesIO(data))

        # Read channels config if needed, default to RGB
        channels = kwargs.get("channels", 0)
        if channels == 1:
            conv_img = image.convert("L")
            arr = np.array(conv_img, dtype=np.uint8)[..., np.newaxis]
        elif channels == 3:
            conv_img = image.convert("RGB")
            arr = np.array(conv_img, dtype=np.uint8)
        elif channels == 4:
            conv_img = image.convert("RGBA")
            arr = np.array(conv_img, dtype=np.uint8)
        else:
            arr = np.array(image, dtype=np.uint8)
            if arr.ndim == 2:
                arr = arr[..., np.newaxis]

        return arr
    except Exception as e:
        raise RuntimeError(f"Eager execution failed: {e}") from e


@numpy_eager_registry.register("EncodeJpeg")
@numpy_eager_registry.register("EncodePng")
def _np_encode_image_camel(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate encode operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.

    Raises:
        RuntimeError: An exception.
    """
    import io

    import numpy as np
    from PIL import Image

    if len(args) == 0:
        return np.array(b"")

    arr = np.asarray(args[0])

    formats = _load_vision_formats()

    # We cheat to get the op name from kwargs or assume JPEG
    op_name = kwargs.get("op_name", "EncodeJpeg")
    fmt = formats.get(op_name, {}).get("format", "JPEG")

    try:
        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr = arr.squeeze(-1)

        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8)

        image = Image.fromarray(arr)
        bio = io.BytesIO()
        image.save(bio, format=fmt)
        return np.array(bio.getvalue())
    except Exception as e:
        raise RuntimeError(f"Eager execution failed: {e}") from e


@numpy_eager_registry.register("ParseExample")
def _np_parse_example_camel(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_parse_example_camel operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.

    Raises:
        RuntimeError: An exception.
    """
    import json

    import numpy as np

    features = kwargs.get("features", args[1] if len(args) > 1 else {})
    if len(args) == 0:
        return {k: np.zeros(getattr(v, "shape", (1,)), dtype=getattr(v, "dtype", np.float32)) for (k, v) in features.items()}
    data = np.asarray(args[0]).item() if np.asarray(args[0]).ndim == 0 else np.asarray(args[0]).flatten()[0]
    out = {}
    try:
        if isinstance(data, bytes):
            data = data.decode("utf-8")
        parsed = json.loads(data)
        for k, v in features.items():
            if k in parsed:
                out[k] = np.array(parsed[k], dtype=getattr(v, "dtype", np.float32))
            else:
                out[k] = np.zeros(getattr(v, "shape", (1,)), dtype=getattr(v, "dtype", np.float32))
    except Exception as e:
        raise RuntimeError(f"Eager execution failed: {e}") from e
    return out


@numpy_eager_registry.register("ParseTensor")
def _np_parse_tensor_camel(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_parse_tensor_camel operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

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
        return np.array(pickle.loads(data), dtype=dtype)
    except Exception as e:
        raise RuntimeError(f"Eager execution failed: {e}") from e


@numpy_eager_registry.register("ReadFile")
def _np_read_file_camel(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_read_file_camel operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

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
def _np_write_file_camel(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_write_file_camel operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

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
