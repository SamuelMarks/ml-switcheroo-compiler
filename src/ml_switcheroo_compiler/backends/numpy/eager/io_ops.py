# ruff: noqa: E402, D100, D103, D104, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, D101, D102, D107, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Numpy I/O operations."""

from typing import Any

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry
from ml_switcheroo_compiler.core.device import Device, DeviceType
from ml_switcheroo_compiler.core.dtype import DType


@numpy_eager_registry.register("Load")
def _np_load(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_load operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.

    Raises:
        ValueError: An exception.
    """
    from ml_switcheroo_compiler.ops.io import _fallback_load

    filepath = args[0] if args else kwargs.get("file", "")
    res = _fallback_load(filepath)
    if res is not None:
        return res
    raise ValueError(f"Format not supported or file not found for load: {filepath}")


@numpy_eager_registry.register("Save")
def _np_save(backend_module: Any, *args: Any, **kwargs: Any) -> None:
    """Evaluate _np_save operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.
    """
    if len(args) >= 2:
        np.save(args[0], args[1])
    elif "filepath" in kwargs and "arr" in kwargs:
        np.save(kwargs["filepath"], kwargs["arr"])
    elif "file" in kwargs and "arr" in kwargs:
        np.save(kwargs["file"], kwargs["arr"])


@numpy_eager_registry.register("SaveGguf")
def _np_save_gguf(backend_module: Any, *args: Any, **kwargs: Any) -> None:
    """Evaluate _np_save_gguf operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Raises:
        RuntimeError: An exception.
    """
    raise RuntimeError("save_gguf requires the MLX backend.")


@numpy_eager_registry.register("Savez")
def _np_savez(backend_module: Any, *args: Any, **kwargs: Any) -> None:
    """Evaluate _np_savez operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.
    """
    if len(args) > 0:
        np.savez(args[0], *args[1:], **kwargs)
    else:
        kw = {k: v for k, v in kwargs.items() if k not in ("filepath", "file")}
        np.savez(kwargs.get("filepath", kwargs.get("file", "out.npz")), **kw)


@numpy_eager_registry.register("SavezCompressed")
def _np_savez_compressed(backend_module: Any, *args: Any, **kwargs: Any) -> None:
    """Evaluate _np_savez_compressed operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.
    """
    if len(args) > 0:
        np.savez_compressed(args[0], *args[1:], **kwargs)
    else:
        kw = {k: v for k, v in kwargs.items() if k not in ("filepath", "file")}
        np.savez_compressed(kwargs.get("filepath", kwargs.get("file", "out.npz")), **kw)


@numpy_eager_registry.register("ReadFile")
def _np_read_file(backend_module: Any, filename: Any, **kwargs: Any) -> Any:
    """Evaluate _np_read_file operation.

    Args:
        backend_module (object): The backend_module parameter.
        filename (object): The filename parameter.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    fname = filename.data if hasattr(filename, "data") else filename
    if not isinstance(fname, str):
        return Tensor(None, TensorConfig((), DType("float32"), Device(DeviceType.CPU)))
    with open(fname, "rb") as f:
        return Tensor(f.read(), TensorConfig((), DType("uint8"), Device(DeviceType.CPU)))


@numpy_eager_registry.register("WriteFile")
def _np_write_file(backend_module: Any, filename: Any, contents: Any, **kwargs: Any) -> None:
    """Evaluate _np_write_file operation.

    Args:
        backend_module (object): The backend_module parameter.
        filename (object): The filename parameter.
        contents (object): The contents parameter.
        **kwargs (object): Keyword args.
    """
    fname = filename.data if hasattr(filename, "data") else filename
    data = contents.data if hasattr(contents, "data") else contents
    if isinstance(fname, str) and isinstance(data, bytes):
        with open(fname, "wb") as f:
            f.write(data)


@numpy_eager_registry.register("DecodeImage")
def _np_decode_image(backend_module: Any, contents: Any, **kwargs: Any) -> Any:
    """Evaluate _np_decode_image operation.

    Args:
        backend_module (object): The backend_module parameter.
        contents (object): The contents parameter.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    dtype = kwargs.get("dtype", "uint8")
    return Tensor(None, TensorConfig((), dtype, Device(DeviceType.CPU)))


@numpy_eager_registry.register("DecodeCsv")
def _np_decode_csv(backend_module: Any, records: Any, **kwargs: Any) -> list:
    """Evaluate _np_decode_csv operation.

    Args:
        backend_module (object): The backend_module parameter.
        records (object): The records parameter.
        **kwargs (object): Keyword args.

    Returns:
        list: Result.
    """
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    record_defaults = kwargs.get("record_defaults", [])
    return [Tensor(None, TensorConfig((), DType("float32"), Device(DeviceType.CPU))) for _ in record_defaults]


@numpy_eager_registry.register("ParseExample")
def _np_parse_example(backend_module: Any, serialized: Any, **kwargs: Any) -> dict:
    """Evaluate _np_parse_example operation.

    Args:
        backend_module (object): The backend_module parameter.
        serialized (object): The serialized parameter.
        **kwargs (object): Keyword args.

    Returns:
        dict: Result.
    """
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    features = kwargs.get("features", {})
    return {k: Tensor(None, TensorConfig((), DType("float32"), Device(DeviceType.CPU))) for k in features}


@numpy_eager_registry.register("SerializeTensor")
def _np_serialize_tensor(backend_module: Any, tensor: Any, **kwargs: Any) -> Any:
    """Evaluate _np_serialize_tensor operation.

    Args:
        backend_module (object): The backend_module parameter.
        tensor (object): The tensor parameter.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    return Tensor(b"", TensorConfig((), DType("uint8"), Device(DeviceType.CPU)))


@numpy_eager_registry.register("ParseTensor")
def _np_parse_tensor(backend_module: Any, serialized: Any, **kwargs: Any) -> Any:
    """Evaluate _np_parse_tensor operation.

    Args:
        backend_module (object): The backend_module parameter.
        serialized (object): The serialized parameter.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    out_type = kwargs.get("out_type", "float32")
    return Tensor(None, TensorConfig((), out_type, Device(DeviceType.CPU)))


@numpy_eager_registry.register("EncodeBase64")
def _np_encode_base64(backend_module: Any, input: Any, **kwargs: Any) -> Any:
    """Evaluate _np_encode_base64 operation.

    Args:
        backend_module (object): The backend_module parameter.
        input (object): The input parameter.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
    from ml_switcheroo_compiler.ops.io import _eager_base64

    if input is None:
        return None
    data = getattr(input, "data", input)
    pad = kwargs.get("pad", False)
    res = _eager_base64("encode", data, pad)
    return Tensor(res, TensorConfig(getattr(input, "shape", ()), DType.Float32, Device(DeviceType.CPU)))


@numpy_eager_registry.register("DecodeBase64")
def _np_decode_base64(backend_module: Any, input: Any, **kwargs: Any) -> Any:
    """Evaluate _np_decode_base64 operation.

    Args:
        backend_module (object): The backend_module parameter.
        input (object): The input parameter.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
    from ml_switcheroo_compiler.ops.io import _eager_base64

    if input is None:
        return None
    data = getattr(input, "data", input)
    res = _eager_base64("decode", data, False)
    return Tensor(res, TensorConfig(getattr(input, "shape", ()), DType.Float32, Device(DeviceType.CPU)))


@numpy_eager_registry.register("ParseSequenceExample")
def _np_parse_sequence_example(backend_module: Any, serialized: Any, **kwargs: Any) -> tuple:
    """Evaluate _np_parse_sequence_example operation.

    Args:
        backend_module (object): The backend_module parameter.
        serialized (object): The serialized parameter.
        **kwargs (object): Keyword args.

    Returns:
        tuple: Result.
    """
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    if serialized is None:
        return ({}, {})
    return ({"dummy": Tensor(None, TensorConfig((), DType.Float32, Device(DeviceType.CPU)))}, {"dummy": Tensor(None, TensorConfig((), DType.Float32, Device(DeviceType.CPU)))})
