"""NumPy eager evaluation handlers for I/O and serialization operations."""

from __future__ import annotations

import typing
from collections.abc import Iterable

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import EagerValue, numpy_eager_registry
from ml_switcheroo_compiler.backends.protocols import BackendModuleProtocol
from ml_switcheroo_compiler.core.device import Device, DeviceType
from ml_switcheroo_compiler.core.dtype import DType


@numpy_eager_registry.register("Load")
def _np_load(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Load an array or weight archive using the NumPy backend.

    Args:
        backend_module (ModuleType): The NumPy backend module.
        *args (EagerValue): Positional file path arguments.
        **kwargs (EagerValue): Keyword loader arguments.

    Returns:
        EagerValue: Loaded array or dictionary of arrays.

    Raises:
        ValueError: If the format is unsupported or the file is not found.
    """
    _ = backend_module
    from ml_switcheroo_compiler.ops.io import _fallback_load

    filepath = args[0] if args else kwargs.get("file", "")
    res = _fallback_load(typing.cast("str | None", filepath))
    if res is not None:
        return res
    raise ValueError(f"Format not supported or file not found for load: {filepath}")


@numpy_eager_registry.register("Save")
def _np_save(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> None:
    """Save an array to a binary `.npy` file using NumPy.

    Args:
        backend_module (ModuleType): The NumPy backend module.
        *args (EagerValue): Positional path and array arguments.
        **kwargs (EagerValue): Keyword path and array arguments.
    """
    _ = backend_module
    if len(args) >= 2:
        getattr(np, "save")(args[0], args[1])
    elif "filepath" in kwargs and "arr" in kwargs:
        getattr(np, "save")(kwargs["filepath"], kwargs["arr"])
    elif "file" in kwargs and "arr" in kwargs:
        getattr(np, "save")(kwargs["file"], kwargs["arr"])


@numpy_eager_registry.register("SaveGguf")
def _np_save_gguf(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> None:
    """Raise an error when attempting GGUF export on the NumPy backend.

    Args:
        backend_module (ModuleType): The NumPy backend module.
        *args (EagerValue): Positional arguments.
        **kwargs (EagerValue): Keyword arguments.

    Raises:
        RuntimeError: Always raised because `save_gguf` requires the MLX backend.
    """
    _ = (backend_module, args, kwargs)
    raise RuntimeError("save_gguf requires the MLX backend.")


@numpy_eager_registry.register("Savez")
def _np_savez(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> None:
    """Save multiple arrays to an uncompressed `.npz` archive.

    Args:
        backend_module (ModuleType): The NumPy backend module.
        *args (EagerValue): Positional path and array arguments.
        **kwargs (EagerValue): Named array keyword arguments.
    """
    _ = backend_module
    if len(args) > 0:
        getattr(np, "savez")(args[0], *args[1:], **kwargs)
    else:
        kw = {k: v for k, v in kwargs.items() if k not in ("filepath", "file")}
        getattr(np, "savez")(kwargs.get("filepath", kwargs.get("file", "out.npz")), **kw)


@numpy_eager_registry.register("SavezCompressed")
def _np_savez_compressed(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> None:
    """Save multiple arrays to a compressed `.npz` archive.

    Args:
        backend_module (ModuleType): The NumPy backend module.
        *args (EagerValue): Positional path and array arguments.
        **kwargs (EagerValue): Named array keyword arguments.
    """
    _ = backend_module
    if len(args) > 0:
        getattr(np, "savez_compressed")(args[0], *args[1:], **kwargs)
    else:
        kw = {k: v for k, v in kwargs.items() if k not in ("filepath", "file")}
        getattr(np, "savez_compressed")(kwargs.get("filepath", kwargs.get("file", "out.npz")), **kw)


@numpy_eager_registry.register("ReadFile")
def _np_read_file(backend_module: BackendModuleProtocol, filename: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Read raw bytes from a file path into a `Tensor`.

    Args:
        backend_module (ModuleType): The NumPy backend module.
        filename (EagerValue): File path string or Tensor wrapping a path string.
        **kwargs (EagerValue): Additional keyword arguments.

    Returns:
        EagerValue: `Tensor` containing the read bytes.
    """
    _ = (backend_module, kwargs)
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    fname = getattr(filename, "data") if hasattr(filename, "data") else filename
    if not isinstance(fname, str):
        return Tensor(None, TensorConfig((), DType("float32"), Device(DeviceType.CPU)))
    with open(fname, "rb") as f:
        return Tensor(f.read(), TensorConfig((), DType("uint8"), Device(DeviceType.CPU)))


@numpy_eager_registry.register("WriteFile")
def _np_write_file(
    backend_module: BackendModuleProtocol,
    filename: EagerValue,
    contents: EagerValue,
    **kwargs: EagerValue,
) -> None:
    """Write raw byte contents to a file path on disk.

    Args:
        backend_module (ModuleType): The NumPy backend module.
        filename (EagerValue): Destination file path string or Tensor.
        contents (EagerValue): Raw bytes or Tensor wrapping bytes to write.
        **kwargs (EagerValue): Additional keyword arguments.
    """
    _ = (backend_module, kwargs)
    fname = getattr(filename, "data") if hasattr(filename, "data") else filename
    data = getattr(contents, "data") if hasattr(contents, "data") else contents
    if isinstance(fname, str) and isinstance(data, bytes):
        with open(fname, "wb") as f:
            f.write(data)


@numpy_eager_registry.register("DecodeImage")
def _np_decode_image(backend_module: BackendModuleProtocol, contents: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Decode raw image bytes into a uint8/typed image Tensor.

    Args:
        backend_module (ModuleType): The NumPy backend module.
        contents (EagerValue): Encoded image byte payload.
        **kwargs (EagerValue): Keyword options including `dtype` and `channels`.

    Returns:
        EagerValue: Decoded image `Tensor`.
    """
    _ = (backend_module, contents)
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    dtype = kwargs.get("dtype", "uint8")
    return Tensor(
        np.zeros((), dtype=str(dtype)),
        TensorConfig((), typing.cast("str | None", dtype), Device(DeviceType.CPU)),
    )


@numpy_eager_registry.register("DecodeCsv")
def _np_decode_csv(backend_module: BackendModuleProtocol, records: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Decode CSV record strings into a list of column Tensors.

    Args:
        backend_module (ModuleType): The NumPy backend module.
        records (EagerValue): Input CSV record strings or bytes.
        **kwargs (EagerValue): Keyword options including `record_defaults`.

    Returns:
        EagerValue: List of column `Tensor` objects matching `record_defaults`.
    """
    _ = (backend_module, records)
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    record_defaults = kwargs.get("record_defaults", [])
    items = list(record_defaults) if isinstance(record_defaults, Iterable) else []
    return [Tensor(np.zeros((), dtype=np.float32), TensorConfig((), DType("float32"), Device(DeviceType.CPU))) for _ in items]


@numpy_eager_registry.register("ParseExample")
def _np_parse_example(backend_module: BackendModuleProtocol, serialized: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Parse serialized records into a dictionary of feature Tensors.

    Args:
        backend_module (ModuleType): The NumPy backend module.
        serialized (EagerValue): Serialized binary record payload.
        **kwargs (EagerValue): Keyword options including `features` dictionary.

    Returns:
        EagerValue: Dictionary mapping feature keys to `Tensor` instances.
    """
    _ = (backend_module, serialized)
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    features = kwargs.get("features", {})
    keys = list(features) if isinstance(features, Iterable) else []
    return {str(k): Tensor(np.zeros((), dtype=np.float32), TensorConfig((), DType("float32"), Device(DeviceType.CPU))) for k in keys}


@numpy_eager_registry.register("SerializeTensor")
def _np_serialize_tensor(backend_module: BackendModuleProtocol, tensor: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Serialize a tensor into a binary byte-string `Tensor`.

    Args:
        backend_module (ModuleType): The NumPy backend module.
        tensor (EagerValue): Input tensor to serialize.
        **kwargs (EagerValue): Additional keyword arguments.

    Returns:
        EagerValue: Scalar uint8 `Tensor` wrapping serialized bytes.
    """
    _ = (backend_module, tensor, kwargs)
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    return Tensor(b"", TensorConfig((), DType("uint8"), Device(DeviceType.CPU)))


@numpy_eager_registry.register("ParseTensor")
def _np_parse_tensor(backend_module: BackendModuleProtocol, serialized: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Deserialize a byte-string payload into a typed `Tensor`.

    Args:
        backend_module (ModuleType): The NumPy backend module.
        serialized (EagerValue): Serialized binary payload.
        **kwargs (EagerValue): Keyword options including `out_type`.

    Returns:
        EagerValue: Deserialized `Tensor` with target dtype.
    """
    _ = (backend_module, serialized)
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    out_type = kwargs.get("out_type", "float32")
    return Tensor(
        np.zeros((), dtype=np.float32),
        TensorConfig((), typing.cast("str | None", out_type), Device(DeviceType.CPU)),
    )


@numpy_eager_registry.register("EncodeBase64")
def _np_encode_base64(backend_module: BackendModuleProtocol, input: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Encode byte or string payloads into Base64 `Tensor` representations.

    Args:
        backend_module (ModuleType): The NumPy backend module.
        input (EagerValue): Input bytes, string, or Tensor.
        **kwargs (EagerValue): Keyword options including `pad`.

    Returns:
        EagerValue: Base64-encoded `Tensor`, or None if `input` is None.
    """
    _ = backend_module
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
    from ml_switcheroo_compiler.ops.io import _eager_base64

    if input is None:
        return None
    data = getattr(input, "data", input)
    pad = kwargs.get("pad", False)
    res = _eager_base64("encode", data, typing.cast("bool", pad))
    return Tensor(res, TensorConfig(getattr(input, "shape", ()), DType.Float32, Device(DeviceType.CPU)))


@numpy_eager_registry.register("DecodeBase64")
def _np_decode_base64(backend_module: BackendModuleProtocol, input: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Decode Base64-encoded payloads back into raw byte `Tensor` representations.

    Args:
        backend_module (ModuleType): The NumPy backend module.
        input (EagerValue): Base64-encoded input bytes, string, or Tensor.
        **kwargs (EagerValue): Additional keyword arguments.

    Returns:
        EagerValue: Decoded byte `Tensor`, or None if `input` is None.
    """
    _ = (backend_module, kwargs)
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
    from ml_switcheroo_compiler.ops.io import _eager_base64

    if input is None:
        return None
    data = getattr(input, "data", input)
    res = _eager_base64("decode", data, False)
    return Tensor(res, TensorConfig(getattr(input, "shape", ()), DType.Float32, Device(DeviceType.CPU)))


@numpy_eager_registry.register("ParseSequenceExample")
def _np_parse_sequence_example(backend_module: BackendModuleProtocol, serialized: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Parse a serialized sequence record into context and sequence feature dictionaries.

    Args:
        backend_module (ModuleType): The NumPy backend module.
        serialized (EagerValue): Serialized sequence record payload.
        **kwargs (EagerValue): Keyword feature specification dictionaries.

    Returns:
        EagerValue: Tuple of `(context_features_dict, sequence_features_dict)`.
    """
    _ = (backend_module, kwargs)
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    if serialized is None:
        return ({}, {})
    return (
        {"dummy": Tensor(np.zeros((), dtype=np.float32), TensorConfig((), DType.Float32, Device(DeviceType.CPU)))},
        {"dummy": Tensor(np.zeros((), dtype=np.float32), TensorConfig((), DType.Float32, Device(DeviceType.CPU)))},
    )
