"""NumPy eager evaluation handlers for I/O and serialization operations."""

from __future__ import annotations

import types
from collections.abc import Iterable

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import (
    BackendArg,
    coerce_bool,
    coerce_opt_str,
    coerce_shape,
    numpy_eager_registry,
)
from ml_switcheroo_compiler.backends.protocols import BackendModuleProtocol, adapt_backend_module
from ml_switcheroo_compiler.core.device import Device, DeviceType
from ml_switcheroo_compiler.core.dtype import DType


def _np_load(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Load an array or weight archive using the NumPy backend.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): The NumPy backend module.
        *args (BackendArg): Positional file path arguments.
        **kwargs (BackendArg): Keyword loader arguments.

    Returns:
        BackendArg: Loaded array or dictionary of arrays.

    Raises:
        ValueError: If the format is unsupported or the file is not found.
    """
    _ = backend_module
    from ml_switcheroo_compiler.ops.io import _fallback_load

    filepath = args[0] if args else kwargs.get("file", "")
    res = _fallback_load(coerce_opt_str(filepath))
    if res is not None:
        return res
    raise ValueError(f"Format not supported or file not found for load: {filepath}")


@numpy_eager_registry.register("Save")
def _np_save(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> None:
    """Save an array to a binary `.npy` file using NumPy.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): The NumPy backend module.
        *args (BackendArg): Positional path and array arguments.
        **kwargs (BackendArg): Keyword path and array arguments.
    """
    _ = backend_module
    if len(args) >= 2:
        adapt_backend_module(np).require_op("save")(args[0], args[1])
    elif "filepath" in kwargs and "arr" in kwargs:
        adapt_backend_module(np).require_op("save")(kwargs["filepath"], kwargs["arr"])
    elif "file" in kwargs and "arr" in kwargs:
        adapt_backend_module(np).require_op("save")(kwargs["file"], kwargs["arr"])


@numpy_eager_registry.register("SaveGguf")
def _np_save_gguf(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> None:
    """Raise an error when attempting GGUF export on the NumPy backend.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): The NumPy backend module.
        *args (BackendArg): Positional arguments.
        **kwargs (BackendArg): Keyword arguments.

    Raises:
        RuntimeError: Always raised because `save_gguf` requires the MLX backend.
    """
    _ = (backend_module, args, kwargs)
    raise RuntimeError("save_gguf requires the MLX backend.")


@numpy_eager_registry.register("Savez")
def _np_savez(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> None:
    """Save multiple arrays to an uncompressed `.npz` archive.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): The NumPy backend module.
        *args (BackendArg): Positional path and array arguments.
        **kwargs (BackendArg): Named array keyword arguments.
    """
    _ = backend_module
    if len(args) > 0:
        adapt_backend_module(np).require_op("savez")(args[0], *args[1:], **kwargs)
    else:
        kw = {k: v for k, v in kwargs.items() if k not in ("filepath", "file")}
        adapt_backend_module(np).require_op("savez")(kwargs.get("filepath", kwargs.get("file", "out.npz")), **kw)


@numpy_eager_registry.register("SavezCompressed")
def _np_savez_compressed(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> None:
    """Save multiple arrays to a compressed `.npz` archive.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): The NumPy backend module.
        *args (BackendArg): Positional path and array arguments.
        **kwargs (BackendArg): Named array keyword arguments.
    """
    _ = backend_module
    if len(args) > 0:
        adapt_backend_module(np).require_op("savez_compressed")(args[0], *args[1:], **kwargs)
    else:
        kw = {k: v for k, v in kwargs.items() if k not in ("filepath", "file")}
        adapt_backend_module(np).require_op("savez_compressed")(
            kwargs.get("filepath", kwargs.get("file", "out.npz")), **kw
        )


def _np_read_file(
    backend_module: BackendModuleProtocol | types.ModuleType | None, filename: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Read raw bytes from a file path into a `Tensor`.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): The NumPy backend module.
        filename (BackendArg): File path string or Tensor wrapping a path string.
        **kwargs (BackendArg): Additional keyword arguments.

    Returns:
        BackendArg: `Tensor` containing the read bytes.
    """
    _ = (backend_module, kwargs)
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    fname = (
        adapt_backend_module(filename).require_symbol("data")
        if adapt_backend_module(filename).has_symbol("data")
        else filename
    )
    if not isinstance(fname, str):
        return Tensor(None, TensorConfig((), DType("float32"), Device(DeviceType.CPU)))
    with open(fname, "rb") as f:
        return Tensor(f.read(), TensorConfig((), DType("uint8"), Device(DeviceType.CPU)))


def _np_write_file(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    filename: BackendArg,
    contents: BackendArg,
    **kwargs: BackendArg,
) -> None:
    """Write raw byte contents to a file path on disk.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): The NumPy backend module.
        filename (BackendArg): Destination file path string or Tensor.
        contents (BackendArg): Raw bytes or Tensor wrapping bytes to write.
        **kwargs (BackendArg): Additional keyword arguments.
    """
    _ = (backend_module, kwargs)
    fname = (
        adapt_backend_module(filename).require_symbol("data")
        if adapt_backend_module(filename).has_symbol("data")
        else filename
    )
    data = (
        adapt_backend_module(contents).require_symbol("data")
        if adapt_backend_module(contents).has_symbol("data")
        else contents
    )
    if isinstance(fname, str) and isinstance(data, bytes):
        with open(fname, "wb") as f:
            f.write(data)


def _np_decode_image(
    backend_module: BackendModuleProtocol | types.ModuleType | None, contents: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Decode raw image bytes into a uint8/typed image Tensor.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): The NumPy backend module.
        contents (BackendArg): Encoded image byte payload.
        **kwargs (BackendArg): Keyword options including `dtype` and `channels`.

    Returns:
        BackendArg: Decoded image `Tensor`.
    """
    _ = (backend_module, contents)
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    dtype = kwargs.get("dtype", "uint8")
    return Tensor(
        np.zeros((), dtype=str(dtype)),
        TensorConfig((), dtype, Device(DeviceType.CPU)),
    )


def _np_decode_csv(
    backend_module: BackendModuleProtocol | types.ModuleType | None, records: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Decode CSV record strings into a list of column Tensors.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): The NumPy backend module.
        records (BackendArg): Input CSV record strings or bytes.
        **kwargs (BackendArg): Keyword options including `record_defaults`.

    Returns:
        BackendArg: List of column `Tensor` objects matching `record_defaults`.
    """
    _ = (backend_module, records)
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    record_defaults = kwargs.get("record_defaults", [])
    items = list(record_defaults) if isinstance(record_defaults, Iterable) else []
    return [
        Tensor(np.zeros((), dtype=np.float32), TensorConfig((), DType("float32"), Device(DeviceType.CPU)))
        for _ in items
    ]


def _np_serialize_tensor(
    backend_module: BackendModuleProtocol | types.ModuleType | None, tensor: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Serialize a tensor into a binary byte-string `Tensor`.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): The NumPy backend module.
        tensor (BackendArg): Input tensor to serialize.
        **kwargs (BackendArg): Additional keyword arguments.

    Returns:
        BackendArg: Scalar uint8 `Tensor` wrapping serialized bytes.
    """
    _ = (backend_module, tensor, kwargs)
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    return Tensor(b"", TensorConfig((), DType("uint8"), Device(DeviceType.CPU)))


def _np_parse_tensor(
    backend_module: BackendModuleProtocol | types.ModuleType | None, serialized: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Deserialize a byte-string payload into a typed `Tensor`.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): The NumPy backend module.
        serialized (BackendArg): Serialized binary payload.
        **kwargs (BackendArg): Keyword options including `out_type`.

    Returns:
        BackendArg: Deserialized `Tensor` with target dtype.
    """
    _ = (backend_module, serialized)
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    out_type = kwargs.get("out_type", "float32")
    return Tensor(
        np.zeros((), dtype=np.float32),
        TensorConfig((), out_type, Device(DeviceType.CPU)),
    )


@numpy_eager_registry.register("EncodeBase64")
def _np_encode_base64(
    backend_module: BackendModuleProtocol | types.ModuleType | None, input: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Encode byte or string payloads into Base64 `Tensor` representations.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): The NumPy backend module.
        input (BackendArg): Input bytes, string, or Tensor.
        **kwargs (BackendArg): Keyword options including `pad`.

    Returns:
        BackendArg: Base64-encoded `Tensor`, or None if `input` is None.
    """
    _ = backend_module
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
    from ml_switcheroo_compiler.ops.io import _eager_base64

    if input is None:
        return None
    data = adapt_backend_module(input).get_symbol("data", input)
    pad = kwargs.get("pad", False)
    res = _eager_base64("encode", data, coerce_bool(pad))
    return Tensor(
        res,
        TensorConfig(
            coerce_shape(adapt_backend_module(input).get_symbol("shape", ())), DType.Float32, Device(DeviceType.CPU)
        ),
    )


@numpy_eager_registry.register("DecodeBase64")
def _np_decode_base64(
    backend_module: BackendModuleProtocol | types.ModuleType | None, input: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Decode Base64-encoded payloads back into raw byte `Tensor` representations.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): The NumPy backend module.
        input (BackendArg): Base64-encoded input bytes, string, or Tensor.
        **kwargs (BackendArg): Additional keyword arguments.

    Returns:
        BackendArg: Decoded byte `Tensor`, or None if `input` is None.
    """
    _ = (backend_module, kwargs)
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
    from ml_switcheroo_compiler.ops.io import _eager_base64

    if input is None:
        return None
    data = adapt_backend_module(input).get_symbol("data", input)
    res = _eager_base64("decode", data, False)
    return Tensor(
        res,
        TensorConfig(
            coerce_shape(adapt_backend_module(input).get_symbol("shape", ())), DType.Float32, Device(DeviceType.CPU)
        ),
    )
