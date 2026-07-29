# ruff: noqa: E402
"""I/O and memory operations."""

from __future__ import annotations

from typing import Any

from ml_switcheroo_compiler.core.config import config as core_config
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.serialization.formats.h5 import H5WeightFormat
from ml_switcheroo_compiler.serialization.formats.safetensors import SafetensorsWeightFormat
from ml_switcheroo_compiler.serialization.utils import load_npz


def _fallback_load(filepath: object) -> object:
    if not isinstance(filepath, str):
        return None
    if filepath.endswith(".safetensors"):
        return SafetensorsWeightFormat().load(filepath)
    if filepath.endswith(".npz"):
        return load_npz(filepath)
    if filepath.endswith(".h5"):
        return H5WeightFormat().load(filepath)
    return None


def load(*args: object, **kwargs: object) -> object:
    """Load arrays or pickled objects from files."""
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("Load", *args, **kwargs)

    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    _first = args[0] if args else (list(kwargs.values())[0] if kwargs else None)
    return _emit_shape_node("Load", list(args), kwargs, getattr(_first, "shape", ()), getattr(_first, "dtype", "float32"))


def save(*args: object, **kwargs: object) -> None:
    """Save.

    Args:
        *args (object): The args.
        **kwargs (object): The kwargs.
    """
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("Save", *args, **kwargs)

    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("Save", list(args), kwargs, (), "float32")


def save_gguf(*args: object, **kwargs: object) -> None:
    """Save gguf.

    Args:
        *args (object): The args.
        **kwargs (object): The kwargs.
    """
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("SaveGguf", *args, **kwargs)

    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("SaveGguf", list(args), kwargs, (), "float32")


def save_safetensors(file: str, arrays: dict[str, object]) -> None:
    """Save a dictionary of arrays to Safetensors format."""
    SafetensorsWeightFormat().save(arrays, file)


def savez(*args: object, **kwargs: object) -> None:
    """Savez.

    Args:
        *args (object): The args.
        **kwargs (object): The kwargs.
    """
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("Savez", *args, **kwargs)

    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("Savez", list(args), kwargs, (), "float32")


def savez_compressed(*args: object, **kwargs: object) -> None:
    """Savez compressed.

    Args:
        *args (object): The args.
        **kwargs (object): The kwargs.
    """
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("SavezCompressed", *args, **kwargs)

    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("SavezCompressed", list(args), kwargs, (), "float32")


def set_default_stream(stream: object) -> None:
    """Set the default stream for the active backend."""
    if core_config.backend == "mlx":
        try:
            from ml_switcheroo_compiler.backends.registry import get_active_backend

            backend_cls = get_active_backend()
            if hasattr(backend_cls, "set_default_stream"):
                backend_cls.set_default_stream(stream)
        except ImportError:
            return


def set_memory_limit(limit: int) -> None:
    """Set the memory limit for the active backend."""
    if core_config.backend == "mlx":
        try:
            from ml_switcheroo_compiler.backends.registry import get_active_backend

            backend_cls = get_active_backend()
            if hasattr(backend_cls, "set_memory_limit"):
                backend_cls.set_memory_limit(limit)
        except ImportError:
            return


def set_wired_limit(limit: int) -> None:
    """Set the wired memory limit for the active backend."""
    if core_config.backend == "mlx":
        try:
            from ml_switcheroo_compiler.backends.registry import get_active_backend

            backend_cls = get_active_backend()
            if hasattr(backend_cls, "set_wired_limit"):
                backend_cls.set_wired_limit(limit)
        except ImportError:
            return


__all__ = [
    "sparse_plus",
    "sparse_sigmoid",
    "decode_csv",
    "decode_image",
    "decode_jpeg",
    "decode_png",
    "decode_gif",
    "decode_bmp",
    "encode_base64",
    "decode_base64",
    "gfile_copy",
    "gfile_glob",
    "gfile_makedirs",
    "gfile_stat",
    "load",
    "parse_example",
    "parse_sequence_example",
    "parse_tensor",
    "read_file",
    "save",
    "save_gguf",
    "save_safetensors",
    "savez",
    "savez_compressed",
    "serialize_tensor",
    "set_default_stream",
    "set_memory_limit",
    "set_wired_limit",
    "write_file",
    "TFRecordOptions",
    "TFRecordWriter",
]


def read_file(filename: str | Tensor, name: str = None) -> Tensor:
    """Read file.

    Args:
        filename (str | Tensor): The filename.
        name (str, optional): The name.

    Returns:
        Tensor: The file content.
    """
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("ReadFile", filename, name=name)

    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("ReadFile", [filename], {"name": name}, getattr(filename, "shape", ()), getattr(filename, "dtype", "float32"))


def write_file(filename: str | Tensor, contents: Tensor, name: str = None) -> None:
    """Write file.

    Args:
        filename (str | Tensor): The filename.
        contents (Tensor): The contents.
        name (str, optional): The name.
    """
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("WriteFile", filename, contents, name=name)

    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("WriteFile", [filename, contents], {"name": name}, (), "float32")


def decode_image(contents: Tensor, channels: int = 0, dtype: DType = DType.UInt8, name: str = None, expand_animations: bool = True) -> Tensor:
    """Decode image.

    Args:
        contents (Tensor): The contents.
        channels (int, optional): The channels.
        dtype (DType, optional): The dtype.
        name (str, optional): The name.
        expand_animations (bool, optional): Expand animations.

    Returns:
        Tensor: The decoded image.
    """
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("DecodeImage", contents, channels=channels, dtype=dtype, name=name, expand_animations=expand_animations)

    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("DecodeImage", [contents], {"channels": channels, "dtype": dtype, "name": name, "expand_animations": expand_animations}, getattr(contents, "shape", ()), getattr(contents, "dtype", "float32"))


def decode_csv(records: Tensor, record_defaults: list[Any], field_delim: str = ",", use_quote_delim: bool = True, na_value: str = "", select_cols: list[int] = None, name: str = None) -> list[Tensor]:
    """Decode csv.

    Args:
        records (Tensor): The records.
        record_defaults (list[Any]): The record defaults.
        field_delim (str, optional): The field delim.
        use_quote_delim (bool, optional): Use quote delim.
        na_value (str, optional): Na value.
        select_cols (list[int], optional): Select cols.
        name (str, optional): The name.

    Returns:
        list[Tensor]: The decoded csv.
    """
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("DecodeCsv", records, record_defaults=record_defaults, field_delim=field_delim, use_quote_delim=use_quote_delim, na_value=na_value, select_cols=select_cols, name=name)

    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("DecodeCsv", [records], {"record_defaults": record_defaults, "field_delim": field_delim, "use_quote_delim": use_quote_delim, "na_value": na_value, "select_cols": select_cols, "name": name}, getattr(records, "shape", ()), getattr(records, "dtype", "float32"))


def parse_example(serialized: Tensor, features: dict[str, Any], example_names: Tensor = None, name: str = None) -> dict[str, Tensor]:
    """Parse example.

    Args:
        serialized (Tensor): The serialized example.
        features (dict[str, Any]): The features.
        example_names (Tensor, optional): The example names.
        name (str, optional): The name.

    Returns:
        dict[str, Tensor]: The parsed example.
    """
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("ParseExample", serialized, features=features, example_names=example_names, name=name)

    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("ParseExample", [serialized], {"features": features, "example_names": example_names, "name": name}, getattr(serialized, "shape", ()), getattr(serialized, "dtype", "float32"))


def serialize_tensor(tensor: Tensor, name: str = None) -> Tensor:
    """Serialize tensor.

    Args:
        tensor (Tensor): The tensor.
        name (str, optional): The name.

    Returns:
        Tensor: The serialized tensor.
    """
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("SerializeTensor", tensor, name=name)

    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("SerializeTensor", [tensor], {"name": name}, getattr(tensor, "shape", ()), getattr(tensor, "dtype", "float32"))


def parse_tensor(serialized: Tensor, out_type: DType, name: str = None) -> Tensor:
    """Parse tensor.

    Args:
        serialized (Tensor): The serialized tensor.
        out_type (DType): The out type.
        name (str, optional): The name.

    Returns:
        Tensor: The parsed tensor.
    """
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("ParseTensor", serialized, out_type=out_type, name=name)

    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("ParseTensor", [serialized], {"out_type": out_type, "name": name}, getattr(serialized, "shape", ()), getattr(serialized, "dtype", "float32"))


import glob
import os
import shutil


def gfile_copy(src: str, dst: str, overwrite: bool = False) -> None:
    """Copy a file."""
    if os.path.exists(dst) and not overwrite:
        raise FileExistsError(f"File {dst} already exists")
    shutil.copy2(src, dst)


def gfile_glob(pattern: str) -> list[str]:
    """Glob pattern."""
    return glob.glob(pattern)


def gfile_stat(path: str) -> dict[str, int]:
    """Stat a file."""
    st = os.stat(path)
    return {"length": st.st_size, "mtime": int(st.st_mtime)}


def gfile_makedirs(path: str) -> None:
    """Make directories."""
    os.makedirs(path, exist_ok=True)


def decode_jpeg(contents: object, channels: int = 0, ratio: int = 1) -> object:
    """Decode JPEG image."""
    from ml_switcheroo_compiler.backends.registry import get_active_backend

    backend_cls = get_active_backend()
    if hasattr(backend_cls, "decode_jpeg"):
        return backend_cls.decode_jpeg(contents, channels=channels, ratio=ratio)
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    return Tensor(None, TensorConfig((), None, None))


def decode_png(contents: object, channels: int = 0, dtype: object = None) -> object:
    """Decode PNG image."""
    from ml_switcheroo_compiler.backends.registry import get_active_backend

    backend_cls = get_active_backend()
    if hasattr(backend_cls, "decode_png"):
        return backend_cls.decode_png(contents, channels=channels, dtype=dtype)
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    return Tensor(None, TensorConfig((), None, None))


def decode_gif(
    contents: object,
) -> object:
    """Decode GIF image."""
    from ml_switcheroo_compiler.backends.registry import get_active_backend

    backend_cls = get_active_backend()
    if hasattr(backend_cls, "decode_gif"):
        return backend_cls.decode_gif(contents)
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    return Tensor(None, TensorConfig((), None, None))


def decode_bmp(contents: object, channels: int = 0) -> object:
    """Decode BMP image."""
    from ml_switcheroo_compiler.backends.registry import get_active_backend

    backend_cls = get_active_backend()
    if hasattr(backend_cls, "decode_bmp"):
        return backend_cls.decode_bmp(contents, channels=channels)
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    return Tensor(None, TensorConfig((), None, None))


def _eager_base64(op: str, data: object, pad: bool = False) -> object:
    import base64

    def _proc(d: object) -> bytes:
        if d is None:
            return b""
        b = d.encode("utf-8") if isinstance(d, str) else d
        res = base64.b64encode(b) if op == "encode" else base64.b64decode(b)
        if op == "encode" and not pad:
            res = res.rstrip(b"=")
        return res

    if isinstance(data, (list, tuple)):
        return [_proc(d) for d in data]
    return _proc(data)


def encode_base64(input: Tensor, pad: bool = False, name: str = None) -> Tensor:
    """Encode base64.

    Args:
        input (Tensor): The input.
        pad (bool, optional): Pad.
        name (str, optional): The name.

    Returns:
        Tensor: The encoded base64.
    """
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("EncodeBase64", input, pad=pad, name=name)

    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("EncodeBase64", [input], {"pad": pad, "name": name}, getattr(input, "shape", ()), getattr(input, "dtype", "float32"))


def decode_base64(input: Tensor, name: str = None) -> Tensor:
    """Decode base64.

    Args:
        input (Tensor): The input.
        name (str, optional): The name.

    Returns:
        Tensor: The decoded base64.
    """
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("DecodeBase64", input, name=name)

    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("DecodeBase64", [input], {"name": name}, getattr(input, "shape", ()), getattr(input, "dtype", "float32"))


def parse_sequence_example(serialized: Tensor, context_features: dict[str, Any] = None, sequence_features: dict[str, Any] = None, example_names: Tensor = None, name: str = None) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    """Parse sequence example.

    Args:
        serialized (Tensor): The serialized sequence example.
        context_features (dict[str, Any], optional): The context features.
        sequence_features (dict[str, Any], optional): The sequence features.
        example_names (Tensor, optional): The example names.
        name (str, optional): The name.

    Returns:
        tuple[dict[str, Tensor], dict[str, Tensor]]: The parsed sequence example.
    """
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("ParseSequenceExample", serialized, context_features=context_features, sequence_features=sequence_features, example_names=example_names, name=name)

    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("ParseSequenceExample", [serialized], {"context_features": context_features, "sequence_features": sequence_features, "example_names": example_names, "name": name}, getattr(serialized, "shape", ()), getattr(serialized, "dtype", "float32"))


class TFRecordOptions:
    """Options for TFRecordWriter."""

    def __init__(self, compression_type: str = "") -> None:
        """Initialize.

        Args:
            compression_type (str, optional): The compression type.
        """
        self.compression_type = compression_type


class TFRecordWriter:
    """Writer for TFRecord format."""

    def __init__(self, path: str, options: TFRecordOptions = None) -> None:
        """Initialize.

        Args:
            path (str): The path.
            options (TFRecordOptions, optional): The options.
        """
        self.path = path
        self.options = options

    def write(self, record: object) -> None:
        """Write record."""
        return None

    def close(self) -> None:
        """Close."""
        return None

    def __enter__(self) -> TFRecordWriter:
        """Enter context manager."""
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        """Exit context manager."""
        self.close()


from ml_switcheroo_compiler.ops.base import OpDef, register_op


@register_op("Load")
class Load(OpDef):
    """Load operation."""

    op_name = "Load"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        from ml_switcheroo_compiler.core.shape import broadcast_shapes

        shapes = [getattr(a, "shape", ()) for a in args if hasattr(a, "shape")]
        if not shapes:
            return ()
        res = shapes[0]
        for s in shapes[1:]:
            res = broadcast_shapes(res, s)
        return res


@register_op("Save")
class Save(OpDef):
    """Save operation."""

    op_name = "Save"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        from ml_switcheroo_compiler.core.shape import broadcast_shapes

        shapes = [getattr(a, "shape", ()) for a in args if hasattr(a, "shape")]
        if not shapes:
            return ()
        res = shapes[0]
        for s in shapes[1:]:
            res = broadcast_shapes(res, s)
        return res


@register_op("SaveGguf")
class SaveGguf(OpDef):
    """SaveGguf operation."""

    op_name = "SaveGguf"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        from ml_switcheroo_compiler.core.shape import broadcast_shapes

        shapes = [getattr(a, "shape", ()) for a in args if hasattr(a, "shape")]
        if not shapes:
            return ()
        res = shapes[0]
        for s in shapes[1:]:
            res = broadcast_shapes(res, s)
        return res


@register_op("Savez")
class Savez(OpDef):
    """Savez operation."""

    op_name = "Savez"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        from ml_switcheroo_compiler.core.shape import broadcast_shapes

        shapes = [getattr(a, "shape", ()) for a in args if hasattr(a, "shape")]
        if not shapes:
            return ()
        res = shapes[0]
        for s in shapes[1:]:
            res = broadcast_shapes(res, s)
        return res


@register_op("SavezCompressed")
class SavezCompressed(OpDef):
    """SavezCompressed operation."""

    op_name = "SavezCompressed"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        from ml_switcheroo_compiler.core.shape import broadcast_shapes

        shapes = [getattr(a, "shape", ()) for a in args if hasattr(a, "shape")]
        if not shapes:
            return ()
        res = shapes[0]
        for s in shapes[1:]:
            res = broadcast_shapes(res, s)
        return res


@register_op("ReadFile")
class ReadFile(OpDef):
    """ReadFile operation."""

    op_name = "ReadFile"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        from ml_switcheroo_compiler.core.shape import broadcast_shapes

        shapes = [getattr(a, "shape", ()) for a in args if hasattr(a, "shape")]
        if not shapes:
            return ()
        res = shapes[0]
        for s in shapes[1:]:
            res = broadcast_shapes(res, s)
        return res


@register_op("WriteFile")
class WriteFile(OpDef):
    """WriteFile operation."""

    op_name = "WriteFile"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        from ml_switcheroo_compiler.core.shape import broadcast_shapes

        shapes = [getattr(a, "shape", ()) for a in args if hasattr(a, "shape")]
        if not shapes:
            return ()
        res = shapes[0]
        for s in shapes[1:]:
            res = broadcast_shapes(res, s)
        return res


@register_op("DecodeImage")
class DecodeImage(OpDef):
    """DecodeImage operation."""

    op_name = "DecodeImage"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        from ml_switcheroo_compiler.core.shape import broadcast_shapes

        shapes = [getattr(a, "shape", ()) for a in args if hasattr(a, "shape")]
        if not shapes:
            return ()
        res = shapes[0]
        for s in shapes[1:]:
            res = broadcast_shapes(res, s)
        return res


@register_op("DecodeCsv")
class DecodeCsv(OpDef):
    """DecodeCsv operation."""

    op_name = "DecodeCsv"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        from ml_switcheroo_compiler.core.shape import broadcast_shapes

        shapes = [getattr(a, "shape", ()) for a in args if hasattr(a, "shape")]
        if not shapes:
            return ()
        res = shapes[0]
        for s in shapes[1:]:
            res = broadcast_shapes(res, s)
        return res


@register_op("ParseExample")
class ParseExample(OpDef):
    """ParseExample operation."""

    op_name = "ParseExample"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        from ml_switcheroo_compiler.core.shape import broadcast_shapes

        shapes = [getattr(a, "shape", ()) for a in args if hasattr(a, "shape")]
        if not shapes:
            return ()
        res = shapes[0]
        for s in shapes[1:]:
            res = broadcast_shapes(res, s)
        return res


@register_op("SerializeTensor")
class SerializeTensor(OpDef):
    """SerializeTensor operation."""

    op_name = "SerializeTensor"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        from ml_switcheroo_compiler.core.shape import broadcast_shapes

        shapes = [getattr(a, "shape", ()) for a in args if hasattr(a, "shape")]
        if not shapes:
            return ()
        res = shapes[0]
        for s in shapes[1:]:
            res = broadcast_shapes(res, s)
        return res


@register_op("ParseTensor")
class ParseTensor(OpDef):
    """ParseTensor operation."""

    op_name = "ParseTensor"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        from ml_switcheroo_compiler.core.shape import broadcast_shapes

        shapes = [getattr(a, "shape", ()) for a in args if hasattr(a, "shape")]
        if not shapes:
            return ()
        res = shapes[0]
        for s in shapes[1:]:
            res = broadcast_shapes(res, s)
        return res


@register_op("EncodeBase64")
class EncodeBase64(OpDef):
    """EncodeBase64 operation."""

    op_name = "EncodeBase64"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        from ml_switcheroo_compiler.core.shape import broadcast_shapes

        shapes = [getattr(a, "shape", ()) for a in args if hasattr(a, "shape")]
        if not shapes:
            return ()
        res = shapes[0]
        for s in shapes[1:]:
            res = broadcast_shapes(res, s)
        return res


@register_op("DecodeBase64")
class DecodeBase64(OpDef):
    """DecodeBase64 operation."""

    op_name = "DecodeBase64"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        from ml_switcheroo_compiler.core.shape import broadcast_shapes

        shapes = [getattr(a, "shape", ()) for a in args if hasattr(a, "shape")]
        if not shapes:
            return ()
        res = shapes[0]
        for s in shapes[1:]:
            res = broadcast_shapes(res, s)
        return res


@register_op("ParseSequenceExample")
class ParseSequenceExample(OpDef):
    """ParseSequenceExample operation."""

    op_name = "ParseSequenceExample"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        from ml_switcheroo_compiler.core.shape import broadcast_shapes

        shapes = [getattr(a, "shape", ()) for a in args if hasattr(a, "shape")]
        if not shapes:
            return ()
        res = shapes[0]
        for s in shapes[1:]:
            res = broadcast_shapes(res, s)
        return res


@register_op("SparsePlus")
class SparsePlus(OpDef):
    """SparsePlus operation."""

    op_name = "SparsePlus"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        from ml_switcheroo_compiler.core.shape import broadcast_shapes

        shapes = [getattr(a, "shape", ()) for a in args if hasattr(a, "shape")]
        if not shapes:
            return ()
        res = shapes[0]
        for s in shapes[1:]:
            res = broadcast_shapes(res, s)
        return res


@register_op("SparseSigmoid")
class SparseSigmoid(OpDef):
    """SparseSigmoid operation."""

    op_name = "SparseSigmoid"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        from ml_switcheroo_compiler.core.shape import broadcast_shapes

        shapes = [getattr(a, "shape", ()) for a in args if hasattr(a, "shape")]
        if not shapes:
            return ()
        res = shapes[0]
        for s in shapes[1:]:
            res = broadcast_shapes(res, s)
        return res


def sparse_plus(*args: object, **kwargs: object) -> object:
    """SparsePlus frontend."""
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("SparsePlus", *args, **kwargs)
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("SparsePlus", list(args), kwargs, getattr(args[0], "shape", ()) if args else (), getattr(args[0], "dtype", "float32") if args else "float32")


def sparse_sigmoid(*args: object, **kwargs: object) -> object:
    """SparseSigmoid frontend."""
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("SparseSigmoid", *args, **kwargs)
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("SparseSigmoid", list(args), kwargs, getattr(args[0], "shape", ()) if args else (), getattr(args[0], "dtype", "float32") if args else "float32")


@register_op("Fromfile")
class Fromfile(OpDef):
    """Construct an array from data in a text or binary file."""

    op_name = "Fromfile"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        count = kwargs.get("count", -1)
        """Infer shape."""
        return (count if count != -1 else None,)


@register_op("Fromstring")
class Fromstring(OpDef):
    """A new 1-D array initialized from text data in a string."""

    op_name = "Fromstring"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        count = kwargs.get("count", -1)
        """Infer shape."""
        return (count if count != -1 else None,)


@register_op("Fromiter")
class Fromiter(OpDef):
    """Create a new 1-dimensional array from an iterable object."""

    op_name = "Fromiter"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        count = kwargs.get("count", -1)
        """Infer shape."""
        return (count if count != -1 else None,)


@register_op("Fromfunction")
class Fromfunction(OpDef):
    """Construct an array by executing a function over each coordinate."""

    op_name = "Fromfunction"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        shape = kwargs.get("shape", args[1] if len(args) > 1 else ())
        """Infer shape."""
        return tuple(shape) if isinstance(shape, (list, tuple)) else (shape,)


def fromfile(file: object, dtype: object = float, count: int = -1, sep: str = "", offset: int = 0, *, like: object = None) -> object:
    """Construct an array from data in a text or binary file."""
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("Fromfile", file, dtype=dtype, count=count, sep=sep, offset=offset, like=like)


def fromstring(string: str, dtype: object = float, count: int = -1, sep: str = "", *, like: object = None) -> object:
    """A new 1-D array initialized from text data in a string."""
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("Fromstring", string, dtype=dtype, count=count, sep=sep, like=like)


def fromiter(iterable: object, dtype: object, count: int = -1, *, like: object = None) -> object:
    """Create a new 1-dimensional array from an iterable object."""
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("Fromiter", iterable, dtype, count=count, like=like)


def fromfunction(function: object, shape: object, *, dtype: object = float, like: object = None, **kwargs: object) -> object:
    """Construct an array by executing a function over each coordinate."""
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("Fromfunction", function, shape, dtype=dtype, like=like, **kwargs)
