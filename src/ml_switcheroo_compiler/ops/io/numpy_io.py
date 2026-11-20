"""Array archive and checkpoint I/O operations."""

from __future__ import annotations

from ml_switcheroo_compiler.core.shape import (
    OpValue,
    ShapeProtocol,
    extract_dtype,
    extract_shape,
)
from ml_switcheroo_compiler.ops.base import OpDef, register_op
from ml_switcheroo_compiler.serialization.formats.h5 import H5WeightFormat
from ml_switcheroo_compiler.serialization.formats.safetensors import SafetensorsWeightFormat
from ml_switcheroo_compiler.serialization.utils import load_npz


def _fallback_load(filepath: str | None) -> OpValue:
    """Load tensor weights based on file extension (.safetensors, .npz, .npy, .h5).

    Args:
        filepath (str | None): Path to the serialized weight file.

    Returns:
        OpValue: Loaded weights dictionary, tensor, or None if unsupported.
    """
    if not isinstance(filepath, str):
        return None
    if filepath.endswith(".safetensors"):
        return SafetensorsWeightFormat().load(filepath)
    if filepath.endswith(".npz"):
        return load_npz(filepath)
    if filepath.endswith(".npy"):
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("Load", filepath)
    if filepath.endswith(".h5"):
        return H5WeightFormat().load(filepath)
    return None


def load(*args: OpValue, **kwargs: OpValue) -> OpValue:
    """Load arrays or serialized tensor archives from disk.

    Args:
        *args (OpValue): Positional file path and loader arguments.
        **kwargs (OpValue): Keyword loader options.

    Returns:
        OpValue: Loaded tensor or dictionary of tensors.
    """
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("Load", *args, **kwargs)
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    _first = args[0] if args else (list(kwargs.values())[0] if kwargs else None)
    return _emit_shape_node(
        "Load",
        list(args),
        kwargs,
        extract_shape(_first),
        extract_dtype(_first),
    )


def save(*args: OpValue, **kwargs: OpValue) -> OpValue:
    """Save a tensor or array to a binary `.npy` file.

    Args:
        *args (OpValue): File path and tensor positional arguments.
        **kwargs (OpValue): Keyword arguments for saving.

    Returns:
        OpValue: Symbolic node when tracing, or backend result in eager mode.
    """
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("Save", *args, **kwargs)
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("Save", list(args), kwargs, (), "float32")


def save_gguf(*args: OpValue, **kwargs: OpValue) -> OpValue:
    """Save weights and metadata in GGUF binary format.

    Args:
        *args (OpValue): Positional path and weight dictionary arguments.
        **kwargs (OpValue): Keyword GGUF serialization options.

    Returns:
        OpValue: Symbolic node when tracing, or backend result in eager mode.
    """
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("SaveGguf", *args, **kwargs)
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("SaveGguf", list(args), kwargs, (), "float32")


def save_safetensors(file: str, arrays: OpValue) -> None:
    """Save a dictionary of tensors to Safetensors format.

    Args:
        file (str): Destination `.safetensors` file path.
        arrays (OpValue): Mapping of tensor names to tensor objects.
    """
    SafetensorsWeightFormat().save(arrays, file)


def savez(*args: OpValue, **kwargs: OpValue) -> OpValue:
    """Save multiple arrays into an uncompressed `.npz` archive.

    Args:
        *args (OpValue): File path and unnamed array positional arguments.
        **kwargs (OpValue): Named arrays to store in the archive.

    Returns:
        OpValue: Symbolic node when tracing, or backend result in eager mode.
    """
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("Savez", *args, **kwargs)
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("Savez", list(args), kwargs, (), "float32")


def savez_compressed(*args: OpValue, **kwargs: OpValue) -> OpValue:
    """Save multiple arrays into a compressed `.npz` archive.

    Args:
        *args (OpValue): File path and unnamed array positional arguments.
        **kwargs (OpValue): Named arrays to store in the compressed archive.

    Returns:
        OpValue: Symbolic node when tracing, or backend result in eager mode.
    """
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("SavezCompressed", *args, **kwargs)
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("SavezCompressed", list(args), kwargs, (), "float32")


@register_op("Load")
class Load(OpDef):
    """Load operation definition."""

    op_name = "Load"

    def infer_shape(self, *args: OpValue, **kwargs: OpValue) -> OpValue:
        """Infer broadcasted output shape for Load.

        Args:
            *args (OpValue): Positional tensor arguments.
            **kwargs (OpValue): Keyword operation attributes.

        Returns:
            OpValue: Inferred output shape tuple.
        """
        from ml_switcheroo_compiler.core.shape import broadcast_shapes

        shapes = [extract_shape(a) for a in args if isinstance(a, ShapeProtocol)]
        if not shapes:
            return ()
        res = shapes[0]
        for s in shapes[1:]:
            res = broadcast_shapes(res, s)
        _typed_ret_1: OpValue = res
        return _typed_ret_1


@register_op("Save")
class Save(OpDef):
    """Save operation definition."""

    op_name = "Save"

    def infer_shape(self, *args: OpValue, **kwargs: OpValue) -> OpValue:
        """Infer broadcasted output shape for Save.

        Args:
            *args (OpValue): Positional tensor arguments.
            **kwargs (OpValue): Keyword operation attributes.

        Returns:
            OpValue: Inferred output shape tuple.
        """
        from ml_switcheroo_compiler.core.shape import broadcast_shapes

        shapes = [extract_shape(a) for a in args if isinstance(a, ShapeProtocol)]
        if not shapes:
            return ()
        res = shapes[0]
        for s in shapes[1:]:
            res = broadcast_shapes(res, s)
        _typed_ret_2: OpValue = res
        return _typed_ret_2


@register_op("SaveGguf")
class SaveGguf(OpDef):
    """SaveGguf operation definition."""

    op_name = "SaveGguf"

    def infer_shape(self, *args: OpValue, **kwargs: OpValue) -> OpValue:
        """Infer broadcasted output shape for SaveGguf.

        Args:
            *args (OpValue): Positional tensor arguments.
            **kwargs (OpValue): Keyword operation attributes.

        Returns:
            OpValue: Inferred output shape tuple.
        """
        from ml_switcheroo_compiler.core.shape import broadcast_shapes

        shapes = [extract_shape(a) for a in args if isinstance(a, ShapeProtocol)]
        if not shapes:
            return ()
        res = shapes[0]
        for s in shapes[1:]:
            res = broadcast_shapes(res, s)
        _typed_ret_3: OpValue = res
        return _typed_ret_3


@register_op("Savez")
class Savez(OpDef):
    """Savez operation definition."""

    op_name = "Savez"

    def infer_shape(self, *args: OpValue, **kwargs: OpValue) -> OpValue:
        """Infer broadcasted output shape for Savez.

        Args:
            *args (OpValue): Positional tensor arguments.
            **kwargs (OpValue): Keyword operation attributes.

        Returns:
            OpValue: Inferred output shape tuple.
        """
        from ml_switcheroo_compiler.core.shape import broadcast_shapes

        shapes = [extract_shape(a) for a in args if isinstance(a, ShapeProtocol)]
        if not shapes:
            return ()
        res = shapes[0]
        for s in shapes[1:]:
            res = broadcast_shapes(res, s)
        _typed_ret_4: OpValue = res
        return _typed_ret_4


@register_op("SavezCompressed")
class SavezCompressed(OpDef):
    """SavezCompressed operation definition."""

    op_name = "SavezCompressed"

    def infer_shape(self, *args: OpValue, **kwargs: OpValue) -> OpValue:
        """Infer broadcasted output shape for SavezCompressed.

        Args:
            *args (OpValue): Positional tensor arguments.
            **kwargs (OpValue): Keyword operation attributes.

        Returns:
            OpValue: Inferred output shape tuple.
        """
        from ml_switcheroo_compiler.core.shape import broadcast_shapes

        shapes = [extract_shape(a) for a in args if isinstance(a, ShapeProtocol)]
        if not shapes:
            return ()
        res = shapes[0]
        for s in shapes[1:]:
            res = broadcast_shapes(res, s)
        _typed_ret_5: OpValue = res
        return _typed_ret_5
