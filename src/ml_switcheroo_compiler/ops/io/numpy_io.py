"""Array archive and checkpoint I/O operations."""

from __future__ import annotations

import typing

from ml_switcheroo_compiler.core.shape import TensorOrScalar
from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.ops.base import OpDef, register_op
from ml_switcheroo_compiler.serialization.formats.h5 import H5WeightFormat
from ml_switcheroo_compiler.serialization.formats.safetensors import SafetensorsWeightFormat
from ml_switcheroo_compiler.serialization.utils import load_npz


def _fallback_load(filepath: str | None) -> TensorOrScalar:
    """Load tensor weights based on file extension (.safetensors, .npz, .npy, .h5).

    Args:
        filepath (str | None): Path to the serialized weight file.

    Returns:
        TensorOrScalar: Loaded weights dictionary, tensor, or None if unsupported.
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


def load(*args: TensorOrScalar, **kwargs: TensorOrScalar) -> TensorOrScalar:
    """Load arrays or serialized tensor archives from disk.

    Args:
        *args (TensorOrScalar): Positional file path and loader arguments.
        **kwargs (TensorOrScalar): Keyword loader options.

    Returns:
        TensorOrScalar: Loaded tensor or dictionary of tensors.
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
        getattr(_first, "shape", ()),
        getattr(_first, "dtype", "float32"),
    )


def save(*args: TensorOrScalar, **kwargs: TensorOrScalar) -> Tensor | None:
    """Save a tensor or array to a binary `.npy` file.

    Args:
        *args (TensorOrScalar): File path and tensor positional arguments.
        **kwargs (TensorOrScalar): Keyword arguments for saving.

    Returns:
        Tensor | None: Symbolic node when tracing, or None in eager mode.
    """
    import typing

    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return typing.cast("Tensor | None", get_active_backend().execute_op("Save", *args, **kwargs))
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("Save", list(args), kwargs, (), "float32")


def save_gguf(*args: TensorOrScalar, **kwargs: TensorOrScalar) -> Tensor | None:
    """Save weights and metadata in GGUF binary format.

    Args:
        *args (TensorOrScalar): Positional path and weight dictionary arguments.
        **kwargs (TensorOrScalar): Keyword GGUF serialization options.

    Returns:
        Tensor | None: Symbolic node when tracing, or None in eager mode.
    """
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return typing.cast("Tensor | None", get_active_backend().execute_op("SaveGguf", *args, **kwargs))
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("SaveGguf", list(args), kwargs, (), "float32")


def save_safetensors(file: str, arrays: TensorOrScalar) -> None:
    """Save a dictionary of tensors to Safetensors format.

    Args:
        file (str): Destination `.safetensors` file path.
        arrays (TensorOrScalar): Mapping of tensor names to tensor objects.
    """
    SafetensorsWeightFormat().save(arrays, file)


def savez(*args: TensorOrScalar, **kwargs: TensorOrScalar) -> Tensor | None:
    """Save multiple arrays into an uncompressed `.npz` archive.

    Args:
        *args (TensorOrScalar): File path and unnamed array positional arguments.
        **kwargs (TensorOrScalar): Named arrays to store in the archive.

    Returns:
        Tensor | None: Symbolic node when tracing, or None in eager mode.
    """
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return typing.cast("Tensor | None", get_active_backend().execute_op("Savez", *args, **kwargs))
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("Savez", list(args), kwargs, (), "float32")


def savez_compressed(*args: TensorOrScalar, **kwargs: TensorOrScalar) -> Tensor | None:
    """Save multiple arrays into a compressed `.npz` archive.

    Args:
        *args (TensorOrScalar): File path and unnamed array positional arguments.
        **kwargs (TensorOrScalar): Named arrays to store in the compressed archive.

    Returns:
        Tensor | None: Symbolic node when tracing, or None in eager mode.
    """
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return typing.cast("Tensor | None", get_active_backend().execute_op("SavezCompressed", *args, **kwargs))
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("SavezCompressed", list(args), kwargs, (), "float32")


@register_op("Load")
class Load(OpDef):
    """Load operation definition."""

    op_name = "Load"

    def infer_shape(self, *args: TensorOrScalar, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Infer broadcasted output shape for Load.

        Args:
            *args (TensorOrScalar): Positional tensor arguments.
            **kwargs (TensorOrScalar): Keyword operation attributes.

        Returns:
            TensorOrScalar: Inferred output shape tuple.
        """
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
    """Save operation definition."""

    op_name = "Save"

    def infer_shape(self, *args: TensorOrScalar, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Infer broadcasted output shape for Save.

        Args:
            *args (TensorOrScalar): Positional tensor arguments.
            **kwargs (TensorOrScalar): Keyword operation attributes.

        Returns:
            TensorOrScalar: Inferred output shape tuple.
        """
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
    """SaveGguf operation definition."""

    op_name = "SaveGguf"

    def infer_shape(self, *args: TensorOrScalar, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Infer broadcasted output shape for SaveGguf.

        Args:
            *args (TensorOrScalar): Positional tensor arguments.
            **kwargs (TensorOrScalar): Keyword operation attributes.

        Returns:
            TensorOrScalar: Inferred output shape tuple.
        """
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
    """Savez operation definition."""

    op_name = "Savez"

    def infer_shape(self, *args: TensorOrScalar, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Infer broadcasted output shape for Savez.

        Args:
            *args (TensorOrScalar): Positional tensor arguments.
            **kwargs (TensorOrScalar): Keyword operation attributes.

        Returns:
            TensorOrScalar: Inferred output shape tuple.
        """
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
    """SavezCompressed operation definition."""

    op_name = "SavezCompressed"

    def infer_shape(self, *args: TensorOrScalar, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Infer broadcasted output shape for SavezCompressed.

        Args:
            *args (TensorOrScalar): Positional tensor arguments.
            **kwargs (TensorOrScalar): Keyword operation attributes.

        Returns:
            TensorOrScalar: Inferred output shape tuple.
        """
        from ml_switcheroo_compiler.core.shape import broadcast_shapes

        shapes = [getattr(a, "shape", ()) for a in args if hasattr(a, "shape")]
        if not shapes:
            return ()
        res = shapes[0]
        for s in shapes[1:]:
            res = broadcast_shapes(res, s)
        return res
