# ruff: noqa: E402, D100, D103, D104, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, D101, D102, D107, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""I/O and memory operations."""

from __future__ import annotations

import glob
import os
import shutil
from typing import Any

from ml_switcheroo_compiler.core.config import config as core_config
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.ops.base import OpDef, register_op
from ml_switcheroo_compiler.serialization.formats.h5 import H5WeightFormat
from ml_switcheroo_compiler.serialization.formats.safetensors import SafetensorsWeightFormat
from ml_switcheroo_compiler.serialization.utils import load_npz


def _fallback_load(filepath: Any) -> Any:
    """Fallback mechanism to load weights based on file extension.

    Args:
        filepath (object): The file path.

    Returns: Any: The loaded weights or None.
    """
    if not isinstance(filepath, str):
        return None
    if filepath.endswith(".safetensors"):
        return SafetensorsWeightFormat().load(filepath)
    if filepath.endswith(".npz"):
        return load_npz(filepath)
    if filepath.endswith(".h5"):
        return H5WeightFormat().load(filepath)
    return None


def load(*args: Any, **kwargs: Any) -> Any:
    """Load arrays or pickled objects from files.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("Load", *args, **kwargs)
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    _first = args[0] if args else (list(kwargs.values())[0] if kwargs else None)
    return _emit_shape_node("Load", list(args), kwargs, getattr(_first, "shape", ()), getattr(_first, "dtype", "float32"))


def save(*args: Any, **kwargs: Any) -> None:
    """Save.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
        NoneType: Result.
    """
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("Save", *args, **kwargs)
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("Save", list(args), kwargs, (), "float32")


def save_gguf(*args: Any, **kwargs: Any) -> None:
    """Save gguf.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
        NoneType: Result.
    """
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("SaveGguf", *args, **kwargs)
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("SaveGguf", list(args), kwargs, (), "float32")


def save_safetensors(file: str, arrays: dict[str, Any]) -> None:
    """Save a dictionary of arrays to Safetensors format.

    Args:
        file (str): The file parameter.
        arrays (dict): The arrays parameter.
    """
    SafetensorsWeightFormat().save(arrays, file)


def savez(*args: Any, **kwargs: Any) -> None:
    """Savez.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
        NoneType: Result.
    """
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("Savez", *args, **kwargs)
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("Savez", list(args), kwargs, (), "float32")


def savez_compressed(*args: Any, **kwargs: Any) -> None:
    """Savez compressed.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
        NoneType: Result.
    """
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("SavezCompressed", *args, **kwargs)
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("SavezCompressed", list(args), kwargs, (), "float32")


@register_op("Load")
class Load(OpDef):
    """Load operation."""

    op_name = "Load"

    def infer_shape(self, *args: Any, **kwargs: Any) -> Any:
        """Infer shape.

        Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

        Returns: Any: Result.
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
    """Save operation."""

    op_name = "Save"

    def infer_shape(self, *args: Any, **kwargs: Any) -> Any:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns: Any: Result.
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
    """SaveGguf operation."""

    op_name = "SaveGguf"

    def infer_shape(self, *args: Any, **kwargs: Any) -> Any:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns: Any: Result.
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
    """Savez operation."""

    op_name = "Savez"

    def infer_shape(self, *args: Any, **kwargs: Any) -> Any:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns: Any: Result.
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
    """SavezCompressed operation."""

    op_name = "SavezCompressed"

    def infer_shape(self, *args: Any, **kwargs: Any) -> Any:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns: Any: Result.
        """
        from ml_switcheroo_compiler.core.shape import broadcast_shapes

        shapes = [getattr(a, "shape", ()) for a in args if hasattr(a, "shape")]
        if not shapes:
            return ()
        res = shapes[0]
        for s in shapes[1:]:
            res = broadcast_shapes(res, s)
        return res
