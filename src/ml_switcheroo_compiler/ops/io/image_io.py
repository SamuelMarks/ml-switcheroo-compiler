# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""I/O and memory operations."""

from __future__ import annotations

import glob
import os
import shutil

from ml_switcheroo_compiler.core.config import config as core_config
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.ops.base import OpDef, register_op
from ml_switcheroo_compiler.serialization.formats.h5 import H5WeightFormat
from ml_switcheroo_compiler.serialization.formats.safetensors import SafetensorsWeightFormat
from ml_switcheroo_compiler.serialization.utils import load_npz


def decode_image(contents: Tensor, channels=0, dtype=DType.UInt8, name=None, expand_animations=True):
    """Decode image.

    Args:
        contents (Tensor): The contents parameter.
        channels (int): The channels parameter.
        dtype (DType): The dtype parameter.
        name (str): The name parameter.
        expand_animations (bool): The expand_animations parameter.

    Returns:
        Tensor: Result.
    """
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("DecodeImage", contents, channels=channels, dtype=dtype, name=name, expand_animations=expand_animations)
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("DecodeImage", [contents], {"channels": channels, "dtype": dtype, "name": name, "expand_animations": expand_animations}, getattr(contents, "shape", ()), getattr(contents, "dtype", "float32"))


def decode_jpeg(contents, channels: int = 0, ratio: int = 1):
    """Decode JPEG image.

    Args:
        contents (object): The contents parameter.
        channels (int): The channels parameter.
        ratio (int): The ratio parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.backends.registry import get_active_backend

    backend_cls = get_active_backend()
    if hasattr(backend_cls, "decode_jpeg"):
        return backend_cls.decode_jpeg(contents, channels=channels, ratio=ratio)
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    return Tensor(None, TensorConfig((), None, None))


def decode_png(contents, channels: int = 0, dtype=None):
    """Decode PNG image.

    Args:
        contents (object): The contents parameter.
        channels (int): The channels parameter.
        like (object): The like parameter.
        dtype (object): The dtype parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.backends.registry import get_active_backend

    backend_cls = get_active_backend()
    if hasattr(backend_cls, "decode_png"):
        return backend_cls.decode_png(contents, channels=channels, dtype=dtype)
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    return Tensor(None, TensorConfig((), None, None))


def decode_gif(
    contents,
):
    """Decode GIF image.

    Args:
        contents (object): The contents parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.backends.registry import get_active_backend

    backend_cls = get_active_backend()
    if hasattr(backend_cls, "decode_gif"):
        return backend_cls.decode_gif(contents)
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    return Tensor(None, TensorConfig((), None, None))


def decode_bmp(contents, channels: int = 0):
    """Decode BMP image.

    Args:
        contents (object): The contents parameter.
        channels (int): The channels parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.backends.registry import get_active_backend

    backend_cls = get_active_backend()
    if hasattr(backend_cls, "decode_bmp"):
        return backend_cls.decode_bmp(contents, channels=channels)
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    return Tensor(None, TensorConfig((), None, None))


@register_op("DecodeImage")
class DecodeImage(OpDef):
    """DecodeImage operation."""

    op_name = "DecodeImage"

    def infer_shape(self, *args, **kwargs):
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        from ml_switcheroo_compiler.core.shape import broadcast_shapes

        shapes = [getattr(a, "shape", ()) for a in args if hasattr(a, "shape")]
        if not shapes:
            return ()
        res = shapes[0]
        for s in shapes[1:]:
            res = broadcast_shapes(res, s)
        return res
