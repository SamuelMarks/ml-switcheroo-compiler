# ruff: noqa: E402, E501, F403
"""I/O and memory operations."""

from __future__ import annotations

import glob  # noqa: F401
import os  # noqa: F401
import shutil  # noqa: F401

from ml_switcheroo_compiler.core.config import config as core_config  # noqa: F401
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.ops.base import OpDef, register_op
from ml_switcheroo_compiler.serialization.formats.h5 import H5WeightFormat  # noqa: F401
from ml_switcheroo_compiler.serialization.formats.safetensors import SafetensorsWeightFormat  # noqa: F401
from ml_switcheroo_compiler.serialization.utils import load_npz  # noqa: F401


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


def _decode_image_with_pil(contents, channels: int = 0):
    """Helper to decode an image buffer using Pillow.

    Args:
        contents (Any): The byte content of the image.
        channels (int): Expected number of channels.

    Returns:
        Tensor: Resulting decoded image tensor.
    """
    import io  # noqa: F401

    import numpy as np  # noqa: F401
    from PIL import Image  # noqa: F401

    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    data = getattr(contents, "data", contents)
    if not data:
        return Tensor(None, TensorConfig((), None, None))

    from ml_switcheroo_compiler.backends.registry import get_active_backend

    backend_cls = get_active_backend()
    if hasattr(backend_cls, "decode_image_bytes"):
        if isinstance(data, str):
            with open(data, "rb") as f:
                data = f.read()
        img_np = backend_cls.decode_image_bytes(data, channels=channels)
        return Tensor(img_np, TensorConfig(img_np.shape, "uint8", "cpu"))
    raise NotImplementedError("PIL is decoupled from core. Active backend lacks decode_image_bytes.")


def decode_jpeg(contents, channels: int = 0, ratio: int = 1):
    """Decode JPEG image.

    Args:
        contents (Any): The contents parameter.
        channels (int): The channels parameter.
        ratio (int): The ratio parameter.

    Returns:
        Tensor: Result.
    """
    from ml_switcheroo_compiler.backends.registry import get_active_backend

    backend_cls = get_active_backend()
    if hasattr(backend_cls, "decode_jpeg"):
        return backend_cls.decode_jpeg(contents, channels=channels, ratio=ratio)
    return _decode_image_with_pil(contents, channels)


def decode_png(contents, channels: int = 0, dtype=None):
    """Decode PNG image.

    Args:
        contents (Any): The contents parameter.
        channels (int): The channels parameter.
        dtype (Any): The dtype parameter.

    Returns:
        Tensor: Result.
    """
    from ml_switcheroo_compiler.backends.registry import get_active_backend

    backend_cls = get_active_backend()
    if hasattr(backend_cls, "decode_png"):
        return backend_cls.decode_png(contents, channels=channels, dtype=dtype)
    return _decode_image_with_pil(contents, channels)


def decode_gif(contents):
    """Decode GIF image.

    Args:
        contents (Any): The contents parameter.

    Returns:
        Tensor: Result.
    """
    from ml_switcheroo_compiler.backends.registry import get_active_backend

    backend_cls = get_active_backend()
    if hasattr(backend_cls, "decode_gif"):
        return backend_cls.decode_gif(contents)
    return _decode_image_with_pil(contents)


def decode_bmp(contents, channels: int = 0):
    """Decode BMP image.

    Args:
        contents (Any): The contents parameter.
        channels (int): The channels parameter.

    Returns:
        Tensor: Result.
    """
    from ml_switcheroo_compiler.backends.registry import get_active_backend

    backend_cls = get_active_backend()
    if hasattr(backend_cls, "decode_bmp"):
        return backend_cls.decode_bmp(contents, channels=channels)
    return _decode_image_with_pil(contents, channels)


@register_op("DecodeImage")
class DecodeImage(OpDef):
    """DecodeImage operation."""

    op_name = "DecodeImage"

    def infer_shape(self, *args, **kwargs):
        """Infer shape.

        Args:
            *args (Any): Positional args.
            **kwargs (Any): Keyword args.

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
