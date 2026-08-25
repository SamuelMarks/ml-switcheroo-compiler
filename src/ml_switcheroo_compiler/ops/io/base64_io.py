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


def _eager_base64(op: str, data: object, pad: bool = False) -> object:
    """Evaluate _eager_base64 operation.

    Args:
        op (str): The op parameter.
        data (object): The data parameter.
        pad (bool): The pad parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    import base64

    def _proc(d: object) -> bytes:
        """Process a single element for base64 operation.

        Args:
            d (object): The element.

        Returns:
            bytes: The base64 bytes.
        """
        if d is None:
            return b""
        b: object = d.encode("utf-8") if isinstance(d, str) else d
        res: object = base64.b64encode(b) if op == "encode" else base64.b64decode(b)
        if op == "encode" and not pad:
            res: object = res.rstrip(b"=")
        return res

    if isinstance(data, (list, tuple)):
        return [_proc(d) for d in data]
    return _proc(data)


def encode_base64(input: Tensor, pad: object = False, name: object = None) -> object:
    """Encode base64.

    Args:
        input (Tensor): The input parameter.
        pad (bool): The pad parameter.
        name (str): The name parameter.

    Returns:
        Tensor: Result.
    """
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("EncodeBase64", input, pad=pad, name=name)
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("EncodeBase64", [input], {"pad": pad, "name": name}, getattr(input, "shape", ()), getattr(input, "dtype", "float32"))


def decode_base64(input: Tensor, name: object = None) -> object:
    """Decode base64.

    Args:
        input (Tensor): The input parameter.
        name (str): The name parameter.

    Returns:
        Tensor: Result.
    """
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("DecodeBase64", input, name=name)
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("DecodeBase64", [input], {"name": name}, getattr(input, "shape", ()), getattr(input, "dtype", "float32"))


@register_op("EncodeBase64")
class EncodeBase64(OpDef):
    """EncodeBase64 operation."""

    op_name: object = "EncodeBase64"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        from ml_switcheroo_compiler.core.shape import broadcast_shapes

        shapes: object = [getattr(a, "shape", ()) for a in args if hasattr(a, "shape")]
        if not shapes:
            return ()
        res: object = shapes[0]
        for s in shapes[1:]:
            res: object = broadcast_shapes(res, s)
        return res


@register_op("DecodeBase64")
class DecodeBase64(OpDef):
    """DecodeBase64 operation."""

    op_name: object = "DecodeBase64"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        from ml_switcheroo_compiler.core.shape import broadcast_shapes

        shapes: object = [getattr(a, "shape", ()) for a in args if hasattr(a, "shape")]
        if not shapes:
            return ()
        res: object = shapes[0]
        for s in shapes[1:]:
            res: object = broadcast_shapes(res, s)
        return res
