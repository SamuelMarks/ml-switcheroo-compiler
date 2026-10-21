"""Base64 encoding and decoding I/O operations."""

from __future__ import annotations

import base64

from ml_switcheroo_compiler.backends.registry import as_bytes
from ml_switcheroo_compiler.core.shape import TensorOrScalar
from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.ops.base import OpDef, register_op


def _eager_base64(op: str, data: TensorOrScalar, pad: bool = False) -> TensorOrScalar:
    """Evaluate Base64 encoding or decoding on eager byte/string inputs.

    Args:
        op (str): Operation mode ('encode' or 'decode').
        data (TensorOrScalar): Input bytes, string, or sequence of bytes/strings.
        pad (bool): Whether to preserve '=' padding characters when encoding.

    Returns:
        TensorOrScalar: Base64-processed bytes or list of bytes.
    """

    def _proc(d: TensorOrScalar) -> bytes:
        """Process a single element for base64 operation.

        Args:
            d (TensorOrScalar): Input element to encode or decode.

        Returns:
            bytes: Processed Base64 byte string.
        """
        if d is None:
            return b""
        raw_bytes = d.encode("utf-8") if isinstance(d, str) else as_bytes(d)
        res = base64.b64encode(raw_bytes) if op == "encode" else base64.b64decode(raw_bytes)
        if op == "encode" and not pad:
            res = res.rstrip(b"=")
        return res

    if isinstance(data, (list, tuple)):
        return [_proc(d) for d in data]
    return _proc(data)


def encode_base64(input: Tensor, pad: TensorOrScalar = False, name: TensorOrScalar = None) -> TensorOrScalar:
    """Encode input tensor strings/bytes into web-safe Base64 representations.

    Args:
        input (Tensor): Input tensor to encode.
        pad (TensorOrScalar): Whether to include padding characters at the end.
        name (TensorOrScalar): Optional operation name.

    Returns:
        TensorOrScalar: Base64-encoded tensor.
    """
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("EncodeBase64", input, pad=pad, name=name)
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node(
        "EncodeBase64",
        [input],
        {"pad": pad, "name": name},
        getattr(input, "shape", ()),
        getattr(input, "dtype", "float32"),
    )


def decode_base64(input: Tensor, name: TensorOrScalar = None) -> TensorOrScalar:
    """Decode Base64-encoded tensor elements back into raw bytes.

    Args:
        input (Tensor): Base64-encoded input tensor.
        name (TensorOrScalar): Optional operation name.

    Returns:
        TensorOrScalar: Decoded byte tensor.
    """
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("DecodeBase64", input, name=name)
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node(
        "DecodeBase64",
        [input],
        {"name": name},
        getattr(input, "shape", ()),
        getattr(input, "dtype", "float32"),
    )


@register_op("EncodeBase64")
class EncodeBase64(OpDef):
    """EncodeBase64 operation definition."""

    op_name = "EncodeBase64"

    def infer_shape(self, *args: TensorOrScalar, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Infer broadcasted output shape for EncodeBase64.

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


@register_op("DecodeBase64")
class DecodeBase64(OpDef):
    """DecodeBase64 operation definition."""

    op_name = "DecodeBase64"

    def infer_shape(self, *args: TensorOrScalar, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Infer broadcasted output shape for DecodeBase64.

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
