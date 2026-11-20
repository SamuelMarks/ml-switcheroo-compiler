"""Base64 encoding and decoding I/O operations."""

from __future__ import annotations

import base64

from ml_switcheroo_compiler.backends.registry import coerce_bytes
from ml_switcheroo_compiler.core.shape import (
    OpValue,
    ShapeProtocol,
    extract_dtype,
    extract_shape,
)
from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.ops.base import OpDef, register_op


def _eager_base64(op: str, data: OpValue, pad: bool = False) -> OpValue:
    """Evaluate Base64 encoding or decoding on eager byte/string inputs.

    Args:
        op (str): Operation mode ('encode' or 'decode').
        data (OpValue): Input bytes, string, or sequence of bytes/strings.
        pad (bool): Whether to preserve '=' padding characters when encoding.

    Returns:
        OpValue: Base64-processed bytes or list of bytes.
    """

    def _proc(d: OpValue) -> bytes:
        """Process a single element for base64 operation.

        Args:
            d (OpValue): Input element to encode or decode.

        Returns:
            bytes: Processed Base64 byte string.
        """
        if d is None:
            return b""
        raw_bytes = d.encode("utf-8") if isinstance(d, str) else coerce_bytes(d)
        res = base64.b64encode(raw_bytes) if op == "encode" else base64.b64decode(raw_bytes)
        if op == "encode" and not pad:
            res = res.rstrip(b"=")
        return res

    if isinstance(data, (list, tuple)):
        return [_proc(d) for d in data]
    return _proc(data)


def encode_base64(input: Tensor | str | bytes, pad: OpValue = False, name: OpValue = None) -> OpValue:
    """Encode input tensor strings/bytes into web-safe Base64 representations.

    Args:
        input (Tensor | str | bytes): Input tensor or string/bytes to encode.
        pad (OpValue): Whether to include padding characters at the end.
        name (OpValue): Optional operation name.

    Returns:
        OpValue: Base64-encoded string/byte tensor.
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
        extract_shape(input),
        extract_dtype(input),
    )


def decode_base64(input: Tensor | str | bytes, name: OpValue = None) -> OpValue:
    """Decode Base64-encoded tensor elements back into raw bytes.

    Args:
        input (Tensor | str | bytes): Base64-encoded input tensor or string/bytes.
        name (OpValue): Optional operation name.

    Returns:
        OpValue: Decoded byte tensor.
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
        extract_shape(input),
        extract_dtype(input),
    )


@register_op("EncodeBase64")
class EncodeBase64(OpDef):
    """EncodeBase64 operation definition."""

    op_name = "EncodeBase64"

    def infer_shape(self, *args: OpValue, **kwargs: OpValue) -> OpValue:
        """Infer broadcasted output shape for EncodeBase64.

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


@register_op("DecodeBase64")
class DecodeBase64(OpDef):
    """DecodeBase64 operation definition."""

    op_name = "DecodeBase64"

    def infer_shape(self, *args: OpValue, **kwargs: OpValue) -> OpValue:
        """Infer broadcasted output shape for DecodeBase64.

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
