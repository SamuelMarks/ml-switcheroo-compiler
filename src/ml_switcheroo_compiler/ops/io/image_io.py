"""Image decoding and I/O operations."""

from __future__ import annotations

import typing
from collections.abc import Sequence

from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.shape import (
    ShapeProtocol,
    TensorOrScalar,
    extract_data,
    extract_dtype,
    extract_shape,
)
from ml_switcheroo_compiler.core.tensor import Tensor, ensure_tensor
from ml_switcheroo_compiler.ops.base import OpDef, register_op


@typing.runtime_checkable
class _DecodeImageBytesBackendProtocol(typing.Protocol):
    """Protocol for backends exposing `decode_image_bytes`."""

    def decode_image_bytes(self, data: TensorOrScalar, channels: int = 0) -> ShapeProtocol:
        """Decode image bytes into an array.

        Args:
            data (TensorOrScalar): Raw image bytes.
            channels (int): Desired channel count.

        Returns:
            ShapeProtocol: Decoded image array.
        """
        ...


@typing.runtime_checkable
class _DecodeJpegBackendProtocol(typing.Protocol):
    """Protocol for backends exposing `decode_jpeg`."""

    def decode_jpeg(self, contents: Tensor | bytes | str | None, channels: int = 0, ratio: int = 1) -> TensorOrScalar:
        """Decode JPEG image.

        Args:
            contents (Tensor | bytes | str | None): JPEG data.
            channels (int): Desired channel count.
            ratio (int): Downscaling ratio.

        Returns:
            TensorOrScalar: Decoded JPEG tensor/array.
        """
        ...


@typing.runtime_checkable
class _DecodePngBackendProtocol(typing.Protocol):
    """Protocol for backends exposing `decode_png`."""

    def decode_png(self, contents: Tensor | bytes | str | None, channels: int = 0, dtype: DType | str | None = None) -> TensorOrScalar:
        """Decode PNG image.

        Args:
            contents (Tensor | bytes | str | None): PNG data.
            channels (int): Desired channel count.
            dtype (DType | str | None): Output dtype.

        Returns:
            TensorOrScalar: Decoded PNG tensor/array.
        """
        ...


@typing.runtime_checkable
class _DecodeGifBackendProtocol(typing.Protocol):
    """Protocol for backends exposing `decode_gif`."""

    def decode_gif(self, contents: Tensor | bytes | str | None) -> TensorOrScalar:
        """Decode GIF image.

        Args:
            contents (Tensor | bytes | str | None): GIF data.

        Returns:
            TensorOrScalar: Decoded GIF tensor/array.
        """
        ...


@typing.runtime_checkable
class _DecodeBmpBackendProtocol(typing.Protocol):
    """Protocol for backends exposing `decode_bmp`."""

    def decode_bmp(self, contents: Tensor | bytes | str | None, channels: int = 0) -> TensorOrScalar:
        """Decode BMP image.

        Args:
            contents (Tensor | bytes | str | None): BMP data.
            channels (int): Desired channel count.

        Returns:
            TensorOrScalar: Decoded BMP tensor/array.
        """
        ...


def decode_image(
    contents: Tensor | bytes | str,
    channels: int = 0,
    dtype: DType | str = DType.UInt8,
    name: str | None = None,
    expand_animations: bool = True,
) -> Tensor:
    """Decode an image byte buffer into a tensor.

    Args:
        contents (Tensor | bytes | str): Encoded image contents or tensor.
        channels (int): Number of color channels for the decoded image.
        dtype (DType | str): Desired output data type.
        name (str | None): Optional operation name.
        expand_animations (bool): Whether to return 4-D tensor for animated formats.

    Returns:
        Tensor: Decoded image tensor.
    """
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return ensure_tensor(
            get_active_backend().execute_op(
                "DecodeImage",
                contents,
                channels=channels,
                dtype=dtype,
                name=name,
                expand_animations=expand_animations,
            )
        )
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node(
        "DecodeImage",
        [contents],
        {
            "channels": channels,
            "dtype": dtype,
            "name": name,
            "expand_animations": expand_animations,
        },
        extract_shape(contents),
        extract_dtype(contents),
    )


def _decode_image_with_pil(
    contents: Tensor | bytes | str | None,
    channels: int = 0,
) -> Tensor:
    """Decode an image byte buffer by delegating to the active backend.

    Args:
        contents (Tensor | bytes | str | None): Encoded byte content or file path of the image.
        channels (int): Expected number of channels.

    Returns:
        Tensor: Decoded image tensor.

    Raises:
        NotImplementedError: If the active backend lacks decode_image_bytes support.
    """
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    data = extract_data(contents)
    if not data:
        return Tensor(b"", TensorConfig((0,), "uint8", "cpu"))

    from ml_switcheroo_compiler.backends.registry import get_active_backend

    backend_cls: TensorOrScalar | _DecodeImageBytesBackendProtocol = get_active_backend()
    if isinstance(backend_cls, _DecodeImageBytesBackendProtocol):
        if isinstance(data, str):
            with open(data, "rb") as f:
                data = f.read()
        img_np = backend_cls.decode_image_bytes(data, channels=channels)
        return Tensor(img_np, TensorConfig(extract_shape(img_np), "uint8", "cpu"))
    raise NotImplementedError("PIL is decoupled from core. Active backend lacks decode_image_bytes.")


def decode_jpeg(
    contents: Tensor | bytes | str | None,
    channels: int = 0,
    ratio: int = 1,
) -> Tensor:
    """Decode a JPEG-encoded image to a uint8 tensor.

    Args:
        contents (Tensor | bytes | str | None): JPEG-encoded bytes or tensor.
        channels (int): Number of color channels for the decoded image.
        ratio (int): Downscaling ratio.

    Returns:
        Tensor: Decoded JPEG image tensor.
    """
    from ml_switcheroo_compiler.backends.registry import get_active_backend

    backend_cls: TensorOrScalar | _DecodeJpegBackendProtocol = get_active_backend()
    if isinstance(backend_cls, _DecodeJpegBackendProtocol):
        return ensure_tensor(backend_cls.decode_jpeg(contents, channels=channels, ratio=ratio))
    return _decode_image_with_pil(contents, channels)


def decode_png(
    contents: Tensor | bytes | str | None,
    channels: int = 0,
    dtype: DType | str | None = None,
) -> Tensor:
    """Decode a PNG-encoded image to a uint8 or uint16 tensor.

    Args:
        contents (Tensor | bytes | str | None): PNG-encoded bytes or tensor.
        channels (int): Number of color channels for the decoded image.
        dtype (DType | str | None): Desired output data type.

    Returns:
        Tensor: Decoded PNG image tensor.
    """
    from ml_switcheroo_compiler.backends.registry import get_active_backend

    backend_cls: TensorOrScalar | _DecodePngBackendProtocol = get_active_backend()
    if isinstance(backend_cls, _DecodePngBackendProtocol):
        return ensure_tensor(backend_cls.decode_png(contents, channels=channels, dtype=dtype))
    return _decode_image_with_pil(contents, channels)


def decode_gif(contents: Tensor | bytes | str | None) -> Tensor:
    """Decode the first frame or all frames of a GIF-encoded image.

    Args:
        contents (Tensor | bytes | str | None): GIF-encoded bytes or tensor.

    Returns:
        Tensor: Decoded GIF image tensor.
    """
    from ml_switcheroo_compiler.backends.registry import get_active_backend

    backend_cls: TensorOrScalar | _DecodeGifBackendProtocol = get_active_backend()
    if isinstance(backend_cls, _DecodeGifBackendProtocol):
        return ensure_tensor(backend_cls.decode_gif(contents))
    return _decode_image_with_pil(contents)


def decode_bmp(
    contents: Tensor | bytes | str | None,
    channels: int = 0,
) -> Tensor:
    """Decode the first frame of a BMP-encoded image to a uint8 tensor.

    Args:
        contents (Tensor | bytes | str | None): BMP-encoded bytes or tensor.
        channels (int): Number of color channels for the decoded image.

    Returns:
        Tensor: Decoded BMP image tensor.
    """
    from ml_switcheroo_compiler.backends.registry import get_active_backend

    backend_cls: TensorOrScalar | _DecodeBmpBackendProtocol = get_active_backend()
    if isinstance(backend_cls, _DecodeBmpBackendProtocol):
        return ensure_tensor(backend_cls.decode_bmp(contents, channels=channels))
    return _decode_image_with_pil(contents, channels)


@register_op("DecodeImage")
class DecodeImage(OpDef):
    """Operation definition for DecodeImage."""

    op_name = "DecodeImage"

    def infer_shape(
        self,
        *args: Tensor,
        **kwargs: int | str | bool | None,
    ) -> tuple[int, ...]:
        """Infer output shape for DecodeImage.

        Args:
            *args (Tensor): Input tensor arguments.
            **kwargs (int | str | bool | None): Keyword configuration arguments.

        Returns:
            tuple[int, ...]: Inferred output shape.
        """
        _ = kwargs
        from ml_switcheroo_compiler.core.shape import broadcast_shapes

        raw_args: Sequence[TensorOrScalar] = args
        shapes = [extract_shape(a) for a in raw_args if isinstance(a, ShapeProtocol)]
        if not shapes:
            return ()
        res = shapes[0]
        for s in shapes[1:]:
            res = broadcast_shapes(res, s)
        _typed_ret_1: tuple[int, ...] = res
        return _typed_ret_1
