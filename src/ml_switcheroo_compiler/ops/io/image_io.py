"""Image decoding and I/O operations."""

from __future__ import annotations

from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, as_tensor_obj
from ml_switcheroo_compiler.ops.base import OpDef, register_op


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

        return as_tensor_obj(
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
        getattr(contents, "shape", ()),
        getattr(contents, "dtype", "float32"),
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

    data = getattr(contents, "data", contents)
    if not data:
        return Tensor(b"", TensorConfig((0,), "uint8", "cpu"))

    from ml_switcheroo_compiler.backends.registry import get_active_backend

    backend_cls = get_active_backend()
    if hasattr(backend_cls, "decode_image_bytes"):
        if isinstance(data, str):
            with open(data, "rb") as f:
                data = f.read()
        img_np = backend_cls.decode_image_bytes(data, channels=channels)
        return Tensor(img_np, TensorConfig(img_np.shape, "uint8", "cpu"))
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

    backend_cls = get_active_backend()
    if hasattr(backend_cls, "decode_jpeg"):
        return as_tensor_obj(backend_cls.decode_jpeg(contents, channels=channels, ratio=ratio))
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

    backend_cls = get_active_backend()
    if hasattr(backend_cls, "decode_png"):
        return as_tensor_obj(backend_cls.decode_png(contents, channels=channels, dtype=dtype))
    return _decode_image_with_pil(contents, channels)


def decode_gif(contents: Tensor | bytes | str | None) -> Tensor:
    """Decode the first frame or all frames of a GIF-encoded image.

    Args:
        contents (Tensor | bytes | str | None): GIF-encoded bytes or tensor.

    Returns:
        Tensor: Decoded GIF image tensor.
    """
    from ml_switcheroo_compiler.backends.registry import get_active_backend

    backend_cls = get_active_backend()
    if hasattr(backend_cls, "decode_gif"):
        return as_tensor_obj(backend_cls.decode_gif(contents))
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

    backend_cls = get_active_backend()
    if hasattr(backend_cls, "decode_bmp"):
        return as_tensor_obj(backend_cls.decode_bmp(contents, channels=channels))
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

        shapes = [getattr(a, "shape", ()) for a in args if hasattr(a, "shape")]
        if not shapes:
            return ()
        res = shapes[0]
        for s in shapes[1:]:
            res = broadcast_shapes(res, s)
        return res
