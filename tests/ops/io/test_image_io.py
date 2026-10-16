"""Tests for test_image_io."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pytest

import ml_switcheroo_compiler.backends.registry as registry
from ml_switcheroo_compiler.ops.io.image_io import _decode_image_with_pil, decode_bmp, decode_gif, decode_png

if TYPE_CHECKING:
    from collections.abc import Callable

    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue


def test_decode_image_with_pil(tmp_path: EagerTensorProtocol, monkeypatch: EagerTensorProtocol) -> None:
    """Test function."""
    # Empty data
    t = _decode_image_with_pil(None)
    assert t.data is None

    # Exception when active backend lacks decode_image_bytes
    with pytest.raises(NotImplementedError):
        _decode_image_with_pil(b"not_an_image")

    with pytest.raises(NotImplementedError):
        _decode_image_with_pil("dummy")

    class DummyImgArray:
        """Test class DummyImgArray."""

        def __init__(self, shape: EagerValue) -> None:
            """Test function __init__."""
            self.shape = shape

    class DummyDecodeBackend:
        """Test class DummyDecodeBackend."""

        @staticmethod
        def decode_image_bytes(data: EagerValue, channels: EagerValue = 0) -> EagerValue:
            """Test function decode_image_bytes."""
            return cast("EagerTensorProtocol", DummyImgArray((10, 10) if channels == 1 else (10, 10, channels or 3)))

    monkeypatch.setattr(registry, "get_active_backend", lambda: DummyDecodeBackend)

    img_file = tmp_path / "sample.png"
    img_file.write_bytes(b"png_bytes")

    t1 = _decode_image_with_pil(str(img_file), channels=1)
    assert t1.shape == (10, 10)

    t3 = _decode_image_with_pil(b"raw_bytes", channels=3)
    assert t3.shape == (10, 10, 3)

    t4 = _decode_image_with_pil(str(img_file), channels=4)
    assert t4.shape == (10, 10, 4)


def test_decode_image_backend(monkeypatch: EagerTensorProtocol) -> None:
    """Test function."""

    class DummyBackend:
        """Test class DummyBackend."""

        @staticmethod
        def decode_png(*args: EagerValue, **kwargs: EagerValue) -> EagerValue:
            """Test function decode_png."""
            return "png"

        @staticmethod
        def decode_gif(*args: EagerValue, **kwargs: EagerValue) -> EagerValue:
            """Test function decode_gif."""
            return "gif"

        @staticmethod
        def decode_bmp(*args: EagerValue, **kwargs: EagerValue) -> EagerValue:
            """Test function decode_bmp."""
            return "bmp"

    monkeypatch.setattr(registry, "get_active_backend", lambda: DummyBackend)

    assert cast("EagerTensorProtocol", decode_png(b"", channels=3)) == cast("EagerTensorProtocol", "png")
    assert cast("EagerTensorProtocol", decode_gif(b"")) == cast("EagerTensorProtocol", "gif")
    assert cast("EagerTensorProtocol", decode_bmp(b"", channels=3)) == cast("EagerTensorProtocol", "bmp")


def test_decode_image_fallback() -> None:
    """Test function."""
    from ml_switcheroo_compiler.ops.io.image_io import decode_jpeg

    # Calling the ops should fallback to PIL since the default mock or no backend has decode_png
    # But we need valid inputs or we get exception.
    t_jpeg = decode_jpeg(None)
    assert t_jpeg.data is None
    t_png = decode_png(None)
    assert t_png.data is None

    t_gif = decode_gif(None)
    assert t_gif.data is None

    t_bmp = decode_bmp(None)
    assert t_bmp.data is None


def test_decode_image_eager_mode(mocker: EagerTensorProtocol) -> None:
    """Test function."""
    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.ops.io.image_io import decode_image, decode_jpeg

    config.eager_mode = True

    class DummyBackend:
        """Test class DummyBackend."""

        def execute_op(self, op_name: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
            """Test function execute_op."""
            return op_name

        @staticmethod
        def decode_jpeg(contents: EagerValue, channels: EagerValue, ratio: EagerValue) -> EagerValue:
            """Test function decode_jpeg."""
            return "jpeg"

    cast("Callable[..., EagerTensorProtocol]", mocker.patch)("ml_switcheroo_compiler.backends.registry.get_active_backend", return_value=cast("Callable[..., EagerTensorProtocol]", DummyBackend)())

    assert cast("EagerTensorProtocol", decode_image("x")) == cast("EagerTensorProtocol", "DecodeImage")
    assert cast("EagerTensorProtocol", decode_jpeg("x")) == cast("EagerTensorProtocol", "jpeg")

    config.eager_mode = False


def test_decode_image_infer_shape() -> None:
    """Test function."""
    from ml_switcheroo_compiler.ops.io.image_io import DecodeImage

    op = DecodeImage()

    assert op.infer_shape() == ()

    class DummyTensor:
        """Test class DummyTensor."""

        def __init__(self, shape: EagerValue) -> None:
            """Test function __init__."""
            self.shape = shape

    t1 = DummyTensor((2, 3))
    t2 = DummyTensor((2, 1))

    assert cast("Callable[..., EagerTensorProtocol]", op.infer_shape)(t1, t2) == (2, 3)


def test_decode_image_non_eager() -> None:
    """Test function."""
    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.ops.io.image_io import decode_image

    config.eager_mode = False

    class DummyNode:
        """Test class DummyNode."""

        shape = ()
        dtype = "float32"

    res = cast("Callable[..., EagerTensorProtocol]", decode_image)(cast("Callable[..., EagerTensorProtocol]", DummyNode)())
    assert res is not None
