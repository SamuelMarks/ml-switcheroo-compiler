import pytest

import ml_switcheroo_compiler.backends.registry as registry
from ml_switcheroo_compiler.ops.io.image_io import _decode_image_with_pil, decode_bmp, decode_gif, decode_png


def test_decode_image_with_pil():
    """Test function."""
    # Empty data
    t = _decode_image_with_pil(None)
    assert t.data is None

    # Exception
    with pytest.raises(ValueError, match="Failed to decode image"):
        _decode_image_with_pil(b"not_an_image")

    # str path
    import os
    import tempfile

    from PIL import Image

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        img = Image.new("RGB", (10, 10))
        img.save(f, format="PNG")
        path = f.name

    try:
        t1 = _decode_image_with_pil(path, channels=1)
        assert t1.shape == (10, 10)  # L

        t3 = _decode_image_with_pil(path, channels=3)
        assert t3.shape == (10, 10, 3)  # RGB

        t4 = _decode_image_with_pil(path, channels=4)
        assert t4.shape == (10, 10, 4)  # RGBA
    finally:
        os.remove(path)


def test_decode_image_backend(monkeypatch):
    """Test function."""

    class DummyBackend:
        @staticmethod
        def decode_png(*args, **kwargs):
            return "png"

        @staticmethod
        def decode_gif(*args, **kwargs):
            return "gif"

        @staticmethod
        def decode_bmp(*args, **kwargs):
            return "bmp"

    monkeypatch.setattr(registry, "get_active_backend", lambda: DummyBackend)

    assert decode_png(b"", channels=3) == "png"
    assert decode_gif(b"") == "gif"
    assert decode_bmp(b"", channels=3) == "bmp"


def test_decode_image_fallback():
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


def test_decode_image_eager_mode(mocker):
    """Test function."""
    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.ops.io.image_io import decode_image, decode_jpeg

    config.eager_mode = True

    class DummyBackend:
        def execute_op(self, op_name, *args, **kwargs):
            return op_name

        @staticmethod
        def decode_jpeg(contents, channels, ratio):
            return "jpeg"

    mocker.patch("ml_switcheroo_compiler.backends.registry.get_active_backend", return_value=DummyBackend())

    assert decode_image("x") == "DecodeImage"
    assert decode_jpeg("x") == "jpeg"

    config.eager_mode = False


def test_decode_image_infer_shape():
    """Test function."""
    from ml_switcheroo_compiler.ops.io.image_io import DecodeImage

    op = DecodeImage()

    assert op.infer_shape() == ()

    class DummyTensor:
        def __init__(self, shape):
            self.shape = shape

    t1 = DummyTensor((2, 3))
    t2 = DummyTensor((2, 1))

    assert op.infer_shape(t1, t2) == (2, 3)


def test_decode_image_non_eager():
    """Test function."""
    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.ops.io.image_io import decode_image

    config.eager_mode = False

    class DummyNode:
        shape = ()
        dtype = "float32"

    res = decode_image(DummyNode())
    assert res is not None
