"""Test module."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ml_switcheroo_compiler.backends.eager_registry import EagerOpRegistry


def test_eager_registry():
    r = EagerOpRegistry()
    assert r.get("foo") is None

    @r.register("foo")
    def foo(x):
        return x + 1

    assert r.get("foo") is foo

    assert r.dispatch("foo", 1) == 2

    with pytest.raises(ValueError):
        r.dispatch("bar", 1)


def test_eager_registry_fallback():
    from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry

    r = EagerOpRegistry()

    @global_eager_registry.register("global_foo_test")
    def gfoo(x):
        return x + 2

    assert r.dispatch("global_foo_test", 3) == 5


def test_global_eager_registry_fallback_missing():
    import pytest

    from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry

    with pytest.raises(ValueError):
        global_eager_registry.dispatch("op_does_not_exist_ever_123")


def test_decode_image_eager():
    from ml_switcheroo_compiler.backends.numpy.eager.math_advanced.math_string_io import _np_decode_image_camel

    # Test empty args
    assert len(_np_decode_image_camel(None)) == 0

    # Test not bytes
    with pytest.raises(RuntimeError, match="Expected bytes"):
        _np_decode_image_camel(None, np.array([1, 2, 3]))

    # Test success path with mocked PIL
    mock_img = MagicMock()
    mock_img.convert.return_value = mock_img
    mock_img.__array__ = MagicMock(return_value=np.ones((10, 10, 3), dtype=np.uint8))

    with patch("PIL.Image.open", return_value=mock_img):
        # Default channels
        res = _np_decode_image_camel(None, b"fake_data")
        assert res.shape == (10, 10, 3)

        # Channels = 1
        res1 = _np_decode_image_camel(None, b"fake_data", channels=1)
        assert res1.shape == (10, 10, 3, 1)  # Because mock array is already 3D

        # Channels = 4
        res4 = _np_decode_image_camel(None, b"fake_data", channels=4)
        assert res4.shape == (10, 10, 3)


def test_encode_image_eager():
    from ml_switcheroo_compiler.backends.numpy.eager.math_advanced.math_string_io import _np_encode_image_camel

    # Test empty args
    res = _np_encode_image_camel(None)
    assert res.item() == b""

    mock_img = MagicMock()

    with patch("PIL.Image.fromarray", return_value=mock_img):
        # Shape missing last channel
        arr = np.ones((10, 10, 1), dtype=np.uint8)
        res = _np_encode_image_camel(None, arr)
        assert isinstance(res, np.ndarray)

        # Non-uint8 dtype
        arr2 = np.ones((10, 10, 3), dtype=np.float32)
        res2 = _np_encode_image_camel(None, arr2)
        assert isinstance(res2, np.ndarray)

        # Trigger exception
        with patch("io.BytesIO", side_effect=Exception("mock error")):
            with pytest.raises(RuntimeError):
                _np_encode_image_camel(None, arr2)


def test_load_vision_formats_missing():
    from ml_switcheroo_compiler.backends.numpy.eager.math_advanced.math_string_io import _load_vision_formats

    with patch("os.path.exists", return_value=False):
        assert _load_vision_formats() == {}
