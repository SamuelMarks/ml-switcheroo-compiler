"""Test vis utils coverage."""

from unittest.mock import MagicMock

import numpy as np

from ml_switcheroo_compiler.utils.vis_utils import array_to_img, img_to_array, load_img, model_to_dot, save_img


def test_array_to_img_coverage(monkeypatch):
    import sys

    mock_pil = MagicMock()
    mock_pil.Image.fromarray = MagicMock(return_value="mock_image")
    mock_pil.Image.new = MagicMock(return_value="mock_new_image")
    monkeypatch.setitem(sys.modules, "PIL", mock_pil)

    assert array_to_img(np.array([1, 2, 3])) == "mock_image"
    assert array_to_img() == "mock_new_image"

    # Test ImportError fallback
    monkeypatch.setitem(sys.modules, "PIL", None)
    assert array_to_img(np.array([1, 2, 3])) is None


def test_load_img_coverage(monkeypatch):
    import sys

    mock_pil = MagicMock()
    mock_pil.Image.open = MagicMock(return_value="mock_opened_image")
    mock_pil.Image.new = MagicMock(return_value="mock_new_image")
    monkeypatch.setitem(sys.modules, "PIL", mock_pil)

    assert load_img("test_path") == "mock_opened_image"
    assert load_img() == "mock_new_image"

    # Test ImportError fallback
    monkeypatch.setitem(sys.modules, "PIL", None)
    assert load_img("test_path") is None


def test_model_to_dot_coverage(monkeypatch):
    import sys

    mock_pydot = MagicMock()
    mock_pydot.Dot = MagicMock(return_value="mock_dot")
    monkeypatch.setitem(sys.modules, "pydot", mock_pydot)

    assert model_to_dot() == "mock_dot"

    # Test ImportError fallback
    monkeypatch.setitem(sys.modules, "pydot", None)
    assert model_to_dot() is None


def test_img_to_array_coverage(monkeypatch):
    import sys

    class MockImg:
        size = (10, 10)

        def getdata(self):
            return [1, 2, 3]

    # has_np
    assert isinstance(img_to_array(MockImg()), np.ndarray)

    monkeypatch.setitem(sys.modules, "numpy", None)
    assert img_to_array(MockImg()) is None


def test_save_img_coverage(monkeypatch):
    import sys

    mock_pil = MagicMock()
    mock_pil.Image.fromarray = MagicMock(return_value=MagicMock())
    monkeypatch.setitem(sys.modules, "PIL", mock_pil)

    class MockTensor1:
        def numpy(self):
            return np.ones((10, 10, 1))

    # Test tensor with numpy
    save_img("path", MockTensor1())

    class MockTensor2:
        def numpy(self):
            return np.ones((10, 10))

    # Test tensor with numpy hitting the missing branch
    save_img("path", MockTensor2())

    class MockTensorMissing:
        def numpy(self):
            return np.ones((10, 10, 1))

        def save(self, *args, **kwargs):
            pass

    # Test numpy array missing
    monkeypatch.setitem(sys.modules, "numpy", None)
    save_img("path", MockTensorMissing())
