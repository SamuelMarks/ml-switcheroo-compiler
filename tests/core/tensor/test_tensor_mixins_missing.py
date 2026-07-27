import sys

import numpy as np

from ml_switcheroo_compiler.core.tensor_mixins import TensorConversionMixin


def test_tensor_conversion_mixin_missing(monkeypatch):
    class MockMixin(TensorConversionMixin):
        def __init__(self, data):
            self._data = data

    import importlib.util

    original_find_spec = importlib.util.find_spec

    def mock_find_spec(name, package=None):
        if name == "ml_switcheroo_compiler.backends.numpy.utils":

            class DummySpec:
                pass

            return DummySpec()
        return original_find_spec(name, package)

    monkeypatch.setattr(sys.modules["importlib.util"], "find_spec", mock_find_spec)

    m = MockMixin(np.array([1, 2]))
    m.numpy()
