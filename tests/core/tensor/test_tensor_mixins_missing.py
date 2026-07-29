import numpy as np

from ml_switcheroo_compiler.core.tensor_mixins import TensorConversionMixin


def test_tensor_conversion_mixin_missing(monkeypatch):
    class MockMixin(TensorConversionMixin):
        def __init__(self, data):
            self._data = data

        @property
        def data(self):
            return self._data

        def eval(self):
            # It expects to evaluate to the underlying tensor or array for conversion
            return self._data

    m = MockMixin(np.array([1, 2]))
    np.testing.assert_array_equal(m.numpy(), np.array([1, 2]))

    # Test __array__
    np.testing.assert_array_equal(np.array(m), np.array([1, 2]))

    # Test __int__ etc using a scalar
    m_scalar = MockMixin(np.array([5]))
    assert int(m_scalar) == 5
    assert float(m_scalar) == 5.0
    assert bool(m_scalar) is True
