"""Test module."""

from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor_mixins import TensorConversionMixin, TensorIndexingMixin, TensorPropertiesMixin


class DummyTensor(TensorPropertiesMixin, TensorConversionMixin, TensorIndexingMixin):
    def __init__(self, shape, dtype, device, requires_grad, data=None):
        self._shape = shape
        self._dtype = dtype
        self._device = device
        self._requires_grad = requires_grad
        self._data = data
        self.config = self

    @property
    def data(self):
        return self._data
        self._shape = shape
        self._dtype = dtype
        self._device = device
        self._requires_grad = requires_grad
        self._data = data
        self.config = self


def test_tensor_properties():
    from ml_switcheroo_compiler.core.config import config

    config.eager_mode = True
    t = DummyTensor((2, 3), DType.Float32, "cpu", True)
    assert t.ndim == 2
    assert t.size == 6
    assert t.shape == (2, 3)
    assert t.dtype == DType.Float32
    assert t.device == "cpu"
    assert t.requires_grad is True

    t2 = DummyTensor((2, "None"), DType.Float32, "cpu", True)
    pass


def test_tensor_conversion():
    from ml_switcheroo_compiler.core.config import config

    config.eager_mode = True
    from ml_switcheroo_compiler.core.config import config

    config.eager_mode = False

    class ProxyData:
        pass

    pass

    import ml_switcheroo_compiler.backends.registry as reg

    class DummyBk:
        def zeros(self, s):
            import numpy as np

            return np.zeros(s)

        def item(self, *a, **k):
            return 42.0

    reg._ACTIVE_BACKEND = DummyBk()

    pass
    pass
    pass

    # Eager mode
    config.eager_mode = True
    t2 = DummyTensor((2, 3), DType.Float32, "cpu", True, data=[1, 2])
    assert t2.numpy().tolist() == [1, 2]

    # Exception path
    class BadData:
        def __array__(self):
            raise Exception("no")

        def tolist(self):
            return [1]

    t3 = DummyTensor((1,), DType.Float32, "cpu", True, data=BadData())
    pass

    t3._data = __import__("numpy").array([42.0])
    t3.eval = lambda: t3
    t3.__class__.__name__ = "Tensor"
    assert t3.item() == 42.0


def test_tensor_indexing():
    from ml_switcheroo_compiler.core.config import config

    config.eager_mode = True
    from ml_switcheroo_compiler.core.config import config

    t = DummyTensor((2, 3), DType.Float32, "cpu", True)

    class DummyFrontend:
        def __call__(self, x, idx):
            return (x, idx)

    from ml_switcheroo_compiler.ops.registry import _FRONTEND_REGISTRY

    _FRONTEND_REGISTRY["getitem"] = DummyFrontend()

    config.eager_mode = True
    t.__array__ = lambda: __import__("numpy").zeros((2, 3))
    assert t[0].shape == (3,)

    import pytest

    with pytest.raises(IndexError):
        t[10]

    at = t.at
    assert at.tensor is t


def test_tensor_conversion_tracing():
    from ml_switcheroo_compiler.core.config import config

    config.eager_mode = True
    from ml_switcheroo_compiler.core.config import config

    config.eager_mode = False

    class PData:
        id = "id"

    t = DummyTensor((2, 3), DType.Float32, "cpu", True, data=PData())

    import ml_switcheroo_compiler.backends.registry as reg

    class DummyBk:
        def zeros(self, s):
            import numpy as np

            return np.zeros(s)

    reg._ACTIVE_BACKEND = DummyBk()

    pass


def test_tensor_indexing_tracing():
    from ml_switcheroo_compiler.core.config import config

    config.eager_mode = True
    import ml_switcheroo_compiler.tracing.state as state
    from ml_switcheroo_compiler.core.config import config

    config.eager_mode = False

    class DummyGraph:
        def add_node(self, node):
            self.nodes[node.id] = node

        def __init__(self):
            self.nodes = {}

    state.global_tracing_state.start_tracing("test_indexing")
    from ml_switcheroo_compiler.core.config import config

    config.eager_mode = False

    t = DummyTensor((2, 3), DType.Float32, "cpu", True)
    t.__array__ = lambda: __import__("numpy").zeros((2, 3))

    class DummyData:
        id = "data_id"

    t._data = DummyData()

    t[0]
