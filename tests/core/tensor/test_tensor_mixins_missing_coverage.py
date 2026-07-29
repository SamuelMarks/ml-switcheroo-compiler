from unittest.mock import MagicMock, patch

import numpy as np
import pytest


def test_tensor_mixins_missing_index():
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    config = TensorConfig(dtype="int32", shape=(1,), device=None)
    t = Tensor(data=[1], config=config)

    with patch.object(t, "item", return_value=7):
        res_index = t.__index__()
        assert res_index == 7


def test_tensor_mixins_size_unknown():
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    t = Tensor(data=None, config=TensorConfig(dtype="int32", shape=("a", 2), device=None))
    assert t.size is None  # property


def test_tensor_mixins_array_with_id():
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    class DataWithId:
        id = "dummy_id"

    t = Tensor(data=DataWithId(), config=TensorConfig(dtype="int32", shape=(1,), device=None))

    with patch("ml_switcheroo_compiler.backends.registry.get_active_backend") as mock_get_backend:
        mock_backend = MagicMock()
        mock_backend.zeros.return_value = np.zeros((1,))
        mock_backend.array.return_value = np.zeros((1,))
        mock_get_backend.return_value = mock_backend

        arr = t.__array__()
        assert arr is not None


def test_tensor_mixins_bool():
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    t1 = Tensor(data=np.array([1]), config=TensorConfig(dtype="int32", shape=(1,), device=None))
    assert bool(t1) is True

    t0 = Tensor(data=np.array([0]), config=TensorConfig(dtype="int32", shape=(1,), device=None))
    assert bool(t0) is False

    t_multi = Tensor(data=np.array([1, 2]), config=TensorConfig(dtype="int32", shape=(2,), device=None))
    with pytest.raises(ValueError, match="truth value of an array with more than one element is ambiguous"):
        bool(t_multi)


def test_tensor_mixins_len():
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    t = Tensor(data=np.array([1, 2]), config=TensorConfig(dtype="int32", shape=(2,), device=None))
    assert len(t) == 2

    t0 = Tensor(data=np.array(5), config=TensorConfig(dtype="int32", shape=(), device=None))
    assert len(t0) == 0


def test_tensor_mixins_iter():
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    t = Tensor(data=np.array([1, 2]), config=TensorConfig(dtype="int32", shape=(2,), device=None))
    it = iter(t)
    assert next(it).item() == 1
    assert next(it).item() == 2

    t0 = Tensor(data=np.array(5), config=TensorConfig(dtype="int32", shape=(), device=None))
    # It seems __iter__ expects to work with Tensor objects correctly
    with patch.object(t0, "__array__", return_value=np.array(5)):
        # Wait, if arr is 0-d, shape is (), arr.shape is ()
        with pytest.raises(TypeError, match="iteration over a 0-d tensor"):
            list(iter(t0))


def test_tensor_mixins_getitem():
    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    config.eager_mode = True
    t_data = np.array([1, 2])
    t = Tensor(data=t_data, config=TensorConfig(dtype="int32", shape=(2,), device=None))

    # getitem with tensor key
    t_key = Tensor(data=np.array([0]), config=TensorConfig(dtype="int32", shape=(1,), device=None))
    res = t[t_key]
    assert isinstance(res, Tensor)

    # tuple of tensors
    t_2d = Tensor(data=np.array([[1, 2], [3, 4]]), config=TensorConfig(dtype="int32", shape=(2, 2), device=None))
    res2 = t_2d[(t_key, t_key)]
    assert isinstance(res2, Tensor)

    # Test index error propagation in eager mode
    with pytest.raises(IndexError):
        t[5]


def test_tensor_mixins_setitem():
    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    t = Tensor(data=np.array([1, 2]), config=TensorConfig(dtype="int32", shape=(2,), device=None))
    config.eager_mode = True

    t[0] = 5
    assert t.data[0] == 5

    t_val = Tensor(data=np.array(7), config=TensorConfig(dtype="int32", shape=(), device=None))
    t[1] = t_val
    assert t.data[1] == 7

    config.eager_mode = False
    with pytest.raises(TypeError, match="does not support item assignment in tracing mode"):
        t[0] = 5

    config.eager_mode = True  # restore


def test_tensor_mixins_add_node_error():
    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
    from ml_switcheroo_compiler.tracing.state import global_tracing_state

    config.eager_mode = False
    t = Tensor(data=np.array([1, 2]), config=TensorConfig(dtype="int32", shape=(2,), device=None))

    global_tracing_state.is_tracing = False

    # Force IndexError not 'too many indices for array'
    class FakeKey:
        pass

    with pytest.raises(RuntimeError, match="not currently tracing"):
        res = t[FakeKey()]

    # Force IndexError with "too many indices for array"
    t_0d = Tensor(data=np.array(5), config=TensorConfig(dtype="int32", shape=(), device=None))
    with pytest.raises(IndexError, match="too many indices for array"):
        res = t_0d[0]

    config.eager_mode = True


def test_tensor_mixins_properties_and_conversions():
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    t = Tensor(data=np.array([1, 2]), config=TensorConfig(dtype="int32", shape=(2,), device=None))
    assert t.ndim == 1
    assert t.requires_grad is False

    t_unknown = Tensor(data=None, config=TensorConfig(dtype="int32", shape=("a", 2), device=None))
    assert t_unknown.size is None

    # numpy conversion
    with patch("ml_switcheroo_compiler.backends.registry.get_active_backend") as mock_get_backend:
        mock_backend = MagicMock()
        mock_backend.numpy.return_value = np.array([1, 2])
        mock_get_backend.return_value = mock_backend

        assert np.array_equal(t.numpy(), np.array([1, 2]))

        # fallback to asarray
        mock_backend.numpy.side_effect = Exception("No numpy")
        mock_backend.asarray.return_value = np.array([1, 2])
        assert np.array_equal(t.numpy(), np.array([1, 2]))


def test_tensor_mixins_array_exceptions():
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    t = Tensor(data=np.array([1, 2]), config=TensorConfig(dtype="int32", shape=(2,), device=None))

    with patch("ml_switcheroo_compiler.backends.registry.get_active_backend") as mock_get_backend:
        mock_backend = MagicMock()
        mock_backend.asarray.side_effect = Exception("No asarray")
        mock_backend.array.return_value = np.array([1, 2])
        mock_get_backend.return_value = mock_backend

        arr = t.__array__()
        assert arr is not None


def test_tensor_mixins_int():
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    t = Tensor(data=np.array([1]), config=TensorConfig(dtype="int32", shape=(1,), device=None))
    with patch.object(t, "item", return_value=5.0):
        assert int(t) == 5


def test_tensor_mixins_float():
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    t = Tensor(data=np.array([1]), config=TensorConfig(dtype="int32", shape=(1,), device=None))
    with patch.object(t, "item", return_value=5.0):
        assert float(t) == 5.0


def test_tensor_mixins_item():
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    t = Tensor(data=np.array([1]), config=TensorConfig(dtype="int32", shape=(1,), device=None))
    with patch("ml_switcheroo_compiler.backends.registry.get_active_backend") as mock_get_backend:
        mock_backend = MagicMock()
        mock_backend.item.return_value = 5.0
        mock_get_backend.return_value = mock_backend
        assert t.item() == 5.0


def test_tensor_mixins_at():
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    t = Tensor(data=np.array([1]), config=TensorConfig(dtype="int32", shape=(1,), device=None))
    indexer = t.at
    assert indexer.tensor is t


def test_tensor_mixins_size_eval():
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    t = Tensor(data=np.array([1, 2]), config=TensorConfig(dtype="int32", shape=(2, 3), device=None))
    assert t.size == 6


def test_tensor_mixins_item_fallback():
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    class DummyEval:
        __name__ = "NotTensor"

    t = Tensor(data=np.array([1]), config=TensorConfig(dtype="int32", shape=(1,), device=None))
    with patch.object(t, "eval", return_value=DummyEval()):
        with patch("ml_switcheroo_compiler.backends.registry.get_active_backend") as mock_get_backend:
            mock_backend = MagicMock()
            mock_backend.item.return_value = 5.0
            mock_get_backend.return_value = mock_backend
            assert t.item() == 5.0


def test_tensor_mixins_iter_loop():
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    t = Tensor(data=np.array([1, 2]), config=TensorConfig(dtype="int32", shape=(2,), device=None))
    res = list(iter(t))
    assert len(res) == 2


def test_tensor_mixins_getitem_tracing():
    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
    from ml_switcheroo_compiler.tracing.state import global_tracing_state

    config.eager_mode = False
    global_tracing_state.is_tracing = True

    class FakeData:
        id = "dummy"

    class MockDtype:
        value = "int32"

    t = Tensor(data=FakeData(), config=TensorConfig(dtype=MockDtype(), shape=(2,), device=None))

    class FakeArray:
        def __getitem__(self, key):
            raise IndexError("Simulated failure")

    with patch.object(t, "__array__", return_value=FakeArray()):
        with patch.object(global_tracing_state, "add_node") as mock_add_node:
            with patch("ml_switcheroo_compiler.core.tensor_mixins.ProxyTensor") as mock_proxy:
                res = t[0]
                assert mock_add_node.called

    config.eager_mode = True  # restore
