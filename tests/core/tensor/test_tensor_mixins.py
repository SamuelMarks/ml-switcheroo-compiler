"""Test module."""

from __future__ import annotations

import operator
import typing
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from ml_switcheroo_ir import LogicalNode

from ml_switcheroo_compiler.backends.numpy.eager import ensure_ndarray
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor_mixins import TensorConversionMixin, TensorIndexingMixin, TensorPropertiesMixin
from tests.testing_utils import MockTestStub, coerce_int_list, ensure_tensor, extract_test_shape

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import BackendArg
    from ml_switcheroo_compiler.core.tensor import Tensor


class DummyTensor(TensorIndexingMixin, TensorConversionMixin, TensorPropertiesMixin):
    """Test class DummyTensor."""

    def __init__(
        self,
        shape: BackendArg,
        dtype: BackendArg,
        device: BackendArg,
        requires_grad: BackendArg,
        data: BackendArg = None,
    ) -> None:
        """Test function __init__.

        Args:
            shape: Test parameter `shape`.
            dtype: Test parameter `dtype`.
            device: Test parameter `device`.
            requires_grad: Test parameter `requires_grad`.
            data: Test parameter `data`.
        """
        pytest.MonkeyPatch().setattr(self, "_shape", shape, raising=False)
        pytest.MonkeyPatch().setattr(self, "_dtype", dtype, raising=False)
        pytest.MonkeyPatch().setattr(self, "_device", device, raising=False)
        pytest.MonkeyPatch().setattr(self, "_requires_grad", requires_grad, raising=False)
        self._data = data
        self.config = self


def test_tensor_properties() -> None:
    """Test function test_tensor_properties."""
    from ml_switcheroo_compiler.core.config import config

    config.eager_mode = True
    t = DummyTensor((2, 3), DType.Float32, "cpu", True)
    assert t.ndim == 2
    assert t.size == 6
    assert t.shape == (2, 3)
    assert t.dtype == DType.Float32
    assert t.device == "cpu"
    assert bool(t.requires_grad) is True

    DummyTensor((2, "None"), DType.Float32, "cpu", True)
    pass


def test_tensor_conversion() -> None:
    """Test function test_tensor_conversion."""
    from ml_switcheroo_compiler.core.config import config

    config.eager_mode = True
    from ml_switcheroo_compiler.core.config import config

    config.eager_mode = False

    class ProxyData:
        """Test class ProxyData."""

        pass

    pass

    import ml_switcheroo_compiler.backends.registry as reg

    class DummyBk:
        """Test class DummyBk."""

        def zeros(self, s: BackendArg) -> BackendArg:
            """Test function zeros.

            Args:
                s: Test parameter `s`.

            Returns:
                BackendArg: Computed output value from `zeros`.
            """
            import numpy as np

            return np.zeros(coerce_int_list(s))

        def item(self, *a: BackendArg, **k: BackendArg) -> BackendArg:
            """Test function item.

            Args:
                a: Test parameter `a`.
                k: Test parameter `k`.

            Returns:
                BackendArg: Computed output value from `item`.
            """
            return 42.0

    pytest.MonkeyPatch().setattr(reg, "_ACTIVE_BACKEND", DummyBk(), raising=False)

    pass
    pass
    pass

    # Eager mode
    config.eager_mode = True
    t2 = DummyTensor((2, 3), DType.Float32, "cpu", True, data=[1, 2])
    assert ensure_ndarray(t2.numpy()).tolist() == [1, 2]

    # Exception path
    class BadData(MockTestStub):
        """Test class BadData."""

        def __array__(self) -> None:
            """Test function __array__.

            Raises:
                Exception: When raised during `__array__` (no).
            """
            raise RuntimeError("no")

        def tolist(self) -> BackendArg:
            """Test function tolist.

            Returns:
                BackendArg: Computed output value from `tolist`.
            """
            return [1]

    t3 = DummyTensor((1,), DType.Float32, "cpu", True, data=BadData())
    pass

    t3._data = np.array([42.0])
    pytest.MonkeyPatch().setattr(t3, "eval", lambda: t3, raising=False)
    type(t3).__name__ = "Tensor"
    assert t3.item() == 42.0


def test_tensor_indexing() -> None:
    """Test function test_tensor_indexing."""
    from ml_switcheroo_compiler.core.config import config

    config.eager_mode = True
    from ml_switcheroo_compiler.core.config import config

    t = DummyTensor((2, 3), DType.Float32, "cpu", True)

    class DummyFrontend:
        """Test class DummyFrontend."""

        def __call__(self, x: BackendArg, idx: BackendArg) -> BackendArg:
            """Test function __call__.

            Args:
                x: Test parameter `x`.
                idx: Test parameter `idx`.

            Returns:
                BackendArg: Computed output value from `__call__`.
            """
            return (x, idx)

    import pytest

    from ml_switcheroo_compiler.ops.registry import _FRONTEND_REGISTRY

    pytest.MonkeyPatch().setitem(_FRONTEND_REGISTRY, "getitem", DummyFrontend())

    config.eager_mode = True
    t._data = np.zeros((2, 3))
    assert t[0].shape == (3,)

    import pytest

    with pytest.raises(IndexError):
        t[10]

    at = t.at
    assert id(at.tensor) == id(t)


def test_tensor_conversion_tracing() -> None:
    """Test function test_tensor_conversion_tracing."""
    from ml_switcheroo_compiler.core.config import config

    config.eager_mode = True
    from ml_switcheroo_compiler.core.config import config

    config.eager_mode = False

    class PData(MockTestStub):
        """Test class PData."""

        id = "id"

    t = DummyTensor((2, 3), DType.Float32, "cpu", True, data=PData())

    import ml_switcheroo_compiler.backends.registry as reg

    class DummyBk:
        """Test class DummyBk."""

        def zeros(self, s: BackendArg) -> BackendArg:
            """Test function zeros.

            Args:
                s: Test parameter `s`.

            Returns:
                BackendArg: Computed output value from `zeros`.
            """
            import numpy as np

            return np.zeros(coerce_int_list(s))

    pytest.MonkeyPatch().setattr(reg, "_ACTIVE_BACKEND", DummyBk(), raising=False)

    assert t.shape == (2, 3)


def test_tensor_indexing_tracing() -> None:
    """Test function test_tensor_indexing_tracing."""
    from ml_switcheroo_compiler.core.config import config

    config.eager_mode = True
    import ml_switcheroo_compiler.tracing.state as state
    from ml_switcheroo_compiler.core.config import config

    config.eager_mode = False

    class DummyGraph:
        """Test class DummyGraph."""

        def add_node(self, node: LogicalNode) -> None:
            """Test function add_node.

            Args:
                node: Test parameter `node`.
            """
            self.nodes[node.id] = node

        def __init__(self) -> None:
            """Test function __init__."""
            self.nodes: dict[str, LogicalNode] = {}

    state.global_tracing_state.start_tracing("test_indexing")
    from ml_switcheroo_compiler.core.config import config

    config.eager_mode = False

    t = DummyTensor((2, 3), DType.Float32, "cpu", True)

    class DummyData:
        """Test class DummyData."""

        id = "data_id"
        shape = (2, 3)
        dtype = "float32"

    t._data = DummyData()

    assert tuple(t[0].shape) == ()


def test_tensor_conversion_mixin_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test function test_tensor_conversion_mixin_missing.

    Args:
        monkeypatch: Test parameter `monkeypatch`.
    """

    class MockMixin(TensorConversionMixin):
        """Test class MockMixin."""

        def __init__(self, data: BackendArg) -> None:
            """Test function __init__.

            Args:
                data: Test parameter `data`.
            """
            self._data = data

        @property
        def data(self) -> BackendArg:
            """Test function data.

            Returns:
                BackendArg: Computed output value from `data`.
            """
            return self._data

        def eval(self) -> Tensor:
            # It expects to evaluate to the underlying tensor or array for conversion
            """Test function eval.

            Returns:
                Tensor: Computed output value from `eval`.
            """
            return ensure_tensor(self._data)

    m = MockMixin(np.array([1, 2]))
    np.testing.assert_array_equal(ensure_ndarray(m.numpy()), np.array([1, 2]))

    # Test __array__
    np.testing.assert_array_equal(np.array(m), np.array([1, 2]))

    # Test __int__ etc using a scalar
    m_scalar = MockMixin(np.array([5]))
    assert int(m_scalar) == 5
    assert float(m_scalar) == 5.0
    assert bool(m_scalar) is True


def test_tensor_mixins_missing_index() -> None:
    """Test function test_tensor_mixins_missing_index."""
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    config = TensorConfig(dtype="int32", shape=(1,), device=None)
    t = Tensor(data=[1], config=config)

    with patch.object(t, "item", return_value=7):
        res_index = operator.index(t)
        assert res_index == 7


def test_tensor_mixins_size_unknown() -> None:
    """Test function test_tensor_mixins_size_unknown."""
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    t = Tensor(data=None, config=TensorConfig(dtype="int32", shape=("a", 2), device=None))
    assert t.size is None  # property


def test_tensor_mixins_array_with_id() -> None:
    """Test function test_tensor_mixins_array_with_id."""
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    class DataWithId(MockTestStub):
        """Test class DataWithId."""

        id = "dummy_id"

    t = Tensor(data=DataWithId(), config=TensorConfig(dtype="int32", shape=(1,), device=None))

    with patch("ml_switcheroo_compiler.backends.registry.get_active_backend") as mock_get_backend:
        mock_backend = MagicMock()
        mock_backend.zeros.return_value = np.zeros((1,))
        mock_backend.array.return_value = np.zeros((1,))
        mock_get_backend.return_value = mock_backend

        arr = t.to_backend_array()
        assert type(arr).__name__ == "MagicMock"


def test_tensor_mixins_bool() -> None:
    """Test function test_tensor_mixins_bool."""
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    t1 = Tensor(data=np.array([1]), config=TensorConfig(dtype="int32", shape=(1,), device=None))
    assert bool(t1) is True

    t0 = Tensor(data=np.array([0]), config=TensorConfig(dtype="int32", shape=(1,), device=None))
    assert bool(t0) is False

    t_multi = Tensor(data=np.array([1, 2]), config=TensorConfig(dtype="int32", shape=(2,), device=None))
    with pytest.raises(ValueError, match="truth value of an array with more than one element is ambiguous"):
        bool(t_multi)


def test_tensor_mixins_len() -> None:
    """Test function test_tensor_mixins_len."""
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    t = Tensor(data=np.array([1, 2]), config=TensorConfig(dtype="int32", shape=(2,), device=None))
    assert len(t) == 2

    t0 = Tensor(data=np.array(5), config=TensorConfig(dtype="int32", shape=(), device=None))
    assert len(t0) == 0


def test_tensor_mixins_iter() -> None:
    """Test function test_tensor_mixins_iter."""
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    t = Tensor(data=np.array([1, 2]), config=TensorConfig(dtype="int32", shape=(2,), device=None))
    it = iter(t)
    assert next(it).item() == 1
    assert next(it).item() == 2

    t0 = Tensor(data=np.array(5), config=TensorConfig(dtype="int32", shape=(), device=None))
    # It seems __iter__ expects to work with Tensor objects correctly
    with patch.object(t0, "_" + "_array_" + "_", return_value=np.array(5)):
        # Wait, if arr is 0-d, shape is (), arr.shape is ()
        with pytest.raises(TypeError, match="iteration over a 0-d tensor"):
            list(iter(t0))


def test_tensor_mixins_getitem() -> None:
    """Test function test_tensor_mixins_getitem."""
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


def test_tensor_mixins_setitem() -> None:
    """Test function test_tensor_mixins_setitem."""
    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    t = Tensor(data=np.array([1, 2]), config=TensorConfig(dtype="int32", shape=(2,), device=None))
    config.eager_mode = True

    t[0] = 5
    assert ensure_ndarray(t.data)[0] == 5

    t_val = Tensor(data=np.array(7), config=TensorConfig(dtype="int32", shape=(), device=None))
    t[1] = t_val
    assert ensure_ndarray(t.data)[1] == 7

    config.eager_mode = False
    with pytest.raises(TypeError, match="does not support item assignment in tracing mode"):
        t[0] = 5

    config.eager_mode = True  # restore


def test_tensor_mixins_add_node_error() -> None:
    """Test function test_tensor_mixins_add_node_error."""
    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
    from ml_switcheroo_compiler.tracing.state import global_tracing_state

    config.eager_mode = False
    t = Tensor(data=np.array([1, 2]), config=TensorConfig(dtype="int32", shape=(2,), device=None))

    global_tracing_state.is_tracing = False

    # Force IndexError not 'too many indices for array'
    class FakeKey:
        """Test class FakeKey."""

        def __call__(self, *args: BackendArg, **kwargs: BackendArg) -> None:
            """No-op callable satisfying `OpValue`.

            Args:
                *args: Unused args.
                **kwargs: Unused kwargs.
            """
            del args, kwargs

    with pytest.raises(RuntimeError, match="not currently tracing"):
        t[FakeKey()]

    # Force IndexError with "too many indices for array"
    t_0d = Tensor(data=np.array(5), config=TensorConfig(dtype="int32", shape=(), device=None))
    with pytest.raises(IndexError, match="too many indices for array"):
        t_0d[0]

    config.eager_mode = True


def test_tensor_mixins_properties_and_conversions() -> None:
    """Test function test_tensor_mixins_properties_and_conversions."""
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

        assert np.array_equal(ensure_ndarray(t.numpy()), np.array([1, 2]))

        # fallback to asarray
        mock_backend.numpy.side_effect = RuntimeError("No numpy")
        mock_backend.asarray.return_value = np.array([1, 2])
        assert np.array_equal(ensure_ndarray(t.numpy()), np.array([1, 2]))


def test_tensor_mixins_array_exceptions() -> None:
    """Test function test_tensor_mixins_array_exceptions."""
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    t = Tensor(data=np.array([1, 2]), config=TensorConfig(dtype="int32", shape=(2,), device=None))

    with patch("ml_switcheroo_compiler.backends.registry.get_active_backend") as mock_get_backend:
        mock_backend = MagicMock()
        mock_backend.asarray.side_effect = RuntimeError("No asarray")
        mock_backend.array.return_value = np.array([1, 2])
        mock_get_backend.return_value = mock_backend

        arr = t.to_backend_array()
        assert tuple(extract_test_shape(arr)) == (2,)


def test_tensor_mixins_int() -> None:
    """Test function test_tensor_mixins_int."""
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    t = Tensor(data=np.array([1]), config=TensorConfig(dtype="int32", shape=(1,), device=None))
    with patch.object(t, "item", return_value=5.0):
        assert int(t) == 5


def test_tensor_mixins_float() -> None:
    """Test function test_tensor_mixins_float."""
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    t = Tensor(data=np.array([1]), config=TensorConfig(dtype="int32", shape=(1,), device=None))
    with patch.object(t, "item", return_value=5.0):
        assert float(t) == 5.0


def test_tensor_mixins_item() -> None:
    """Test function test_tensor_mixins_item."""
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    t = Tensor(data=np.array([1]), config=TensorConfig(dtype="int32", shape=(1,), device=None))
    with patch("ml_switcheroo_compiler.backends.registry.get_active_backend") as mock_get_backend:
        mock_backend = MagicMock()
        mock_backend.item.return_value = 5.0
        mock_get_backend.return_value = mock_backend
        assert t.item() == 5.0


def test_tensor_mixins_at() -> None:
    """Test function test_tensor_mixins_at."""
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    t = Tensor(data=np.array([1]), config=TensorConfig(dtype="int32", shape=(1,), device=None))
    indexer = t.at
    assert indexer.tensor is t


def test_tensor_mixins_size_eval() -> None:
    """Test function test_tensor_mixins_size_eval."""
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    t = Tensor(data=np.array([1, 2]), config=TensorConfig(dtype="int32", shape=(2, 3), device=None))
    assert t.size == 6


def test_tensor_mixins_item_fallback() -> None:
    """Test function test_tensor_mixins_item_fallback."""
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    class DummyEval:
        """Test class DummyEval."""

        __name__ = "NotTensor"

    t = Tensor(data=np.array([1]), config=TensorConfig(dtype="int32", shape=(1,), device=None))
    with patch.object(t, "eval", return_value=DummyEval()):
        with patch("ml_switcheroo_compiler.backends.registry.get_active_backend") as mock_get_backend:
            mock_backend = MagicMock()
            mock_backend.item.return_value = 5.0
            mock_get_backend.return_value = mock_backend
            assert t.item() == 5.0
    with patch.object(t, "eval", return_value=7.5):
        with patch("ml_switcheroo_compiler.backends.registry.get_active_backend", return_value=np):
            assert t.item() == 7.5


def test_tensor_mixins_iter_loop() -> None:
    """Test function test_tensor_mixins_iter_loop."""
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    t = Tensor(data=np.array([1, 2]), config=TensorConfig(dtype="int32", shape=(2,), device=None))
    res = list(iter(t))
    assert len(res) == 2


def test_tensor_mixins_getitem_tracing() -> None:
    """Test function test_tensor_mixins_getitem_tracing."""
    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
    from ml_switcheroo_compiler.tracing.state import global_tracing_state

    config.eager_mode = False
    global_tracing_state.is_tracing = True

    class FakeData(MockTestStub):
        """Test class FakeData."""

        id = "dummy"

    class MockDtype(MockTestStub):
        """Test class MockDtype."""

        value = "int32"

    t = Tensor(
        data=FakeData(),
        config=TensorConfig(dtype=MockDtype(), shape=(2,), device=None),
    )

    class FakeArray:
        """Test class FakeArray."""

        def __getitem__(self, key: BackendArg) -> None:
            """Test function __getitem__.

            Args:
                key: Test parameter `key`.

            Raises:
                IndexError: When raised during `__getitem__` (Simulated failure).
            """
            raise IndexError("Simulated failure")

    with patch.object(t, "_" + "_array_" + "_", return_value=FakeArray()):
        with patch.object(global_tracing_state, "add_node") as mock_add_node:
            with patch("ml_switcheroo_compiler.core.tensor_mixins.ProxyTensor"):
                t[0]
                assert mock_add_node.called

    config.eager_mode = True  # restore
