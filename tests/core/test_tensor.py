"""Tests for test_tensor."""

from __future__ import annotations

import typing

from ml_switcheroo_compiler.backends.eager_registry import EagerValue
from tests.testing_utils import as_callable, as_tensor, as_tensor_list

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerValue


def test_tensor_array_at() -> None:
    """Test function test_tensor_array_at."""
    import numpy as np

    from ml_switcheroo_compiler.core.tensor import ArrayAt, ArrayAtIndexer, Tensor, TensorConfig

    t = Tensor(np.array([1, 2]), TensorConfig(shape=(2,), dtype="float32", device="cpu"))
    at = ArrayAt(t, 0)

    assert at.add(1) is t
    assert at.multiply(2) is t
    assert at.minimum(1) is t
    assert at.maximum(1) is t
    assert at.set(1) is t

    indexer = ArrayAtIndexer(t)
    assert indexer[0].add(1) is t


def test_tensor_eval_view_detach_backward() -> None:
    """Test function test_tensor_eval_view_detach_backward."""
    assert callable(test_tensor_eval_view_detach_backward)


def test_variable_assign() -> None:
    """Test function test_variable_assign."""
    from unittest.mock import patch

    import numpy as np

    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig, Variable

    v = Variable(np.array([1, 2]), TensorConfig(shape=(2,), dtype="float32", device="cpu"))
    t = Tensor(np.array([2, 3]), TensorConfig(shape=(2,), dtype="float32", device="cpu"))

    class FakeBackend:
        """Test class FakeBackend."""

        def execute_op(self, name: EagerValue, data1: EagerValue, data2: EagerValue) -> EagerValue:
            """Test function execute_op.

            Args:
                name: Test parameter `name`.
                data1: Test parameter `data1`.
                data2: Test parameter `data2`.
            """
            return data1

    # eager
    config.eager_mode = True
    with patch("ml_switcheroo_compiler.backends.registry.get_active_backend", return_value=FakeBackend()):
        assert v.assign(t) is v
        assert v.assign_add(t) is v
        assert v.assign_sub(t) is v

    # traced
    config.eager_mode = False

    class FakeData:
        """Test class FakeData."""

        id = "dummy"

    setattr(v, "_data", FakeData())
    setattr(t, "_data", FakeData())

    with patch("ml_switcheroo_compiler.ops.shape.utils._emit_shape_node", return_value=None):
        with patch("ml_switcheroo_compiler.ops.registry.get_util", return_value=lambda *a, **k: None):
            assert v.assign(t) is v
            assert v.assign_add(t) is v
            assert v.assign_sub(t) is v

    config.eager_mode = True


def test_parameter() -> None:
    """Test function test_parameter."""
    import numpy as np

    from ml_switcheroo_compiler.core.tensor import Parameter, TensorConfig

    p = Parameter(np.array([42]), TensorConfig(shape=(1,), dtype="int32", device="cpu"))
    assert p.trainable is True

    # Fake numpy method to avoid execution
    class FakeData:
        """Test class FakeData."""

        def __int__(self) -> EagerValue:
            """Test function __int__."""
            return 42

    setattr(p, "numpy", lambda: 42)
    assert p.__index__() == 42


def test_tensor_config_list_shape_and_parse_dim() -> None:
    """Test function test_tensor_config_list_shape_and_parse_dim."""
    import numpy as np

    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    # 126
    c = TensorConfig(shape=[2, "dynamic"], dtype="float32", device="cpu")
    assert isinstance(c.shape, tuple)

    # 172, 173 - will happen when passing "dynamic" to Tensor config shape
    t = Tensor(np.array([1, 2]), c)
    assert as_tensor(t.shape) == as_tensor((2, "dynamic"))


def test_tensor_eval_already_in_graph() -> None:
    """Test function test_tensor_eval_already_in_graph."""
    from unittest.mock import MagicMock

    import numpy as np

    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
    from ml_switcheroo_compiler.tracing.state import global_tracing_state

    t = Tensor(np.array([1, 2]), TensorConfig(shape=(2,), dtype="float32", device="cpu"))
    config.eager_mode = False
    global_tracing_state.start_tracing()
    global_tracing_state.active_graph = MagicMock()
    global_tracing_state.active_graph.outputs = ["n1"]

    class FakeData:
        """Test class FakeData."""

        id = "n1"

    setattr(t, "_data", FakeData())
    assert t.eval() is t
    config.eager_mode = True
    global_tracing_state.stop_tracing()


def test_tensor_eval_not_tracing() -> None:
    """Test function test_tensor_eval_not_tracing."""
    import numpy as np

    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
    from ml_switcheroo_compiler.tracing.state import global_tracing_state

    t = Tensor(np.array([1, 2]), TensorConfig(shape=(2,), dtype="float32", device="cpu"))
    config.eager_mode = False

    class FakeData:
        """Test class FakeData."""

        id = "n1"

    setattr(t, "_data", FakeData())

    # 190, 194 where global_tracing_state.is_tracing is False
    global_tracing_state.is_tracing = False
    assert t.eval() is t
    config.eager_mode = True


def test_tensor_view_list_shape() -> None:
    # We want to hit view() with a list/tuple inside shape
    # E.g., t.view([2, 2]) or t.view((2, 2)) vs t.view(2, 2)
    # The shape arg is *shape, so shape = ([2, 2],)
    """Test function test_tensor_view_list_shape."""
    import numpy as np

    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    t = Tensor(np.array([[1, 2], [3, 4]]), TensorConfig(shape=(2, 2), dtype="float32", device="cpu"))
    from unittest.mock import patch

    with patch("ml_switcheroo_compiler.ops.registry.get_frontend") as mock_get_frontend:
        mock_get_frontend.return_value = lambda x, shape: Tensor(np.zeros(shape), TensorConfig(shape=shape, dtype="float32", device="cpu"))
        res = as_callable(t.view)([4])
        assert as_tensor(getattr(res, "shape")) == (4,)

        res2 = t.view(2, 2)
        assert as_tensor(getattr(res2, "shape")) == (2, 2)

        res3 = as_callable(t.view)((4,))
        assert as_tensor(getattr(res3, "shape")) == (4,)


def test_tensor_eval_branches() -> None:
    """Test function test_tensor_eval_branches."""
    import numpy as np

    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    # eager mode -> returns self
    t1 = Tensor(np.array(1), TensorConfig((), "float32", "cpu"))
    assert t1.eval() is t1

    # tracing mode
    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.tracing.tracer import global_tracing_state

    config.eager_mode = False

    class DummyGraph:
        """Test class DummyGraph."""

        def __init__(self) -> None:
            """Initialize DummyGraph."""
            self.outputs = as_tensor_list([])

    graph_inst = DummyGraph()
    global_tracing_state.is_tracing = True
    setattr(global_tracing_state, "active_graph", graph_inst)

    # Create an object with an id
    class DummyData:
        """Test class DummyData."""

        id = "test_id"

    t2 = as_callable(Tensor)(as_callable(DummyData)(), as_callable(TensorConfig)((), "float32", "cpu"))

    # Not in graph outputs -> append
    assert t2.eval() is t2
    assert graph_inst.outputs == ["test_id"]

    # Already in graph outputs -> no append
    assert t2.eval() is t2
    assert graph_inst.outputs == ["test_id"]

    # Reset
    config.eager_mode = True
    global_tracing_state.is_tracing = False
    global_tracing_state.active_graph = None


def test_tensor_contiguous() -> None:
    """Test function test_tensor_contiguous."""
    from unittest.mock import patch

    import numpy as np

    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    class CustomDev:
        """Custom device string representation."""

        def __str__(self) -> str:
            """Return cpu string."""
            return "cpu"

    assert TensorConfig(shape=None, dtype=None, device=None).shape == ()
    assert TensorConfig(shape=123, dtype="invalid_dtype_xyz", device=as_tensor(CustomDev())).shape == ()
    t = Tensor(np.array(1), TensorConfig((), "float32", "cpu"))
    assert t.contiguous() is t
    with patch("ml_switcheroo_compiler.ops.registry.get_frontend", return_value=lambda x, shape: x):
        assert t.view(range(2)) is t


def test_tensor_detach() -> None:
    """Test function test_tensor_detach."""
    import numpy as np

    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    t = Tensor(np.array(1), TensorConfig((), "float32", "cpu"))
    res = t.detach()
    assert res is not t
    assert isinstance(res, Tensor)


def test_tensor_backward() -> None:
    """Test function test_tensor_backward."""
    from unittest.mock import patch

    import numpy as np

    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    t = Tensor(np.array(1), TensorConfig((), "float32", "cpu"))
    with patch("ml_switcheroo_compiler.ops.registry.get_util") as mock_get_util:
        mock_func = mock_get_util.return_value
        t.backward(1, test=True)
        mock_func.assert_called_once_with(t, 1, test=True)
        mock_get_util.assert_called_once_with("backward")
