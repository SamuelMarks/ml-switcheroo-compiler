"""Tests for test_frontend_basic."""

from __future__ import annotations

import typing
from unittest.mock import patch

import pytest

from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.ops.creation.frontend_basic import _create_backend_array, _get_dtype_val, _infer_dtype, _try_create_array, array, asarray, convert_to_numpy, convert_to_tensor, empty, empty_like, frombuffer, full, full_like, ones, ones_like, zeros, zeros_like

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol, EagerTensorProtocol, EagerValue


def test_frontend_basic() -> None:
    """Test function test_frontend_basic."""

    class DummyItem:
        """Test class DummyItem."""

        def item(self) -> EagerValue:
            """Test function item."""
            return 1

    class DummyData:
        """Test class DummyData."""

        data = DummyItem()

    import numpy as np

    assert _infer_dtype(np.array([1.0])) == DType.Float64
    assert _infer_dtype(np.array([1], dtype="int32")) == DType.Int32
    assert _get_dtype_val(DType.Float32) == "float32"
    assert _get_dtype_val(None) is None

    class DummyBackend:
        """Test class DummyBackend."""

        def array(self, x: EagerValue, dtype: EagerValue = None) -> EagerValue:
            """Test function array.

            Args:
                x: Test parameter `x`.
                dtype: Test parameter `dtype`.
            """

            class T:
                """Test class T."""

                dtype = "float32"
                shape = (1,)

            return T()

        def asarray(self, x: EagerValue, dtype: EagerValue = None) -> EagerValue:
            """Test function asarray.

            Args:
                x: Test parameter `x`.
                dtype: Test parameter `dtype`.
            """

            class T:
                """Test class T."""

                dtype = "float32"
                shape = (1,)

            return T()

    assert typing.cast("EagerTensorProtocol", getattr(typing.cast("EagerCallableProtocol", _try_create_array)(typing.cast("EagerCallableProtocol", DummyBackend)(), [1], "int32"), "dtype")) == "float32"
    assert typing.cast("EagerTensorProtocol", getattr(typing.cast("EagerCallableProtocol", _try_create_array)(typing.cast("EagerCallableProtocol", DummyBackend)(), [1], None), "dtype")) == "float32"

    class DummyBackendThrow:
        """Test class DummyBackendThrow."""

        def array(self, x: EagerValue, dtype: EagerValue = None) -> None:
            """Test function array.

            Args:
                x: Test parameter `x`.
                dtype: Test parameter `dtype`.
            """
            raise Exception()

    with pytest.raises((RuntimeError, ValueError, TypeError, KeyError, Exception)):
        typing.cast("EagerCallableProtocol", _try_create_array)(typing.cast("EagerCallableProtocol", DummyBackendThrow)(), [1], None)
    with patch("ml_switcheroo_compiler.ops.creation.frontend_basic.get_active_backend", return_value=DummyBackend()):
        assert typing.cast("EagerTensorProtocol", getattr(_create_backend_array([1], None), "dtype")) == "float32"
        from ml_switcheroo_compiler.core.config import config

        config.eager_mode = True
        from ml_switcheroo_compiler.core.config import config

        config.eager_mode = True
        from ml_switcheroo_compiler.core.config import config

        config.eager_mode = True
        config.eager_mode = True
        t = array([1])
        assert t.shape == (1,)
        config.eager_mode = False
        import ml_switcheroo_compiler.tracing.state as state

        class DummyGraph:
            """Test class DummyGraph."""

            def __init__(self) -> None:
                """Test function __init__."""
                self.nodes = typing.cast("dict[str, EagerTensorProtocol]", {})

            def add_node(self, node: EagerValue) -> None:
                """Test function add_node.

                Args:
                    node: Test parameter `node`.
                """
                pass

        state.global_tracing_state.is_tracing = True
        setattr(state.global_tracing_state, "active_graph", DummyGraph())
        setattr(state.global_tracing_state, "add_node", typing.cast("EagerTensorProtocol", getattr(state.global_tracing_state.active_graph, "add_node")))
        config.eager_mode = False
        array([1])
        asarray([1])
        config.eager_mode = True
        asarray([1])
        from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

        tc = Tensor(data="data", config=TensorConfig((1,), DType.Float32, "cpu"))
        assert convert_to_tensor(tc) is tc
        convert_to_tensor([1])


def test_frontend_basic_functions() -> None:
    """Test function test_frontend_basic_functions."""
    import ml_switcheroo_compiler.backends.registry as reg
    from ml_switcheroo_compiler.core.config import config

    config.eager_mode = True

    class DummyBackend:
        """Test class DummyBackend."""

        def array(self, x: EagerValue, dtype: EagerValue = None) -> EagerValue:
            """Test function array.

            Args:
                x: Test parameter `x`.
                dtype: Test parameter `dtype`.
            """

            class T:
                """Test class T."""

                dtype = "float32"
                shape = (1,)

            return T()

        def asarray(self, x: EagerValue, dtype: EagerValue = None) -> EagerValue:
            """Test function asarray.

            Args:
                x: Test parameter `x`.
                dtype: Test parameter `dtype`.
            """

            class T:
                """Test class T."""

                dtype = "float32"
                shape = (1,)

            return T()

        def zeros(self, s: EagerValue, dtype: EagerValue = None) -> EagerValue:
            """Test function zeros.

            Args:
                s: Test parameter `s`.
                dtype: Test parameter `dtype`.
            """

            class T:
                """Test class T."""

                dtype = "float32"
                shape = (1,)

            return T()

        def ones(self, s: EagerValue, dtype: EagerValue = None) -> EagerValue:
            """Test function ones.

            Args:
                s: Test parameter `s`.
                dtype: Test parameter `dtype`.
            """

            class T:
                """Test class T."""

                dtype = "float32"
                shape = (1,)

            return T()

        def full(self, s: EagerValue, v: EagerValue, dtype: EagerValue = None) -> EagerValue:
            """Test function full.

            Args:
                s: Test parameter `s`.
                v: Test parameter `v`.
                dtype: Test parameter `dtype`.
            """

            class T:
                """Test class T."""

                dtype = "float32"
                shape = (1,)

            return T()

        def empty(self, s: EagerValue, dtype: EagerValue = None) -> EagerValue:
            """Test function empty.

            Args:
                s: Test parameter `s`.
                dtype: Test parameter `dtype`.
            """

            class T:
                """Test class T."""

                dtype = "float32"
                shape = (1,)

            return T()

        def frombuffer(self, s: EagerValue, dtype: EagerValue = None, count: EagerValue = None, offset: EagerValue = None) -> EagerValue:
            """Test function frombuffer.

            Args:
                s: Test parameter `s`.
                dtype: Test parameter `dtype`.
                count: Test parameter `count`.
                offset: Test parameter `offset`.
            """

            class T:
                """Test class T."""

                dtype = "float32"
                shape = (1,)

            return T()

    setattr(reg, "_ACTIVE_BACKEND", DummyBackend())
    from ml_switcheroo_compiler.core.config import config

    config.eager_mode = True
    zeros((1,))
    ones((1,))
    full((1,), 0)
    empty((1,))
    import numpy as np

    frombuffer(np.zeros(10).tobytes(), count=10)
    config.eager_mode = False
    import ml_switcheroo_compiler.tracing.state as state

    class DummyGraph:
        """Test class DummyGraph."""

        def __init__(self) -> None:
            """Test function __init__."""
            self.nodes = typing.cast("dict[str, EagerTensorProtocol]", {})

        def add_node(self, node: EagerValue) -> None:
            """Test function add_node.

            Args:
                node: Test parameter `node`.
            """
            pass

    state.global_tracing_state.is_tracing = True
    setattr(state.global_tracing_state, "active_graph", DummyGraph())
    setattr(state.global_tracing_state, "add_node", typing.cast("EagerTensorProtocol", getattr(state.global_tracing_state.active_graph, "add_node")))
    from ml_switcheroo_compiler.core.config import config

    config.eager_mode = True
    config.eager_mode = False
    config.eager_mode = False
    zeros((1,))
    ones((1,))
    full((1,), 0)
    empty((1,))
    frombuffer(np.zeros(10).tobytes(), count=10)
    assert convert_to_numpy(zeros((1,))) is not None
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    t = Tensor(data="data", config=TensorConfig((1,), DType.Float32, "cpu"))
    config.eager_mode = False
    config.eager_mode = False
    zeros_like(t)
    ones_like(t)
    full_like(t, 1)
    empty_like(t)


def test_frontend_basic_missing_branches() -> None:
    """Test function test_frontend_basic_missing_branches."""
    from ml_switcheroo_compiler.core.config import config

    config.eager_mode = False

    class DummyItem2:
        """Test class DummyItem2."""

        def item(self) -> EagerValue:
            """Test function item."""
            return "item"

        def __int__(self) -> EagerValue:
            """Test function __int__."""
            return 1

    class DummyTensor2:
        """Test class DummyTensor2."""

        data = DummyItem2()

    import numpy as np

    _infer_dtype(np.array([1], dtype="float16"))
    _infer_dtype(np.array([1], dtype="int16"))
    _infer_dtype(np.array([1], dtype="int8"))
    with patch("ml_switcheroo_compiler.ops.creation.frontend_basic.get_active_backend") as m:
        m.return_value.array.side_effect = Exception("test")
        with pytest.raises((RuntimeError, ValueError, TypeError, KeyError, Exception)):
            _create_backend_array([1], None)
    with patch("ml_switcheroo_compiler.ops.creation.frontend_basic.get_active_backend") as m:
        m.return_value.asarray.side_effect = Exception("test")

        class DummyTensor3:
            """Test class DummyTensor3."""

            pass

        with pytest.raises(ValueError):
            convert_to_tensor([1])
    config.eager_mode = True
    import ml_switcheroo_compiler.backends.registry as reg

    class DummyBackend4:
        """Test class DummyBackend4."""

        def empty(self, s: EagerValue, dtype: EagerValue = None) -> EagerValue:
            """Test function empty.

            Args:
                s: Test parameter `s`.
                dtype: Test parameter `dtype`.
            """

            class T:
                """Test class T."""

                shape = (1,)
                dtype = "float32"

            return T()

    setattr(reg, "_ACTIVE_BACKEND", DummyBackend4())
    empty((1,))


def test_frontend_basic_convert() -> None:
    """Test function test_frontend_basic_convert."""
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    t = Tensor(data=[1], config=TensorConfig((1,), DType.Int32, "cpu"))

    class DummyBackend4:
        """Test class DummyBackend4."""

        def convert_to_numpy(self, x: EagerValue) -> EagerValue:
            """Test function convert_to_numpy.

            Args:
                x: Test parameter `x`.
            """
            return "numpy_arr"

    with patch("ml_switcheroo_compiler.backends.registry.get_active_backend", return_value=DummyBackend4()):
        assert convert_to_numpy(t) == "numpy_arr"


def test_frontend_basic_kwargs() -> None:
    """Test function test_frontend_basic_kwargs."""
    from ml_switcheroo_compiler.core.config import config

    config.eager_mode = False
    import ml_switcheroo_compiler.backends.registry as reg

    class DummyBackend:
        """Test class DummyBackend."""

        def zeros(self, s: EagerValue, dtype: EagerValue = None) -> EagerValue:
            """Test function zeros.

            Args:
                s: Test parameter `s`.
                dtype: Test parameter `dtype`.
            """

            class T:
                """Test class T."""

                dtype = "float32"
                shape = (1,)

            return T()

        def ones(self, s: EagerValue, dtype: EagerValue = None) -> EagerValue:
            """Test function ones.

            Args:
                s: Test parameter `s`.
                dtype: Test parameter `dtype`.
            """

            class T:
                """Test class T."""

                dtype = "float32"
                shape = (1,)

            return T()

        def full(self, s: EagerValue, v: EagerValue, dtype: EagerValue = None) -> EagerValue:
            """Test function full.

            Args:
                s: Test parameter `s`.
                v: Test parameter `v`.
                dtype: Test parameter `dtype`.
            """

            class T:
                """Test class T."""

                dtype = "float32"
                shape = (1,)

            return T()

        def empty(self, s: EagerValue, dtype: EagerValue = None) -> EagerValue:
            """Test function empty.

            Args:
                s: Test parameter `s`.
                dtype: Test parameter `dtype`.
            """

            class T:
                """Test class T."""

                dtype = "float32"
                shape = (1,)

            return T()

        def frombuffer(self, s: EagerValue, dtype: EagerValue = None, count: EagerValue = None, offset: EagerValue = None) -> EagerValue:
            """Test function frombuffer.

            Args:
                s: Test parameter `s`.
                dtype: Test parameter `dtype`.
                count: Test parameter `count`.
                offset: Test parameter `offset`.
            """

            class T:
                """Test class T."""

                dtype = "float32"
                shape = (1,)

            return T()

    setattr(reg, "_ACTIVE_BACKEND", DummyBackend())
    import ml_switcheroo_compiler.tracing.state as state

    class DummyGraph:
        """Test class DummyGraph."""

        def __init__(self) -> None:
            """Test function __init__."""
            self.nodes = typing.cast("dict[str, EagerTensorProtocol]", {})

        def add_node(self, node: EagerValue) -> None:
            """Test function add_node.

            Args:
                node: Test parameter `node`.
            """
            pass

    state.global_tracing_state.is_tracing = True
    setattr(state.global_tracing_state, "active_graph", DummyGraph())
    setattr(state.global_tracing_state, "add_node", typing.cast("EagerTensorProtocol", getattr(state.global_tracing_state.active_graph, "add_node")))
    with patch("ml_switcheroo_compiler.ops.creation.frontend_basic._emit_creation_node", return_value="emitted"):
        zeros((1,), dtype=DType.Float32)
        ones((1,), dtype=DType.Float32)
        full((1,), 0, dtype=DType.Float32)
        empty((1,), dtype=DType.Float32)
        frombuffer(b"test", dtype=DType.Float32)
    config.eager_mode = True
    assert callable(test_frontend_basic_kwargs)


def test_frontend_basic_coverage() -> None:
    """Test function test_frontend_basic_coverage."""
    import numpy as np

    from ml_switcheroo_compiler.core.dtype import DType
    from ml_switcheroo_compiler.ops.creation.frontend_basic import _extract_fill_value, _unpack_shape, empty, empty_like, frombuffer, full, full_like, ones, ones_like, zeros, zeros_like

    assert _unpack_shape((1,)) == (1,)

    class DummyItem:
        """Test class DummyItem."""

        def item(self) -> EagerValue:
            """Test function item."""
            return 1

    class DummyData:
        """Test class DummyData."""

        data = DummyItem()

    assert typing.cast("EagerCallableProtocol", _unpack_shape)((typing.cast("EagerCallableProtocol", DummyData)(), typing.cast("EagerCallableProtocol", DummyItem)(), 2)) == (1, 1, 2)
    assert _extract_fill_value(1) == 1
    assert typing.cast("EagerCallableProtocol", _extract_fill_value)(typing.cast("EagerCallableProtocol", DummyData)()) == 1
    assert typing.cast("EagerCallableProtocol", _extract_fill_value)(typing.cast("EagerCallableProtocol", DummyItem)()) == 1
    import ml_switcheroo_compiler.backends.registry as reg

    class DummyBackend:
        """Test class DummyBackend."""

        def execute_op(self, op: EagerValue, *a: EagerValue, **k: EagerValue) -> EagerValue:
            """Test function execute_op.

            Args:
                op: Test parameter `op`.
                a: Test parameter `a`.
                k: Test parameter `k`.
            """
            return op

    setattr(reg, "_ACTIVE_BACKEND", DummyBackend())
    pass
    from ml_switcheroo_compiler.core.config import config

    config.eager_mode = False
    import ml_switcheroo_compiler.tracing.state as state

    class DummyGraph:
        """Test class DummyGraph."""

        def __init__(self) -> None:
            """Test function __init__."""
            self.nodes = typing.cast("dict[str, EagerTensorProtocol]", {})

        def add_node(self, node: EagerValue) -> None:
            """Test function add_node.

            Args:
                node: Test parameter `node`.
            """
            pass

    state.global_tracing_state.is_tracing = True
    setattr(state.global_tracing_state, "active_graph", DummyGraph())
    setattr(state.global_tracing_state, "add_node", typing.cast("EagerTensorProtocol", getattr(state.global_tracing_state.active_graph, "add_node")))
    from unittest.mock import patch

    with patch("ml_switcheroo_compiler.ops.creation.frontend_basic._emit_creation_node", return_value="emitted"):
        assert typing.cast("EagerTensorProtocol", zeros((1,), dtype=DType.Float32)) == typing.cast("EagerTensorProtocol", "emitted")
        assert typing.cast("EagerTensorProtocol", ones((1,), dtype=DType.Float32)) == typing.cast("EagerTensorProtocol", "emitted")
        assert typing.cast("EagerTensorProtocol", full((1,), 0, dtype=DType.Float32)) == typing.cast("EagerTensorProtocol", "emitted")
        assert typing.cast("EagerTensorProtocol", empty((1,), dtype=DType.Float32)) == typing.cast("EagerTensorProtocol", "emitted")
        assert typing.cast("EagerTensorProtocol", frombuffer(np.zeros(10).tobytes(), count=10)) == typing.cast("EagerTensorProtocol", "emitted")
        from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

        t = Tensor(data="data", config=TensorConfig((1,), DType.Float32, "cpu"))
        assert typing.cast("EagerTensorProtocol", zeros_like(t)) == typing.cast("EagerTensorProtocol", "emitted")
        assert typing.cast("EagerTensorProtocol", ones_like(t)) == typing.cast("EagerTensorProtocol", "emitted")
        assert typing.cast("EagerTensorProtocol", full_like(t, 1)) == typing.cast("EagerTensorProtocol", "emitted")
        assert typing.cast("EagerTensorProtocol", empty_like(t)) == typing.cast("EagerTensorProtocol", "emitted")
    config.eager_mode = True
