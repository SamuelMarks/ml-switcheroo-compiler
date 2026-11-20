"""Tests for test_frontend_basic."""

from __future__ import annotations

import typing
from unittest.mock import patch

import pytest
from ml_switcheroo_ir import LogicalNode

from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.ops.creation.api_basic import (
    _create_backend_array,
    _get_dtype_val,
    _infer_dtype,
    _try_create_array,
    array,
    asarray,
    convert_to_numpy,
    convert_to_tensor,
    empty,
    empty_like,
    frombuffer,
    full,
    full_like,
    ones,
    ones_like,
    zeros,
    zeros_like,
)
from tests.testing_utils import (
    MockTensorStub,
    MockTestStub,
    adapt_backend_module,
    extract_test_dtype,
    extract_test_shape,
)

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import BackendArg


def test_frontend_basic() -> None:
    """Test function test_frontend_basic."""

    class DummyItem(MockTensorStub):
        """Test class DummyItem."""

        def item(self) -> float:
            """Test function item.

            Returns:
                float: Computed output value from `item`.
            """
            return 1

    class DummyData(MockTensorStub):
        """Test class DummyData."""

        data = DummyItem()

    import numpy as np

    assert _infer_dtype(np.array([1.0])) == DType.Float64
    assert _infer_dtype(np.array([1], dtype="int32")) == DType.Int32
    assert _get_dtype_val(DType.Float32) == "float32"
    assert _get_dtype_val(None) is None

    class DummyBackend(MockTestStub):
        """Test class DummyBackend."""

        def array(self, x: BackendArg, dtype: BackendArg = None) -> BackendArg:
            """Test function array.

            Args:
                x: Test parameter `x`.
                dtype: Test parameter `dtype`.

            Returns:
                BackendArg: Computed output value from `array`.
            """

            class T:
                """Test class T."""

                dtype = "float32"
                shape = (1,)

            return T()

        def asarray(self, x: BackendArg, dtype: BackendArg = None) -> BackendArg:
            """Test function asarray.

            Args:
                x: Test parameter `x`.
                dtype: Test parameter `dtype`.

            Returns:
                BackendArg: Computed output value from `asarray`.
            """

            class T:
                """Test class T."""

                dtype = "float32"
                shape = (1,)

            return T()

    assert extract_test_dtype(_try_create_array(DummyBackend(), [1], "int32")) == "float32"
    assert extract_test_dtype(_try_create_array(DummyBackend(), [1], None)) == "float32"

    class DummyBackendThrow(MockTestStub):
        """Test class DummyBackendThrow."""

        def array(self, x: BackendArg, dtype: BackendArg = None) -> None:
            """Test function array.

            Args:
                x: Test parameter `x`.
                dtype: Test parameter `dtype`.

            Raises:
                Exception: If execution or validation fails in `array`.
            """
            raise RuntimeError()

    with pytest.raises((RuntimeError, ValueError, TypeError, KeyError, Exception)):
        _try_create_array(DummyBackendThrow(), [1], None)
    with patch("ml_switcheroo_compiler.ops.creation.api_basic.get_active_backend", return_value=DummyBackend()):
        assert extract_test_dtype(_create_backend_array([1], None)) == "float32"
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
                self.nodes: dict[str, LogicalNode] = {}

            def add_node(self, node: BackendArg) -> None:
                """Test function add_node.

                Args:
                    node: Test parameter `node`.
                """
                pass

        state.global_tracing_state.is_tracing = True
        pytest.MonkeyPatch().setattr(state.global_tracing_state, "active_graph", DummyGraph(), raising=False)
        pytest.MonkeyPatch().setattr(
            state.global_tracing_state,
            "add_node",
            adapt_backend_module(state.global_tracing_state.active_graph).require_symbol("add_node"),
            raising=False,
        )
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

    class DummyBackend(MockTestStub):
        """Test class DummyBackend."""

        def array(self, x: BackendArg, dtype: BackendArg = None) -> BackendArg:
            """Test function array.

            Args:
                x: Test parameter `x`.
                dtype: Test parameter `dtype`.

            Returns:
                BackendArg: Computed output value from `array`.
            """

            class T:
                """Test class T."""

                dtype = "float32"
                shape = (1,)

            return T()

        def asarray(self, x: BackendArg, dtype: BackendArg = None) -> BackendArg:
            """Test function asarray.

            Args:
                x: Test parameter `x`.
                dtype: Test parameter `dtype`.

            Returns:
                BackendArg: Computed output value from `asarray`.
            """

            class T:
                """Test class T."""

                dtype = "float32"
                shape = (1,)

            return T()

        def zeros(self, s: BackendArg, dtype: BackendArg = None) -> BackendArg:
            """Test function zeros.

            Args:
                s: Test parameter `s`.
                dtype: Test parameter `dtype`.

            Returns:
                BackendArg: Computed output value from `zeros`.
            """

            class T:
                """Test class T."""

                dtype = "float32"
                shape = (1,)

            return T()

        def ones(self, s: BackendArg, dtype: BackendArg = None) -> BackendArg:
            """Test function ones.

            Args:
                s: Test parameter `s`.
                dtype: Test parameter `dtype`.

            Returns:
                BackendArg: Computed output value from `ones`.
            """

            class T:
                """Test class T."""

                dtype = "float32"
                shape = (1,)

            return T()

        def full(self, s: BackendArg, v: BackendArg, dtype: BackendArg = None) -> BackendArg:
            """Test function full.

            Args:
                s: Test parameter `s`.
                v: Test parameter `v`.
                dtype: Test parameter `dtype`.

            Returns:
                BackendArg: Computed output value from `full`.
            """

            class T:
                """Test class T."""

                dtype = "float32"
                shape = (1,)

            return T()

        def empty(self, s: BackendArg, dtype: BackendArg = None) -> BackendArg:
            """Test function empty.

            Args:
                s: Test parameter `s`.
                dtype: Test parameter `dtype`.

            Returns:
                BackendArg: Computed output value from `empty`.
            """

            class T:
                """Test class T."""

                dtype = "float32"
                shape = (1,)

            return T()

        def frombuffer(
            self, s: BackendArg, dtype: BackendArg = None, count: BackendArg = None, offset: BackendArg = None
        ) -> BackendArg:
            """Test function frombuffer.

            Args:
                s: Test parameter `s`.
                dtype: Test parameter `dtype`.
                count: Test parameter `count`.
                offset: Test parameter `offset`.

            Returns:
                BackendArg: Computed output value from `frombuffer`.
            """

            class T:
                """Test class T."""

                dtype = "float32"
                shape = (1,)

            return T()

    pytest.MonkeyPatch().setattr(reg, "_ACTIVE_BACKEND", DummyBackend(), raising=False)
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
            self.nodes: dict[str, LogicalNode] = {}

        def add_node(self, node: BackendArg) -> None:
            """Test function add_node.

            Args:
                node: Test parameter `node`.
            """
            pass

    state.global_tracing_state.is_tracing = True
    pytest.MonkeyPatch().setattr(state.global_tracing_state, "active_graph", DummyGraph(), raising=False)
    pytest.MonkeyPatch().setattr(
        state.global_tracing_state,
        "add_node",
        adapt_backend_module(state.global_tracing_state.active_graph).require_symbol("add_node"),
        raising=False,
    )
    from ml_switcheroo_compiler.core.config import config

    config.eager_mode = True
    config.eager_mode = False
    config.eager_mode = False
    zeros((1,))
    ones((1,))
    full((1,), 0)
    empty((1,))
    frombuffer(np.zeros(10).tobytes(), count=10)
    assert tuple(extract_test_shape(convert_to_numpy(zeros((1,))))) == ()
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

        def item(self) -> BackendArg:
            """Test function item.

            Returns:
                BackendArg: Computed output value from `item`.
            """
            return "item"

        def __int__(self) -> BackendArg:
            """Test function __int__.

            Returns:
                BackendArg: Computed output value from `__int__`.
            """
            return 1

    class DummyTensor2:
        """Test class DummyTensor2."""

        data = DummyItem2()

    import numpy as np

    _infer_dtype(np.array([1], dtype="float16"))
    _infer_dtype(np.array([1], dtype="int16"))
    _infer_dtype(np.array([1], dtype="int8"))
    with patch("ml_switcheroo_compiler.ops.creation.api_basic.get_active_backend") as m:
        m.return_value.array.side_effect = RuntimeError("test")
        with pytest.raises((RuntimeError, ValueError, TypeError, KeyError, Exception)):
            _create_backend_array([1], None)
    with patch("ml_switcheroo_compiler.ops.creation.api_basic.get_active_backend") as m:
        m.return_value.asarray.side_effect = RuntimeError("test")

        class DummyTensor3:
            """Test class DummyTensor3."""

            pass

        with pytest.raises(ValueError):
            convert_to_tensor([1])
    config.eager_mode = True
    import ml_switcheroo_compiler.backends.registry as reg

    class DummyBackend4:
        """Test class DummyBackend4."""

        def empty(self, s: BackendArg, dtype: BackendArg = None) -> BackendArg:
            """Test function empty.

            Args:
                s: Test parameter `s`.
                dtype: Test parameter `dtype`.

            Returns:
                BackendArg: Computed output value from `empty`.
            """

            class T:
                """Test class T."""

                shape = (1,)
                dtype = "float32"

            return T()

    pytest.MonkeyPatch().setattr(reg, "_ACTIVE_BACKEND", DummyBackend4(), raising=False)
    empty((1,))


def test_frontend_basic_convert() -> None:
    """Test function test_frontend_basic_convert."""
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    t = Tensor(data=[1], config=TensorConfig((1,), DType.Int32, "cpu"))

    class DummyBackend4:
        """Test class DummyBackend4."""

        def convert_to_numpy(self, x: BackendArg) -> BackendArg:
            """Test function convert_to_numpy.

            Args:
                x: Test parameter `x`.

            Returns:
                BackendArg: Computed output value from `convert_to_numpy`.
            """
            return "numpy_arr"

    with patch("ml_switcheroo_compiler.backends.registry.get_active_backend", return_value=DummyBackend4()):
        assert convert_to_numpy(t) == "numpy_arr"


def test_frontend_basic_kwargs() -> None:
    """Test function test_frontend_basic_kwargs."""
    from ml_switcheroo_compiler.core.config import config

    config.eager_mode = False
    import ml_switcheroo_compiler.backends.registry as reg

    class DummyBackend(MockTestStub):
        """Test class DummyBackend."""

        def zeros(self, s: BackendArg, dtype: BackendArg = None) -> BackendArg:
            """Test function zeros.

            Args:
                s: Test parameter `s`.
                dtype: Test parameter `dtype`.

            Returns:
                BackendArg: Computed output value from `zeros`.
            """

            class T:
                """Test class T."""

                dtype = "float32"
                shape = (1,)

            return T()

        def ones(self, s: BackendArg, dtype: BackendArg = None) -> BackendArg:
            """Test function ones.

            Args:
                s: Test parameter `s`.
                dtype: Test parameter `dtype`.

            Returns:
                BackendArg: Computed output value from `ones`.
            """

            class T:
                """Test class T."""

                dtype = "float32"
                shape = (1,)

            return T()

        def full(self, s: BackendArg, v: BackendArg, dtype: BackendArg = None) -> BackendArg:
            """Test function full.

            Args:
                s: Test parameter `s`.
                v: Test parameter `v`.
                dtype: Test parameter `dtype`.

            Returns:
                BackendArg: Computed output value from `full`.
            """

            class T:
                """Test class T."""

                dtype = "float32"
                shape = (1,)

            return T()

        def empty(self, s: BackendArg, dtype: BackendArg = None) -> BackendArg:
            """Test function empty.

            Args:
                s: Test parameter `s`.
                dtype: Test parameter `dtype`.

            Returns:
                BackendArg: Computed output value from `empty`.
            """

            class T:
                """Test class T."""

                dtype = "float32"
                shape = (1,)

            return T()

        def frombuffer(
            self, s: BackendArg, dtype: BackendArg = None, count: BackendArg = None, offset: BackendArg = None
        ) -> BackendArg:
            """Test function frombuffer.

            Args:
                s: Test parameter `s`.
                dtype: Test parameter `dtype`.
                count: Test parameter `count`.
                offset: Test parameter `offset`.

            Returns:
                BackendArg: Computed output value from `frombuffer`.
            """

            class T:
                """Test class T."""

                dtype = "float32"
                shape = (1,)

            return T()

    pytest.MonkeyPatch().setattr(reg, "_ACTIVE_BACKEND", DummyBackend(), raising=False)
    import ml_switcheroo_compiler.tracing.state as state

    class DummyGraph:
        """Test class DummyGraph."""

        def __init__(self) -> None:
            """Test function __init__."""
            self.nodes: dict[str, LogicalNode] = {}

        def add_node(self, node: BackendArg) -> None:
            """Test function add_node.

            Args:
                node: Test parameter `node`.
            """
            pass

    state.global_tracing_state.is_tracing = True
    pytest.MonkeyPatch().setattr(state.global_tracing_state, "active_graph", DummyGraph(), raising=False)
    pytest.MonkeyPatch().setattr(
        state.global_tracing_state,
        "add_node",
        adapt_backend_module(state.global_tracing_state.active_graph).require_symbol("add_node"),
        raising=False,
    )
    with patch(
        "ml_switcheroo_compiler.ops.creation.api_basic._emit_creation_node", return_value="emitted"
    ) as mock_emit:
        zeros((1,), dtype=DType.Float32)
        ones((1,), dtype=DType.Float32)
        full((1,), 0, dtype=DType.Float32)
        empty((1,), dtype=DType.Float32)
        frombuffer(b"test", dtype=DType.Float32)
        assert mock_emit.call_count == 5
    config.eager_mode = True


def test_frontend_basic_coverage() -> None:
    """Test function test_frontend_basic_coverage."""
    import numpy as np

    from ml_switcheroo_compiler.core.dtype import DType
    from ml_switcheroo_compiler.ops.creation.api_basic import (
        _extract_fill_value,
        _unpack_shape,
        empty,
        empty_like,
        frombuffer,
        full,
        full_like,
        ones,
        ones_like,
        zeros,
        zeros_like,
    )

    assert _unpack_shape((1,)) == (1,)

    class DummyItem(MockTensorStub):
        """Test class DummyItem."""

        def item(self) -> float:
            """Test function item.

            Returns:
                float: Computed output value from `item`.
            """
            return 1

    class DummyData(MockTensorStub):
        """Test class DummyData."""

        data = DummyItem()

    assert _unpack_shape((DummyData(), DummyItem(), 2)) == (1, 1, 2)
    assert _extract_fill_value(1) == 1
    assert _extract_fill_value(DummyData()) == 1
    assert _extract_fill_value(DummyItem()) == 1
    import ml_switcheroo_compiler.backends.registry as reg

    class DummyBackend(MockTestStub):
        """Test class DummyBackend."""

        def execute_op(self, op: BackendArg, *a: BackendArg, **k: BackendArg) -> BackendArg:
            """Test function execute_op.

            Args:
                op: Test parameter `op`.
                a: Test parameter `a`.
                k: Test parameter `k`.

            Returns:
                BackendArg: Computed output value from `execute_op`.
            """
            return op

    pytest.MonkeyPatch().setattr(reg, "_ACTIVE_BACKEND", DummyBackend(), raising=False)
    pass
    from ml_switcheroo_compiler.core.config import config

    config.eager_mode = False
    import ml_switcheroo_compiler.tracing.state as state

    class DummyGraph:
        """Test class DummyGraph."""

        def __init__(self) -> None:
            """Test function __init__."""
            self.nodes: dict[str, LogicalNode] = {}

        def add_node(self, node: BackendArg) -> None:
            """Test function add_node.

            Args:
                node: Test parameter `node`.
            """
            pass

    state.global_tracing_state.is_tracing = True
    pytest.MonkeyPatch().setattr(state.global_tracing_state, "active_graph", DummyGraph(), raising=False)
    pytest.MonkeyPatch().setattr(
        state.global_tracing_state,
        "add_node",
        adapt_backend_module(state.global_tracing_state.active_graph).require_symbol("add_node"),
        raising=False,
    )
    from unittest.mock import patch

    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    emitted_sentinel = Tensor(data="emitted", config=TensorConfig((1,), DType.Float32, "cpu"))
    with patch("ml_switcheroo_compiler.ops.creation.api_basic._emit_creation_node", return_value=emitted_sentinel):
        assert zeros((1,), dtype=DType.Float32) is emitted_sentinel
        assert ones((1,), dtype=DType.Float32) is emitted_sentinel
        assert full((1,), 0, dtype=DType.Float32) is emitted_sentinel
        assert empty((1,), dtype=DType.Float32) is emitted_sentinel
        assert frombuffer(np.zeros(10).tobytes(), count=10) is emitted_sentinel

        t = Tensor(data="data", config=TensorConfig((1,), DType.Float32, "cpu"))
        assert zeros_like(t) is emitted_sentinel
        assert ones_like(t) is emitted_sentinel
        assert full_like(t, 1) is emitted_sentinel
        assert empty_like(t) is emitted_sentinel
    config.eager_mode = True
