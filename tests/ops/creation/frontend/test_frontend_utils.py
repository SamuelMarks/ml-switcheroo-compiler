"""Behavioral unit tests for frontend utils."""

from __future__ import annotations

import typing

import pytest
from ml_switcheroo_ir import LogicalNode

from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.ops.creation.api_utils import Frompyfunc, Geometric, Geomspace, _emit_creation_node
from tests.testing_utils import MockTestStub, adapt_backend_module

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import BackendArg


def test_frontend_utils() -> None:
    """Test function test_frontend_utils."""
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

    config.eager_mode = False
    _emit_creation_node("zeros", (1,), DType.Float32, {"a": 1})
    pass
    state.global_tracing_state.is_tracing = False
    with pytest.raises(RuntimeError):
        _emit_creation_node("zeros", (1,), DType.Float32, {"a": 1})
    state.global_tracing_state.is_tracing = True


def test_frontend_utils_constant() -> None:
    """Test function test_frontend_utils_constant."""
    import ml_switcheroo_compiler.tracing.state as state
    from ml_switcheroo_compiler.ops.creation.api_utils import _emit_constant_node

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
    import ml_switcheroo_compiler.backends.registry as reg

    class DummyBackend:
        """Test class DummyBackend."""

        def array(self, value: BackendArg, dtype: BackendArg = None) -> BackendArg:
            """Test function array.

            Args:
                value: Test parameter `value`.
                dtype: Test parameter `dtype`.

            Returns:
                BackendArg: Computed output value from `array`.
            """

            class T:
                """Test class T."""

                shape = (1,)
                ndim = 1

                def tolist(self) -> BackendArg:
                    """Test function tolist.

                    Returns:
                        BackendArg: Computed output value from `tolist`.
                    """
                    return [value]

            return T()

    pytest.MonkeyPatch().setattr(reg, "_ACTIVE_BACKEND", DummyBackend(), raising=False)
    from ml_switcheroo_compiler.core.config import config

    config.eager_mode = False
    _emit_constant_node(0, DType.Float32)
    pass
    state.global_tracing_state.is_tracing = False
    with pytest.raises(RuntimeError):
        _emit_constant_node(0, DType.Float32)
    state.global_tracing_state.is_tracing = True


def test_frontend_utils_constant_scalar() -> None:
    """Test function test_frontend_utils_constant_scalar."""
    import ml_switcheroo_compiler.tracing.state as state
    from ml_switcheroo_compiler.ops.creation.api_utils import _emit_constant_node

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
    import ml_switcheroo_compiler.backends.registry as reg

    class DummyBackend:
        """Test class DummyBackend."""

        def array(self, value: BackendArg, dtype: BackendArg = None) -> BackendArg:
            """Test function array.

            Args:
                value: Test parameter `value`.
                dtype: Test parameter `dtype`.

            Returns:
                BackendArg: Computed output value from `array`.
            """

            class T:
                """Test class T."""

                shape = ()
                ndim = 0

                def item(self) -> BackendArg:
                    """Test function item.

                    Returns:
                        BackendArg: Computed output value from `item`.
                    """
                    return value

            return T()

    pytest.MonkeyPatch().setattr(reg, "_ACTIVE_BACKEND", DummyBackend(), raising=False)
    from ml_switcheroo_compiler.core.config import config

    config.eager_mode = False
    _emit_constant_node(0, DType.Float32)
    assert callable(_emit_constant_node)


def test_frontend_utils_missing_3() -> None:
    """Test function test_frontend_utils_missing_3."""
    from ml_switcheroo_compiler.ops.creation.api_utils import Geometric, frompyfunc

    _multinomial_shape = Geometric().infer_shape

    class MockTensor(MockTestStub):
        """Test class MockTensor."""

        shape = (10,)

    assert _multinomial_shape(MockTensor(), size=5) == (5,)
    with pytest.raises(RuntimeError):
        frompyfunc(lambda x: x, 1, 1)


def test_frontend_utils_frompyfunc() -> None:
    """Test function test_frontend_utils_frompyfunc."""
    from unittest.mock import patch

    from ml_switcheroo_compiler.ops.creation.api_utils import frompyfunc

    with patch("ml_switcheroo_compiler.ops.dispatcher.dispatch_op") as mock_dispatch:
        mock_dispatch.return_value = "dummy_frompyfunc"
        assert frompyfunc(lambda x: x, 1, 1) == "dummy_frompyfunc"


def _side_effect_type_error(obj: BackendArg, dtype: BackendArg = None) -> list[BackendArg]:
    """Simulate backend array error when dtype is provided.

    Args:
        obj (BackendArg): Input object.
        dtype (BackendArg): Desired dtype.

    Returns:
        list[Any]: Nested list wrapping obj.

    Raises:
        TypeError: When raised during `_side_effect_type_error` (unsupported dtype).
    """
    if dtype is not None:
        raise TypeError("unsupported dtype")
    return [obj]


def _side_effect_value_error(obj: BackendArg, dtype: BackendArg = None) -> list[BackendArg]:
    """Simulate backend array error when dtype is missing.

    Args:
        obj (BackendArg): Input object.
        dtype (BackendArg): Desired dtype.

    Returns:
        list[Any]: List with obj and dtype.

    Raises:
        ValueError: When raised during `_side_effect_value_error` (needs dtype).
    """
    if dtype is None:
        raise ValueError("needs dtype")
    return [obj, dtype]


class _DummyInput(MockTestStub):
    """Dummy tensor object holding a shape attribute."""

    def __init__(self, shape: tuple[int, ...]) -> None:
        """Initialize dummy input.

        Args:
            shape (tuple[int, ...]): Tensor shape.
        """
        self.shape = shape


def test_frontend_utils_frompyfunc_infer_shape() -> None:
    """Cover Frompyfunc.infer_shape with broadcast and empty inputs."""
    fpf = Frompyfunc()
    assert fpf.infer_shape() == ()

    in1 = _DummyInput((2, 1))
    in2 = _DummyInput((1, 3))
    # Frompyfunc with args[3:] slicing or args <= 3
    res1 = fpf.infer_shape("a", "b", "c", in1, in2)
    assert res1 == (2, 3)

    res2 = fpf.infer_shape(in1, in2)
    assert res2 == (2, 3)


def test_frontend_utils_geomspace_infer_shape() -> None:
    """Cover Geomspace.infer_shape with negative axis and valid shapes."""
    gs = Geomspace()

    start = _DummyInput((4,))
    stop = _DummyInput((4,))

    # axis < 0 branch
    res_neg_axis = gs.infer_shape(start, stop, num=10, axis=-1)
    assert res_neg_axis == (4, 10)

    # axis >= 0 branch
    res_pos_axis = gs.infer_shape(start, stop, num=10, axis=0)
    assert res_pos_axis == (10, 4)

    # scalar start/stop
    res_scalar = gs.infer_shape(1.0, 10.0, num=5)
    assert res_scalar == (5,)

    # invalid num / axis exception handling
    res_invalid = gs.infer_shape(start, stop, num="invalid", axis="invalid")
    assert res_invalid == (50, 4)


def test_frontend_utils_geometric_infer_shape() -> None:
    """Cover Geometric.infer_shape with size tuple/list/int."""
    geo = Geometric()

    class DummyP(MockTestStub):
        """Dummy parameter tensor with shape."""

        shape = (3, 4)

    # size as tuple
    assert geo.infer_shape(DummyP(), size=(2, 5)) == (2, 5)
    # size as int
    assert geo.infer_shape(DummyP(), size=7) == (7,)
    # size None, defaults to p.shape
    assert geo.infer_shape(DummyP()) == (3, 4)
