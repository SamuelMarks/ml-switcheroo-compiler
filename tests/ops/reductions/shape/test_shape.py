"""Tests for test_shape."""

from __future__ import annotations

import types
import typing
from unittest.mock import patch

import numpy as np
import pytest
from ml_switcheroo_ir import LogicalNode

from ml_switcheroo_compiler.backends.numpy.eager import ensure_ndarray
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Device, DType, Tensor, TensorConfig
from ml_switcheroo_compiler.ops.reductions.api_pool import (
    adaptive_avg_pool2d,
    adaptive_avg_pool3d,
    adaptive_max_pool2d,
    adaptive_max_pool3d,
    fold,
    fractional_max_pool2d,
    fractional_max_pool3d,
    unfold,
)
from ml_switcheroo_compiler.ops.shape.api import (
    argpartition,
    argwhere,
    atleast_1d,
    atleast_2d,
    atleast_3d,
    block,
    compress,
    delete,
    diag_indices,
    diag_indices_from,
    diagflat,
    expand_dims,
    fill_diagonal,
    flip,
    fliplr,
    flipud,
    insert,
    moveaxis,
    partition,
    permute,
    reshape,
    roll,
    size,
    squeeze,
    swapaxes,
)
from ml_switcheroo_compiler.ops.shape.utils import compute_reduction_shape
from ml_switcheroo_compiler.tracing.state import TracingState
from tests.testing_utils import (
    EagerBoundCallable,
    MockTestStub,
    adapt_backend_module,
    require_callable,
    unpack_pair,
)

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import BackendArg


class MockShapeBackend:
    """Mock backend used for shape frontend coverage tests."""

    def __init__(self) -> None:
        """Initialize mock backend with a default 2x2 result array."""
        self.res: BackendArg = np.zeros((2, 2))

    def execute_op(self, op: BackendArg, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
        """Return preconfigured mock array result.

        Args:
            op (BackendArg): Operation name.
            *args (BackendArg): Positional operation arguments.
            **kwargs (BackendArg): Keyword operation arguments.

        Returns:
            BackendArg: Mock result array.
        """
        del op, args, kwargs
        return self.res

    def array(self, x: BackendArg, dtype: BackendArg = None) -> BackendArg:
        """Convert input `x` into a NumPy array.

        Args:
            x (BackendArg): Input array-like value.
            dtype (BackendArg): Optional dtype parameter.

        Returns:
            BackendArg: NumPy array.
        """
        del dtype
        return np.array(x)

    def item(self, x: BackendArg) -> BackendArg:
        """Extract scalar item when supported.

        Args:
            x (BackendArg): Input scalar or array.

        Returns:
            BackendArg: Extracted Python scalar or original input.
        """
        return ensure_ndarray(x).item() if adapt_backend_module(x).has_symbol("item") else x


class _TestFrontendPoolCoverageState(typing.TypedDict, total=False):
    """Typed shared state for `_test_frontend_pool_coverage` split helpers."""

    DummyOp: type
    fu: types.ModuleType
    global_tracing_state: TracingState
    orig_emit: EagerBoundCallable
    t_2d: Tensor
    t_3d: Tensor


def _test_frontend_pool_coverage_sub_1(_state: _TestFrontendPoolCoverageState) -> None:
    """Execute sub-part 1 of test_frontend_pool_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (t_2d, t_3d) = (_state["t_2d"], _state["t_3d"])
    assert adaptive_avg_pool2d(t_2d, (2, 2)) is not None
    assert adaptive_max_pool2d(t_2d, (2, 2)) is not None
    with pytest.raises((RuntimeError, ValueError, TypeError, KeyError, Exception)):
        (p, i) = unpack_pair(fractional_max_pool3d(t_3d, (2, 2, 2)))
    with pytest.raises(TypeError):
        (p, i) = adaptive_max_pool3d(t_3d, (2, 2, 2), return_indices=True)
    assert adaptive_avg_pool3d(t_3d, (2, 2, 2)) is not None
    assert adaptive_max_pool3d(t_3d, (2, 2, 2)) is not None
    config.eager_mode = False
    from ml_switcheroo_compiler.tracing.state import global_tracing_state

    global_tracing_state.is_tracing = True
    _state.update({"global_tracing_state": global_tracing_state})
    _state.update({"t_2d": t_2d, "t_3d": t_3d})


def _test_frontend_pool_coverage_sub_2(_state: _TestFrontendPoolCoverageState) -> None:
    """Execute sub-part 2 of test_frontend_pool_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    global_tracing_state = _state["global_tracing_state"]

    class DummyGraph:
        """Test class DummyGraph."""

        name = "dummy"
        nodes: dict[str, LogicalNode] = {}

        def add_node(self, node: BackendArg) -> None:
            """Test function add_node.

            Args:
                node: Test parameter `node`.
            """
            pass

    pytest.MonkeyPatch().setattr(global_tracing_state, "active_graph", DummyGraph(), raising=False)
    import ml_switcheroo_compiler.ops.reductions.api_utils as fu

    orig_emit = fu._emit_reduction_node

    class DummyOp:
        """Test class DummyOp."""

        def __call__(self, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
            """Test function __call__.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.

            Returns:
                BackendArg: Computed output value from `__call__`.
            """
            return "pool"

        def infer_shape(self, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
            """Test function infer_shape.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.

            Returns:
                BackendArg: Computed output value from `infer_shape`.
            """
            return ()

    _state.update({"DummyOp": DummyOp, "fu": fu, "orig_emit": require_callable(orig_emit)})
    _state.update({"global_tracing_state": global_tracing_state})


def _test_frontend_pool_coverage_sub_3(_state: _TestFrontendPoolCoverageState) -> None:
    """Execute sub-part 3 of test_frontend_pool_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (DummyOp, fu, global_tracing_state, orig_emit) = (
        _state["DummyOp"],
        _state["fu"],
        _state["global_tracing_state"],
        _state["orig_emit"],
    )
    (t_2d, t_3d) = (_state["t_2d"], _state["t_3d"])
    pytest.MonkeyPatch().setattr(fu, "_emit_reduction_node", lambda *args, **kwargs: DummyOp(), raising=False)
    try:
        assert fractional_max_pool2d(t_2d, (2, 2)) is not None
        assert adaptive_avg_pool2d(t_2d, (2, 2)) is not None
        assert adaptive_max_pool2d(t_2d, (2, 2)) is not None
        assert unfold(t_2d, (2, 2)) is not None
        assert fold(t_2d, (4, 4), (2, 2)) is not None
        (p, i) = unpack_pair(fractional_max_pool3d(t_3d, (2, 2, 2)))
        assert adaptive_avg_pool3d(t_3d, (2, 2, 2)) is not None
        assert adaptive_max_pool3d(t_3d, (2, 2, 2)) is not None
    finally:
        pytest.MonkeyPatch().setattr(fu, "_emit_reduction_node", orig_emit, raising=False)
        config.eager_mode = True
        pytest.MonkeyPatch().setattr(global_tracing_state, "is_tracing", False, raising=False)
    _state.update({"DummyOp": DummyOp, "fu": fu, "global_tracing_state": global_tracing_state})
    _state.update({"orig_emit": require_callable(orig_emit), "t_2d": t_2d, "t_3d": t_3d})


def test_frontend_pool_coverage() -> None:
    """Test function test_frontend_pool_coverage."""
    config.eager_mode = True

    class DummyDataId(MockTestStub):
        """Test class DummyDataId."""

        id = "dummy"

    t_2d = Tensor(
        DummyDataId(),
        TensorConfig(shape=(1, 1, 4, 4), dtype=DType("float32"), device=Device("cpu")),
    )
    t_3d = Tensor(
        DummyDataId(),
        TensorConfig(shape=(1, 1, 4, 4, 4), dtype=DType("float32"), device=Device("cpu")),
    )
    import ml_switcheroo_compiler.backends.registry as registry_mod

    class MockBackend:
        """Test class MockBackend."""

        def execute_op(self, op: BackendArg, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
            """Test function execute_op.

            Args:
                op: Test parameter `op`.
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.

            Returns:
                BackendArg: Computed output value from `execute_op`.
            """
            if op == "FractionalMaxPool3D" or (op == "AdaptiveMaxPool3D" and kwargs.get("return_indices")):
                return (np.zeros((1, 1, 2, 2, 2)), np.zeros((1, 1, 2, 2, 2)))
            if "3D" in str(op):
                return np.zeros((1, 1, 2, 2, 2))
            return np.zeros((1, 1, 2, 2))

        def array(self, x: BackendArg, dtype: BackendArg = None) -> BackendArg:
            """Test function array.

            Args:
                x: Test parameter `x`.
                dtype: Test parameter `dtype`.

            Returns:
                BackendArg: Computed output value from `array`.
            """
            return np.array(x)

    orig_backend = registry_mod.get_active_backend
    pytest.MonkeyPatch().setattr(registry_mod, "get_active_backend", lambda: MockBackend(), raising=False)
    try:
        _state: _TestFrontendPoolCoverageState = {}
        _state.update({"t_2d": t_2d})
        _state.update({"t_3d": t_3d})
        for _sub_fn in (
            _test_frontend_pool_coverage_sub_1,
            _test_frontend_pool_coverage_sub_2,
            _test_frontend_pool_coverage_sub_3,
        ):
            _sub_fn(_state)
        assert len(_state) > 0
    finally:
        registry_mod.get_active_backend = orig_backend


class _TestShapeFrontendCoverageState(typing.TypedDict, total=False):
    """Typed shared state for `_test_shape_frontend_coverage` split helpers."""

    DummyOp: type
    backend_inst: MockShapeBackend
    global_tracing_state: TracingState
    orig_emit: EagerBoundCallable
    shape_utils: types.ModuleType
    t: Tensor


def _test_shape_frontend_coverage_sub_1(_state: _TestShapeFrontendCoverageState) -> None:
    """Execute sub-part 1 of test_shape_frontend_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    t = _state["t"]
    assert expand_dims(t, 0) is not None
    assert argwhere(t) is not None
    assert argpartition(t, 1) is not None
    assert partition(t, 1) is not None
    assert compress([True, False], t) is not None
    assert insert(t, 0, 1) is not None
    assert fill_diagonal(t, 0) is not None
    assert moveaxis(t, 0, 1) is not None
    assert permute(t, [1, 0]) is not None
    assert swapaxes(t, 0, 1) is not None
    assert roll(t, 1) is not None
    assert atleast_1d(t) is not None
    assert atleast_2d(t) is not None
    assert atleast_3d(t) is not None
    assert squeeze(t) is not None
    assert diagflat(t) is not None
    assert block([[t, t]]) is not None
    assert delete(t, 0) is not None
    _state.update({"t": t})


def _test_shape_frontend_coverage_sub_2(_state: _TestShapeFrontendCoverageState) -> None:
    """Execute sub-part 2 of test_shape_frontend_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (backend_inst, t) = (_state["backend_inst"], _state["t"])
    pytest.MonkeyPatch().setattr(backend_inst, "res", [np.zeros((2,)), np.zeros((2,))], raising=False)
    assert diag_indices(2) is not None
    assert diag_indices_from(t) is not None
    pytest.MonkeyPatch().setattr(backend_inst, "res", np.array(4), raising=False)
    assert size(t) is not None
    pytest.MonkeyPatch().setattr(backend_inst, "res", np.zeros((4,)), raising=False)
    assert reshape(t, (4,)) is not None
    assert flip(t) is not None
    assert fliplr(t) is not None
    assert flipud(t) is not None
    config.eager_mode = False
    from ml_switcheroo_compiler.tracing.state import global_tracing_state

    global_tracing_state.is_tracing = True
    _state.update({"global_tracing_state": global_tracing_state})
    _state.update({"backend_inst": backend_inst, "t": t})


def _test_shape_frontend_coverage_sub_3(_state: _TestShapeFrontendCoverageState) -> None:
    """Execute sub-part 3 of test_shape_frontend_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    global_tracing_state = _state["global_tracing_state"]

    class DummyGraph:
        """Test class DummyGraph."""

        name = "dummy"
        nodes: dict[str, LogicalNode] = {}

        def add_node(self, node: BackendArg) -> None:
            """Test function add_node.

            Args:
                node: Test parameter `node`.
            """
            pass

    pytest.MonkeyPatch().setattr(global_tracing_state, "active_graph", DummyGraph(), raising=False)
    import ml_switcheroo_compiler.ops.shape.utils as shape_utils

    orig_emit = shape_utils._emit_shape_node

    class DummyOp:
        """Test class DummyOp."""

        def __call__(self, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
            """Test function __call__.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.

            Returns:
                BackendArg: Computed output value from `__call__`.
            """
            return "shape"

        def infer_shape(self, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
            """Test function infer_shape.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.

            Returns:
                BackendArg: Computed output value from `infer_shape`.
            """
            return ()

    _state.update({"DummyOp": DummyOp, "orig_emit": require_callable(orig_emit), "shape_utils": shape_utils})
    _state.update({"global_tracing_state": global_tracing_state})


def _test_shape_frontend_coverage_sub_4(_state: _TestShapeFrontendCoverageState) -> None:
    """Execute sub-part 4 of test_shape_frontend_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (DummyOp, shape_utils) = (_state["DummyOp"], _state["shape_utils"])
    pytest.MonkeyPatch().setattr(shape_utils, "_emit_shape_node", lambda *args, **kwargs: DummyOp(), raising=False)
    _state.update({"DummyOp": DummyOp, "shape_utils": shape_utils})


def _test_shape_frontend_coverage_sub_5(_state: _TestShapeFrontendCoverageState) -> None:
    """Execute sub-part 5 of test_shape_frontend_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (DummyOp, global_tracing_state, orig_emit, shape_utils) = (
        _state["DummyOp"],
        _state["global_tracing_state"],
        _state["orig_emit"],
        _state["shape_utils"],
    )
    t = _state["t"]
    try:
        assert expand_dims(t, 0) is not None
        assert argwhere(t) is not None
        assert argpartition(t, 1) is not None
        assert partition(t, 1) is not None
        assert compress([True, False], t) is not None
        assert insert(t, 0, 1) is not None
        assert fill_diagonal(t, 0) is not None
        assert moveaxis(t, 0, 1) is not None
        assert permute(t, [1, 0]) is not None
        assert swapaxes(t, 0, 1) is not None
        assert roll(t, 1) is not None
        assert atleast_1d(t) is not None
        assert atleast_2d(t) is not None
        assert atleast_3d(t) is not None
        assert squeeze(t) is not None
        assert diagflat(t) is not None
        assert block([[t, t]]) is not None
        assert delete(t, 0) is not None
        pytest.MonkeyPatch().setattr(shape_utils, "_emit_shape_node", orig_emit, raising=False)
        with patch("ml_switcheroo_compiler.tracing.builder.TracingNodeBuilder.emit_tracing_node") as mock_emit:

            class DummyItemNode:
                """Test class DummyItemNode."""

                _shape = ()
                config = None

            mock_emit.return_value = DummyItemNode()
            assert diag_indices(2) is not None
            assert diag_indices_from(t) is not None
        pytest.MonkeyPatch().setattr(shape_utils, "_emit_shape_node", lambda *args, **kwargs: DummyOp(), raising=False)
        assert size(t) is not None
        assert reshape(t, (4,)) is not None
        assert flip(t) is not None
        assert fliplr(t) is not None
        assert flipud(t) is not None
    finally:
        pytest.MonkeyPatch().setattr(shape_utils, "_emit_shape_node", orig_emit, raising=False)
        config.eager_mode = True
        pytest.MonkeyPatch().setattr(global_tracing_state, "is_tracing", False, raising=False)
    _state.update(
        {"DummyOp": DummyOp, "global_tracing_state": global_tracing_state, "orig_emit": require_callable(orig_emit)}
    )
    _state.update({"shape_utils": shape_utils, "t": t})


def test_shape_frontend_coverage() -> None:
    """Test function test_shape_frontend_coverage."""
    config.eager_mode = True

    class DummyDataId(MockTestStub):
        """Test class DummyDataId."""

        id = "dummy"

    t = Tensor(np.ones((2, 2)), TensorConfig(shape=(2, 2), dtype=DType("float32"), device=Device("cpu")))
    import ml_switcheroo_compiler.backends.registry as registry_mod

    backend_inst = MockShapeBackend()
    orig_backend = registry_mod.get_active_backend
    pytest.MonkeyPatch().setattr(registry_mod, "get_active_backend", lambda: backend_inst, raising=False)
    try:
        _state: _TestShapeFrontendCoverageState = {}
        _state.update({"backend_inst": backend_inst})
        _state.update({"t": t})
        for _sub_fn in (
            _test_shape_frontend_coverage_sub_1,
            _test_shape_frontend_coverage_sub_2,
            _test_shape_frontend_coverage_sub_3,
            _test_shape_frontend_coverage_sub_4,
            _test_shape_frontend_coverage_sub_5,
        ):
            _sub_fn(_state)
        assert len(_state) > 0
    finally:
        registry_mod.get_active_backend = orig_backend


def test_compute_reduction_shape() -> None:
    """Test function test_compute_reduction_shape."""
    assert compute_reduction_shape((2, 3, 4), (1,), keepdims=True) == (2, 1, 4)
    assert compute_reduction_shape((2, 3, 4), (1,), keepdims=False) == (2, 4)
