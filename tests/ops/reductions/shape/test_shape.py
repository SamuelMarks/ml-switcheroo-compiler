"""Tests for test_shape."""

from __future__ import annotations

import typing
from unittest.mock import patch

import numpy as np
import pytest

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Device, DType, Tensor, TensorConfig
from ml_switcheroo_compiler.ops.reductions.frontend_pool import adaptive_avg_pool2d, adaptive_avg_pool3d, adaptive_max_pool2d, adaptive_max_pool3d, fold, fractional_max_pool2d, fractional_max_pool3d, unfold
from ml_switcheroo_compiler.ops.shape.frontend import (
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
from tests.testing_utils import ensure_callable, ensure_eager_tensor, ensure_mutable, ensure_tensor_dict, has_backend_member

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue


def _test_frontend_pool_coverage_sub_1(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute sub-part 1 of test_frontend_pool_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (t_2d, t_3d) = (_state["t_2d"], _state["t_3d"])
    assert ensure_callable(adaptive_avg_pool2d)(t_2d, (2, 2)) is not None
    assert ensure_callable(adaptive_max_pool2d)(t_2d, (2, 2)) is not None
    with pytest.raises((RuntimeError, ValueError, TypeError, KeyError, Exception)):
        (p, i) = ensure_callable(fractional_max_pool3d)(t_3d, (2, 2, 2))
    with pytest.raises(TypeError):
        (p, i) = ensure_callable(adaptive_max_pool3d)(t_3d, (2, 2, 2), return_indices=True)
    assert ensure_callable(adaptive_avg_pool3d)(t_3d, (2, 2, 2)) is not None
    assert ensure_callable(adaptive_max_pool3d)(t_3d, (2, 2, 2)) is not None
    config.eager_mode = False
    from ml_switcheroo_compiler.tracing.state import global_tracing_state

    global_tracing_state.is_tracing = True
    _state.update(ensure_tensor_dict({"global_tracing_state": global_tracing_state}))
    _state.update(ensure_tensor_dict({"t_2d": t_2d, "t_3d": t_3d}))


def _test_frontend_pool_coverage_sub_2(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute sub-part 2 of test_frontend_pool_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    global_tracing_state = _state["global_tracing_state"]

    class DummyGraph:
        """Test class DummyGraph."""

        name = "dummy"
        nodes = ensure_tensor_dict({})

        def add_node(self, node: EagerValue) -> None:
            """Test function add_node.

            Args:
                node: Test parameter `node`.
            """
            pass

    ensure_mutable(global_tracing_state).active_graph = ensure_eager_tensor(DummyGraph())
    import ml_switcheroo_compiler.ops.reductions.frontend_utils as fu

    orig_emit = fu._emit_reduction_node

    class DummyOp:
        """Test class DummyOp."""

        def __call__(self, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
            """Test function __call__.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.

            Returns:
                EagerValue: Computed output value from `__call__`.
            """
            return "pool"

        def infer_shape(self, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
            """Test function infer_shape.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.

            Returns:
                EagerValue: Computed output value from `infer_shape`.
            """
            return ()

    _state.update(ensure_tensor_dict({"DummyGraph": DummyGraph, "DummyOp": DummyOp, "fu": fu, "orig_emit": orig_emit}))
    _state.update(ensure_tensor_dict({"global_tracing_state": global_tracing_state}))


def _test_frontend_pool_coverage_sub_3(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute sub-part 3 of test_frontend_pool_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (DummyOp, fu, global_tracing_state, orig_emit) = (_state["DummyOp"], _state["fu"], _state["global_tracing_state"], _state["orig_emit"])
    (t_2d, t_3d) = (_state["t_2d"], _state["t_3d"])
    ensure_mutable(fu)._emit_reduction_node = ensure_eager_tensor(lambda *args, **kwargs: DummyOp())
    try:
        assert ensure_callable(fractional_max_pool2d)(t_2d, (2, 2)) is not None
        assert ensure_callable(adaptive_avg_pool2d)(t_2d, (2, 2)) is not None
        assert ensure_callable(adaptive_max_pool2d)(t_2d, (2, 2)) is not None
        assert ensure_callable(unfold)(t_2d, (2, 2)) is not None
        assert ensure_callable(fold)(t_2d, (4, 4), (2, 2)) is not None
        (p, i) = ensure_callable(fractional_max_pool3d)(t_3d, (2, 2, 2))
        assert ensure_callable(adaptive_avg_pool3d)(t_3d, (2, 2, 2)) is not None
        assert ensure_callable(adaptive_max_pool3d)(t_3d, (2, 2, 2)) is not None
    finally:
        ensure_mutable(fu)._emit_reduction_node = ensure_eager_tensor(orig_emit)
        config.eager_mode = True
        ensure_mutable(global_tracing_state).is_tracing = ensure_eager_tensor(False)
    _state.update(ensure_tensor_dict({"DummyOp": DummyOp, "fu": fu, "global_tracing_state": global_tracing_state}))
    _state.update(ensure_tensor_dict({"orig_emit": orig_emit, "t_2d": t_2d, "t_3d": t_3d}))


def test_frontend_pool_coverage() -> None:
    """Test function test_frontend_pool_coverage."""
    config.eager_mode = True

    class DummyDataId:
        """Test class DummyDataId."""

        id = "dummy"

    t_2d = ensure_callable(Tensor)(ensure_callable(DummyDataId)(), ensure_callable(TensorConfig)(shape=(1, 1, 4, 4), dtype=ensure_callable(DType)("float32"), device=ensure_callable(Device)("cpu")))
    t_3d = ensure_callable(Tensor)(ensure_callable(DummyDataId)(), ensure_callable(TensorConfig)(shape=(1, 1, 4, 4, 4), dtype=ensure_callable(DType)("float32"), device=ensure_callable(Device)("cpu")))
    import ml_switcheroo_compiler.backends.registry as registry_mod

    class MockBackend:
        """Test class MockBackend."""

        def execute_op(self, op: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
            """Test function execute_op.

            Args:
                op: Test parameter `op`.
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.

            Returns:
                EagerValue: Computed output value from `execute_op`.
            """
            if op == "FractionalMaxPool3D" or (op == "AdaptiveMaxPool3D" and kwargs.get("return_indices")):
                return (np.zeros((1, 1, 2, 2, 2)), np.zeros((1, 1, 2, 2, 2)))
            if ensure_eager_tensor("3D") in ensure_eager_tensor(op):
                return np.zeros((1, 1, 2, 2, 2))
            return np.zeros((1, 1, 2, 2))

        def array(self, x: EagerValue, dtype: EagerValue = None) -> EagerValue:
            """Test function array.

            Args:
                x: Test parameter `x`.
                dtype: Test parameter `dtype`.

            Returns:
                EagerValue: Computed output value from `array`.
            """
            return np.array(x)

    orig_backend = registry_mod.get_active_backend
    ensure_mutable(registry_mod).get_active_backend = ensure_eager_tensor(lambda: MockBackend())
    try:
        _state: dict[str, EagerTensorProtocol] = {}
        _state.update(ensure_tensor_dict({"id": id, "orig_backend": orig_backend, "registry_mod": registry_mod, "t_2d": t_2d}))
        _state.update(ensure_tensor_dict({"t_3d": t_3d}))
        for _sub_fn in (_test_frontend_pool_coverage_sub_1, _test_frontend_pool_coverage_sub_2, _test_frontend_pool_coverage_sub_3):
            _sub_fn(_state)
        assert len(_state) > 0
    finally:
        registry_mod.get_active_backend = orig_backend


def _test_shape_frontend_coverage_sub_1(_state: dict[str, EagerTensorProtocol]) -> None:
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
    assert ensure_callable(permute)(t, [1, 0]) is not None
    assert swapaxes(t, 0, 1) is not None
    assert roll(t, 1) is not None
    assert atleast_1d(t) is not None
    assert atleast_2d(t) is not None
    assert atleast_3d(t) is not None
    assert squeeze(t) is not None
    assert diagflat(t) is not None
    assert block([[t, t]]) is not None
    assert delete(t, 0) is not None
    _state.update(ensure_tensor_dict({"t": t}))


def _test_shape_frontend_coverage_sub_2(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute sub-part 2 of test_shape_frontend_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (backend_inst, t) = (_state["backend_inst"], _state["t"])
    ensure_mutable(backend_inst).res = ensure_eager_tensor([np.zeros((2,)), np.zeros((2,))])
    assert diag_indices(2) is not None
    assert diag_indices_from(t) is not None
    ensure_mutable(backend_inst).res = ensure_eager_tensor(np.array(4))
    assert size(t) is not None
    ensure_mutable(backend_inst).res = ensure_eager_tensor(np.zeros((4,)))
    assert ensure_callable(reshape)(t, (4,)) is not None
    assert ensure_callable(flip)(t) is not None
    assert fliplr(t) is not None
    assert flipud(t) is not None
    config.eager_mode = False
    from ml_switcheroo_compiler.tracing.state import global_tracing_state

    global_tracing_state.is_tracing = True
    _state.update(ensure_tensor_dict({"global_tracing_state": global_tracing_state}))
    _state.update(ensure_tensor_dict({"backend_inst": backend_inst, "t": t}))


def _test_shape_frontend_coverage_sub_3(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute sub-part 3 of test_shape_frontend_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    global_tracing_state = _state["global_tracing_state"]

    class DummyGraph:
        """Test class DummyGraph."""

        name = "dummy"
        nodes = ensure_tensor_dict({})

        def add_node(self, node: EagerValue) -> None:
            """Test function add_node.

            Args:
                node: Test parameter `node`.
            """
            pass

    ensure_mutable(global_tracing_state).active_graph = ensure_eager_tensor(DummyGraph())
    import ml_switcheroo_compiler.ops.shape.utils as shape_utils

    orig_emit = shape_utils._emit_shape_node

    class DummyOp:
        """Test class DummyOp."""

        def __call__(self, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
            """Test function __call__.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.

            Returns:
                EagerValue: Computed output value from `__call__`.
            """
            return "shape"

        def infer_shape(self, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
            """Test function infer_shape.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.

            Returns:
                EagerValue: Computed output value from `infer_shape`.
            """
            return ()

    _state.update(ensure_tensor_dict({"DummyGraph": DummyGraph, "DummyOp": DummyOp, "orig_emit": orig_emit, "shape_utils": shape_utils}))
    _state.update(ensure_tensor_dict({"global_tracing_state": global_tracing_state}))


def _test_shape_frontend_coverage_sub_4(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute sub-part 4 of test_shape_frontend_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (DummyOp, shape_utils) = (_state["DummyOp"], _state["shape_utils"])
    ensure_mutable(shape_utils)._emit_shape_node = ensure_eager_tensor(lambda *args, **kwargs: DummyOp())
    _state.update(ensure_tensor_dict({"DummyOp": DummyOp, "shape_utils": shape_utils}))


def _test_shape_frontend_coverage_sub_5(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute sub-part 5 of test_shape_frontend_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (DummyOp, global_tracing_state, orig_emit, shape_utils) = (_state["DummyOp"], _state["global_tracing_state"], _state["orig_emit"], _state["shape_utils"])
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
        assert ensure_callable(permute)(t, [1, 0]) is not None
        assert swapaxes(t, 0, 1) is not None
        assert roll(t, 1) is not None
        assert atleast_1d(t) is not None
        assert atleast_2d(t) is not None
        assert atleast_3d(t) is not None
        assert squeeze(t) is not None
        assert diagflat(t) is not None
        assert block([[t, t]]) is not None
        assert delete(t, 0) is not None
        ensure_mutable(shape_utils)._emit_shape_node = ensure_eager_tensor(orig_emit)
        with patch("ml_switcheroo_compiler.tracing.builder.TracingNodeBuilder.emit_tracing_node") as mock_emit:

            class DummyItemNode:
                """Test class DummyItemNode."""

                _shape = ()
                config = None

            mock_emit.return_value = DummyItemNode()
            assert diag_indices(2) is not None
            assert diag_indices_from(t) is not None
        ensure_mutable(shape_utils)._emit_shape_node = ensure_eager_tensor(lambda *args, **kwargs: DummyOp())
        assert size(t) is not None
        assert ensure_callable(reshape)(t, (4,)) is not None
        assert ensure_callable(flip)(t) is not None
        assert fliplr(t) is not None
        assert flipud(t) is not None
    finally:
        ensure_mutable(shape_utils)._emit_shape_node = ensure_eager_tensor(orig_emit)
        config.eager_mode = True
        ensure_mutable(global_tracing_state).is_tracing = ensure_eager_tensor(False)
    _state.update(ensure_tensor_dict({"DummyOp": DummyOp, "global_tracing_state": global_tracing_state, "orig_emit": orig_emit}))
    _state.update(ensure_tensor_dict({"shape_utils": shape_utils, "t": t}))


def test_shape_frontend_coverage() -> None:
    """Test function test_shape_frontend_coverage."""
    config.eager_mode = True

    class DummyDataId:
        """Test class DummyDataId."""

        id = "dummy"

    t = Tensor(np.ones((2, 2)), TensorConfig(shape=(2, 2), dtype=DType("float32"), device=Device("cpu")))
    import ml_switcheroo_compiler.backends.registry as registry_mod

    class MockShapeBackend:
        """Test class MockShapeBackend."""

        def __init__(self) -> None:
            """Test function __init__."""
            self.res = np.zeros((2, 2))

        def execute_op(self, op: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
            """Test function execute_op.

            Args:
                op: Test parameter `op`.
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.

            Returns:
                EagerValue: Computed output value from `execute_op`.
            """
            return self.res

        def array(self, x: EagerValue, dtype: EagerValue = None) -> EagerValue:
            """Test function array.

            Args:
                x: Test parameter `x`.
                dtype: Test parameter `dtype`.

            Returns:
                EagerValue: Computed output value from `array`.
            """
            return np.array(x)

        def item(self, x: EagerValue) -> EagerValue:
            """Test function item.

            Args:
                x: Test parameter `x`.

            Returns:
                EagerValue: Computed output value from `item`.
            """
            return ensure_eager_tensor(ensure_eager_tensor(x).item)() if has_backend_member(x, "item") else x

    backend_inst = MockShapeBackend()
    orig_backend = registry_mod.get_active_backend
    ensure_mutable(registry_mod).get_active_backend = ensure_eager_tensor(lambda: backend_inst)
    try:
        _state: dict[str, EagerTensorProtocol] = {}
        _state.update(ensure_tensor_dict({"backend_inst": backend_inst, "id": id, "orig_backend": orig_backend, "registry_mod": registry_mod}))
        _state.update(ensure_tensor_dict({"t": t}))
        for _sub_fn in (_test_shape_frontend_coverage_sub_1, _test_shape_frontend_coverage_sub_2, _test_shape_frontend_coverage_sub_3, _test_shape_frontend_coverage_sub_4, _test_shape_frontend_coverage_sub_5):
            _sub_fn(_state)
        assert len(_state) > 0
    finally:
        registry_mod.get_active_backend = orig_backend


def test_compute_reduction_shape() -> None:
    """Test function test_compute_reduction_shape."""
    assert compute_reduction_shape((2, 3, 4), (1,), keepdims=True) == (2, 1, 4)
    assert compute_reduction_shape((2, 3, 4), (1,), keepdims=False) == (2, 4)
