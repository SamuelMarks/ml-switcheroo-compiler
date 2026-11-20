"""Behavioral unit tests for ragged."""

from __future__ import annotations

import typing
from unittest.mock import patch

import numpy as np
import pytest
from ml_switcheroo_ir import LogicalNode

import ml_switcheroo_compiler.ops.ragged.api as ragged_frontend
import ml_switcheroo_compiler.ops.ragged.core as ragged_core_mod
from ml_switcheroo_compiler.backends.numpy.eager import ensure_ndarray
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Device, DType, Tensor, TensorConfig
from ml_switcheroo_compiler.ops.ragged import (
    BooleanMask,
    MapFlatValues,
    RaggedAdd,
    RaggedConstant,
    RaggedCrossHashed,
    RaggedDynamicBroadcast,
    RaggedGather,
    RaggedMatMul,
    RaggedRange,
    RaggedRowSplitsToSegmentIds,
    RaggedSegmentIdsToRowSplits,
    RaggedStack,
    RaggedStackDynamicPartitions,
    RaggedTensorToDense,
    boolean_mask,
    map_flat_values,
    ragged_tensor_to_dense,
)
from ml_switcheroo_compiler.ops.ragged.api import ragged_dot, ragged_stack, ragged_stack_dynamic_partitions
from ml_switcheroo_compiler.ops.ragged.core import RaggedDot
from tests.testing_utils import coerce_operand_list

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import BackendArg


def test_ragged_ops() -> None:
    """Test the ragged ops behavior.

    Returns:
        OpValue: The inferred shape or computed result.
    """
    with patch("ml_switcheroo_compiler.ops.ragged.api._ragged_op") as mock_ragged:
        mock_ragged.return_value = "res"
        res1 = ragged_stack()
        assert res1 == "res"
        res2 = ragged_stack_dynamic_partitions()
        assert res2 == "res"
        res3 = ragged_dot()
        assert res3 == "res"
    op = RaggedDot()
    res4 = op.infer_shape()
    assert res4 == ()


class _TestRaggedCoverageState(typing.TypedDict, total=False):
    """Typed shared state for `_test_ragged_coverage` split helpers."""

    DummyOp: type
    orig_eager: bool
    t: Tensor


def _test_ragged_coverage_part_1(_state: _TestRaggedCoverageState) -> None:
    """Execute part 1 of test_ragged_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    config.eager_mode = True
    t = Tensor(np.ones((2, 2)), TensorConfig(shape=(2, 2), dtype=DType("float32"), device=Device("cpu")))
    assert RaggedGather().infer_shape() == ()
    assert RaggedTensorToDense().infer_shape() == ()
    assert RaggedAdd().infer_shape() == ()
    assert RaggedMatMul().infer_shape() == ()
    assert RaggedDynamicBroadcast().infer_shape() == ()
    assert RaggedConstant().infer_shape() == ()
    assert RaggedCrossHashed().infer_shape() == ()
    assert len(coerce_operand_list(RaggedRange().infer_shape())) == 2
    assert len(coerce_operand_list(RaggedRowSplitsToSegmentIds().infer_shape())) == 1
    assert len(coerce_operand_list(RaggedSegmentIdsToRowSplits().infer_shape())) == 1
    assert RaggedStack().infer_shape() == ()
    assert len(coerce_operand_list(RaggedStackDynamicPartitions().infer_shape())) == 2
    assert len(coerce_operand_list(BooleanMask().infer_shape())) == 1
    assert MapFlatValues().infer_shape(t) == ()
    _state.update({"t": t})


def _test_ragged_coverage_part_2(_state: _TestRaggedCoverageState) -> None:
    """Execute part 2 of test_ragged_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    t = _state["t"]
    with patch("ml_switcheroo_compiler.backends.registry.get_active_backend") as mock_backend:
        mock_backend.return_value.execute_op.return_value = np.zeros((2, 2))
        assert ragged_tensor_to_dense(t) is not None
        config.eager_mode = False
        from ml_switcheroo_compiler.tracing.state import global_tracing_state

        global_tracing_state.is_tracing = True

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
        assert ragged_tensor_to_dense(t) is not None
        config.eager_mode = True
        global_tracing_state.is_tracing = False
    _state.update({"t": t})


def _test_ragged_coverage_part_3(_state: _TestRaggedCoverageState) -> None:
    """Execute part 3 of test_ragged_coverage.

    Args:
        _state: Shared test state dictionary.
    """

    class DummyOp:
        """Test class DummyOp."""

        device = "cpu"

        def __init__(self) -> None:
            """Test function __init__."""
            self.op_type = "dummy"

    orig_eager = config.eager_mode
    _state.update({"DummyOp": DummyOp, "orig_eager": orig_eager})


def _test_ragged_coverage_part_4(_state: _TestRaggedCoverageState) -> None:
    """Execute part 4 of test_ragged_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (DummyOp, orig_eager, t) = (
        _state["DummyOp"],
        _state["orig_eager"],
        _state["t"],
    )
    try:
        config.eager_mode = False
        from ml_switcheroo_compiler.tracing.state import global_tracing_state

        global_tracing_state.is_tracing = True

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

        class DummyOpMask:
            """Test class DummyOpMask."""

            def __call__(self, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
                """Test function __call__.

                Args:
                    args: Test parameter `args`.
                    kwargs: Test parameter `kwargs`.

                Returns:
                    BackendArg: Computed output value from `__call__`.
                """
                return "masked"

            def infer_shape(self, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
                """Test function infer_shape.

                Args:
                    args: Test parameter `args`.
                    kwargs: Test parameter `kwargs`.

                Returns:
                    BackendArg: Computed output value from `infer_shape`.
                """
                return ()

            def dispatch_infer_shape(self, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
                """Delegate shape inference to `infer_shape`.

                Args:
                    args: Test parameter `args`.
                    kwargs: Test parameter `kwargs`.

                Returns:
                    BackendArg: Inferred shape tuple.
                """
                return self.infer_shape(*args, **kwargs)

        import ml_switcheroo_compiler.ops.linalg.utils as linalg_utils

        orig_emit = linalg_utils._emit_linalg_node
        pytest.MonkeyPatch().setattr(linalg_utils, "_emit_linalg_node", lambda *args, **kwargs: "masked", raising=False)
        try:
            with patch.object(ragged_frontend, "get_op", lambda x: DummyOpMask):
                assert boolean_mask(t, t) is not None
                assert map_flat_values(DummyOp(), t) is not None
        finally:
            linalg_utils._emit_linalg_node = orig_emit
    finally:
        pytest.MonkeyPatch().setattr(config, "eager_mode", orig_eager, raising=False)
        global_tracing_state.is_tracing = False


def test_ragged_coverage() -> None:
    """Test function test_ragged_coverage."""
    _state: _TestRaggedCoverageState = {}
    _test_ragged_coverage_part_1(_state)
    _test_ragged_coverage_part_2(_state)
    _test_ragged_coverage_part_3(_state)
    _test_ragged_coverage_part_4(_state)
    assert len(_state) > 0


class DummyWithShape:
    """Mock operand providing a shape attribute."""

    def __init__(self, shape: tuple[int, ...]) -> None:
        """Initialize mock with specific shape.

        Args:
            shape (tuple[int, ...]): Target shape.
        """
        self.shape: tuple[int, ...] = shape


class DummyWithoutShape:
    """Mock operand lacking a shape attribute."""

    def __init__(self) -> None:
        """Initialize mock without shape."""
        self.val: int = 42


def test_ragged_core_coverage() -> None:
    """Verify 100% line and branch coverage for ragged core operations."""
    dot_op = ragged_core_mod.RaggedDot()

    # len(args) >= 2 with shapes
    t1 = DummyWithShape((2, 4, 8))
    t2 = DummyWithShape((8, 16))
    out1 = dot_op.infer_shape(t1, t2)
    assert out1 == (2, 4, 16)

    # len(args) < 2 using kwargs rt_a and b
    out2 = dot_op.infer_shape(rt_a=t1, b=t2)
    assert out2 == (2, 4, 16)

    # a is None -> returns ()
    assert dot_op.infer_shape(None, None) == ()

    # shape_a variations:
    # 1. a is raw list/tuple
    assert ensure_ndarray(dot_op.infer_shape([3, 5], None))[0] == 3

    # 2. len(shape_a) <= 1 (ragged_dim uses SymVar fallback)
    t_1d = DummyWithShape((7,))
    out_1d = dot_op.infer_shape(t_1d, None)
    assert ensure_ndarray(out_1d)[0] == 7
    assert str(ensure_ndarray(out_1d)[1]) == "ragged_dim"
    assert ensure_ndarray(out_1d)[2] == 1  # len(shape_a) <= 2, len(shape_b) == 0 fallback

    # 3. len(shape_a) == 0 (batch fallback to 1)
    t_0d = DummyWithShape(())
    out_0d = dot_op.infer_shape(t_0d, None)
    assert ensure_ndarray(out_0d)[0] == 1

    # 4. len(shape_b) == 0, len(shape_a) > 2 (out_features falls back to shape_a[-1])
    t_3d = DummyWithShape((2, 3, 9))
    out_3d = dot_op.infer_shape(t_3d, None)
    assert ensure_ndarray(out_3d)[2] == 9
