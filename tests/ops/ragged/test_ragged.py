"""Tests for test_ragged."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast
from unittest.mock import patch

import numpy as np

import ml_switcheroo_compiler.ops.ragged.frontend as ragged_frontend
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
from ml_switcheroo_compiler.ops.ragged.core import RaggedDot
from ml_switcheroo_compiler.ops.ragged.frontend import ragged_dot, ragged_stack, ragged_stack_dynamic_partitions

if TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol, EagerTensorProtocol, EagerValue


def test_ragged_ops() -> None:
    """Test the ragged ops behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    with patch("ml_switcheroo_compiler.ops.ragged.frontend._ragged_op") as mock_ragged:
        mock_ragged.return_value = "res"
        res1 = ragged_stack()
        assert cast("EagerTensorProtocol", res1) == cast("EagerTensorProtocol", "res")
        res2 = ragged_stack_dynamic_partitions()
        assert cast("EagerTensorProtocol", res2) == cast("EagerTensorProtocol", "res")
        res3 = ragged_dot()
        assert cast("EagerTensorProtocol", res3) == cast("EagerTensorProtocol", "res")
    op = RaggedDot()
    res4 = op.infer_shape()
    assert res4 == ()


def _test_ragged_coverage_part_1(_state: dict[str, EagerTensorProtocol]) -> None:
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
    assert cast("EagerCallableProtocol", len)(cast("EagerCallableProtocol", RaggedRange)().infer_shape()) == 2
    assert cast("EagerCallableProtocol", len)(cast("EagerCallableProtocol", RaggedRowSplitsToSegmentIds)().infer_shape()) == 1
    assert cast("EagerCallableProtocol", len)(cast("EagerCallableProtocol", RaggedSegmentIdsToRowSplits)().infer_shape()) == 1
    assert RaggedStack().infer_shape() == ()
    assert cast("EagerCallableProtocol", len)(cast("EagerCallableProtocol", RaggedStackDynamicPartitions)().infer_shape()) == 2
    assert cast("EagerCallableProtocol", len)(cast("EagerCallableProtocol", BooleanMask)().infer_shape()) == 1
    assert MapFlatValues().infer_shape(t) == ()
    getattr(_state, "update")({"t": t})
    _state.update(locals())


def _test_ragged_coverage_part_2(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 2 of test_ragged_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    t = _state["t"]
    with patch("ml_switcheroo_compiler.backends.registry.get_active_backend") as mock_backend:
        mock_backend.return_value.execute_op.return_value = np.zeros((2, 2))
        assert cast("EagerCallableProtocol", ragged_tensor_to_dense)(t) is not None
        config.eager_mode = False
        from ml_switcheroo_compiler.tracing.state import global_tracing_state

        global_tracing_state.is_tracing = True

        class DummyGraph:
            """Test class DummyGraph."""

            name = "dummy"
            nodes = cast("dict[str, EagerTensorProtocol]", {})

            def add_node(self, node: EagerValue) -> None:
                """Test function add_node.

                Args:
                    node: Test parameter `node`.
                """
                pass

        setattr(global_tracing_state, "active_graph", DummyGraph())
        assert cast("EagerCallableProtocol", ragged_tensor_to_dense)(t) is not None
        config.eager_mode = True
        global_tracing_state.is_tracing = False
    getattr(_state, "update")({"DummyGraph": DummyGraph, "global_tracing_state": global_tracing_state, "mock_backend": mock_backend})
    _state.update(locals())


def _test_ragged_coverage_part_3(_state: dict[str, EagerTensorProtocol]) -> None:
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
    orig_get_op = cast("EagerTensorProtocol", getattr(ragged_frontend, "get_op"))
    getattr(_state, "update")({"DummyOp": DummyOp, "orig_eager": orig_eager, "orig_get_op": orig_get_op})
    _state.update(locals())


def _test_ragged_coverage_part_4(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 4 of test_ragged_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (DummyOp, orig_eager, orig_get_op, t) = (_state["DummyOp"], _state["orig_eager"], _state["orig_get_op"], _state["t"])
    try:
        config.eager_mode = False
        from ml_switcheroo_compiler.tracing.state import global_tracing_state

        global_tracing_state.is_tracing = True

        class DummyGraph:
            """Test class DummyGraph."""

            name = "dummy"
            nodes = cast("dict[str, EagerTensorProtocol]", {})

            def add_node(self, node: EagerValue) -> None:
                """Test function add_node.

                Args:
                    node: Test parameter `node`.
                """
                pass

        setattr(global_tracing_state, "active_graph", DummyGraph())

        class DummyOpMask:
            """Test class DummyOpMask."""

            def __call__(self, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
                """Test function __call__.

                Args:
                    args: Test parameter `args`.
                    kwargs: Test parameter `kwargs`.
                """
                return "masked"

            def infer_shape(self, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
                """Test function infer_shape.

                Args:
                    args: Test parameter `args`.
                    kwargs: Test parameter `kwargs`.
                """
                return ()

        setattr(ragged_frontend, "get_op", lambda x: DummyOpMask)
        import ml_switcheroo_compiler.ops.linalg.utils as linalg_utils

        orig_emit = linalg_utils._emit_linalg_node
        setattr(linalg_utils, "_emit_linalg_node", lambda *args, **kwargs: "masked")
        try:
            assert cast("EagerCallableProtocol", boolean_mask)(t, t) is not None
            assert cast("EagerCallableProtocol", map_flat_values)(cast("EagerCallableProtocol", DummyOp)(), t) is not None
        finally:
            linalg_utils._emit_linalg_node = orig_emit
    finally:
        setattr(config, "eager_mode", orig_eager)
        setattr(ragged_frontend, "get_op", orig_get_op)
        global_tracing_state.is_tracing = False


def test_ragged_coverage() -> None:
    """Test function test_ragged_coverage."""
    _state: dict[str, EagerTensorProtocol] = {}
    _test_ragged_coverage_part_1(_state)
    _test_ragged_coverage_part_2(_state)
    _test_ragged_coverage_part_3(_state)
    _test_ragged_coverage_part_4(_state)
    assert len(_state) > 0
