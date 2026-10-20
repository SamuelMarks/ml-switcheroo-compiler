"""Tests for test_frontend_pool_tracing."""

from __future__ import annotations

import typing

import ml_switcheroo_compiler.tracing.state as state
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.ops.reductions.frontend_pool import UnpoolOptions, adaptive_avg_pool2d, adaptive_avg_pool3d, adaptive_max_pool2d, adaptive_max_pool3d, fold, fractional_max_pool2d, fractional_max_pool3d, max_unpool1d, max_unpool2d, max_unpool3d, unfold

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol, EagerTensorProtocol, EagerValue


def test_frontend_pool_tracing() -> None:
    """Test function test_frontend_pool_tracing."""
    orig_eager = config.eager_mode
    orig_tracing = state.global_tracing_state.is_tracing
    orig_add_node = state.global_tracing_state.add_node

    try:
        config.eager_mode = False
        state.global_tracing_state.is_tracing = True
        setattr(state.global_tracing_state, "add_node", lambda node: None)

        class MockData:
            """Test class MockData."""

            id = "t1"

        class MockTensor:
            """Test class MockTensor."""

            def __init__(self, shape: EagerValue) -> None:
                """Test function __init__.

                Args:
                    shape: Test parameter `shape`.
                """
                self.data = MockData()
                self.shape = shape
                self.dtype = DType("float32")
                self.device = Device("cpu")

            def eval(self) -> EagerValue:
                """Test function eval."""
                return typing.cast("EagerTensorProtocol", self)

        mt3 = MockTensor((1, 1, 1))
        mt4 = MockTensor((1, 1, 1, 1))
        mt5 = MockTensor((1, 1, 1, 1, 1))

        typing.cast("EagerCallableProtocol", fractional_max_pool2d)(mt4, output_size=[1, 1])
        typing.cast("EagerCallableProtocol", adaptive_avg_pool2d)(mt4, output_size=[1, 1])
        typing.cast("EagerCallableProtocol", adaptive_max_pool2d)(mt4, output_size=[1, 1])
        typing.cast("EagerCallableProtocol", unfold)(mt4, kernel_size=(1, 1))
        typing.cast("EagerCallableProtocol", fold)(mt3, output_size=(1, 1), kernel_size=(1, 1))
        typing.cast("EagerCallableProtocol", fractional_max_pool3d)(mt5, output_size=[1, 1, 1])
        typing.cast("EagerCallableProtocol", adaptive_avg_pool3d)(mt5, output_size=[1, 1, 1])
        typing.cast("EagerCallableProtocol", adaptive_max_pool3d)(mt5, output_size=[1, 1, 1], return_indices=True)
        typing.cast("EagerCallableProtocol", adaptive_max_pool3d)(mt5, output_size=[1, 1, 1], return_indices=False)
        typing.cast("EagerCallableProtocol", max_unpool1d)(mt3, mt3, typing.cast("EagerCallableProtocol", UnpoolOptions)(kernel_size=(1,), output_size=(1,)))
        typing.cast("EagerCallableProtocol", max_unpool2d)(mt4, mt4, typing.cast("EagerCallableProtocol", UnpoolOptions)(kernel_size=(1, 1), output_size=(1, 1)))
        typing.cast("EagerCallableProtocol", max_unpool3d)(mt5, mt5, typing.cast("EagerCallableProtocol", UnpoolOptions)(kernel_size=(1, 1, 1), output_size=(1, 1, 1)))

    finally:
        config.eager_mode = orig_eager
        state.global_tracing_state.is_tracing = orig_tracing
        setattr(state.global_tracing_state, "add_node", orig_add_node)
    assert getattr(state.global_tracing_state, "add_node") == orig_add_node
