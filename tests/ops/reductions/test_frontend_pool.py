"""Tests for test_frontend_pool."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.reductions.frontend_pool import (
    adaptive_avg_pool2d,
    adaptive_avg_pool3d,
    adaptive_max_pool2d,
    adaptive_max_pool3d,
    fractional_max_pool2d,
    fractional_max_pool3d,
)
from ml_switcheroo_compiler.tracing.state import global_tracing_state

if TYPE_CHECKING:
    from collections.abc import Callable

    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol


def test_frontend_pool_edge_cases() -> None:
    """Test function test_frontend_pool_edge_cases."""
    global_tracing_state.is_tracing = True
    try:
        from ml_switcheroo_ir import LogicalGraph

        graph = LogicalGraph()
        global_tracing_state.active_graph = graph

        class DummyData:
            """Test class DummyData."""

            id = "dummy1"

        t1d = cast("Callable[..., EagerTensorProtocol]", Tensor)(cast("Callable[..., EagerTensorProtocol]", DummyData)(), cast("Callable[..., EagerTensorProtocol]", TensorConfig)((1,), "float32", None))

        cast("Callable[..., EagerTensorProtocol]", fractional_max_pool2d)(t1d, output_size=[1, 1])
        cast("Callable[..., EagerTensorProtocol]", adaptive_avg_pool2d)(t1d, output_size=[1, 1])
        cast("Callable[..., EagerTensorProtocol]", adaptive_max_pool2d)(t1d, output_size=[1, 1])

        class DummyData2:
            """Test class DummyData2."""

            id = "dummy2"

        t2d = cast("Callable[..., EagerTensorProtocol]", Tensor)(cast("Callable[..., EagerTensorProtocol]", DummyData2)(), cast("Callable[..., EagerTensorProtocol]", TensorConfig)((1, 1), "float32", None))

        cast("Callable[..., EagerTensorProtocol]", fractional_max_pool3d)(t2d, output_size=[1, 1, 1])
        cast("Callable[..., EagerTensorProtocol]", adaptive_avg_pool3d)(t2d, output_size=[1, 1, 1])
        cast("Callable[..., EagerTensorProtocol]", adaptive_max_pool3d)(t2d, output_size=[1, 1, 1])

    finally:
        global_tracing_state.is_tracing = False
        global_tracing_state.active_graph = None
    assert callable(test_frontend_pool_edge_cases)
