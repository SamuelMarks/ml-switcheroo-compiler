"""Tests for test_frontend_pool."""

from __future__ import annotations

import typing

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
from tests.testing_utils import as_callable

if typing.TYPE_CHECKING:
    pass


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

        t1d = as_callable(Tensor)(as_callable(DummyData)(), as_callable(TensorConfig)((1,), "float32", None))

        as_callable(fractional_max_pool2d)(t1d, output_size=[1, 1])
        as_callable(adaptive_avg_pool2d)(t1d, output_size=[1, 1])
        as_callable(adaptive_max_pool2d)(t1d, output_size=[1, 1])

        class DummyData2:
            """Test class DummyData2."""

            id = "dummy2"

        t2d = as_callable(Tensor)(as_callable(DummyData2)(), as_callable(TensorConfig)((1, 1), "float32", None))

        as_callable(fractional_max_pool3d)(t2d, output_size=[1, 1, 1])
        as_callable(adaptive_avg_pool3d)(t2d, output_size=[1, 1, 1])
        as_callable(adaptive_max_pool3d)(t2d, output_size=[1, 1, 1])

    finally:
        global_tracing_state.is_tracing = False
        global_tracing_state.active_graph = None
    assert callable(test_frontend_pool_edge_cases)
