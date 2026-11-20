"""Tests for test_reductions."""

from __future__ import annotations

import typing

# ruff: noqa: E501
from unittest.mock import patch

import numpy as np
import pytest
from ml_switcheroo_ir import LogicalNode

import ml_switcheroo_compiler.ops.reductions.api as rd
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Device, DType, Tensor, TensorConfig

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import BackendArg


def test_reductions_coverage() -> None:
    """Test function test_reductions_coverage."""
    config.eager_mode = True
    t = Tensor(np.ones((2, 2)), TensorConfig(shape=(2, 2), dtype=DType("float32"), device=Device("cpu")))

    with patch("ml_switcheroo_compiler.backends.registry.get_active_backend") as mock_backend:
        mock_backend.return_value.execute_op.return_value = np.zeros((2, 2))
        assert tuple(rd.sum(t).shape) == ()
        assert tuple(rd.max(t).shape) == ()
        assert tuple(rd.min(t).shape) == ()

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
                return "reduction"

            def infer_shape(self, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
                """Test function infer_shape.

                Args:
                    args: Test parameter `args`.
                    kwargs: Test parameter `kwargs`.

                Returns:
                    BackendArg: Computed output value from `infer_shape`.
                """
                return ()

        pytest.MonkeyPatch().setattr(shape_utils, "_emit_shape_node", lambda *args, **kwargs: DummyOp(), raising=False)
        try:
            assert type(rd.sum(t)).__name__ == "Tensor"
            assert type(rd.max(t)).__name__ == "Tensor"
            assert type(rd.min(t)).__name__ == "Tensor"
        finally:
            shape_utils._emit_shape_node = orig_emit
            config.eager_mode = True
