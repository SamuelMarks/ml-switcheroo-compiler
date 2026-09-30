"""Tests for test_sequence_coverage."""

from __future__ import annotations

import importlib
from unittest.mock import MagicMock, patch

import numpy as np

import ml_switcheroo_compiler.ops.creation.frontend_sequence as seq_mod
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.device import Device

gru_mod = importlib.import_module("ml_switcheroo_compiler.ops.nn.gru")


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


def test_frontend_sequence_coverage() -> None:
    """Verify 100% line and branch coverage for frontend_sequence operations."""
    orig_eager = config.eager_mode
    try:
        config.eager_mode = True
        mock_backend = MagicMock()
        mock_backend.execute_op.return_value = np.array([0.0, 1.0, 2.0], dtype=np.float32)

        with patch("ml_switcheroo_compiler.ops.creation.frontend_sequence.get_active_backend", return_value=mock_backend):
            t1 = seq_mod.arange(stop=None, start=3.0)
            assert t1.shape == (3,)
            assert t1.dtype == config.default_float_dtype
            mock_backend.execute_op.assert_called_with("Arange", 0, 3.0, 1, dtype=config.default_float_dtype.value)

            # Eager mode with custom dtype string without .value
            class DummyDtypeWithoutValue:
                """Mock dtype without value attribute."""

                name: str = "float32"

            t2 = seq_mod.arange(start=1.0, stop=5.0, step=2.0, dtype=DummyDtypeWithoutValue(), device=Device("cpu"))
            assert t2.shape == (2,)

        # Non-eager mode
        config.eager_mode = False
        with patch("ml_switcheroo_compiler.ops.creation.frontend_sequence._emit_creation_node", return_value="emitted_arange") as mock_emit:
            res_graph = seq_mod.arange(0.0, 10.0, 2.0)
            assert res_graph == "emitted_arange"
            mock_emit.assert_called_once_with(
                "Arange",
                (5,),
                config.default_float_dtype,
                {"start": 0.0, "stop": 10.0, "step": 2.0},
            )

        # linspace eager mode
        config.eager_mode = True
        mock_backend.execute_op.return_value = np.array([0.0, 0.5, 1.0], dtype=np.float32)
        with patch("ml_switcheroo_compiler.ops.creation.frontend_sequence.get_active_backend", return_value=mock_backend):
            t3 = seq_mod.linspace(0.0, 1.0, 3)
            assert t3.shape == (3,)

            # with custom dtype without .value
            t4 = seq_mod.linspace(0.0, 1.0, 3, dtype=DummyDtypeWithoutValue(), device=Device("cpu"))
            assert t4.shape == (3,)

        # linspace non-eager mode
        config.eager_mode = False
        with patch("ml_switcheroo_compiler.ops.creation.frontend_sequence._emit_creation_node", return_value="emitted_linspace") as mock_emit:
            res_graph_ls = seq_mod.linspace(0.0, 2.0, 5)
            assert res_graph_ls == "emitted_linspace"
            mock_emit.assert_called_once_with(
                "LinSpace",
                (5,),
                config.default_float_dtype,
                {"start": 0.0, "stop": 2.0, "steps": 5},
            )
    finally:
        config.eager_mode = orig_eager
