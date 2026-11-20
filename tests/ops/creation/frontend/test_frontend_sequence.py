"""Behavioral unit tests for frontend sequence."""

from __future__ import annotations

import importlib
import typing
from unittest.mock import MagicMock, patch

import numpy as np

import ml_switcheroo_compiler.ops.creation.api_sequence as seq_mod
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.ops.creation.api_sequence import arange, linspace
from tests.testing_utils import extract_test_dtype, extract_test_shape, require_callable

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import BackendArg


def test_frontend_sequence() -> None:
    """Test function test_frontend_sequence."""
    from ml_switcheroo_compiler.core.config import config

    config.eager_mode = True

    class DummyBackend:
        """Test class DummyBackend."""

        def execute_op(self, op: BackendArg, *a: BackendArg, **k: BackendArg) -> BackendArg:
            """Test function execute_op.

            Args:
                op (BackendArg): Test parameter `op`.
                *a (BackendArg): Test parameter `a`.
                **k (BackendArg): Test parameter `k`.

            Returns:
                BackendArg: Computed output value from `execute_op`.
            """

            class T:
                """Test class T."""

                shape = (1,)

            return T()

    import ml_switcheroo_compiler.ops.creation.api_sequence

    with patch.object(
        ml_switcheroo_compiler.ops.creation.api_sequence,
        "get_active_backend",
        return_value=DummyBackend(),
    ):
        assert extract_test_shape(arange(10)) == (10,)
        assert extract_test_shape(arange(0, 10)) == (10,)
        assert extract_test_shape(linspace(1, 10, 50)) == (50,)

        config.eager_mode = False

        with patch.object(
            ml_switcheroo_compiler.ops.creation.api_sequence, "_emit_creation_node", return_value="emitted"
        ):
            assert arange(10) == "emitted"
            assert arange(0, 10) == "emitted"
            assert linspace(1, 10, 50) == "emitted"

    config.eager_mode = True


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

        with patch("ml_switcheroo_compiler.ops.creation.api_sequence.get_active_backend", return_value=mock_backend):
            t1 = seq_mod.arange(stop=None, start=3.0)
            assert extract_test_shape(t1) == (3,)
            assert extract_test_dtype(t1) == config.default_float_dtype
            mock_backend.execute_op.assert_called_with("Arange", 0, 3.0, 1, dtype=config.default_float_dtype.value)

            # Eager mode with custom dtype string without .value
            class DummyDtypeWithoutValue:
                """Mock dtype without value attribute."""

                shape: tuple[int, ...] = ()
                name: str = "float32"

            t2 = require_callable(seq_mod.arange)(
                start=1.0,
                stop=5.0,
                step=2.0,
                dtype=DummyDtypeWithoutValue(),
                device=Device("cpu"),
            )
            assert extract_test_shape(t2) == (2,)

        # Non-eager mode
        config.eager_mode = False
        with patch(
            "ml_switcheroo_compiler.ops.creation.api_sequence._emit_creation_node", return_value="emitted_arange"
        ) as mock_emit:
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
        with patch("ml_switcheroo_compiler.ops.creation.api_sequence.get_active_backend", return_value=mock_backend):
            t3 = seq_mod.linspace(0.0, 1.0, 3)
            assert extract_test_shape(t3) == (3,)

            # with custom dtype without .value
            t4 = require_callable(seq_mod.linspace)(0.0, 1.0, 3, dtype=DummyDtypeWithoutValue(), device=Device("cpu"))
            assert extract_test_shape(t4) == (3,)

        # linspace non-eager mode
        config.eager_mode = False
        with patch(
            "ml_switcheroo_compiler.ops.creation.api_sequence._emit_creation_node", return_value="emitted_linspace"
        ) as mock_emit:
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
