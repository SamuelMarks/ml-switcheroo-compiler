"""Tests for test_linalg_utils_coverage."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

import ml_switcheroo_compiler.ops.linalg.utils as linalg_utils_mod
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.tracing import global_tracing_state


class DummyWithShape:
    """Mock operand providing shape attribute."""

    def __init__(self, shape: tuple[int, ...]) -> None:
        """Initialize mock with specific shape.

        Args:
            shape (tuple[int, ...]): Target shape.
        """
        self.shape: tuple[int, ...] = shape


class DummyWithoutShape:
    """Mock operand lacking shape attribute."""

    def __init__(self) -> None:
        """Initialize mock without shape."""
        self.val: int = 123


def test_linalg_utils_full_coverage() -> None:
    """Test full coverage for linear algebra tracing node emitters and output tensor construction."""
    with pytest.raises(RuntimeError, match="outside of a tracing context"):
        with patch.object(global_tracing_state, "is_tracing", False):
            linalg_utils_mod._emit_linalg_node("TestOp", [], {}, [[]], ["float32"])

    class MockTracingBuilder:
        """Mock tracing node builder for proxy extraction."""

        @staticmethod
        def extract_proxy_inputs(inputs: tuple[Tensor, ...]) -> tuple[list[str], list[str], dict[str, str]]:
            """Extract proxy identifiers from inputs.

            Args:
                inputs (tuple[Tensor, ...]): Input tensor sequence.

            Returns:
                tuple[list[str], list[str], dict[str, str]]: IDs, names, kwargs.
            """
            return ["input_id_0"], [], {}

    t_in = Tensor(np.ones((2, 2), dtype=np.float32), TensorConfig((2, 2), "float32", "cpu"))

    with (
        patch.object(global_tracing_state, "is_tracing", True),
        patch.object(global_tracing_state, "add_node") as mock_add,
        patch("ml_switcheroo_compiler.ops.linalg.utils.TracingNodeBuilder", MockTracingBuilder),
    ):
        res_single = linalg_utils_mod._emit_linalg_node(
            "MatrixSolve",
            [t_in],
            {"attr": 1},
            out_shapes=[[2, 2]],
            out_dtypes=[DType("float32")],
        )
        assert isinstance(res_single, Tensor)
        assert res_single.shape == (2, 2)
        mock_add.assert_called_once()

        res_multi = linalg_utils_mod._emit_linalg_node(
            "QR",
            [],
            {},
            out_shapes=[[2, 2], [2, 2]],
            out_dtypes=["float32", "float32"],
        )
        assert isinstance(res_multi, tuple)
        assert len(res_multi) == 2
