"""Tests for test_control_flow_mixin_coverage."""

from __future__ import annotations

from unittest.mock import patch

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.ops.control_flow.cond import cond
from ml_switcheroo_compiler.ops.control_flow.scan import scan
from ml_switcheroo_compiler.ops.control_flow.while_loop import while_loop


class _MockDimWithId:
    """Mock dimension exposing an id attribute."""

    def __init__(self, dim_id: str) -> None:
        """Initialize mock dimension.

        Args:
            dim_id (str): Identifier for dimension.
        """
        self.id = dim_id


class _MockShapeContainer:
    """Mock container exposing a shape attribute."""

    def __init__(self, shape: tuple[int, ...]) -> None:
        """Initialize mock shape container.

        Args:
            shape (tuple[int, ...]): Shape tuple.
        """
        self.shape = shape


class _MockShapeMetaContainer:
    """Mock container exposing a shape_metadata attribute."""

    def __init__(self, shape_metadata: tuple[int, ...]) -> None:
        """Initialize mock shape metadata container.

        Args:
            shape_metadata (tuple[int, ...]): Shape metadata tuple.
        """
        self.shape_metadata = shape_metadata


class _MockGenerator:
    """Mock generator for variable AST visitor."""

    def get_fallback_prefix(self) -> str:
        """Get fallback prefix for AST emissions.

        Returns:
            str: Fallback prefix string.
        """
        return "mock_backend"


def test_control_flow_cond_scan_while_exhaustive() -> None:
    """Verify eager and tracing execution branches for cond, scan, and while_loop."""
    # 1. cond
    with patch("ml_switcheroo_compiler.ops.control_flow.cond_eager", return_value="cond_eager_res") as mock_ce:
        with patch("ml_switcheroo_compiler.ops.control_flow.cond_tracing", return_value="cond_tracing_res") as mock_ct:
            config.eager_mode = True
            assert cond(None, None, None) == "cond_eager_res"
            mock_ce.assert_called_once()

            config.eager_mode = False
            assert cond(None, None, None) == "cond_tracing_res"
            mock_ct.assert_called_once()

    # 2. scan
    with patch("ml_switcheroo_compiler.ops.control_flow.scan_eager", return_value="scan_eager_res") as mock_se:
        with patch("ml_switcheroo_compiler.ops.control_flow.scan_tracing", return_value="scan_tracing_res") as mock_st:
            config.eager_mode = True
            assert scan(None, None, None) == "scan_eager_res"
            mock_se.assert_called_once()

            config.eager_mode = False
            assert scan(None, None, None) == "scan_tracing_res"
            mock_st.assert_called_once()

    # 3. while_loop
    with patch("ml_switcheroo_compiler.ops.control_flow.while_loop_eager", return_value="while_eager_res") as mock_we:
        with patch("ml_switcheroo_compiler.ops.control_flow.while_loop_tracing", return_value="while_tracing_res") as mock_wt:
            config.eager_mode = True
            assert while_loop(None, None, None) == "while_eager_res"
            mock_we.assert_called_once()

            config.eager_mode = False
            assert while_loop(None, None, None) == "while_tracing_res"
            mock_wt.assert_called_once()
