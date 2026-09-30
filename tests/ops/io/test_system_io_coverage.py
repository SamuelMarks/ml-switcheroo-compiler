"""Tests for test_system_io_coverage."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import ml_switcheroo_compiler.ops.io.system_io as sys_io_mod
from ml_switcheroo_compiler.core.config import config


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


def test_system_io_coverage() -> None:
    """Verify 100% line and branch coverage for system_io operations."""
    orig_backend = config.backend
    try:
        # Non-mlx backend: should be no-op
        config.backend = "numpy"
        sys_io_mod.set_default_stream("stream1")
        sys_io_mod.set_memory_limit(1024)
        sys_io_mod.set_wired_limit(2048)

        # mlx backend with methods present
        config.backend = "mlx"
        mock_mlx = MagicMock()
        with patch("ml_switcheroo_compiler.backends.registry.get_active_backend", return_value=mock_mlx):
            sys_io_mod.set_default_stream("stream1")
            mock_mlx.set_default_stream.assert_called_once_with("stream1")

            sys_io_mod.set_memory_limit(1024)
            mock_mlx.set_memory_limit.assert_called_once_with(1024)

            sys_io_mod.set_wired_limit(2048)
            mock_mlx.set_wired_limit.assert_called_once_with(2048)

        # mlx backend without methods present
        class BackendWithoutMethods:
            """Mock backend lacking system stream and memory limits."""

        with patch("ml_switcheroo_compiler.backends.registry.get_active_backend", return_value=BackendWithoutMethods()):
            sys_io_mod.set_default_stream("stream1")
            sys_io_mod.set_memory_limit(1024)
            sys_io_mod.set_wired_limit(2048)

        # mlx backend triggering ImportError
        with patch("ml_switcheroo_compiler.backends.registry.get_active_backend", side_effect=ImportError("no mlx")):
            sys_io_mod.set_default_stream("stream1")
            sys_io_mod.set_memory_limit(1024)
            sys_io_mod.set_wired_limit(2048)
    finally:
        config.backend = orig_backend
