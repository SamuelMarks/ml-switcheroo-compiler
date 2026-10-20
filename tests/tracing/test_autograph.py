"""Tests for AutoGraph primitives."""

from __future__ import annotations

import typing

from ml_switcheroo_compiler.tracing.autograph import LoopOptions, do_not_convert, set_loop_options

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol, EagerTensorProtocol


def test_loop_options() -> None:
    """Test LoopOptions."""
    opts = LoopOptions(parallel_iterations=10, swap_memory=True, maximum_iterations=100, shape_invariants=[])
    assert opts.parallel_iterations == 10
    assert opts.swap_memory is True
    assert opts.maximum_iterations == 100
    assert opts.shape_invariants == []

    set_loop_options(parallel_iterations=5)


def test_do_not_convert() -> None:
    """Test do_not_convert decorator."""

    @do_not_convert
    def dummy_func(x: int) -> int:
        """Test function dummy_func.

        Args:
            x: Test parameter `x`.
        """
        return x + 1

    assert typing.cast("EagerCallableProtocol", dummy_func)(5) == 6
    assert bool(typing.cast("EagerTensorProtocol", getattr(dummy_func, "__wrapped__"))._autograph_do_not_convert) is True

    @typing.cast("EagerCallableProtocol", do_not_convert)()
    def dummy_func2(x: int) -> int:
        """Test function dummy_func2.

        Args:
            x: Test parameter `x`.
        """
        return x + 2

    assert dummy_func2(5) == 7
    assert bool(dummy_func2.__wrapped__._autograph_do_not_convert) is True
