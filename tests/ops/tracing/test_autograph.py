"""Tests for test_autograph."""

from __future__ import annotations

import typing

from ml_switcheroo_compiler.tracing.autograph import LoopOptions, do_not_convert, set_loop_options

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import BackendArg


def test_autograph_coverage() -> None:
    """Test function test_autograph_coverage."""
    set_loop_options(parallel_iterations=1, swap_memory=True, maximum_iterations=10, shape_invariants=None)

    @do_not_convert
    def f() -> BackendArg:
        """Test function f.

        Returns:
            BackendArg: Computed output value from `f`.
        """
        return 1

    assert f() == 1
    assert f._autograph_do_not_convert

    def f2() -> BackendArg:
        """Test function f2.

        Returns:
            BackendArg: Computed output value from `f2`.
        """
        return 2

    f2_d = do_not_convert(f2)
    assert f2_d() == 2
    assert f2_d._autograph_do_not_convert

    lo = LoopOptions(parallel_iterations=1, swap_memory=True, maximum_iterations=10, shape_invariants=None)
    assert lo.parallel_iterations == 1
