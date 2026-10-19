"""Tests for test_autograph."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from ml_switcheroo_compiler.tracing.autograph import LoopOptions, do_not_convert, set_loop_options

if TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol, EagerTensorProtocol, EagerValue


def test_autograph_coverage() -> None:
    """Test function test_autograph_coverage."""
    set_loop_options(parallel_iterations=1, swap_memory=True, maximum_iterations=10, shape_invariants=None)

    @do_not_convert
    def f() -> EagerValue:
        """Test function f."""
        return 1

    assert cast("EagerCallableProtocol", f)() == 1
    assert cast("EagerTensorProtocol", getattr(f, "_autograph_do_not_convert"))

    def f2() -> EagerValue:
        """Test function f2."""
        return 2

    f2_d = do_not_convert(f2)
    assert cast("EagerCallableProtocol", f2_d)() == 2
    assert cast("EagerTensorProtocol", getattr(f2_d, "_autograph_do_not_convert"))

    lo = LoopOptions(parallel_iterations=1, swap_memory=True, maximum_iterations=10, shape_invariants=None)
    assert lo.parallel_iterations == 1
