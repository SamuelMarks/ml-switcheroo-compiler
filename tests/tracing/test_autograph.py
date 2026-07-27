"""Tests for AutoGraph primitives."""

from ml_switcheroo_compiler.tracing.autograph import LoopOptions, do_not_convert, set_loop_options


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
        return x + 1

    assert dummy_func(5) == 6
    assert dummy_func.__wrapped__._autograph_do_not_convert is True

    @do_not_convert()
    def dummy_func2(x: int) -> int:
        return x + 2

    assert dummy_func2(5) == 7
    assert dummy_func2.__wrapped__._autograph_do_not_convert is True
