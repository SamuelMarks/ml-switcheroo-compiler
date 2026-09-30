"""Tests for test_jit_coverage."""

from __future__ import annotations

import importlib

import ml_switcheroo_compiler.grad as grad_pkg


class _MockBackendModule:
    """Mock backend module implementing zeros, array, and asarray."""

    def zeros(self, shape: tuple[int, ...]) -> tuple[str, tuple[int, ...]]:
        """Mock zeros.

        Args:
            shape (tuple[int, ...]): Output shape.

        Returns:
            tuple[str, tuple[int, ...]]: Mock zeros descriptor.
        """
        return ("zeros", shape)

    def array(self, data: object, dtype: object = None) -> tuple[str, object, object]:
        """Mock array.

        Args:
            data (object): Input data.
            dtype (object): Optional data type.

        Returns:
            tuple[str, object, object]: Mock array descriptor.
        """
        return ("array", data, dtype)

    def asarray(self, data: object) -> tuple[str, object]:
        """Mock asarray.

        Args:
            data (object): Input data.

        Returns:
            tuple[str, object]: Mock asarray descriptor.
        """
        return ("asarray", data)


class _MockItemObject:
    """Mock object implementing item()."""

    def item(self) -> float:
        """Return scalar item value.

        Returns:
            float: Scalar float value.
        """
        return 42.5


class _MockDaskDtypeWrapper:
    """Mock dtype object with value attribute."""

    def __init__(self, val: str) -> None:
        """Initialize mock dtype wrapper.

        Args:
            val (str): Type name.
        """
        self.value = val


def _sample_add_fn(x: int, y: int = 10) -> int:
    """Add two integers.

    Args:
        x (int): First int.
        y (int): Second int.

    Returns:
        int: Sum of x and y.
    """
    return x + y


def _sample_shape_fn(a: int, b: int) -> tuple[int, int]:
    """Sample shape function returning tuple.

    Args:
        a (int): First dimension.
        b (int): Second dimension.

    Returns:
        tuple[int, int]: Pair (a, b).
    """
    return (a, b)


def test_grad_init_and_jit() -> None:
    """Verify grad package exports, JitOptions, jit wrapper, disable_jit context, and eval_shape."""
    jit_mod = importlib.import_module("ml_switcheroo_compiler.grad.jit")

    # grad __init__ re-exports
    assert hasattr(grad_pkg, "grad")
    assert hasattr(grad_pkg, "value_and_grad")
    assert hasattr(grad_pkg, "jit")
    assert hasattr(grad_pkg, "disable_jit")
    assert hasattr(grad_pkg, "eval_shape")
    assert "grad" in grad_pkg.__all__
    assert "jit" in grad_pkg.__all__

    # JitOptions defaults
    opts = jit_mod.JitOptions()
    assert opts.static_argnums is None
    assert opts.static_argnames is None
    assert opts.donate_argnums is None
    assert opts.donate_argnames is None
    assert opts.keep_unused is False
    assert opts.device is None
    assert opts.backend is None
    assert opts.inline is False
    assert opts.abstracted_axes is None

    jitted1 = jit_mod.jit(_sample_add_fn)
    assert jitted1(5, y=15) == 20

    jitted2 = jit_mod.jit(_sample_add_fn, options=opts)
    assert jitted2(3) == 13

    # disable_jit context manager
    with jit_mod.disable_jit():
        pass

    assert jit_mod.eval_shape(_sample_shape_fn, 3, 4) == (3, 4)
