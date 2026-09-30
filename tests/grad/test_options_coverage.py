"""Tests for test_options_coverage."""

from __future__ import annotations

import ml_switcheroo_compiler.grad.options as grad_options


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


def test_grad_options() -> None:
    """Verify GradOptions, GradCheckOptions, and JitOptions dataclasses in grad/options.py."""
    g_opts = grad_options.GradOptions()
    assert g_opts.argnums == 0
    assert g_opts.has_aux is False
    assert g_opts.holistic is False
    assert g_opts.reduce_axes == ()
    assert g_opts.return_value is False

    gc_opts = grad_options.GradCheckOptions(order=2, atol=1e-4, rtol=1e-4, step=1e-5)
    assert gc_opts.order == 2
    assert gc_opts.atol == 1e-4
    assert gc_opts.rtol == 1e-4
    assert gc_opts.step == 1e-5

    j_opts = grad_options.JitOptions(static_argnums=(0,), device="cpu", backend="numpy", inline=True)
    assert j_opts.static_argnums == (0,)
    assert j_opts.device == "cpu"
    assert j_opts.backend == "numpy"
    assert j_opts.inline is True
