"""Tests for test_unimplemented_ops."""

from __future__ import annotations

import typing

import ml_switcheroo_compiler.ops.linalg as mod

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol


def test_linalg_dummy_ops_infer_shape() -> None:
    """Test function test_linalg_dummy_ops_infer_shape."""

    class MockTensor:
        """Test class MockTensor."""

        def __init__(self, shape: tuple[int, ...]) -> None:
            """Test function __init__.

            Args:
                shape: Test parameter `shape`.
            """
            self.shape = shape

    t1 = MockTensor((2, 3))
    t2 = MockTensor((2, 3))
    mat = MockTensor((2, 3, 3))
    vec = MockTensor((2, 3))

    # Test infer_shape with args
    assert typing.cast("EagerCallableProtocol", mod.Vecdot)().infer_shape(t1, t2) == (2,)
    assert typing.cast("EagerCallableProtocol", mod.CustomLinearSolve)().infer_shape(mat, vec) == (2, 3)
    assert typing.cast("EagerCallableProtocol", mod.CustomRoot)().infer_shape(None, t1) == (2, 3)

    # Test infer_shape without args (the missing branches)
    assert mod.Vecdot().infer_shape() == ()
    assert mod.CustomLinearSolve().infer_shape() == ()
    assert mod.CustomRoot().infer_shape() == ()
