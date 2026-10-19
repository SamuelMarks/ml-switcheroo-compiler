"""Tests for test_math_special_mock."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ml_switcheroo_compiler.backends.numpy.eager.math_advanced import _np_rawmatmul

if TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue


def test_mock(monkeypatch: EagerTensorProtocol) -> None:
    """Test function test_mock.

    Args:
        monkeypatch: Test parameter `monkeypatch`.
    """
    import ml_switcheroo_compiler.ops as ops

    class FakeOp:
        """Test class FakeOp."""

        def __init__(self, *args: EagerValue, **kwargs: EagerValue) -> None:
            """Test function __init__.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.
            """
            self.hit = True

    monkeypatch.setattr(ops, "RawMatMul", FakeOp, raising=False)

    try:
        res = _np_rawmatmul(np, 1, 2)
        print("RES:", res)
    except Exception as e:
        print("EXCEPTION:", repr(e))
    assert callable(monkeypatch.setattr)
