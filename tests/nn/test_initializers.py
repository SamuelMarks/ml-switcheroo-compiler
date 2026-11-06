"""Tests for neural network parameter initializers."""

from __future__ import annotations

import typing

import pytest

import ml_switcheroo_compiler.nn.initializers as init
from tests.testing_utils import ensure_callable, ensure_eager_tensor

if typing.TYPE_CHECKING:
    pass


class DummyOps:
    """Mock ops module recording invoked operation names."""

    zeros = staticmethod(lambda *args, **kwargs: "zeros")
    ones = staticmethod(lambda *args, **kwargs: "ones")
    full = staticmethod(lambda *args, **kwargs: "full")
    scalar_mul = staticmethod(lambda *args, **kwargs: "scalar_mul")
    rand = staticmethod(lambda *args, **kwargs: "rand")
    randn = staticmethod(lambda *args, **kwargs: "randn")
    clip = staticmethod(lambda *args, **kwargs: "clip")
    subtract = staticmethod(lambda *args, **kwargs: "subtract")
    qr = staticmethod(lambda *args, **kwargs: ("qr_q", "qr_r"))


def test_initializers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that initializers invoke the appropriate random and linear algebra operations.

    Args:
        monkeypatch (pytest.MonkeyPatch): Pytest monkeypatch fixture.
    """
    monkeypatch.setattr(init, "ops", DummyOps())

    assert init._compute_fans(()) == (1, 1)
    assert init._compute_fans((4,)) == (4, 4)
    assert init._compute_fans((3, 3, 16, 32)) == (144, 288)

    assert ensure_eager_tensor(init.zeros(None, (1,))) == ensure_eager_tensor("zeros")
    assert ensure_eager_tensor(init.ones(None, (1,))) == ensure_eager_tensor("ones")
    assert ensure_callable(init.constant)(42)(None, (1,)) == "full"
    assert ensure_callable(init.uniform)(1.0)(None, (2, 3)) == "scalar_mul"
    assert ensure_callable(init.normal)(1.0, "float32")(None, (2, 3)) == "scalar_mul"
    assert ensure_callable(init.truncated_normal)(1.0, "float32", -2.0, 2.0)(None, (2, 3)) == "scalar_mul"

    assert ensure_callable(init.orthogonal)(1.0)(None, (1,)) == "scalar_mul"
    assert ensure_callable(init.orthogonal)(1.0)(None, (3, 3)) == "scalar_mul"
    assert ensure_callable(init.delta_orthogonal)(1.0)(None, (3, 3, 4)) == "scalar_mul"

    assert ensure_callable(init.variance_scaling)(ensure_callable(ensure_eager_tensor(ensure_eager_tensor(init).InitializerConfig))(scale=1.0, mode="fan_in", distribution="truncated_normal"))(None, (2, 4)) == "scalar_mul"
    assert ensure_callable(init.variance_scaling)(ensure_callable(ensure_eager_tensor(init).InitializerConfig)(scale=1.0, mode="fan_out", distribution="normal"))(None, (2, 4), "float32") == "scalar_mul"
    assert ensure_callable(init.variance_scaling)(ensure_callable(ensure_eager_tensor(ensure_eager_tensor(init).InitializerConfig))(scale=1.0, mode="fan_avg", distribution="uniform"))(None, (2, 4)) == "scalar_mul"

    assert ensure_callable(init.glorot_uniform)()(None, (2, 4)) == "scalar_mul"
    assert ensure_callable(init.glorot_normal)()(None, (2, 4)) == "scalar_mul"
    assert ensure_callable(init.he_uniform)()(None, (2, 4)) == "scalar_mul"
    assert ensure_callable(init.he_normal)()(None, (2, 4)) == "scalar_mul"
    assert ensure_callable(init.lecun_uniform)()(None, (2, 4)) == "scalar_mul"
    assert ensure_callable(init.lecun_normal)()(None, (2, 4)) == "scalar_mul"
