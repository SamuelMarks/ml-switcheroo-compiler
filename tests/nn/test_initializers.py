"""Tests for neural network parameter initializers."""

from __future__ import annotations

import typing

import pytest

import ml_switcheroo_compiler.nn.initializers as init

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
    reshape = staticmethod(lambda *args, **kwargs: "reshaped_q")


def test_initializers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that initializers invoke the appropriate random and linear algebra operations.

    Args:
        monkeypatch (pytest.MonkeyPatch): Pytest monkeypatch fixture.
    """
    monkeypatch.setattr(init, "ops", DummyOps())

    assert init._compute_fans(()) == (1, 1)
    assert init._compute_fans((4,)) == (4, 4)
    assert init._compute_fans((3, 3, 16, 32)) == (144, 288)

    assert init.zeros(None, (1,)) == "zeros"
    assert init.ones(None, (1,)) == "ones"
    assert init.constant(42)(None, (1,)) == "full"
    assert init.uniform(1.0)(None, (2, 3)) == "scalar_mul"
    assert init.normal(1.0, "float32")(None, (2, 3)) == "scalar_mul"
    assert init.truncated_normal(1.0, "float32", -2.0, 2.0)(None, (2, 3)) == "scalar_mul"

    assert init.orthogonal(1.0)(None, (1,)) == "scalar_mul"
    assert init.orthogonal(1.0)(None, (3, 3)) == "scalar_mul"
    assert init.delta_orthogonal(1.0)(None, (3, 3, 4)) == "scalar_mul"

    assert (
        init.variance_scaling(init.InitializerConfig(scale=1.0, mode="fan_in", distribution="truncated_normal"))(
            None, (2, 4)
        )
        == "scalar_mul"
    )
    assert (
        init.variance_scaling(init.InitializerConfig(scale=1.0, mode="fan_out", distribution="normal"))(
            None, (2, 4), "float32"
        )
        == "scalar_mul"
    )
    assert (
        init.variance_scaling(init.InitializerConfig(scale=1.0, mode="fan_avg", distribution="uniform"))(None, (2, 4))
        == "scalar_mul"
    )

    assert init.glorot_uniform()(None, (2, 4)) == "scalar_mul"
    assert init.glorot_normal()(None, (2, 4)) == "scalar_mul"
    assert init.he_uniform()(None, (2, 4)) == "scalar_mul"
    assert init.he_normal()(None, (2, 4)) == "scalar_mul"
    assert init.lecun_uniform()(None, (2, 4)) == "scalar_mul"
    assert init.lecun_normal()(None, (2, 4)) == "scalar_mul"


def test_orthogonal_eager_orthogonality(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify eager orthogonal initializer produces Q^T Q == scale^2 I and exercises fallback callables.

    Args:
        monkeypatch (pytest.MonkeyPatch): Pytest monkeypatch fixture.
    """
    import numpy as np

    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.core.tensor import Tensor

    monkeypatch.setattr(config, "eager_mode", True)
    q_tensor = init.orthogonal(scale=2.0)(0, (4, 4))
    assert isinstance(q_tensor, Tensor)
    q_np = np.asarray(q_tensor.numpy(), dtype=np.float64)
    np.testing.assert_allclose(q_np.T @ q_np, 4.0 * np.eye(4), atol=1e-4)

    q_rect = init.orthogonal(scale=1.5)(0, (4, 2))
    assert q_rect.shape == (4, 2)
    q_rect_np = np.asarray(q_rect.numpy(), dtype=np.float64)
    np.testing.assert_allclose(q_rect_np.T @ q_rect_np, 2.25 * np.eye(2), atol=1e-4)

    class DummyOpsWithoutQr:
        """Ops stub omitting qr and reshape to exercise direct op fallbacks."""

        scalar_mul = staticmethod(lambda *args, **kwargs: "scalar_mul_fallback")
        randn = staticmethod(lambda *args, **kwargs: "randn")

    monkeypatch.setattr(init, "ops", DummyOpsWithoutQr())
    monkeypatch.setattr(init, "_qr_op", lambda x: "q_single")
    monkeypatch.setattr(init, "_reshape_op", lambda x, s: "reshaped_fb")
    assert init.orthogonal(1.0)(None, (3, 3)) == "scalar_mul_fallback"
