"""Tests for neural network parameter initializers."""

from __future__ import annotations

import typing

import pytest

import ml_switcheroo_compiler.nn.initializers as init
from tests.testing_utils import as_callable, as_tensor

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerValue


class DummyOps:
    """Mock ops module recording invoked operation names."""

    def __getattr__(self, name: str) -> EagerValue:
        """Return a mock callable for the requested operation name.

        Args:
            name (str): Operation name.

        Returns:
            Callable: Mock operation callable.
        """

        def _dummy(*args: EagerValue, **kwargs: EagerValue) -> EagerValue:
            """Execute mock operation.

            Args:
                *args (EagerValue): Positional arguments.
                **kwargs (EagerValue): Keyword arguments.

            Returns:
                EagerValue: Operation name or tuple for QR decomposition.
            """
            _ = (args, kwargs)
            if name == "qr":
                return ("qr_q", "qr_r")
            return name

        return _dummy


def test_initializers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that initializers invoke the appropriate random and linear algebra operations.

    Args:
        monkeypatch (pytest.MonkeyPatch): Pytest monkeypatch fixture.
    """
    monkeypatch.setattr(init, "ops", DummyOps())

    assert init._compute_fans(()) == (1, 1)
    assert init._compute_fans((4,)) == (4, 4)
    assert init._compute_fans((3, 3, 16, 32)) == (144, 288)

    assert as_tensor(init.zeros(None, (1,))) == as_tensor("zeros")
    assert as_tensor(init.ones(None, (1,))) == as_tensor("ones")
    assert as_callable(init.constant)(42)(None, (1,)) == "full"
    assert as_callable(init.uniform)(1.0)(None, (2, 3)) == "scalar_mul"
    assert as_callable(init.normal)(1.0, "float32")(None, (2, 3)) == "scalar_mul"
    assert as_callable(init.truncated_normal)(1.0, "float32", -2.0, 2.0)(None, (2, 3)) == "scalar_mul"

    assert as_callable(init.orthogonal)(1.0)(None, (1,)) == "scalar_mul"
    assert as_callable(init.orthogonal)(1.0)(None, (3, 3)) == "scalar_mul"
    assert as_callable(init.delta_orthogonal)(1.0)(None, (3, 3, 4)) == "scalar_mul"

    assert as_callable(init.variance_scaling)(as_callable(as_tensor(getattr(init, "InitializerConfig")))(scale=1.0, mode="fan_in", distribution="truncated_normal"))(None, (2, 4)) == "scalar_mul"
    assert as_callable(init.variance_scaling)(as_callable(getattr(init, "InitializerConfig"))(scale=1.0, mode="fan_out", distribution="normal"))(None, (2, 4), "float32") == "scalar_mul"
    assert as_callable(init.variance_scaling)(as_callable(as_tensor(getattr(init, "InitializerConfig")))(scale=1.0, mode="fan_avg", distribution="uniform"))(None, (2, 4)) == "scalar_mul"

    assert as_callable(init.glorot_uniform)()(None, (2, 4)) == "scalar_mul"
    assert as_callable(init.glorot_normal)()(None, (2, 4)) == "scalar_mul"
    assert as_callable(init.he_uniform)()(None, (2, 4)) == "scalar_mul"
    assert as_callable(init.he_normal)()(None, (2, 4)) == "scalar_mul"
    assert as_callable(init.lecun_uniform)()(None, (2, 4)) == "scalar_mul"
    assert as_callable(init.lecun_normal)()(None, (2, 4)) == "scalar_mul"
