"""Tests for test_jax_ragged_eager."""

from __future__ import annotations

import typing

import numpy as np
import pytest

from tests.testing_utils import ensure_eager_tensor

if typing.TYPE_CHECKING:
    pass


def test_jax_ragged_eager(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test function test_jax_ragged_eager.

    Args:
        monkeypatch (pytest.MonkeyPatch): Pytest monkeypatch fixture for temporary attribute and environment overrides.
    """
    import types

    import ml_switcheroo_compiler.backends.jax.eager as jax_eager
    from ml_switcheroo_compiler.backends.jax.eager import _execute_ragged_tensor_to_dense

    jnp: types.ModuleType
    try:
        import jax.numpy as _jnp

        jnp = _jnp
    except ImportError:
        jnp = np
        monkeypatch.setattr(jax_eager, "jnp", np)

    a = [jnp.array([1, 2]), jnp.array([1])]
    res = _execute_ragged_tensor_to_dense(a)
    assert ensure_eager_tensor(res).shape == (2, 2)
    assert _execute_ragged_tensor_to_dense(np.array([1, 2])) is not None
