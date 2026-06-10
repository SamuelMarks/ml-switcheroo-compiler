"""Tests for Random operations."""

import pytest
from ml_switcheroo.core import ConfigContext
from ml_switcheroo.random import (
    seed,
    PRNGKey,
    split,
    fold_in,
    uniform,
    normal,
    bernoulli,
    truncated_normal,
    randint,
)


def test_random_eager() -> None:
    """Docstring."""
    with ConfigContext(eager_mode=True):
        seed(42)
        key = PRNGKey(42)
        assert key.shape == (2,)

        keys = split(key, 3)  # noqa: F841
        assert keys.shape == (3, 2)

        folded = fold_in(key, 123)  # noqa: F841
        assert folded.shape == (2,)

        u = uniform(key, (2, 2))
        assert u.shape == (2, 2)

        n = normal(key, (3, 3))  # noqa: F841
        assert n.shape == (3, 3)

        b = bernoulli(key, 0.5, (4,))  # noqa: F841
        assert b.shape == (4,)

        b2 = bernoulli(key, u)  # noqa: F841
        assert b2.shape == (2, 2)

        tn = truncated_normal(key, -1.0, 1.0, (2,))  # noqa: F841
        assert tn.shape == (2,)

        ri = randint(key, (5,), 0, 10)  # noqa: F841
        assert ri.shape == (5,)


def test_random_tracing() -> None:
    """Docstring."""
    from ml_switcheroo.tracing import _tracer

    with ConfigContext(eager_mode=False):
        graph = _tracer.start_tracing()  # noqa: F841
        try:
            key = PRNGKey(42)
            keys = split(key, 3)  # noqa: F841
            folded = fold_in(key, 123)  # noqa: F841
            u = uniform(key, (2, 2))
            n = normal(key, (3, 3))  # noqa: F841
            b = bernoulli(key, 0.5, (4,))  # noqa: F841
            b2 = bernoulli(key, u)  # noqa: F841
            tn = truncated_normal(key, -1.0, 1.0, (2,))  # noqa: F841
            ri = randint(key, (5,), 0, 10)  # noqa: F841
        finally:
            _tracer.stop_tracing()

        with pytest.raises(RuntimeError):
            PRNGKey(42)
        with pytest.raises(RuntimeError):
            split(key, 3)
        with pytest.raises(RuntimeError):
            fold_in(key, 123)
        with pytest.raises(RuntimeError):
            uniform(key, (2, 2))
        with pytest.raises(RuntimeError):
            normal(key, (3, 3))
        with pytest.raises(RuntimeError):
            bernoulli(key, 0.5, (4,))
        with pytest.raises(RuntimeError):
            truncated_normal(key, -1.0, 1.0, (2,))
        with pytest.raises(RuntimeError):
            randint(key, (5,), 0, 10)
