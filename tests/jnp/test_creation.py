"""Tests for Creation operations."""

import pytest
import numpy as np
from ml_switcheroo.core import ConfigContext, Tensor, Device, DeviceType
from ml_switcheroo.ops import (
    zeros,
    ones,
    full,
    zeros_like,
    ones_like,
    full_like,
    arange,
    linspace,
    eye,
    identity,
    diag,
    empty,
)


def test_creation_eager() -> None:
    """Docstring."""
    with ConfigContext(eager_mode=True):
        z = zeros((2, 2))
        assert np.array_equal(z.data, np.zeros((2, 2)))

        o = ones((2, 2))
        assert np.array_equal(o.data, np.ones((2, 2)))

        f = full((2, 2), 7)  # noqa: F841
        assert np.array_equal(f.data, np.full((2, 2), 7))

        zl = zeros_like(o)  # noqa: F841
        assert np.array_equal(zl.data, np.zeros((2, 2)))

        ol = ones_like(z)  # noqa: F841
        assert np.array_equal(ol.data, np.ones((2, 2)))

        fl = full_like(z, 5)  # noqa: F841
        assert np.array_equal(fl.data, np.full((2, 2), 5))

        ar = arange(0, 10, 2)
        assert np.array_equal(ar.data, np.arange(0, 10, 2))

        ar2 = arange(10)  # noqa: F841
        assert np.array_equal(ar2.data, np.arange(10))

        ls = linspace(0, 1, 5)  # noqa: F841
        assert np.array_equal(ls.data, np.linspace(0, 1, 5))

        ey = eye(3)  # noqa: F841
        assert np.array_equal(ey.data, np.eye(3))

        id_mat = identity(4)  # noqa: F841
        assert np.array_equal(id_mat.data, np.eye(4))

        dg = diag(ar)  # noqa: F841
        assert np.array_equal(dg.data, np.diag(np.arange(0, 10, 2)))

        dg2 = diag(o)  # noqa: F841
        assert np.array_equal(dg2.data, np.diag(np.ones((2, 2))))

        with pytest.raises(ValueError):
            diag(empty((2, 2, 2)))

        em = empty((2, 2))  # noqa: F841
        assert em.shape == (2, 2)


def test_creation_tracing() -> None:
    """Docstring."""
    from ml_switcheroo.tracing import _tracer

    with ConfigContext(eager_mode=False):
        graph = _tracer.start_tracing()  # noqa: F841
        try:
            z = zeros((2, 2))
            assert z.shape == (2, 2)
            assert len(graph.nodes) == 1

            o = ones((2, 2))
            f = full((2, 2), 7)  # noqa: F841
            zl = zeros_like(o)  # noqa: F841
            ol = ones_like(z)  # noqa: F841
            fl = full_like(z, 5)  # noqa: F841
            ar = arange(0, 10, 2)
            ar2 = arange(10)  # noqa: F841
            ls = linspace(0, 1, 5)  # noqa: F841
            ey = eye(3)  # noqa: F841
            id_mat = identity(4)  # noqa: F841
            dg = diag(ar)  # noqa: F841
            dg2 = diag(o)  # noqa: F841
            em = empty((2, 2))  # noqa: F841

            with pytest.raises(ValueError):
                diag(empty((2, 2, 2)))

        finally:
            _tracer.stop_tracing()

        with pytest.raises(RuntimeError):
            zeros((2, 2))
        with pytest.raises(RuntimeError):
            from ml_switcheroo.tracing import ProxyTensor
            from ml_switcheroo.core import DType

            t = Tensor(
                data=ProxyTensor("a", (2, 2)),
                shape=(2, 2),
                dtype=DType.Float32,
                device=Device(DeviceType.CPU),
            )
            diag(t)
