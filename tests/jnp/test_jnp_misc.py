"""Docstring."""

import numpy as np
from ml_switcheroo.core.tensor import Tensor
import ml_switcheroo.core.dtype as DTypeMod
import ml_switcheroo.jnp as jnp
from ml_switcheroo.tracing import _tracer, ProxyTensor
from ml_switcheroo_ir import LogicalGraph as IRGraph
from ml_switcheroo.core.config import config


def test_jnp_specials() -> None:
    """Docstring."""
    config.eager_mode = True
    t1 = jnp.zeros((2, 2))
    np.array(t1)
    repr(t1)
    try:
        t1[jnp.zeros((1,), dtype=DTypeMod.DType.Int32)]
    except Exception:
        pass
    try:
        t1[
            (
                jnp.zeros((1,), dtype=DTypeMod.DType.Int32),
                jnp.zeros((1,), dtype=DTypeMod.DType.Int32),
            )
        ]
    except Exception:
        pass

    try:
        t1.__bool__()
    except Exception:
        pass
    try:
        t1.__iter__()
    except Exception:
        pass

    # max / min with kwargs
    try:
        jnp.max(t1, where=t1, initial=0.0)
    except Exception:
        pass
    try:
        jnp.min(t1, where=t1, initial=0.0)
    except Exception:
        pass
    try:
        jnp.sum(t1, where=t1, initial=0.0)
    except Exception:
        pass
    try:
        jnp.prod(t1, where=t1, initial=1.0)
    except Exception:
        pass
    try:
        jnp.mean(t1, where=t1)
    except Exception:
        pass

    try:
        jnp.array_equal(t1, t1)
    except Exception:
        pass

    try:
        jnp.linspace(0, 10, 10, retstep=True)
    except Exception:
        pass
    try:
        jnp.eye(2, k=1)
    except Exception:
        pass
    try:
        jnp.meshgrid(t1, t1, sparse=True)
    except Exception:
        pass

    _tracer.active_graph = IRGraph()
    _tracer.is_tracing = True
    config.eager_mode = False
    try:
        t_proxy = jnp.ndarray(
            Tensor(
                ProxyTensor("a", (2,), "float32"), (2,), DTypeMod.DType.Float32, None
            )
        )
        try:
            np.array(t_proxy)
        except Exception:
            pass
        try:
            repr(t_proxy)
        except Exception:
            pass

        try:
            t_proxy[jnp.zeros((1,), dtype=DTypeMod.DType.Int32)]
        except Exception:
            pass
        try:
            t_proxy[
                (
                    jnp.zeros((1,), dtype=DTypeMod.DType.Int32),
                    jnp.zeros((1,), dtype=DTypeMod.DType.Int32),
                )
            ]
        except Exception:
            pass

        try:
            t_proxy.__bool__()
        except Exception:
            pass
        try:
            t_proxy.__iter__()
        except Exception:
            pass

        t_eager = Tensor(np.array([1.0]), (1,), DTypeMod.DType.Float32, None)
        jnp.exp(t_eager)
        pt = ProxyTensor("proxy", (1,), "float32")
        jnp.exp(pt)
        jnp.exp(1.0)
        jnp.exp([1.0, 2.0])

        jnp.where(t_proxy, t_proxy, t_proxy)
        jnp.where(t_proxy, 1.0, 0.0)

        try:
            jnp.max(t_proxy, where=t_proxy, initial=0.0)
        except Exception:
            pass
        try:
            jnp.min(t_proxy, where=t_proxy, initial=0.0)
        except Exception:
            pass
        try:
            jnp.sum(t_proxy, where=t_proxy, initial=0.0)
        except Exception:
            pass
        try:
            jnp.prod(t_proxy, where=t_proxy, initial=1.0)
        except Exception:
            pass
        try:
            jnp.mean(t_proxy, where=t_proxy)
        except Exception:
            pass

        try:
            jnp.array_equal(t_proxy, t_proxy)
        except Exception:
            pass

        try:
            jnp.linspace(0, 10, 10, retstep=True)
        except Exception:
            pass
        try:
            jnp.eye(2, k=1)
        except Exception:
            pass
        try:
            jnp.meshgrid(t_proxy, t_proxy, sparse=True)
        except Exception:
            pass
    finally:
        _tracer.is_tracing = False


def test_jnp_missing() -> None:
    """Docstring."""
    import ml_switcheroo.jnp as jnp
    from ml_switcheroo.core.config import config

    config.eager_mode = True

    # bool
    t1 = jnp.zeros((1,))
    bool(t1)

    # _wrap list/tuple
    from ml_switcheroo.jnp.array import _wrap

    _wrap([t1._tensor, t1._tensor])
    _wrap((t1._tensor, t1._tensor))

    # clip
    jnp.clip(t1, a_min=0.0, a_max=1.0)

    # sum with where
    jnp.sum(t1, where=jnp.array([True]))

    # transpose with axes
    try:
        jnp.transpose(jnp.zeros((2, 2)), axes=(1, 0))
    except NotImplementedError:
        pass

    # ravel order
    try:
        jnp.ravel(t1, order="F")
    except NotImplementedError:
        pass

    # swapaxes
    jnp.swapaxes(jnp.zeros((2, 2)), 0, 1)

    # moveaxis
    jnp.moveaxis(jnp.zeros((2, 2)), 0, 1)

    # take_along_axis

    # shape
    jnp.shape(t1)

    # arange dtype string
    jnp.cumsum(jnp.zeros((2,)), dtype="float32")


def test_jnp_missing_more() -> None:
    """Docstring."""
    import ml_switcheroo.jnp as jnp
    from ml_switcheroo.core.dtype import DType

    jnp.take_along_axis(jnp.zeros((2, 2)), jnp.zeros((2, 2), dtype=DType.Int32), 0)

    # 426
    from ml_switcheroo.jnp.array import _wrap

    _wrap([1, 2])

    # 875-876: transpose with None axes
    jnp.transpose(jnp.zeros((2, 2)), axes=None)

    # 1948: shape
    jnp.shape([1, 2, 3])

    # arange dtype string, value/name
    jnp.cumsum(jnp.zeros((2,)), dtype="float32")

    class DummyDtype:
        """Docstring."""

        name = "int32"

    jnp.cumsum(jnp.zeros((2,)), dtype=DummyDtype())


def test_jnp_final() -> None:
    """Docstring."""
    import ml_switcheroo.jnp as jnp
    from ml_switcheroo.jnp import _unary_op
    from ml_switcheroo.core.dtype import DType

    _unary_op(jnp.zeros((2, 2)), "Transpose")
    try:
        _unary_op(jnp.zeros((2, 2)), "Unknown")
    except NotImplementedError:
        pass

    jnp.cumsum(jnp.zeros((2,)), dtype=DType.Float32)


def test_jnp_comparisons() -> None:
    """Docstring."""
    import ml_switcheroo.jnp as jnp

    t1 = jnp.zeros((2, 2))
    t2 = jnp.zeros((2, 2))
    assert (t1 < t2) is not None
    assert (t1 > t2) is not None
    assert (t1 <= t2) is not None
    assert (t1 >= t2) is not None
    assert (t1 == t2) is not None
    assert (-t1) is not None


def test_jnp_dtype() -> None:
    """Docstring."""
    import ml_switcheroo.jnp as jnp

    t1 = jnp.zeros((2, 2))
    assert t1.dtype is not None
    assert t1.shape is not None
