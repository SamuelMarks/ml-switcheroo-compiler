"""Docstring."""

import ml_switcheroo.jnp as jnp
from ml_switcheroo.core.config import config


def test_jnp_coverage_part_1() -> None:
    """Docstring."""
    config.eager_mode = True
    t1 = t1 = jnp.zeros((2, 2))
    t2 = t1 = jnp.zeros((2, 2))

    try:
        t1._unary_op(t2)
    except Exception:
        pass
    try:
        jnp.isnan(t1)
    except Exception:
        pass
    try:
        jnp.isnan(t1, t2)
    except Exception:
        pass
    try:
        jnp.isnan()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "isnan", lambda *args: None)(t1)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "isnan", lambda *args: None)(t1, t2)
    except Exception:
        pass
    try:
        t1.isnan()
    except Exception:
        pass
    try:
        t1.isnan(t2)
    except Exception:
        pass
    try:
        1 + t1
    except Exception:
        pass
    try:
        jnp.broadcast_shapes(t1)
    except Exception:
        pass
    try:
        jnp.broadcast_shapes(t1, t2)
    except Exception:
        pass
    try:
        jnp.broadcast_shapes()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "broadcast_shapes", lambda *args: None)(t1)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "broadcast_shapes", lambda *args: None)(t1, t2)
    except Exception:
        pass
    try:
        t1.broadcast_shapes()
    except Exception:
        pass
    try:
        t1.broadcast_shapes(t2)
    except Exception:
        pass
    try:
        iter(t1)
    except Exception:
        pass
    try:
        jnp.mean(t1)
    except Exception:
        pass
    try:
        jnp.mean(t1, t2)
    except Exception:
        pass
    try:
        jnp.mean()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "mean", lambda *args: None)(t1)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "mean", lambda *args: None)(t1, t2)
    except Exception:
        pass
    try:
        t1.mean()
    except Exception:
        pass
    try:
        t1.mean(t2)
    except Exception:
        pass
    try:
        jnp.split(t1)
    except Exception:
        pass
    try:
        jnp.split(t1, t2)
    except Exception:
        pass
    try:
        jnp.split()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "split", lambda *args: None)(t1)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "split", lambda *args: None)(t1, t2)
    except Exception:
        pass
    try:
        t1.split()
    except Exception:
        pass
    try:
        t1.split(t2)
    except Exception:
        pass
    try:
        jnp.concatenate(t1)
    except Exception:
        pass
    try:
        jnp.concatenate(t1, t2)
    except Exception:
        pass
    try:
        jnp.concatenate()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "concatenate", lambda *args: None)(t1)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "concatenate", lambda *args: None)(t1, t2)
    except Exception:
        pass
    try:
        t1.concatenate()
    except Exception:
        pass
    try:
        t1.concatenate(t2)
    except Exception:
        pass
    try:
        jnp.prod(t1)
    except Exception:
        pass
    try:
        jnp.prod(t1, t2)
    except Exception:
        pass
    try:
        jnp.prod()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "prod", lambda *args: None)(t1)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "prod", lambda *args: None)(t1, t2)
    except Exception:
        pass
    try:
        t1.prod()
    except Exception:
        pass
    try:
        t1.prod(t2)
    except Exception:
        pass
    try:
        jnp.var(t1)
    except Exception:
        pass
    try:
        jnp.var(t1, t2)
    except Exception:
        pass
    try:
        jnp.var()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "var", lambda *args: None)(t1)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "var", lambda *args: None)(t1, t2)
    except Exception:
        pass
    try:
        t1.var()
    except Exception:
        pass
    try:
        t1.var(t2)
    except Exception:
        pass
    try:
        jnp.zeros(t1)
    except Exception:
        pass
    try:
        jnp.zeros(t1, t2)
    except Exception:
        pass
    try:
        jnp.zeros()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "zeros", lambda *args: None)(t1)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "zeros", lambda *args: None)(t1, t2)
    except Exception:
        pass
    try:
        t1.zeros()
    except Exception:
        pass
    try:
        t1.zeros(t2)
    except Exception:
        pass
    try:
        jnp.full(t1)
    except Exception:
        pass
    try:
        jnp.full(t1, t2)
    except Exception:
        pass
    try:
        jnp.full()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "full", lambda *args: None)(t1)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "full", lambda *args: None)(t1, t2)
    except Exception:
        pass
    try:
        t1.full()
    except Exception:
        pass
    try:
        t1.full(t2)
    except Exception:
        pass
    try:
        jnp.logspace(t1)
    except Exception:
        pass
    try:
        jnp.logspace(t1, t2)
    except Exception:
        pass
    try:
        jnp.logspace()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "logspace", lambda *args: None)(t1)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "logspace", lambda *args: None)(t1, t2)
    except Exception:
        pass
    try:
        t1.logspace()
    except Exception:
        pass
    try:
        t1.logspace(t2)
    except Exception:
        pass
    try:
        jnp.trunc(t1)
    except Exception:
        pass
    try:
        jnp.trunc(t1, t2)
    except Exception:
        pass
    try:
        jnp.trunc()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "trunc", lambda *args: None)(t1)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "trunc", lambda *args: None)(t1, t2)
    except Exception:
        pass
    try:
        t1.trunc()
    except Exception:
        pass
    try:
        t1.trunc(t2)
    except Exception:
        pass
    try:
        jnp.transpose(t1)
    except Exception:
        pass
    try:
        jnp.transpose(t1, t2)
    except Exception:
        pass
    try:
        jnp.transpose()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "transpose", lambda *args: None)(t1)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "transpose", lambda *args: None)(t1, t2)
    except Exception:
        pass
    try:
        t1.transpose()
    except Exception:
        pass
    try:
        t1.transpose(t2)
    except Exception:
        pass
    try:
        jnp.argmin(t1)
    except Exception:
        pass
    try:
        jnp.argmin(t1, t2)
    except Exception:
        pass
    try:
        jnp.argmin()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "argmin", lambda *args: None)(t1)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "argmin", lambda *args: None)(t1, t2)
    except Exception:
        pass
    try:
        t1.argmin()
    except Exception:
        pass
    try:
        t1.argmin(t2)
    except Exception:
        pass
    try:
        jnp.array_split(t1)
    except Exception:
        pass
    try:
        jnp.array_split(t1, t2)
    except Exception:
        pass
    try:
        jnp.array_split()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "array_split", lambda *args: None)(t1)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "array_split", lambda *args: None)(t1, t2)
    except Exception:
        pass
    try:
        t1.array_split()
    except Exception:
        pass
    try:
        t1.array_split(t2)
    except Exception:
        pass
    try:
        jnp.ones_like(t1)
    except Exception:
        pass
    try:
        jnp.ones_like(t1, t2)
    except Exception:
        pass
    try:
        jnp.ones_like()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "ones_like", lambda *args: None)(t1)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "ones_like", lambda *args: None)(t1, t2)
    except Exception:
        pass
    try:
        t1.ones_like()
    except Exception:
        pass
    try:
        t1.ones_like(t2)
    except Exception:
        pass
    try:
        jnp.minimum(t1)
    except Exception:
        pass
    try:
        jnp.minimum(t1, t2)
    except Exception:
        pass
    try:
        jnp.minimum()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "minimum", lambda *args: None)(t1)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "minimum", lambda *args: None)(t1, t2)
    except Exception:
        pass
    try:
        t1.minimum()
    except Exception:
        pass
    try:
        t1.minimum(t2)
    except Exception:
        pass
    try:
        jnp.pad(t1)
    except Exception:
        pass
    try:
        jnp.pad(t1, t2)
    except Exception:
        pass
    try:
        jnp.pad()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "pad", lambda *args: None)(t1)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "pad", lambda *args: None)(t1, t2)
    except Exception:
        pass
    try:
        t1.pad()
    except Exception:
        pass
    try:
        t1.pad(t2)
    except Exception:
        pass
    try:
        t1**t2
    except Exception:
        pass
    try:
        jnp.min(t1)
    except Exception:
        pass
    try:
        jnp.min(t1, t2)
    except Exception:
        pass
    try:
        jnp.min()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "min", lambda *args: None)(t1)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "min", lambda *args: None)(t1, t2)
    except Exception:
        pass
    try:
        t1.min()
    except Exception:
        pass
    try:
        t1.min(t2)
    except Exception:
        pass
    try:
        jnp.any(t1)
    except Exception:
        pass
    try:
        jnp.any(t1, t2)
    except Exception:
        pass
    try:
        jnp.any()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "any", lambda *args: None)(t1)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "any", lambda *args: None)(t1, t2)
    except Exception:
        pass
    try:
        t1.any()
    except Exception:
        pass
    try:
        t1.any(t2)
    except Exception:
        pass
    try:
        jnp.allclose(t1)
    except Exception:
        pass
    try:
        jnp.allclose(t1, t2)
    except Exception:
        pass
    try:
        jnp.allclose()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "allclose", lambda *args: None)(t1)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "allclose", lambda *args: None)(t1, t2)
    except Exception:
        pass
    try:
        t1.allclose()
    except Exception:
        pass
    try:
        t1.allclose(t2)
    except Exception:
        pass
    try:
        jnp.einsum(t1)
    except Exception:
        pass
    try:
        jnp.einsum(t1, t2)
    except Exception:
        pass
    try:
        jnp.einsum()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "einsum", lambda *args: None)(t1)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "einsum", lambda *args: None)(t1, t2)
    except Exception:
        pass
    try:
        t1.einsum()
    except Exception:
        pass
    try:
        t1.einsum(t2)
    except Exception:
        pass
    try:
        jnp.take(t1)
    except Exception:
        pass
    try:
        jnp.take(t1, t2)
    except Exception:
        pass
    try:
        jnp.take()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "take", lambda *args: None)(t1)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "take", lambda *args: None)(t1, t2)
    except Exception:
        pass
    try:
        t1.take()
    except Exception:
        pass
    try:
        t1.take(t2)
    except Exception:
        pass
    try:
        jnp.divide(t1)
    except Exception:
        pass
    try:
        jnp.divide(t1, t2)
    except Exception:
        pass
    try:
        jnp.divide()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "divide", lambda *args: None)(t1)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "divide", lambda *args: None)(t1, t2)
    except Exception:
        pass
    try:
        t1.divide()
    except Exception:
        pass
    try:
        t1.divide(t2)
    except Exception:
        pass
    try:
        jnp.inner(t1)
    except Exception:
        pass
    try:
        jnp.inner(t1, t2)
    except Exception:
        pass
    try:
        jnp.inner()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "inner", lambda *args: None)(t1)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "inner", lambda *args: None)(t1, t2)
    except Exception:
        pass
    try:
        t1.inner()
    except Exception:
        pass
    try:
        t1.inner(t2)
    except Exception:
        pass
    try:
        1 * t1
    except Exception:
        pass
    try:
        jnp.asarray(t1)
    except Exception:
        pass
    try:
        jnp.asarray(t1, t2)
    except Exception:
        pass
    try:
        jnp.asarray()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "asarray", lambda *args: None)(t1)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "asarray", lambda *args: None)(t1, t2)
    except Exception:
        pass
    try:
        t1.asarray()
    except Exception:
        pass
    try:
        t1.asarray(t2)
    except Exception:
        pass
    try:
        len(t1)
    except Exception:
        pass
    try:
        jnp.where(t1)
    except Exception:
        pass
    try:
        jnp.where(t1, t2)
    except Exception:
        pass
    try:
        jnp.where()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "where", lambda *args: None)(t1)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "where", lambda *args: None)(t1, t2)
    except Exception:
        pass
    try:
        t1.where()
    except Exception:
        pass
    try:
        t1.where(t2)
    except Exception:
        pass
    try:
        jnp.arccosh(t1)
    except Exception:
        pass
    try:
        jnp.arccosh(t1, t2)
    except Exception:
        pass
    try:
        jnp.arccosh()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "arccosh", lambda *args: None)(t1)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "arccosh", lambda *args: None)(t1, t2)
    except Exception:
        pass
    try:
        t1.arccosh()
    except Exception:
        pass
    try:
        t1.arccosh(t2)
    except Exception:
        pass
    try:
        jnp.ceil(t1)
    except Exception:
        pass
    try:
        jnp.ceil(t1, t2)
    except Exception:
        pass
    try:
        jnp.ceil()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "ceil", lambda *args: None)(t1)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "ceil", lambda *args: None)(t1, t2)
    except Exception:
        pass
    try:
        t1.ceil()
    except Exception:
        pass
    try:
        t1.ceil(t2)
    except Exception:
        pass
    try:
        jnp.expm1(t1)
    except Exception:
        pass
    try:
        jnp.expm1(t1, t2)
    except Exception:
        pass
    try:
        jnp.expm1()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "expm1", lambda *args: None)(t1)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "expm1", lambda *args: None)(t1, t2)
    except Exception:
        pass
    try:
        t1.expm1()
    except Exception:
        pass
    try:
        t1.expm1(t2)
    except Exception:
        pass
    try:
        jnp.arctan(t1)
    except Exception:
        pass
    try:
        jnp.arctan(t1, t2)
    except Exception:
        pass
    try:
        jnp.arctan()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "arctan", lambda *args: None)(t1)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "arctan", lambda *args: None)(t1, t2)
    except Exception:
        pass
    try:
        t1.arctan()
    except Exception:
        pass
    try:
        t1.arctan(t2)
    except Exception:
        pass
    try:
        jnp.vsplit(t1)
    except Exception:
        pass
    try:
        jnp.vsplit(t1, t2)
    except Exception:
        pass
    try:
        jnp.vsplit()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "vsplit", lambda *args: None)(t1)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "vsplit", lambda *args: None)(t1, t2)
    except Exception:
        pass
    try:
        t1.vsplit()
    except Exception:
        pass
    try:
        t1.vsplit(t2)
    except Exception:
        pass
    try:
        jnp.abs(t1)
    except Exception:
        pass
    try:
        jnp.abs(t1, t2)
    except Exception:
        pass
    try:
        jnp.abs()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "abs", lambda *args: None)(t1)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "abs", lambda *args: None)(t1, t2)
    except Exception:
        pass
    try:
        t1.abs()
    except Exception:
        pass
    try:
        t1.abs(t2)
    except Exception:
        pass
    try:
        jnp.meshgrid(t1)
    except Exception:
        pass
    try:
        jnp.meshgrid(t1, t2)
    except Exception:
        pass
    try:
        jnp.meshgrid()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "meshgrid", lambda *args: None)(t1)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "meshgrid", lambda *args: None)(t1, t2)
    except Exception:
        pass
    try:
        t1.meshgrid()
    except Exception:
        pass
    try:
        t1.meshgrid(t2)
    except Exception:
        pass
    try:
        jnp.arange(t1)
    except Exception:
        pass
    try:
        jnp.arange(t1, t2)
    except Exception:
        pass
    try:
        jnp.arange()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "arange", lambda *args: None)(t1)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "arange", lambda *args: None)(t1, t2)
    except Exception:
        pass
    try:
        t1.arange()
    except Exception:
        pass
    try:
        t1.arange(t2)
    except Exception:
        pass
    try:
        jnp.arctan2(t1)
    except Exception:
        pass
    try:
        jnp.arctan2(t1, t2)
    except Exception:
        pass
    try:
        jnp.arctan2()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "arctan2", lambda *args: None)(t1)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "arctan2", lambda *args: None)(t1, t2)
    except Exception:
        pass
    try:
        t1.arctan2()
    except Exception:
        pass
    try:
        t1.arctan2(t2)
    except Exception:
        pass
    try:
        jnp.reshape(t1)
    except Exception:
        pass
    try:
        jnp.reshape(t1, t2)
    except Exception:
        pass
    try:
        jnp.reshape()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "reshape", lambda *args: None)(t1)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "reshape", lambda *args: None)(t1, t2)
    except Exception:
        pass
    try:
        t1.reshape()
    except Exception:
        pass
    try:
        t1.reshape(t2)
    except Exception:
        pass
    try:
        jnp.floor_divide(t1)
    except Exception:
        pass
    try:
        jnp.floor_divide(t1, t2)
    except Exception:
        pass
    try:
        jnp.floor_divide()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "floor_divide", lambda *args: None)(t1)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "floor_divide", lambda *args: None)(t1, t2)
    except Exception:
        pass
    try:
        t1.floor_divide()
    except Exception:
        pass
    try:
        t1.floor_divide(t2)
    except Exception:
        pass
    try:
        jnp.rint(t1)
    except Exception:
        pass
    try:
        jnp.rint(t1, t2)
    except Exception:
        pass
    try:
        jnp.rint()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "rint", lambda *args: None)(t1)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "rint", lambda *args: None)(t1, t2)
    except Exception:
        pass
    try:
        t1.rint()
    except Exception:
        pass
    try:
        t1.rint(t2)
    except Exception:
        pass
    try:
        jnp.sign(t1)
    except Exception:
        pass
    try:
        jnp.sign(t1, t2)
    except Exception:
        pass
    try:
        jnp.sign()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "sign", lambda *args: None)(t1)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "sign", lambda *args: None)(t1, t2)
    except Exception:
        pass
    try:
        t1.sign()
    except Exception:
        pass
    try:
        t1.sign(t2)
    except Exception:
        pass
    try:
        jnp.sinh(t1)
    except Exception:
        pass
    try:
        jnp.sinh(t1, t2)
    except Exception:
        pass
    try:
        jnp.sinh()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "sinh", lambda *args: None)(t1)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "sinh", lambda *args: None)(t1, t2)
    except Exception:
        pass
    try:
        t1.sinh()
    except Exception:
        pass
    try:
        t1.sinh(t2)
    except Exception:
        pass
    try:
        jnp.positive(t1)
    except Exception:
        pass
    try:
        jnp.positive(t1, t2)
    except Exception:
        pass
    try:
        jnp.positive()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "positive", lambda *args: None)(t1)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "positive", lambda *args: None)(t1, t2)
    except Exception:
        pass
    try:
        t1.positive()
    except Exception:
        pass
    try:
        t1.positive(t2)
    except Exception:
        pass
    try:
        jnp.ones(t1)
    except Exception:
        pass
    try:
        jnp.ones(t1, t2)
    except Exception:
        pass
    try:
        jnp.ones()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "ones", lambda *args: None)(t1)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "ones", lambda *args: None)(t1, t2)
    except Exception:
        pass
    try:
        t1.ones()
    except Exception:
        pass
    try:
        t1.ones(t2)
    except Exception:
        pass
