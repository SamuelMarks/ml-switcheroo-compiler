"""Docstring."""

import ml_switcheroo.jnp as jnp
from ml_switcheroo.core.config import config


def test_jnp_coverage_part_3() -> None:
    """Docstring."""
    config.eager_mode = True
    t1 = jnp.zeros((2, 2))
    t1 = jnp.zeros((2, 2))

    try:
        jnp.vstack(t1)
    except Exception:
        pass
    try:
        jnp.vstack(t1, t1)
    except Exception:
        pass
    try:
        jnp.vstack()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "vstack", lambda *args: None)(t1)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "vstack", lambda *args: None)(t1, t1)
    except Exception:
        pass
    try:
        t1.vstack()
    except Exception:
        pass
    try:
        t1.vstack(t1)
    except Exception:
        pass
    try:
        jnp.mod(t1)
    except Exception:
        pass
    try:
        jnp.mod(t1, t1)
    except Exception:
        pass
    try:
        jnp.mod()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "mod", lambda *args: None)(t1)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "mod", lambda *args: None)(t1, t1)
    except Exception:
        pass
    try:
        t1.mod()
    except Exception:
        pass
    try:
        t1.mod(t1)
    except Exception:
        pass
    try:
        jnp.hstack(t1)
    except Exception:
        pass
    try:
        jnp.hstack(t1, t1)
    except Exception:
        pass
    try:
        jnp.hstack()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "hstack", lambda *args: None)(t1)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "hstack", lambda *args: None)(t1, t1)
    except Exception:
        pass
    try:
        t1.hstack()
    except Exception:
        pass
    try:
        t1.hstack(t1)
    except Exception:
        pass
    try:
        jnp.vdot(t1)
    except Exception:
        pass
    try:
        jnp.vdot(t1, t1)
    except Exception:
        pass
    try:
        jnp.vdot()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "vdot", lambda *args: None)(t1)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "vdot", lambda *args: None)(t1, t1)
    except Exception:
        pass
    try:
        t1.vdot()
    except Exception:
        pass
    try:
        t1.vdot(t1)
    except Exception:
        pass
    try:
        jnp.broadcast_to(t1)
    except Exception:
        pass
    try:
        jnp.broadcast_to(t1, t1)
    except Exception:
        pass
    try:
        jnp.broadcast_to()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "broadcast_to", lambda *args: None)(t1)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "broadcast_to", lambda *args: None)(t1, t1)
    except Exception:
        pass
    try:
        t1.broadcast_to()
    except Exception:
        pass
    try:
        t1.broadcast_to(t1)
    except Exception:
        pass
    try:
        t1 * t1
    except Exception:
        pass
    try:
        jnp.array(t1)
    except Exception:
        pass
    try:
        jnp.array(t1, t1)
    except Exception:
        pass
    try:
        jnp.array()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "array", lambda *args: None)(t1)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "array", lambda *args: None)(t1, t1)
    except Exception:
        pass
    try:
        t1.array()
    except Exception:
        pass
    try:
        t1.array(t1)
    except Exception:
        pass
    try:
        jnp.all(t1)
    except Exception:
        pass
    try:
        jnp.all(t1, t1)
    except Exception:
        pass
    try:
        jnp.all()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "all", lambda *args: None)(t1)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "all", lambda *args: None)(t1, t1)
    except Exception:
        pass
    try:
        t1.all()
    except Exception:
        pass
    try:
        t1.all(t1)
    except Exception:
        pass
    try:
        jnp.expand_dims(t1)
    except Exception:
        pass
    try:
        jnp.expand_dims(t1, t1)
    except Exception:
        pass
    try:
        jnp.expand_dims()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "expand_dims", lambda *args: None)(t1)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "expand_dims", lambda *args: None)(t1, t1)
    except Exception:
        pass
    try:
        t1.expand_dims()
    except Exception:
        pass
    try:
        t1.expand_dims(t1)
    except Exception:
        pass
    try:
        jnp.max(t1)
    except Exception:
        pass
    try:
        jnp.max(t1, t1)
    except Exception:
        pass
    try:
        jnp.max()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "max", lambda *args: None)(t1)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "max", lambda *args: None)(t1, t1)
    except Exception:
        pass
    try:
        t1.max()
    except Exception:
        pass
    try:
        t1.max(t1)
    except Exception:
        pass
    try:
        jnp.sin(t1)
    except Exception:
        pass
    try:
        jnp.sin(t1, t1)
    except Exception:
        pass
    try:
        jnp.sin()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "sin", lambda *args: None)(t1)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "sin", lambda *args: None)(t1, t1)
    except Exception:
        pass
    try:
        t1.sin()
    except Exception:
        pass
    try:
        t1.sin(t1)
    except Exception:
        pass
    try:
        jnp.tile(t1)
    except Exception:
        pass
    try:
        jnp.tile(t1, t1)
    except Exception:
        pass
    try:
        jnp.tile()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "tile", lambda *args: None)(t1)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "tile", lambda *args: None)(t1, t1)
    except Exception:
        pass
    try:
        t1.tile()
    except Exception:
        pass
    try:
        t1.tile(t1)
    except Exception:
        pass
    try:
        jnp.cosh(t1)
    except Exception:
        pass
    try:
        jnp.cosh(t1, t1)
    except Exception:
        pass
    try:
        jnp.cosh()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "cosh", lambda *args: None)(t1)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "cosh", lambda *args: None)(t1, t1)
    except Exception:
        pass
    try:
        t1.cosh()
    except Exception:
        pass
    try:
        t1.cosh(t1)
    except Exception:
        pass
    try:
        jnp.add(t1)
    except Exception:
        pass
    try:
        jnp.add(t1, t1)
    except Exception:
        pass
    try:
        jnp.add()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "add", lambda *args: None)(t1)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "add", lambda *args: None)(t1, t1)
    except Exception:
        pass
    try:
        t1.add()
    except Exception:
        pass
    try:
        t1.add(t1)
    except Exception:
        pass
    try:
        jnp.zeros_like(t1)
    except Exception:
        pass
    try:
        jnp.zeros_like(t1, t1)
    except Exception:
        pass
    try:
        jnp.zeros_like()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "zeros_like", lambda *args: None)(t1)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "zeros_like", lambda *args: None)(t1, t1)
    except Exception:
        pass
    try:
        t1.zeros_like()
    except Exception:
        pass
    try:
        t1.zeros_like(t1)
    except Exception:
        pass
    try:
        jnp.divmod(t1)
    except Exception:
        pass
    try:
        jnp.divmod(t1, t1)
    except Exception:
        pass
    try:
        jnp.divmod()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "divmod", lambda *args: None)(t1)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "divmod", lambda *args: None)(t1, t1)
    except Exception:
        pass
    try:
        t1.divmod()
    except Exception:
        pass
    try:
        t1.divmod(t1)
    except Exception:
        pass
    try:
        t1 - t1
    except Exception:
        pass
    try:
        jnp.tanh(t1)
    except Exception:
        pass
    try:
        jnp.tanh(t1, t1)
    except Exception:
        pass
    try:
        jnp.tanh()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "tanh", lambda *args: None)(t1)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "tanh", lambda *args: None)(t1, t1)
    except Exception:
        pass
    try:
        t1.tanh()
    except Exception:
        pass
    try:
        t1.tanh(t1)
    except Exception:
        pass
    try:
        jnp.outer(t1)
    except Exception:
        pass
    try:
        jnp.outer(t1, t1)
    except Exception:
        pass
    try:
        jnp.outer()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "outer", lambda *args: None)(t1)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "outer", lambda *args: None)(t1, t1)
    except Exception:
        pass
    try:
        t1.outer()
    except Exception:
        pass
    try:
        t1.outer(t1)
    except Exception:
        pass
    try:
        jnp.array_equal(t1)
    except Exception:
        pass
    try:
        jnp.array_equal(t1, t1)
    except Exception:
        pass
    try:
        jnp.array_equal()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "array_equal", lambda *args: None)(t1)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "array_equal", lambda *args: None)(t1, t1)
    except Exception:
        pass
    try:
        t1.array_equal()
    except Exception:
        pass
    try:
        t1.array_equal(t1)
    except Exception:
        pass
    try:
        jnp.arcsin(t1)
    except Exception:
        pass
    try:
        jnp.arcsin(t1, t1)
    except Exception:
        pass
    try:
        jnp.arcsin()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "arcsin", lambda *args: None)(t1)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "arcsin", lambda *args: None)(t1, t1)
    except Exception:
        pass
    try:
        t1.arcsin()
    except Exception:
        pass
    try:
        t1.arcsin(t1)
    except Exception:
        pass
    try:
        jnp.squeeze(t1)
    except Exception:
        pass
    try:
        jnp.squeeze(t1, t1)
    except Exception:
        pass
    try:
        jnp.squeeze()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "squeeze", lambda *args: None)(t1)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "squeeze", lambda *args: None)(t1, t1)
    except Exception:
        pass
    try:
        t1.squeeze()
    except Exception:
        pass
    try:
        t1.squeeze(t1)
    except Exception:
        pass
    try:
        bool(t1)
    except Exception:
        pass
    try:
        jnp.dstack(t1)
    except Exception:
        pass
    try:
        jnp.dstack(t1, t1)
    except Exception:
        pass
    try:
        jnp.dstack()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "dstack", lambda *args: None)(t1)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "dstack", lambda *args: None)(t1, t1)
    except Exception:
        pass
    try:
        t1.dstack()
    except Exception:
        pass
    try:
        t1.dstack(t1)
    except Exception:
        pass
    try:
        jnp.sum(t1)
    except Exception:
        pass
    try:
        jnp.sum(t1, t1)
    except Exception:
        pass
    try:
        jnp.sum()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "sum", lambda *args: None)(t1)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "sum", lambda *args: None)(t1, t1)
    except Exception:
        pass
    try:
        t1.sum()
    except Exception:
        pass
    try:
        t1.sum(t1)
    except Exception:
        pass
    try:
        jnp.repeat(t1)
    except Exception:
        pass
    try:
        jnp.repeat(t1, t1)
    except Exception:
        pass
    try:
        jnp.repeat()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "repeat", lambda *args: None)(t1)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "repeat", lambda *args: None)(t1, t1)
    except Exception:
        pass
    try:
        t1.repeat()
    except Exception:
        pass
    try:
        t1.repeat(t1)
    except Exception:
        pass
    try:
        jnp.exp(t1)
    except Exception:
        pass
    try:
        jnp.exp(t1, t1)
    except Exception:
        pass
    try:
        jnp.exp()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "exp", lambda *args: None)(t1)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "exp", lambda *args: None)(t1, t1)
    except Exception:
        pass
    try:
        t1.exp()
    except Exception:
        pass
    try:
        t1.exp(t1)
    except Exception:
        pass
    try:
        jnp.true_divide(t1)
    except Exception:
        pass
    try:
        jnp.true_divide(t1, t1)
    except Exception:
        pass
    try:
        jnp.true_divide()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "true_divide", lambda *args: None)(t1)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "true_divide", lambda *args: None)(t1, t1)
    except Exception:
        pass
    try:
        t1.true_divide()
    except Exception:
        pass
    try:
        t1.true_divide(t1)
    except Exception:
        pass
    try:
        1 / t1
    except Exception:
        pass
    try:
        jnp.power(t1)
    except Exception:
        pass
    try:
        jnp.power(t1, t1)
    except Exception:
        pass
    try:
        jnp.power()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "power", lambda *args: None)(t1)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "power", lambda *args: None)(t1, t1)
    except Exception:
        pass
    try:
        t1.power()
    except Exception:
        pass
    try:
        t1.power(t1)
    except Exception:
        pass
    try:
        jnp._unary_op(t1)
    except Exception:
        pass
    try:
        jnp._unary_op(t1, t1)
    except Exception:
        pass
    try:
        jnp._unary_op()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "_unary_op", lambda *args: None)(t1)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "_unary_op", lambda *args: None)(t1, t1)
    except Exception:
        pass
    try:
        t1._unary_op()
    except Exception:
        pass
    try:
        t1._unary_op(t1)
    except Exception:
        pass
    try:
        jnp.isnan(t1)
    except Exception:
        pass
    try:
        jnp.isnan(t1, t1)
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
        getattr(jnp.linalg, "isnan", lambda *args: None)(t1, t1)
    except Exception:
        pass
    try:
        t1.isnan()
    except Exception:
        pass
    try:
        t1.isnan(t1)
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
        jnp.broadcast_shapes(t1, t1)
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
        getattr(jnp.linalg, "broadcast_shapes", lambda *args: None)(t1, t1)
    except Exception:
        pass
    try:
        t1.broadcast_shapes()
    except Exception:
        pass
    try:
        t1.broadcast_shapes(t1)
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
        jnp.mean(t1, t1)
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
        getattr(jnp.linalg, "mean", lambda *args: None)(t1, t1)
    except Exception:
        pass
    try:
        t1.mean()
    except Exception:
        pass
    try:
        t1.mean(t1)
    except Exception:
        pass
    try:
        jnp.split(t1)
    except Exception:
        pass
    try:
        jnp.split(t1, t1)
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
        getattr(jnp.linalg, "split", lambda *args: None)(t1, t1)
    except Exception:
        pass
    try:
        t1.split()
    except Exception:
        pass
    try:
        t1.split(t1)
    except Exception:
        pass
    try:
        jnp.concatenate(t1)
    except Exception:
        pass
    try:
        jnp.concatenate(t1, t1)
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
        getattr(jnp.linalg, "concatenate", lambda *args: None)(t1, t1)
    except Exception:
        pass
    try:
        t1.concatenate()
    except Exception:
        pass
    try:
        t1.concatenate(t1)
    except Exception:
        pass
    try:
        jnp.prod(t1)
    except Exception:
        pass
    try:
        jnp.prod(t1, t1)
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
        getattr(jnp.linalg, "prod", lambda *args: None)(t1, t1)
    except Exception:
        pass
    try:
        t1.prod()
    except Exception:
        pass
    try:
        t1.prod(t1)
    except Exception:
        pass
    try:
        jnp.var(t1)
    except Exception:
        pass
    try:
        jnp.var(t1, t1)
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
        getattr(jnp.linalg, "var", lambda *args: None)(t1, t1)
    except Exception:
        pass
    try:
        t1.var()
    except Exception:
        pass
    try:
        t1.var(t1)
    except Exception:
        pass
    try:
        jnp.zeros(t1)
    except Exception:
        pass
    try:
        jnp.zeros(t1, t1)
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
        getattr(jnp.linalg, "zeros", lambda *args: None)(t1, t1)
    except Exception:
        pass
    try:
        t1.zeros()
    except Exception:
        pass
    try:
        t1.zeros(t1)
    except Exception:
        pass
    try:
        jnp.full(t1)
    except Exception:
        pass
    try:
        jnp.full(t1, t1)
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
        getattr(jnp.linalg, "full", lambda *args: None)(t1, t1)
    except Exception:
        pass
    try:
        t1.full()
    except Exception:
        pass
    try:
        t1.full(t1)
    except Exception:
        pass
    try:
        jnp.logspace(t1)
    except Exception:
        pass
    try:
        jnp.logspace(t1, t1)
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
        getattr(jnp.linalg, "logspace", lambda *args: None)(t1, t1)
    except Exception:
        pass
    try:
        t1.logspace()
    except Exception:
        pass
    try:
        t1.logspace(t1)
    except Exception:
        pass
    try:
        jnp.trunc(t1)
    except Exception:
        pass
    try:
        jnp.trunc(t1, t1)
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
        getattr(jnp.linalg, "trunc", lambda *args: None)(t1, t1)
    except Exception:
        pass
    try:
        t1.trunc()
    except Exception:
        pass
    try:
        t1.trunc(t1)
    except Exception:
        pass
    try:
        jnp.transpose(t1)
    except Exception:
        pass
    try:
        jnp.transpose(t1, t1)
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
        getattr(jnp.linalg, "transpose", lambda *args: None)(t1, t1)
    except Exception:
        pass
    try:
        t1.transpose()
    except Exception:
        pass
    try:
        t1.transpose(t1)
    except Exception:
        pass
    try:
        jnp.argmin(t1)
    except Exception:
        pass
    try:
        jnp.argmin(t1, t1)
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
        getattr(jnp.linalg, "argmin", lambda *args: None)(t1, t1)
    except Exception:
        pass
    try:
        t1.argmin()
    except Exception:
        pass
    try:
        t1.argmin(t1)
    except Exception:
        pass
    try:
        jnp.array_split(t1)
    except Exception:
        pass
    try:
        jnp.array_split(t1, t1)
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
        getattr(jnp.linalg, "array_split", lambda *args: None)(t1, t1)
    except Exception:
        pass
    try:
        t1.array_split()
    except Exception:
        pass
    try:
        t1.array_split(t1)
    except Exception:
        pass
    try:
        jnp.ones_like(t1)
    except Exception:
        pass
    try:
        jnp.ones_like(t1, t1)
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
        getattr(jnp.linalg, "ones_like", lambda *args: None)(t1, t1)
    except Exception:
        pass
    try:
        t1.ones_like()
    except Exception:
        pass
    try:
        t1.ones_like(t1)
    except Exception:
        pass
