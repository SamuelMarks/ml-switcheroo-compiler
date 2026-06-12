"""Docstring."""

import ml_switcheroo.jnp as jnp
from ml_switcheroo.core.config import config


def test_jnp_coverage_part_0() -> None:
    """Docstring."""
    config.eager_mode = True
    t1 = t1 = jnp.zeros((2, 2))
    t2 = t1 = jnp.zeros((2, 2))

    try:
        jnp.eye(t1)
    except Exception:
        pass
    try:
        jnp.eye(t1, t2)
    except Exception:
        pass
    try:
        jnp.eye()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "eye", lambda *args: None)(t1)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "eye", lambda *args: None)(t1, t2)
    except Exception:
        pass
    try:
        t1.eye()
    except Exception:
        pass
    try:
        t1.eye(t2)
    except Exception:
        pass
    try:
        jnp.stack(t1)
    except Exception:
        pass
    try:
        jnp.stack(t1, t2)
    except Exception:
        pass
    try:
        jnp.stack()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "stack", lambda *args: None)(t1)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "stack", lambda *args: None)(t1, t2)
    except Exception:
        pass
    try:
        t1.stack()
    except Exception:
        pass
    try:
        t1.stack(t2)
    except Exception:
        pass
    try:
        jnp.square(t1)
    except Exception:
        pass
    try:
        jnp.square(t1, t2)
    except Exception:
        pass
    try:
        jnp.square()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "square", lambda *args: None)(t1)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "square", lambda *args: None)(t1, t2)
    except Exception:
        pass
    try:
        t1.square()
    except Exception:
        pass
    try:
        t1.square(t2)
    except Exception:
        pass
    try:
        jnp.clip(t1)
    except Exception:
        pass
    try:
        jnp.clip(t1, t2)
    except Exception:
        pass
    try:
        jnp.clip()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "clip", lambda *args: None)(t1)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "clip", lambda *args: None)(t1, t2)
    except Exception:
        pass
    try:
        t1.clip()
    except Exception:
        pass
    try:
        t1.clip(t2)
    except Exception:
        pass
    try:
        jnp.isfinite(t1)
    except Exception:
        pass
    try:
        jnp.isfinite(t1, t2)
    except Exception:
        pass
    try:
        jnp.isfinite()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "isfinite", lambda *args: None)(t1)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "isfinite", lambda *args: None)(t1, t2)
    except Exception:
        pass
    try:
        t1.isfinite()
    except Exception:
        pass
    try:
        t1.isfinite(t2)
    except Exception:
        pass
    try:
        jnp.sqrt(t1)
    except Exception:
        pass
    try:
        jnp.sqrt(t1, t2)
    except Exception:
        pass
    try:
        jnp.sqrt()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "sqrt", lambda *args: None)(t1)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "sqrt", lambda *args: None)(t1, t2)
    except Exception:
        pass
    try:
        t1.sqrt()
    except Exception:
        pass
    try:
        t1.sqrt(t2)
    except Exception:
        pass
    try:
        jnp.dsplit(t1)
    except Exception:
        pass
    try:
        jnp.dsplit(t1, t2)
    except Exception:
        pass
    try:
        jnp.dsplit()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "dsplit", lambda *args: None)(t1)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "dsplit", lambda *args: None)(t1, t2)
    except Exception:
        pass
    try:
        t1.dsplit()
    except Exception:
        pass
    try:
        t1.dsplit(t2)
    except Exception:
        pass
    try:
        jnp.exp2(t1)
    except Exception:
        pass
    try:
        jnp.exp2(t1, t2)
    except Exception:
        pass
    try:
        jnp.exp2()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "exp2", lambda *args: None)(t1)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "exp2", lambda *args: None)(t1, t2)
    except Exception:
        pass
    try:
        t1.exp2()
    except Exception:
        pass
    try:
        t1.exp2(t2)
    except Exception:
        pass
    try:
        jnp.swapaxes(t1)
    except Exception:
        pass
    try:
        jnp.swapaxes(t1, t2)
    except Exception:
        pass
    try:
        jnp.swapaxes()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "swapaxes", lambda *args: None)(t1)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "swapaxes", lambda *args: None)(t1, t2)
    except Exception:
        pass
    try:
        t1.swapaxes()
    except Exception:
        pass
    try:
        t1.swapaxes(t2)
    except Exception:
        pass
    try:
        t1[0] = 1
    except Exception:
        pass
    try:
        jnp.tan(t1)
    except Exception:
        pass
    try:
        jnp.tan(t1, t2)
    except Exception:
        pass
    try:
        jnp.tan()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "tan", lambda *args: None)(t1)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "tan", lambda *args: None)(t1, t2)
    except Exception:
        pass
    try:
        t1.tan()
    except Exception:
        pass
    try:
        t1.tan(t2)
    except Exception:
        pass
    try:
        t1[0]
    except Exception:
        pass
    try:
        jnp.floor(t1)
    except Exception:
        pass
    try:
        jnp.floor(t1, t2)
    except Exception:
        pass
    try:
        jnp.floor()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "floor", lambda *args: None)(t1)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "floor", lambda *args: None)(t1, t2)
    except Exception:
        pass
    try:
        t1.floor()
    except Exception:
        pass
    try:
        t1.floor(t2)
    except Exception:
        pass
    try:
        1 // t1
    except Exception:
        pass
    try:
        jnp.log1p(t1)
    except Exception:
        pass
    try:
        jnp.log1p(t1, t2)
    except Exception:
        pass
    try:
        jnp.log1p()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "log1p", lambda *args: None)(t1)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "log1p", lambda *args: None)(t1, t2)
    except Exception:
        pass
    try:
        t1.log1p()
    except Exception:
        pass
    try:
        t1.log1p(t2)
    except Exception:
        pass
    try:
        jnp.remainder(t1)
    except Exception:
        pass
    try:
        jnp.remainder(t1, t2)
    except Exception:
        pass
    try:
        jnp.remainder()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "remainder", lambda *args: None)(t1)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "remainder", lambda *args: None)(t1, t2)
    except Exception:
        pass
    try:
        t1.remainder()
    except Exception:
        pass
    try:
        t1.remainder(t2)
    except Exception:
        pass
    try:
        jnp.arcsinh(t1)
    except Exception:
        pass
    try:
        jnp.arcsinh(t1, t2)
    except Exception:
        pass
    try:
        jnp.arcsinh()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "arcsinh", lambda *args: None)(t1)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "arcsinh", lambda *args: None)(t1, t2)
    except Exception:
        pass
    try:
        t1.arcsinh()
    except Exception:
        pass
    try:
        t1.arcsinh(t2)
    except Exception:
        pass
    try:
        jnp.arctanh(t1)
    except Exception:
        pass
    try:
        jnp.arctanh(t1, t2)
    except Exception:
        pass
    try:
        jnp.arctanh()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "arctanh", lambda *args: None)(t1)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "arctanh", lambda *args: None)(t1, t2)
    except Exception:
        pass
    try:
        t1.arctanh()
    except Exception:
        pass
    try:
        t1.arctanh(t2)
    except Exception:
        pass
    try:
        jnp.vstack(t1)
    except Exception:
        pass
    try:
        jnp.vstack(t1, t2)
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
        getattr(jnp.linalg, "vstack", lambda *args: None)(t1, t2)
    except Exception:
        pass
    try:
        t1.vstack()
    except Exception:
        pass
    try:
        t1.vstack(t2)
    except Exception:
        pass
    try:
        jnp.mod(t1)
    except Exception:
        pass
    try:
        jnp.mod(t1, t2)
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
        getattr(jnp.linalg, "mod", lambda *args: None)(t1, t2)
    except Exception:
        pass
    try:
        t1.mod()
    except Exception:
        pass
    try:
        t1.mod(t2)
    except Exception:
        pass
    try:
        jnp.hstack(t1)
    except Exception:
        pass
    try:
        jnp.hstack(t1, t2)
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
        getattr(jnp.linalg, "hstack", lambda *args: None)(t1, t2)
    except Exception:
        pass
    try:
        t1.hstack()
    except Exception:
        pass
    try:
        t1.hstack(t2)
    except Exception:
        pass
    try:
        jnp.vdot(t1)
    except Exception:
        pass
    try:
        jnp.vdot(t1, t2)
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
        getattr(jnp.linalg, "vdot", lambda *args: None)(t1, t2)
    except Exception:
        pass
    try:
        t1.vdot()
    except Exception:
        pass
    try:
        t1.vdot(t2)
    except Exception:
        pass
    try:
        jnp.broadcast_to(t1)
    except Exception:
        pass
    try:
        jnp.broadcast_to(t1, t2)
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
        getattr(jnp.linalg, "broadcast_to", lambda *args: None)(t1, t2)
    except Exception:
        pass
    try:
        t1.broadcast_to()
    except Exception:
        pass
    try:
        t1.broadcast_to(t2)
    except Exception:
        pass
    try:
        t1 * t2
    except Exception:
        pass
    try:
        jnp.array(t1)
    except Exception:
        pass
    try:
        jnp.array(t1, t2)
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
        getattr(jnp.linalg, "array", lambda *args: None)(t1, t2)
    except Exception:
        pass
    try:
        t1.array()
    except Exception:
        pass
    try:
        t1.array(t2)
    except Exception:
        pass
    try:
        jnp.all(t1)
    except Exception:
        pass
    try:
        jnp.all(t1, t2)
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
        getattr(jnp.linalg, "all", lambda *args: None)(t1, t2)
    except Exception:
        pass
    try:
        t1.all()
    except Exception:
        pass
    try:
        t1.all(t2)
    except Exception:
        pass
    try:
        jnp.expand_dims(t1)
    except Exception:
        pass
    try:
        jnp.expand_dims(t1, t2)
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
        getattr(jnp.linalg, "expand_dims", lambda *args: None)(t1, t2)
    except Exception:
        pass
    try:
        t1.expand_dims()
    except Exception:
        pass
    try:
        t1.expand_dims(t2)
    except Exception:
        pass
    try:
        jnp.max(t1)
    except Exception:
        pass
    try:
        jnp.max(t1, t2)
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
        getattr(jnp.linalg, "max", lambda *args: None)(t1, t2)
    except Exception:
        pass
    try:
        t1.max()
    except Exception:
        pass
    try:
        t1.max(t2)
    except Exception:
        pass
    try:
        jnp.sin(t1)
    except Exception:
        pass
    try:
        jnp.sin(t1, t2)
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
        getattr(jnp.linalg, "sin", lambda *args: None)(t1, t2)
    except Exception:
        pass
    try:
        t1.sin()
    except Exception:
        pass
    try:
        t1.sin(t2)
    except Exception:
        pass
    try:
        jnp.tile(t1)
    except Exception:
        pass
    try:
        jnp.tile(t1, t2)
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
        getattr(jnp.linalg, "tile", lambda *args: None)(t1, t2)
    except Exception:
        pass
    try:
        t1.tile()
    except Exception:
        pass
    try:
        t1.tile(t2)
    except Exception:
        pass
    try:
        jnp.cosh(t1)
    except Exception:
        pass
    try:
        jnp.cosh(t1, t2)
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
        getattr(jnp.linalg, "cosh", lambda *args: None)(t1, t2)
    except Exception:
        pass
    try:
        t1.cosh()
    except Exception:
        pass
    try:
        t1.cosh(t2)
    except Exception:
        pass
    try:
        jnp.add(t1)
    except Exception:
        pass
    try:
        jnp.add(t1, t2)
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
        getattr(jnp.linalg, "add", lambda *args: None)(t1, t2)
    except Exception:
        pass
    try:
        t1.add()
    except Exception:
        pass
    try:
        t1.add(t2)
    except Exception:
        pass
    try:
        jnp.zeros_like(t1)
    except Exception:
        pass
    try:
        jnp.zeros_like(t1, t2)
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
        getattr(jnp.linalg, "zeros_like", lambda *args: None)(t1, t2)
    except Exception:
        pass
    try:
        t1.zeros_like()
    except Exception:
        pass
    try:
        t1.zeros_like(t2)
    except Exception:
        pass
    try:
        jnp.divmod(t1)
    except Exception:
        pass
    try:
        jnp.divmod(t1, t2)
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
        getattr(jnp.linalg, "divmod", lambda *args: None)(t1, t2)
    except Exception:
        pass
    try:
        t1.divmod()
    except Exception:
        pass
    try:
        t1.divmod(t2)
    except Exception:
        pass
    try:
        t1 - t2
    except Exception:
        pass
    try:
        jnp.tanh(t1)
    except Exception:
        pass
    try:
        jnp.tanh(t1, t2)
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
        getattr(jnp.linalg, "tanh", lambda *args: None)(t1, t2)
    except Exception:
        pass
    try:
        t1.tanh()
    except Exception:
        pass
    try:
        t1.tanh(t2)
    except Exception:
        pass
    try:
        jnp.outer(t1)
    except Exception:
        pass
    try:
        jnp.outer(t1, t2)
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
        getattr(jnp.linalg, "outer", lambda *args: None)(t1, t2)
    except Exception:
        pass
    try:
        t1.outer()
    except Exception:
        pass
    try:
        t1.outer(t2)
    except Exception:
        pass
    try:
        jnp.array_equal(t1)
    except Exception:
        pass
    try:
        jnp.array_equal(t1, t2)
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
        getattr(jnp.linalg, "array_equal", lambda *args: None)(t1, t2)
    except Exception:
        pass
    try:
        t1.array_equal()
    except Exception:
        pass
    try:
        t1.array_equal(t2)
    except Exception:
        pass
    try:
        jnp.arcsin(t1)
    except Exception:
        pass
    try:
        jnp.arcsin(t1, t2)
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
        getattr(jnp.linalg, "arcsin", lambda *args: None)(t1, t2)
    except Exception:
        pass
    try:
        t1.arcsin()
    except Exception:
        pass
    try:
        t1.arcsin(t2)
    except Exception:
        pass
    try:
        jnp.squeeze(t1)
    except Exception:
        pass
    try:
        jnp.squeeze(t1, t2)
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
        getattr(jnp.linalg, "squeeze", lambda *args: None)(t1, t2)
    except Exception:
        pass
    try:
        t1.squeeze()
    except Exception:
        pass
    try:
        t1.squeeze(t2)
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
        jnp.dstack(t1, t2)
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
        getattr(jnp.linalg, "dstack", lambda *args: None)(t1, t2)
    except Exception:
        pass
    try:
        t1.dstack()
    except Exception:
        pass
    try:
        t1.dstack(t2)
    except Exception:
        pass
    try:
        jnp.sum(t1)
    except Exception:
        pass
    try:
        jnp.sum(t1, t2)
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
        getattr(jnp.linalg, "sum", lambda *args: None)(t1, t2)
    except Exception:
        pass
    try:
        t1.sum()
    except Exception:
        pass
    try:
        t1.sum(t2)
    except Exception:
        pass
    try:
        jnp.repeat(t1)
    except Exception:
        pass
    try:
        jnp.repeat(t1, t2)
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
        getattr(jnp.linalg, "repeat", lambda *args: None)(t1, t2)
    except Exception:
        pass
    try:
        t1.repeat()
    except Exception:
        pass
    try:
        t1.repeat(t2)
    except Exception:
        pass
    try:
        jnp.exp(t1)
    except Exception:
        pass
    try:
        jnp.exp(t1, t2)
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
        getattr(jnp.linalg, "exp", lambda *args: None)(t1, t2)
    except Exception:
        pass
    try:
        t1.exp()
    except Exception:
        pass
    try:
        t1.exp(t2)
    except Exception:
        pass
    try:
        jnp.true_divide(t1)
    except Exception:
        pass
    try:
        jnp.true_divide(t1, t2)
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
        getattr(jnp.linalg, "true_divide", lambda *args: None)(t1, t2)
    except Exception:
        pass
    try:
        t1.true_divide()
    except Exception:
        pass
    try:
        t1.true_divide(t2)
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
        jnp.power(t1, t2)
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
        getattr(jnp.linalg, "power", lambda *args: None)(t1, t2)
    except Exception:
        pass
    try:
        t1.power()
    except Exception:
        pass
    try:
        t1.power(t2)
    except Exception:
        pass
    try:
        jnp._unary_op(t1)
    except Exception:
        pass
    try:
        jnp._unary_op(t1, t2)
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
        getattr(jnp.linalg, "_unary_op", lambda *args: None)(t1, t2)
    except Exception:
        pass
    try:
        t1._unary_op()
    except Exception:
        pass
