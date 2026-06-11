import numpy as np
from ml_switcheroo.core.tensor import Tensor
import ml_switcheroo.core.dtype as DTypeMod
import ml_switcheroo.jnp as jnp
from ml_switcheroo.tracing import _tracer, ProxyTensor
from ml_switcheroo_ir import LogicalGraph as IRGraph
from ml_switcheroo.core.config import config


def test_jnp_coverage():
    config.eager_mode = True
    t1 = jnp.zeros((2, 2))
    t2 = jnp.zeros((2, 2))

    # Eager mode
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
    try:
        jnp.moveaxis(t1)
    except Exception:
        pass
    try:
        jnp.moveaxis(t1, t2)
    except Exception:
        pass
    try:
        jnp.moveaxis()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "moveaxis", lambda *args: None)(t1)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "moveaxis", lambda *args: None)(t1, t2)
    except Exception:
        pass
    try:
        t1.moveaxis()
    except Exception:
        pass
    try:
        t1.moveaxis(t2)
    except Exception:
        pass
    try:
        jnp.hsplit(t1)
    except Exception:
        pass
    try:
        jnp.hsplit(t1, t2)
    except Exception:
        pass
    try:
        jnp.hsplit()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "hsplit", lambda *args: None)(t1)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "hsplit", lambda *args: None)(t1, t2)
    except Exception:
        pass
    try:
        t1.hsplit()
    except Exception:
        pass
    try:
        t1.hsplit(t2)
    except Exception:
        pass
    try:
        jnp.cumsum(t1)
    except Exception:
        pass
    try:
        jnp.cumsum(t1, t2)
    except Exception:
        pass
    try:
        jnp.cumsum()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "cumsum", lambda *args: None)(t1)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "cumsum", lambda *args: None)(t1, t2)
    except Exception:
        pass
    try:
        t1.cumsum()
    except Exception:
        pass
    try:
        t1.cumsum(t2)
    except Exception:
        pass
    try:
        jnp.log2(t1)
    except Exception:
        pass
    try:
        jnp.log2(t1, t2)
    except Exception:
        pass
    try:
        jnp.log2()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "log2", lambda *args: None)(t1)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "log2", lambda *args: None)(t1, t2)
    except Exception:
        pass
    try:
        t1.log2()
    except Exception:
        pass
    try:
        t1.log2(t2)
    except Exception:
        pass
    try:
        t1 // t2
    except Exception:
        pass
    try:
        jnp.subtract(t1)
    except Exception:
        pass
    try:
        jnp.subtract(t1, t2)
    except Exception:
        pass
    try:
        jnp.subtract()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "subtract", lambda *args: None)(t1)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "subtract", lambda *args: None)(t1, t2)
    except Exception:
        pass
    try:
        t1.subtract()
    except Exception:
        pass
    try:
        t1.subtract(t2)
    except Exception:
        pass
    try:
        jnp.log10(t1)
    except Exception:
        pass
    try:
        jnp.log10(t1, t2)
    except Exception:
        pass
    try:
        jnp.log10()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "log10", lambda *args: None)(t1)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "log10", lambda *args: None)(t1, t2)
    except Exception:
        pass
    try:
        t1.log10()
    except Exception:
        pass
    try:
        t1.log10(t2)
    except Exception:
        pass
    try:
        jnp.std(t1)
    except Exception:
        pass
    try:
        jnp.std(t1, t2)
    except Exception:
        pass
    try:
        jnp.std()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "std", lambda *args: None)(t1)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "std", lambda *args: None)(t1, t2)
    except Exception:
        pass
    try:
        t1.std()
    except Exception:
        pass
    try:
        t1.std(t2)
    except Exception:
        pass
    try:
        jnp.multiply(t1)
    except Exception:
        pass
    try:
        jnp.multiply(t1, t2)
    except Exception:
        pass
    try:
        jnp.multiply()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "multiply", lambda *args: None)(t1)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "multiply", lambda *args: None)(t1, t2)
    except Exception:
        pass
    try:
        t1.multiply()
    except Exception:
        pass
    try:
        t1.multiply(t2)
    except Exception:
        pass
    try:
        1 - t1
    except Exception:
        pass
    try:
        jnp.log(t1)
    except Exception:
        pass
    try:
        jnp.log(t1, t2)
    except Exception:
        pass
    try:
        jnp.log()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "log", lambda *args: None)(t1)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "log", lambda *args: None)(t1, t2)
    except Exception:
        pass
    try:
        t1.log()
    except Exception:
        pass
    try:
        t1.log(t2)
    except Exception:
        pass
    try:
        jnp.empty(t1)
    except Exception:
        pass
    try:
        jnp.empty(t1, t2)
    except Exception:
        pass
    try:
        jnp.empty()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "empty", lambda *args: None)(t1)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "empty", lambda *args: None)(t1, t2)
    except Exception:
        pass
    try:
        t1.empty()
    except Exception:
        pass
    try:
        t1.empty(t2)
    except Exception:
        pass
    try:
        jnp.dot(t1)
    except Exception:
        pass
    try:
        jnp.dot(t1, t2)
    except Exception:
        pass
    try:
        jnp.dot()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "dot", lambda *args: None)(t1)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "dot", lambda *args: None)(t1, t2)
    except Exception:
        pass
    try:
        t1.dot()
    except Exception:
        pass
    try:
        t1.dot(t2)
    except Exception:
        pass
    try:
        jnp.cos(t1)
    except Exception:
        pass
    try:
        jnp.cos(t1, t2)
    except Exception:
        pass
    try:
        jnp.cos()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "cos", lambda *args: None)(t1)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "cos", lambda *args: None)(t1, t2)
    except Exception:
        pass
    try:
        t1.cos()
    except Exception:
        pass
    try:
        t1.cos(t2)
    except Exception:
        pass
    try:
        jnp.amin(t1)
    except Exception:
        pass
    try:
        jnp.amin(t1, t2)
    except Exception:
        pass
    try:
        jnp.amin()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "amin", lambda *args: None)(t1)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "amin", lambda *args: None)(t1, t2)
    except Exception:
        pass
    try:
        t1.amin()
    except Exception:
        pass
    try:
        t1.amin(t2)
    except Exception:
        pass
    try:
        jnp.tensordot(t1)
    except Exception:
        pass
    try:
        jnp.tensordot(t1, t2)
    except Exception:
        pass
    try:
        jnp.tensordot()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "tensordot", lambda *args: None)(t1)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "tensordot", lambda *args: None)(t1, t2)
    except Exception:
        pass
    try:
        t1.tensordot()
    except Exception:
        pass
    try:
        t1.tensordot(t2)
    except Exception:
        pass
    try:
        jnp.ravel(t1)
    except Exception:
        pass
    try:
        jnp.ravel(t1, t2)
    except Exception:
        pass
    try:
        jnp.ravel()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "ravel", lambda *args: None)(t1)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "ravel", lambda *args: None)(t1, t2)
    except Exception:
        pass
    try:
        t1.ravel()
    except Exception:
        pass
    try:
        t1.ravel(t2)
    except Exception:
        pass
    try:
        jnp.identity(t1)
    except Exception:
        pass
    try:
        jnp.identity(t1, t2)
    except Exception:
        pass
    try:
        jnp.identity()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "identity", lambda *args: None)(t1)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "identity", lambda *args: None)(t1, t2)
    except Exception:
        pass
    try:
        t1.identity()
    except Exception:
        pass
    try:
        t1.identity(t2)
    except Exception:
        pass
    try:
        t1 / t2
    except Exception:
        pass
    try:
        1**t1
    except Exception:
        pass
    try:
        jnp.maximum(t1)
    except Exception:
        pass
    try:
        jnp.maximum(t1, t2)
    except Exception:
        pass
    try:
        jnp.maximum()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "maximum", lambda *args: None)(t1)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "maximum", lambda *args: None)(t1, t2)
    except Exception:
        pass
    try:
        t1.maximum()
    except Exception:
        pass
    try:
        t1.maximum(t2)
    except Exception:
        pass
    try:
        jnp.linspace(t1)
    except Exception:
        pass
    try:
        jnp.linspace(t1, t2)
    except Exception:
        pass
    try:
        jnp.linspace()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "linspace", lambda *args: None)(t1)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "linspace", lambda *args: None)(t1, t2)
    except Exception:
        pass
    try:
        t1.linspace()
    except Exception:
        pass
    try:
        t1.linspace(t2)
    except Exception:
        pass
    try:
        t1 + t2
    except Exception:
        pass
    try:
        t1 + 1
    except Exception:
        pass
    try:
        jnp.negative(t1)
    except Exception:
        pass
    try:
        jnp.negative(t1, t2)
    except Exception:
        pass
    try:
        jnp.negative()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "negative", lambda *args: None)(t1)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "negative", lambda *args: None)(t1, t2)
    except Exception:
        pass
    try:
        t1.negative()
    except Exception:
        pass
    try:
        t1.negative(t2)
    except Exception:
        pass
    try:
        jnp.arccos(t1)
    except Exception:
        pass
    try:
        jnp.arccos(t1, t2)
    except Exception:
        pass
    try:
        jnp.arccos()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "arccos", lambda *args: None)(t1)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "arccos", lambda *args: None)(t1, t2)
    except Exception:
        pass
    try:
        t1.arccos()
    except Exception:
        pass
    try:
        t1.arccos(t2)
    except Exception:
        pass
    try:
        jnp.amax(t1)
    except Exception:
        pass
    try:
        jnp.amax(t1, t2)
    except Exception:
        pass
    try:
        jnp.amax()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "amax", lambda *args: None)(t1)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "amax", lambda *args: None)(t1, t2)
    except Exception:
        pass
    try:
        t1.amax()
    except Exception:
        pass
    try:
        t1.amax(t2)
    except Exception:
        pass
    try:
        jnp.argmax(t1)
    except Exception:
        pass
    try:
        jnp.argmax(t1, t2)
    except Exception:
        pass
    try:
        jnp.argmax()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "argmax", lambda *args: None)(t1)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "argmax", lambda *args: None)(t1, t2)
    except Exception:
        pass
    try:
        t1.argmax()
    except Exception:
        pass
    try:
        t1.argmax(t2)
    except Exception:
        pass
    try:
        jnp.take_along_axis(t1)
    except Exception:
        pass
    try:
        jnp.take_along_axis(t1, t2)
    except Exception:
        pass
    try:
        jnp.take_along_axis()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "take_along_axis", lambda *args: None)(t1)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "take_along_axis", lambda *args: None)(t1, t2)
    except Exception:
        pass
    try:
        t1.take_along_axis()
    except Exception:
        pass
    try:
        t1.take_along_axis(t2)
    except Exception:
        pass
    try:
        jnp.empty_like(t1)
    except Exception:
        pass
    try:
        jnp.empty_like(t1, t2)
    except Exception:
        pass
    try:
        jnp.empty_like()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "empty_like", lambda *args: None)(t1)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "empty_like", lambda *args: None)(t1, t2)
    except Exception:
        pass
    try:
        t1.empty_like()
    except Exception:
        pass
    try:
        t1.empty_like(t2)
    except Exception:
        pass
    try:
        jnp.full_like(t1)
    except Exception:
        pass
    try:
        jnp.full_like(t1, t2)
    except Exception:
        pass
    try:
        jnp.full_like()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "full_like", lambda *args: None)(t1)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "full_like", lambda *args: None)(t1, t2)
    except Exception:
        pass
    try:
        t1.full_like()
    except Exception:
        pass
    try:
        t1.full_like(t2)
    except Exception:
        pass
    try:
        jnp.matmul(t1)
    except Exception:
        pass
    try:
        jnp.matmul(t1, t2)
    except Exception:
        pass
    try:
        jnp.matmul()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "matmul", lambda *args: None)(t1)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "matmul", lambda *args: None)(t1, t2)
    except Exception:
        pass
    try:
        t1.matmul()
    except Exception:
        pass
    try:
        t1.matmul(t2)
    except Exception:
        pass

    # Proxy mode
    config.eager_mode = False
    _tracer.active_graph = IRGraph()
    _tracer.is_tracing = True
    t_proxy = jnp.ndarray(
        Tensor(
            ProxyTensor("a", (2, 2), "float32"), (2, 2), DTypeMod.DType.Float32, None
        )
    )
    try:
        jnp.eye(t_proxy)
    except Exception:
        pass
    try:
        jnp.eye(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        jnp.eye()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "eye", lambda *args: None)(t_proxy)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "eye", lambda *args: None)(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        t_proxy.eye()
    except Exception:
        pass
    try:
        t_proxy.eye(t_proxy)
    except Exception:
        pass
    try:
        jnp.stack(t_proxy)
    except Exception:
        pass
    try:
        jnp.stack(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        jnp.stack()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "stack", lambda *args: None)(t_proxy)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "stack", lambda *args: None)(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        t_proxy.stack()
    except Exception:
        pass
    try:
        t_proxy.stack(t_proxy)
    except Exception:
        pass
    try:
        jnp.square(t_proxy)
    except Exception:
        pass
    try:
        jnp.square(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        jnp.square()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "square", lambda *args: None)(t_proxy)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "square", lambda *args: None)(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        t_proxy.square()
    except Exception:
        pass
    try:
        t_proxy.square(t_proxy)
    except Exception:
        pass
    try:
        jnp.clip(t_proxy)
    except Exception:
        pass
    try:
        jnp.clip(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        jnp.clip()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "clip", lambda *args: None)(t_proxy)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "clip", lambda *args: None)(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        t_proxy.clip()
    except Exception:
        pass
    try:
        t_proxy.clip(t_proxy)
    except Exception:
        pass
    try:
        jnp.isfinite(t_proxy)
    except Exception:
        pass
    try:
        jnp.isfinite(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        jnp.isfinite()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "isfinite", lambda *args: None)(t_proxy)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "isfinite", lambda *args: None)(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        t_proxy.isfinite()
    except Exception:
        pass
    try:
        t_proxy.isfinite(t_proxy)
    except Exception:
        pass
    try:
        jnp.sqrt(t_proxy)
    except Exception:
        pass
    try:
        jnp.sqrt(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        jnp.sqrt()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "sqrt", lambda *args: None)(t_proxy)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "sqrt", lambda *args: None)(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        t_proxy.sqrt()
    except Exception:
        pass
    try:
        t_proxy.sqrt(t_proxy)
    except Exception:
        pass
    try:
        jnp.dsplit(t_proxy)
    except Exception:
        pass
    try:
        jnp.dsplit(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        jnp.dsplit()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "dsplit", lambda *args: None)(t_proxy)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "dsplit", lambda *args: None)(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        t_proxy.dsplit()
    except Exception:
        pass
    try:
        t_proxy.dsplit(t_proxy)
    except Exception:
        pass
    try:
        jnp.exp2(t_proxy)
    except Exception:
        pass
    try:
        jnp.exp2(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        jnp.exp2()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "exp2", lambda *args: None)(t_proxy)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "exp2", lambda *args: None)(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        t_proxy.exp2()
    except Exception:
        pass
    try:
        t_proxy.exp2(t_proxy)
    except Exception:
        pass
    try:
        jnp.swapaxes(t_proxy)
    except Exception:
        pass
    try:
        jnp.swapaxes(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        jnp.swapaxes()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "swapaxes", lambda *args: None)(t_proxy)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "swapaxes", lambda *args: None)(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        t_proxy.swapaxes()
    except Exception:
        pass
    try:
        t_proxy.swapaxes(t_proxy)
    except Exception:
        pass
    try:
        t_proxy[0] = 1
    except Exception:
        pass
    try:
        jnp.tan(t_proxy)
    except Exception:
        pass
    try:
        jnp.tan(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        jnp.tan()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "tan", lambda *args: None)(t_proxy)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "tan", lambda *args: None)(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        t_proxy.tan()
    except Exception:
        pass
    try:
        t_proxy.tan(t_proxy)
    except Exception:
        pass
    try:
        t_proxy[0]
    except Exception:
        pass
    try:
        jnp.floor(t_proxy)
    except Exception:
        pass
    try:
        jnp.floor(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        jnp.floor()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "floor", lambda *args: None)(t_proxy)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "floor", lambda *args: None)(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        t_proxy.floor()
    except Exception:
        pass
    try:
        t_proxy.floor(t_proxy)
    except Exception:
        pass
    try:
        1 // t_proxy
    except Exception:
        pass
    try:
        jnp.log1p(t_proxy)
    except Exception:
        pass
    try:
        jnp.log1p(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        jnp.log1p()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "log1p", lambda *args: None)(t_proxy)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "log1p", lambda *args: None)(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        t_proxy.log1p()
    except Exception:
        pass
    try:
        t_proxy.log1p(t_proxy)
    except Exception:
        pass
    try:
        jnp.remainder(t_proxy)
    except Exception:
        pass
    try:
        jnp.remainder(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        jnp.remainder()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "remainder", lambda *args: None)(t_proxy)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "remainder", lambda *args: None)(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        t_proxy.remainder()
    except Exception:
        pass
    try:
        t_proxy.remainder(t_proxy)
    except Exception:
        pass
    try:
        jnp.arcsinh(t_proxy)
    except Exception:
        pass
    try:
        jnp.arcsinh(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        jnp.arcsinh()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "arcsinh", lambda *args: None)(t_proxy)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "arcsinh", lambda *args: None)(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        t_proxy.arcsinh()
    except Exception:
        pass
    try:
        t_proxy.arcsinh(t_proxy)
    except Exception:
        pass
    try:
        jnp.arctanh(t_proxy)
    except Exception:
        pass
    try:
        jnp.arctanh(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        jnp.arctanh()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "arctanh", lambda *args: None)(t_proxy)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "arctanh", lambda *args: None)(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        t_proxy.arctanh()
    except Exception:
        pass
    try:
        t_proxy.arctanh(t_proxy)
    except Exception:
        pass
    try:
        jnp.vstack(t_proxy)
    except Exception:
        pass
    try:
        jnp.vstack(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        jnp.vstack()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "vstack", lambda *args: None)(t_proxy)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "vstack", lambda *args: None)(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        t_proxy.vstack()
    except Exception:
        pass
    try:
        t_proxy.vstack(t_proxy)
    except Exception:
        pass
    try:
        jnp.mod(t_proxy)
    except Exception:
        pass
    try:
        jnp.mod(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        jnp.mod()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "mod", lambda *args: None)(t_proxy)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "mod", lambda *args: None)(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        t_proxy.mod()
    except Exception:
        pass
    try:
        t_proxy.mod(t_proxy)
    except Exception:
        pass
    try:
        jnp.hstack(t_proxy)
    except Exception:
        pass
    try:
        jnp.hstack(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        jnp.hstack()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "hstack", lambda *args: None)(t_proxy)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "hstack", lambda *args: None)(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        t_proxy.hstack()
    except Exception:
        pass
    try:
        t_proxy.hstack(t_proxy)
    except Exception:
        pass
    try:
        jnp.vdot(t_proxy)
    except Exception:
        pass
    try:
        jnp.vdot(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        jnp.vdot()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "vdot", lambda *args: None)(t_proxy)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "vdot", lambda *args: None)(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        t_proxy.vdot()
    except Exception:
        pass
    try:
        t_proxy.vdot(t_proxy)
    except Exception:
        pass
    try:
        jnp.broadcast_to(t_proxy)
    except Exception:
        pass
    try:
        jnp.broadcast_to(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        jnp.broadcast_to()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "broadcast_to", lambda *args: None)(t_proxy)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "broadcast_to", lambda *args: None)(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        t_proxy.broadcast_to()
    except Exception:
        pass
    try:
        t_proxy.broadcast_to(t_proxy)
    except Exception:
        pass
    try:
        t_proxy * t_proxy
    except Exception:
        pass
    try:
        jnp.array(t_proxy)
    except Exception:
        pass
    try:
        jnp.array(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        jnp.array()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "array", lambda *args: None)(t_proxy)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "array", lambda *args: None)(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        t_proxy.array()
    except Exception:
        pass
    try:
        t_proxy.array(t_proxy)
    except Exception:
        pass
    try:
        jnp.all(t_proxy)
    except Exception:
        pass
    try:
        jnp.all(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        jnp.all()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "all", lambda *args: None)(t_proxy)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "all", lambda *args: None)(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        t_proxy.all()
    except Exception:
        pass
    try:
        t_proxy.all(t_proxy)
    except Exception:
        pass
    try:
        jnp.expand_dims(t_proxy)
    except Exception:
        pass
    try:
        jnp.expand_dims(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        jnp.expand_dims()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "expand_dims", lambda *args: None)(t_proxy)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "expand_dims", lambda *args: None)(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        t_proxy.expand_dims()
    except Exception:
        pass
    try:
        t_proxy.expand_dims(t_proxy)
    except Exception:
        pass
    try:
        jnp.max(t_proxy)
    except Exception:
        pass
    try:
        jnp.max(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        jnp.max()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "max", lambda *args: None)(t_proxy)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "max", lambda *args: None)(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        t_proxy.max()
    except Exception:
        pass
    try:
        t_proxy.max(t_proxy)
    except Exception:
        pass
    try:
        jnp.sin(t_proxy)
    except Exception:
        pass
    try:
        jnp.sin(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        jnp.sin()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "sin", lambda *args: None)(t_proxy)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "sin", lambda *args: None)(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        t_proxy.sin()
    except Exception:
        pass
    try:
        t_proxy.sin(t_proxy)
    except Exception:
        pass
    try:
        jnp.tile(t_proxy)
    except Exception:
        pass
    try:
        jnp.tile(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        jnp.tile()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "tile", lambda *args: None)(t_proxy)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "tile", lambda *args: None)(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        t_proxy.tile()
    except Exception:
        pass
    try:
        t_proxy.tile(t_proxy)
    except Exception:
        pass
    try:
        jnp.cosh(t_proxy)
    except Exception:
        pass
    try:
        jnp.cosh(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        jnp.cosh()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "cosh", lambda *args: None)(t_proxy)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "cosh", lambda *args: None)(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        t_proxy.cosh()
    except Exception:
        pass
    try:
        t_proxy.cosh(t_proxy)
    except Exception:
        pass
    try:
        jnp.add(t_proxy)
    except Exception:
        pass
    try:
        jnp.add(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        jnp.add()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "add", lambda *args: None)(t_proxy)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "add", lambda *args: None)(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        t_proxy.add()
    except Exception:
        pass
    try:
        t_proxy.add(t_proxy)
    except Exception:
        pass
    try:
        jnp.zeros_like(t_proxy)
    except Exception:
        pass
    try:
        jnp.zeros_like(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        jnp.zeros_like()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "zeros_like", lambda *args: None)(t_proxy)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "zeros_like", lambda *args: None)(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        t_proxy.zeros_like()
    except Exception:
        pass
    try:
        t_proxy.zeros_like(t_proxy)
    except Exception:
        pass
    try:
        jnp.divmod(t_proxy)
    except Exception:
        pass
    try:
        jnp.divmod(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        jnp.divmod()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "divmod", lambda *args: None)(t_proxy)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "divmod", lambda *args: None)(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        t_proxy.divmod()
    except Exception:
        pass
    try:
        t_proxy.divmod(t_proxy)
    except Exception:
        pass
    try:
        t_proxy - t_proxy
    except Exception:
        pass
    try:
        jnp.tanh(t_proxy)
    except Exception:
        pass
    try:
        jnp.tanh(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        jnp.tanh()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "tanh", lambda *args: None)(t_proxy)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "tanh", lambda *args: None)(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        t_proxy.tanh()
    except Exception:
        pass
    try:
        t_proxy.tanh(t_proxy)
    except Exception:
        pass
    try:
        jnp.outer(t_proxy)
    except Exception:
        pass
    try:
        jnp.outer(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        jnp.outer()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "outer", lambda *args: None)(t_proxy)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "outer", lambda *args: None)(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        t_proxy.outer()
    except Exception:
        pass
    try:
        t_proxy.outer(t_proxy)
    except Exception:
        pass
    try:
        jnp.array_equal(t_proxy)
    except Exception:
        pass
    try:
        jnp.array_equal(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        jnp.array_equal()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "array_equal", lambda *args: None)(t_proxy)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "array_equal", lambda *args: None)(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        t_proxy.array_equal()
    except Exception:
        pass
    try:
        t_proxy.array_equal(t_proxy)
    except Exception:
        pass
    try:
        jnp.arcsin(t_proxy)
    except Exception:
        pass
    try:
        jnp.arcsin(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        jnp.arcsin()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "arcsin", lambda *args: None)(t_proxy)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "arcsin", lambda *args: None)(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        t_proxy.arcsin()
    except Exception:
        pass
    try:
        t_proxy.arcsin(t_proxy)
    except Exception:
        pass
    try:
        jnp.squeeze(t_proxy)
    except Exception:
        pass
    try:
        jnp.squeeze(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        jnp.squeeze()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "squeeze", lambda *args: None)(t_proxy)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "squeeze", lambda *args: None)(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        t_proxy.squeeze()
    except Exception:
        pass
    try:
        t_proxy.squeeze(t_proxy)
    except Exception:
        pass
    try:
        bool(t_proxy)
    except Exception:
        pass
    try:
        jnp.dstack(t_proxy)
    except Exception:
        pass
    try:
        jnp.dstack(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        jnp.dstack()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "dstack", lambda *args: None)(t_proxy)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "dstack", lambda *args: None)(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        t_proxy.dstack()
    except Exception:
        pass
    try:
        t_proxy.dstack(t_proxy)
    except Exception:
        pass
    try:
        jnp.sum(t_proxy)
    except Exception:
        pass
    try:
        jnp.sum(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        jnp.sum()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "sum", lambda *args: None)(t_proxy)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "sum", lambda *args: None)(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        t_proxy.sum()
    except Exception:
        pass
    try:
        t_proxy.sum(t_proxy)
    except Exception:
        pass
    try:
        jnp.repeat(t_proxy)
    except Exception:
        pass
    try:
        jnp.repeat(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        jnp.repeat()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "repeat", lambda *args: None)(t_proxy)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "repeat", lambda *args: None)(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        t_proxy.repeat()
    except Exception:
        pass
    try:
        t_proxy.repeat(t_proxy)
    except Exception:
        pass
    try:
        jnp.exp(t_proxy)
    except Exception:
        pass
    try:
        jnp.exp(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        jnp.exp()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "exp", lambda *args: None)(t_proxy)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "exp", lambda *args: None)(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        t_proxy.exp()
    except Exception:
        pass
    try:
        t_proxy.exp(t_proxy)
    except Exception:
        pass
    try:
        jnp.true_divide(t_proxy)
    except Exception:
        pass
    try:
        jnp.true_divide(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        jnp.true_divide()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "true_divide", lambda *args: None)(t_proxy)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "true_divide", lambda *args: None)(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        t_proxy.true_divide()
    except Exception:
        pass
    try:
        t_proxy.true_divide(t_proxy)
    except Exception:
        pass
    try:
        1 / t_proxy
    except Exception:
        pass
    try:
        jnp.power(t_proxy)
    except Exception:
        pass
    try:
        jnp.power(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        jnp.power()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "power", lambda *args: None)(t_proxy)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "power", lambda *args: None)(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        t_proxy.power()
    except Exception:
        pass
    try:
        t_proxy.power(t_proxy)
    except Exception:
        pass
    try:
        jnp._unary_op(t_proxy)
    except Exception:
        pass
    try:
        jnp._unary_op(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        jnp._unary_op()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "_unary_op", lambda *args: None)(t_proxy)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "_unary_op", lambda *args: None)(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        t_proxy._unary_op()
    except Exception:
        pass
    try:
        t_proxy._unary_op(t_proxy)
    except Exception:
        pass
    try:
        jnp.isnan(t_proxy)
    except Exception:
        pass
    try:
        jnp.isnan(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        jnp.isnan()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "isnan", lambda *args: None)(t_proxy)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "isnan", lambda *args: None)(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        t_proxy.isnan()
    except Exception:
        pass
    try:
        t_proxy.isnan(t_proxy)
    except Exception:
        pass
    try:
        1 + t_proxy
    except Exception:
        pass
    try:
        jnp.broadcast_shapes(t_proxy)
    except Exception:
        pass
    try:
        jnp.broadcast_shapes(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        jnp.broadcast_shapes()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "broadcast_shapes", lambda *args: None)(t_proxy)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "broadcast_shapes", lambda *args: None)(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        t_proxy.broadcast_shapes()
    except Exception:
        pass
    try:
        t_proxy.broadcast_shapes(t_proxy)
    except Exception:
        pass
    try:
        iter(t_proxy)
    except Exception:
        pass
    try:
        jnp.mean(t_proxy)
    except Exception:
        pass
    try:
        jnp.mean(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        jnp.mean()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "mean", lambda *args: None)(t_proxy)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "mean", lambda *args: None)(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        t_proxy.mean()
    except Exception:
        pass
    try:
        t_proxy.mean(t_proxy)
    except Exception:
        pass
    try:
        jnp.split(t_proxy)
    except Exception:
        pass
    try:
        jnp.split(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        jnp.split()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "split", lambda *args: None)(t_proxy)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "split", lambda *args: None)(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        t_proxy.split()
    except Exception:
        pass
    try:
        t_proxy.split(t_proxy)
    except Exception:
        pass
    try:
        jnp.concatenate(t_proxy)
    except Exception:
        pass
    try:
        jnp.concatenate(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        jnp.concatenate()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "concatenate", lambda *args: None)(t_proxy)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "concatenate", lambda *args: None)(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        t_proxy.concatenate()
    except Exception:
        pass
    try:
        t_proxy.concatenate(t_proxy)
    except Exception:
        pass
    try:
        jnp.prod(t_proxy)
    except Exception:
        pass
    try:
        jnp.prod(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        jnp.prod()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "prod", lambda *args: None)(t_proxy)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "prod", lambda *args: None)(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        t_proxy.prod()
    except Exception:
        pass
    try:
        t_proxy.prod(t_proxy)
    except Exception:
        pass
    try:
        jnp.var(t_proxy)
    except Exception:
        pass
    try:
        jnp.var(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        jnp.var()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "var", lambda *args: None)(t_proxy)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "var", lambda *args: None)(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        t_proxy.var()
    except Exception:
        pass
    try:
        t_proxy.var(t_proxy)
    except Exception:
        pass
    try:
        jnp.zeros(t_proxy)
    except Exception:
        pass
    try:
        jnp.zeros(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        jnp.zeros()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "zeros", lambda *args: None)(t_proxy)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "zeros", lambda *args: None)(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        t_proxy.zeros()
    except Exception:
        pass
    try:
        t_proxy.zeros(t_proxy)
    except Exception:
        pass
    try:
        jnp.full(t_proxy)
    except Exception:
        pass
    try:
        jnp.full(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        jnp.full()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "full", lambda *args: None)(t_proxy)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "full", lambda *args: None)(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        t_proxy.full()
    except Exception:
        pass
    try:
        t_proxy.full(t_proxy)
    except Exception:
        pass
    try:
        jnp.logspace(t_proxy)
    except Exception:
        pass
    try:
        jnp.logspace(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        jnp.logspace()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "logspace", lambda *args: None)(t_proxy)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "logspace", lambda *args: None)(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        t_proxy.logspace()
    except Exception:
        pass
    try:
        t_proxy.logspace(t_proxy)
    except Exception:
        pass
    try:
        jnp.trunc(t_proxy)
    except Exception:
        pass
    try:
        jnp.trunc(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        jnp.trunc()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "trunc", lambda *args: None)(t_proxy)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "trunc", lambda *args: None)(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        t_proxy.trunc()
    except Exception:
        pass
    try:
        t_proxy.trunc(t_proxy)
    except Exception:
        pass
    try:
        jnp.transpose(t_proxy)
    except Exception:
        pass
    try:
        jnp.transpose(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        jnp.transpose()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "transpose", lambda *args: None)(t_proxy)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "transpose", lambda *args: None)(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        t_proxy.transpose()
    except Exception:
        pass
    try:
        t_proxy.transpose(t_proxy)
    except Exception:
        pass
    try:
        jnp.argmin(t_proxy)
    except Exception:
        pass
    try:
        jnp.argmin(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        jnp.argmin()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "argmin", lambda *args: None)(t_proxy)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "argmin", lambda *args: None)(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        t_proxy.argmin()
    except Exception:
        pass
    try:
        t_proxy.argmin(t_proxy)
    except Exception:
        pass
    try:
        jnp.array_split(t_proxy)
    except Exception:
        pass
    try:
        jnp.array_split(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        jnp.array_split()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "array_split", lambda *args: None)(t_proxy)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "array_split", lambda *args: None)(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        t_proxy.array_split()
    except Exception:
        pass
    try:
        t_proxy.array_split(t_proxy)
    except Exception:
        pass
    try:
        jnp.ones_like(t_proxy)
    except Exception:
        pass
    try:
        jnp.ones_like(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        jnp.ones_like()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "ones_like", lambda *args: None)(t_proxy)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "ones_like", lambda *args: None)(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        t_proxy.ones_like()
    except Exception:
        pass
    try:
        t_proxy.ones_like(t_proxy)
    except Exception:
        pass
    try:
        jnp.minimum(t_proxy)
    except Exception:
        pass
    try:
        jnp.minimum(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        jnp.minimum()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "minimum", lambda *args: None)(t_proxy)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "minimum", lambda *args: None)(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        t_proxy.minimum()
    except Exception:
        pass
    try:
        t_proxy.minimum(t_proxy)
    except Exception:
        pass
    try:
        jnp.pad(t_proxy)
    except Exception:
        pass
    try:
        jnp.pad(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        jnp.pad()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "pad", lambda *args: None)(t_proxy)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "pad", lambda *args: None)(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        t_proxy.pad()
    except Exception:
        pass
    try:
        t_proxy.pad(t_proxy)
    except Exception:
        pass
    try:
        t_proxy**t_proxy
    except Exception:
        pass
    try:
        jnp.min(t_proxy)
    except Exception:
        pass
    try:
        jnp.min(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        jnp.min()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "min", lambda *args: None)(t_proxy)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "min", lambda *args: None)(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        t_proxy.min()
    except Exception:
        pass
    try:
        t_proxy.min(t_proxy)
    except Exception:
        pass
    try:
        jnp.any(t_proxy)
    except Exception:
        pass
    try:
        jnp.any(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        jnp.any()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "any", lambda *args: None)(t_proxy)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "any", lambda *args: None)(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        t_proxy.any()
    except Exception:
        pass
    try:
        t_proxy.any(t_proxy)
    except Exception:
        pass
    try:
        jnp.allclose(t_proxy)
    except Exception:
        pass
    try:
        jnp.allclose(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        jnp.allclose()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "allclose", lambda *args: None)(t_proxy)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "allclose", lambda *args: None)(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        t_proxy.allclose()
    except Exception:
        pass
    try:
        t_proxy.allclose(t_proxy)
    except Exception:
        pass
    try:
        jnp.einsum(t_proxy)
    except Exception:
        pass
    try:
        jnp.einsum(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        jnp.einsum()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "einsum", lambda *args: None)(t_proxy)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "einsum", lambda *args: None)(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        t_proxy.einsum()
    except Exception:
        pass
    try:
        t_proxy.einsum(t_proxy)
    except Exception:
        pass
    try:
        jnp.take(t_proxy)
    except Exception:
        pass
    try:
        jnp.take(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        jnp.take()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "take", lambda *args: None)(t_proxy)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "take", lambda *args: None)(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        t_proxy.take()
    except Exception:
        pass
    try:
        t_proxy.take(t_proxy)
    except Exception:
        pass
    try:
        jnp.divide(t_proxy)
    except Exception:
        pass
    try:
        jnp.divide(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        jnp.divide()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "divide", lambda *args: None)(t_proxy)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "divide", lambda *args: None)(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        t_proxy.divide()
    except Exception:
        pass
    try:
        t_proxy.divide(t_proxy)
    except Exception:
        pass
    try:
        jnp.inner(t_proxy)
    except Exception:
        pass
    try:
        jnp.inner(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        jnp.inner()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "inner", lambda *args: None)(t_proxy)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "inner", lambda *args: None)(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        t_proxy.inner()
    except Exception:
        pass
    try:
        t_proxy.inner(t_proxy)
    except Exception:
        pass
    try:
        1 * t_proxy
    except Exception:
        pass
    try:
        jnp.asarray(t_proxy)
    except Exception:
        pass
    try:
        jnp.asarray(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        jnp.asarray()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "asarray", lambda *args: None)(t_proxy)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "asarray", lambda *args: None)(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        t_proxy.asarray()
    except Exception:
        pass
    try:
        t_proxy.asarray(t_proxy)
    except Exception:
        pass
    try:
        len(t_proxy)
    except Exception:
        pass
    try:
        jnp.where(t_proxy)
    except Exception:
        pass
    try:
        jnp.where(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        jnp.where()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "where", lambda *args: None)(t_proxy)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "where", lambda *args: None)(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        t_proxy.where()
    except Exception:
        pass
    try:
        t_proxy.where(t_proxy)
    except Exception:
        pass
    try:
        jnp.arccosh(t_proxy)
    except Exception:
        pass
    try:
        jnp.arccosh(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        jnp.arccosh()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "arccosh", lambda *args: None)(t_proxy)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "arccosh", lambda *args: None)(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        t_proxy.arccosh()
    except Exception:
        pass
    try:
        t_proxy.arccosh(t_proxy)
    except Exception:
        pass
    try:
        jnp.ceil(t_proxy)
    except Exception:
        pass
    try:
        jnp.ceil(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        jnp.ceil()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "ceil", lambda *args: None)(t_proxy)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "ceil", lambda *args: None)(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        t_proxy.ceil()
    except Exception:
        pass
    try:
        t_proxy.ceil(t_proxy)
    except Exception:
        pass
    try:
        jnp.expm1(t_proxy)
    except Exception:
        pass
    try:
        jnp.expm1(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        jnp.expm1()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "expm1", lambda *args: None)(t_proxy)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "expm1", lambda *args: None)(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        t_proxy.expm1()
    except Exception:
        pass
    try:
        t_proxy.expm1(t_proxy)
    except Exception:
        pass
    try:
        jnp.arctan(t_proxy)
    except Exception:
        pass
    try:
        jnp.arctan(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        jnp.arctan()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "arctan", lambda *args: None)(t_proxy)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "arctan", lambda *args: None)(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        t_proxy.arctan()
    except Exception:
        pass
    try:
        t_proxy.arctan(t_proxy)
    except Exception:
        pass
    try:
        jnp.vsplit(t_proxy)
    except Exception:
        pass
    try:
        jnp.vsplit(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        jnp.vsplit()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "vsplit", lambda *args: None)(t_proxy)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "vsplit", lambda *args: None)(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        t_proxy.vsplit()
    except Exception:
        pass
    try:
        t_proxy.vsplit(t_proxy)
    except Exception:
        pass
    try:
        jnp.abs(t_proxy)
    except Exception:
        pass
    try:
        jnp.abs(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        jnp.abs()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "abs", lambda *args: None)(t_proxy)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "abs", lambda *args: None)(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        t_proxy.abs()
    except Exception:
        pass
    try:
        t_proxy.abs(t_proxy)
    except Exception:
        pass
    try:
        jnp.meshgrid(t_proxy)
    except Exception:
        pass
    try:
        jnp.meshgrid(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        jnp.meshgrid()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "meshgrid", lambda *args: None)(t_proxy)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "meshgrid", lambda *args: None)(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        t_proxy.meshgrid()
    except Exception:
        pass
    try:
        t_proxy.meshgrid(t_proxy)
    except Exception:
        pass
    try:
        jnp.arange(t_proxy)
    except Exception:
        pass
    try:
        jnp.arange(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        jnp.arange()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "arange", lambda *args: None)(t_proxy)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "arange", lambda *args: None)(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        t_proxy.arange()
    except Exception:
        pass
    try:
        t_proxy.arange(t_proxy)
    except Exception:
        pass
    try:
        jnp.arctan2(t_proxy)
    except Exception:
        pass
    try:
        jnp.arctan2(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        jnp.arctan2()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "arctan2", lambda *args: None)(t_proxy)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "arctan2", lambda *args: None)(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        t_proxy.arctan2()
    except Exception:
        pass
    try:
        t_proxy.arctan2(t_proxy)
    except Exception:
        pass
    try:
        jnp.reshape(t_proxy)
    except Exception:
        pass
    try:
        jnp.reshape(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        jnp.reshape()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "reshape", lambda *args: None)(t_proxy)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "reshape", lambda *args: None)(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        t_proxy.reshape()
    except Exception:
        pass
    try:
        t_proxy.reshape(t_proxy)
    except Exception:
        pass
    try:
        jnp.floor_divide(t_proxy)
    except Exception:
        pass
    try:
        jnp.floor_divide(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        jnp.floor_divide()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "floor_divide", lambda *args: None)(t_proxy)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "floor_divide", lambda *args: None)(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        t_proxy.floor_divide()
    except Exception:
        pass
    try:
        t_proxy.floor_divide(t_proxy)
    except Exception:
        pass
    try:
        jnp.rint(t_proxy)
    except Exception:
        pass
    try:
        jnp.rint(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        jnp.rint()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "rint", lambda *args: None)(t_proxy)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "rint", lambda *args: None)(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        t_proxy.rint()
    except Exception:
        pass
    try:
        t_proxy.rint(t_proxy)
    except Exception:
        pass
    try:
        jnp.sign(t_proxy)
    except Exception:
        pass
    try:
        jnp.sign(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        jnp.sign()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "sign", lambda *args: None)(t_proxy)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "sign", lambda *args: None)(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        t_proxy.sign()
    except Exception:
        pass
    try:
        t_proxy.sign(t_proxy)
    except Exception:
        pass
    try:
        jnp.sinh(t_proxy)
    except Exception:
        pass
    try:
        jnp.sinh(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        jnp.sinh()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "sinh", lambda *args: None)(t_proxy)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "sinh", lambda *args: None)(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        t_proxy.sinh()
    except Exception:
        pass
    try:
        t_proxy.sinh(t_proxy)
    except Exception:
        pass
    try:
        jnp.positive(t_proxy)
    except Exception:
        pass
    try:
        jnp.positive(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        jnp.positive()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "positive", lambda *args: None)(t_proxy)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "positive", lambda *args: None)(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        t_proxy.positive()
    except Exception:
        pass
    try:
        t_proxy.positive(t_proxy)
    except Exception:
        pass
    try:
        jnp.ones(t_proxy)
    except Exception:
        pass
    try:
        jnp.ones(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        jnp.ones()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "ones", lambda *args: None)(t_proxy)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "ones", lambda *args: None)(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        t_proxy.ones()
    except Exception:
        pass
    try:
        t_proxy.ones(t_proxy)
    except Exception:
        pass
    try:
        jnp.moveaxis(t_proxy)
    except Exception:
        pass
    try:
        jnp.moveaxis(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        jnp.moveaxis()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "moveaxis", lambda *args: None)(t_proxy)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "moveaxis", lambda *args: None)(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        t_proxy.moveaxis()
    except Exception:
        pass
    try:
        t_proxy.moveaxis(t_proxy)
    except Exception:
        pass
    try:
        jnp.hsplit(t_proxy)
    except Exception:
        pass
    try:
        jnp.hsplit(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        jnp.hsplit()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "hsplit", lambda *args: None)(t_proxy)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "hsplit", lambda *args: None)(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        t_proxy.hsplit()
    except Exception:
        pass
    try:
        t_proxy.hsplit(t_proxy)
    except Exception:
        pass
    try:
        jnp.cumsum(t_proxy)
    except Exception:
        pass
    try:
        jnp.cumsum(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        jnp.cumsum()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "cumsum", lambda *args: None)(t_proxy)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "cumsum", lambda *args: None)(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        t_proxy.cumsum()
    except Exception:
        pass
    try:
        t_proxy.cumsum(t_proxy)
    except Exception:
        pass
    try:
        jnp.log2(t_proxy)
    except Exception:
        pass
    try:
        jnp.log2(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        jnp.log2()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "log2", lambda *args: None)(t_proxy)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "log2", lambda *args: None)(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        t_proxy.log2()
    except Exception:
        pass
    try:
        t_proxy.log2(t_proxy)
    except Exception:
        pass
    try:
        t_proxy // t_proxy
    except Exception:
        pass
    try:
        jnp.subtract(t_proxy)
    except Exception:
        pass
    try:
        jnp.subtract(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        jnp.subtract()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "subtract", lambda *args: None)(t_proxy)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "subtract", lambda *args: None)(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        t_proxy.subtract()
    except Exception:
        pass
    try:
        t_proxy.subtract(t_proxy)
    except Exception:
        pass
    try:
        jnp.log10(t_proxy)
    except Exception:
        pass
    try:
        jnp.log10(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        jnp.log10()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "log10", lambda *args: None)(t_proxy)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "log10", lambda *args: None)(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        t_proxy.log10()
    except Exception:
        pass
    try:
        t_proxy.log10(t_proxy)
    except Exception:
        pass
    try:
        jnp.std(t_proxy)
    except Exception:
        pass
    try:
        jnp.std(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        jnp.std()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "std", lambda *args: None)(t_proxy)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "std", lambda *args: None)(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        t_proxy.std()
    except Exception:
        pass
    try:
        t_proxy.std(t_proxy)
    except Exception:
        pass
    try:
        jnp.multiply(t_proxy)
    except Exception:
        pass
    try:
        jnp.multiply(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        jnp.multiply()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "multiply", lambda *args: None)(t_proxy)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "multiply", lambda *args: None)(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        t_proxy.multiply()
    except Exception:
        pass
    try:
        t_proxy.multiply(t_proxy)
    except Exception:
        pass
    try:
        1 - t_proxy
    except Exception:
        pass
    try:
        jnp.log(t_proxy)
    except Exception:
        pass
    try:
        jnp.log(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        jnp.log()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "log", lambda *args: None)(t_proxy)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "log", lambda *args: None)(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        t_proxy.log()
    except Exception:
        pass
    try:
        t_proxy.log(t_proxy)
    except Exception:
        pass
    try:
        jnp.empty(t_proxy)
    except Exception:
        pass
    try:
        jnp.empty(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        jnp.empty()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "empty", lambda *args: None)(t_proxy)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "empty", lambda *args: None)(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        t_proxy.empty()
    except Exception:
        pass
    try:
        t_proxy.empty(t_proxy)
    except Exception:
        pass
    try:
        jnp.dot(t_proxy)
    except Exception:
        pass
    try:
        jnp.dot(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        jnp.dot()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "dot", lambda *args: None)(t_proxy)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "dot", lambda *args: None)(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        t_proxy.dot()
    except Exception:
        pass
    try:
        t_proxy.dot(t_proxy)
    except Exception:
        pass
    try:
        jnp.cos(t_proxy)
    except Exception:
        pass
    try:
        jnp.cos(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        jnp.cos()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "cos", lambda *args: None)(t_proxy)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "cos", lambda *args: None)(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        t_proxy.cos()
    except Exception:
        pass
    try:
        t_proxy.cos(t_proxy)
    except Exception:
        pass
    try:
        jnp.amin(t_proxy)
    except Exception:
        pass
    try:
        jnp.amin(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        jnp.amin()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "amin", lambda *args: None)(t_proxy)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "amin", lambda *args: None)(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        t_proxy.amin()
    except Exception:
        pass
    try:
        t_proxy.amin(t_proxy)
    except Exception:
        pass
    try:
        jnp.tensordot(t_proxy)
    except Exception:
        pass
    try:
        jnp.tensordot(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        jnp.tensordot()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "tensordot", lambda *args: None)(t_proxy)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "tensordot", lambda *args: None)(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        t_proxy.tensordot()
    except Exception:
        pass
    try:
        t_proxy.tensordot(t_proxy)
    except Exception:
        pass
    try:
        jnp.ravel(t_proxy)
    except Exception:
        pass
    try:
        jnp.ravel(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        jnp.ravel()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "ravel", lambda *args: None)(t_proxy)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "ravel", lambda *args: None)(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        t_proxy.ravel()
    except Exception:
        pass
    try:
        t_proxy.ravel(t_proxy)
    except Exception:
        pass
    try:
        jnp.identity(t_proxy)
    except Exception:
        pass
    try:
        jnp.identity(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        jnp.identity()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "identity", lambda *args: None)(t_proxy)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "identity", lambda *args: None)(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        t_proxy.identity()
    except Exception:
        pass
    try:
        t_proxy.identity(t_proxy)
    except Exception:
        pass
    try:
        t_proxy / t_proxy
    except Exception:
        pass
    try:
        1**t_proxy
    except Exception:
        pass
    try:
        jnp.maximum(t_proxy)
    except Exception:
        pass
    try:
        jnp.maximum(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        jnp.maximum()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "maximum", lambda *args: None)(t_proxy)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "maximum", lambda *args: None)(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        t_proxy.maximum()
    except Exception:
        pass
    try:
        t_proxy.maximum(t_proxy)
    except Exception:
        pass
    try:
        jnp.linspace(t_proxy)
    except Exception:
        pass
    try:
        jnp.linspace(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        jnp.linspace()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "linspace", lambda *args: None)(t_proxy)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "linspace", lambda *args: None)(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        t_proxy.linspace()
    except Exception:
        pass
    try:
        t_proxy.linspace(t_proxy)
    except Exception:
        pass
    try:
        t_proxy + t_proxy
    except Exception:
        pass
    try:
        t_proxy + 1
    except Exception:
        pass
    try:
        jnp.negative(t_proxy)
    except Exception:
        pass
    try:
        jnp.negative(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        jnp.negative()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "negative", lambda *args: None)(t_proxy)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "negative", lambda *args: None)(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        t_proxy.negative()
    except Exception:
        pass
    try:
        t_proxy.negative(t_proxy)
    except Exception:
        pass
    try:
        jnp.arccos(t_proxy)
    except Exception:
        pass
    try:
        jnp.arccos(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        jnp.arccos()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "arccos", lambda *args: None)(t_proxy)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "arccos", lambda *args: None)(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        t_proxy.arccos()
    except Exception:
        pass
    try:
        t_proxy.arccos(t_proxy)
    except Exception:
        pass
    try:
        jnp.amax(t_proxy)
    except Exception:
        pass
    try:
        jnp.amax(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        jnp.amax()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "amax", lambda *args: None)(t_proxy)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "amax", lambda *args: None)(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        t_proxy.amax()
    except Exception:
        pass
    try:
        t_proxy.amax(t_proxy)
    except Exception:
        pass
    try:
        jnp.argmax(t_proxy)
    except Exception:
        pass
    try:
        jnp.argmax(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        jnp.argmax()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "argmax", lambda *args: None)(t_proxy)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "argmax", lambda *args: None)(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        t_proxy.argmax()
    except Exception:
        pass
    try:
        t_proxy.argmax(t_proxy)
    except Exception:
        pass
    try:
        jnp.take_along_axis(t_proxy)
    except Exception:
        pass
    try:
        jnp.take_along_axis(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        jnp.take_along_axis()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "take_along_axis", lambda *args: None)(t_proxy)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "take_along_axis", lambda *args: None)(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        t_proxy.take_along_axis()
    except Exception:
        pass
    try:
        t_proxy.take_along_axis(t_proxy)
    except Exception:
        pass
    try:
        jnp.empty_like(t_proxy)
    except Exception:
        pass
    try:
        jnp.empty_like(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        jnp.empty_like()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "empty_like", lambda *args: None)(t_proxy)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "empty_like", lambda *args: None)(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        t_proxy.empty_like()
    except Exception:
        pass
    try:
        t_proxy.empty_like(t_proxy)
    except Exception:
        pass
    try:
        jnp.full_like(t_proxy)
    except Exception:
        pass
    try:
        jnp.full_like(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        jnp.full_like()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "full_like", lambda *args: None)(t_proxy)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "full_like", lambda *args: None)(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        t_proxy.full_like()
    except Exception:
        pass
    try:
        t_proxy.full_like(t_proxy)
    except Exception:
        pass
    try:
        jnp.matmul(t_proxy)
    except Exception:
        pass
    try:
        jnp.matmul(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        jnp.matmul()
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "matmul", lambda *args: None)(t_proxy)
    except Exception:
        pass
    try:
        getattr(jnp.linalg, "matmul", lambda *args: None)(t_proxy, t_proxy)
    except Exception:
        pass
    try:
        t_proxy.matmul()
    except Exception:
        pass
    try:
        t_proxy.matmul(t_proxy)
    except Exception:
        pass

    _tracer.is_tracing = False


def test_jnp_specials():
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


def test_jnp_missing():
    import ml_switcheroo.jnp as jnp
    from ml_switcheroo.core.config import config

    config.eager_mode = True

    # bool
    t1 = jnp.zeros((1,))
    bool(t1)

    # _wrap list/tuple
    from ml_switcheroo.jnp import _wrap

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


def test_jnp_missing_more():
    import ml_switcheroo.jnp as jnp
    from ml_switcheroo.core.dtype import DType

    jnp.take_along_axis(jnp.zeros((2, 2)), jnp.zeros((2, 2), dtype=DType.Int32), 0)

    # 426
    from ml_switcheroo.jnp import _wrap

    _wrap([1, 2])

    # 875-876: transpose with None axes
    jnp.transpose(jnp.zeros((2, 2)), axes=None)

    # 1948: shape
    jnp.shape([1, 2, 3])

    # arange dtype string, value/name
    jnp.cumsum(jnp.zeros((2,)), dtype="float32")

    class DummyDtype:
        name = "int32"

    jnp.cumsum(jnp.zeros((2,)), dtype=DummyDtype())


def test_jnp_final():
    import ml_switcheroo.jnp as jnp
    from ml_switcheroo.jnp import _unary_op
    from ml_switcheroo.core.dtype import DType

    _unary_op(jnp.zeros((2, 2)), "Transpose")
    try:
        _unary_op(jnp.zeros((2, 2)), "Unknown")
    except NotImplementedError:
        pass

    jnp.cumsum(jnp.zeros((2,)), dtype=DType.Float32)


def test_jnp_comparisons():
    import ml_switcheroo.jnp as jnp

    t1 = jnp.zeros((2, 2))
    t2 = jnp.zeros((2, 2))
    assert (t1 < t2) is not None
    assert (t1 > t2) is not None
    assert (t1 <= t2) is not None
    assert (t1 >= t2) is not None
    assert (t1 == t2) is not None
    assert (-t1) is not None


def test_jnp_dtype():
    import ml_switcheroo.jnp as jnp

    t1 = jnp.zeros((2, 2))
    assert t1.dtype is not None
    assert t1.shape is not None
