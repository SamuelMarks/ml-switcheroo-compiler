"""Docstring."""

from ml_switcheroo.core.tensor import Tensor
import ml_switcheroo.core.dtype as DTypeMod
import ml_switcheroo.jnp as jnp
from ml_switcheroo.tracing import _tracer, ProxyTensor
from ml_switcheroo_ir import LogicalGraph as IRGraph
from ml_switcheroo.core.config import config


def test_jnp_coverage_part_2() -> None:
    """Docstring."""
    config.eager_mode = True
    t1 = t1 = jnp.zeros((2, 2))
    t2 = t1 = jnp.zeros((2, 2))

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
    config.eager_mode = False
    _tracer.active_graph = IRGraph()
    _tracer.is_tracing = True
    t1 = jnp.ndarray(
        Tensor(
            ProxyTensor("a", (2, 2), "float32"), (2, 2), DTypeMod.DType.Float32, None
        )
    )
    try:
        jnp.eye(t1)
    except Exception:
        pass
    try:
        jnp.eye(t1, t1)
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
        getattr(jnp.linalg, "eye", lambda *args: None)(t1, t1)
    except Exception:
        pass
    try:
        t1.eye()
    except Exception:
        pass
    try:
        t1.eye(t1)
    except Exception:
        pass
    try:
        jnp.stack(t1)
    except Exception:
        pass
    try:
        jnp.stack(t1, t1)
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
        getattr(jnp.linalg, "stack", lambda *args: None)(t1, t1)
    except Exception:
        pass
    try:
        t1.stack()
    except Exception:
        pass
    try:
        t1.stack(t1)
    except Exception:
        pass
    try:
        jnp.square(t1)
    except Exception:
        pass
    try:
        jnp.square(t1, t1)
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
        getattr(jnp.linalg, "square", lambda *args: None)(t1, t1)
    except Exception:
        pass
    try:
        t1.square()
    except Exception:
        pass
    try:
        t1.square(t1)
    except Exception:
        pass
    try:
        jnp.clip(t1)
    except Exception:
        pass
    try:
        jnp.clip(t1, t1)
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
        getattr(jnp.linalg, "clip", lambda *args: None)(t1, t1)
    except Exception:
        pass
    try:
        t1.clip()
    except Exception:
        pass
    try:
        t1.clip(t1)
    except Exception:
        pass
    try:
        jnp.isfinite(t1)
    except Exception:
        pass
    try:
        jnp.isfinite(t1, t1)
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
        getattr(jnp.linalg, "isfinite", lambda *args: None)(t1, t1)
    except Exception:
        pass
    try:
        t1.isfinite()
    except Exception:
        pass
    try:
        t1.isfinite(t1)
    except Exception:
        pass
    try:
        jnp.sqrt(t1)
    except Exception:
        pass
    try:
        jnp.sqrt(t1, t1)
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
        getattr(jnp.linalg, "sqrt", lambda *args: None)(t1, t1)
    except Exception:
        pass
    try:
        t1.sqrt()
    except Exception:
        pass
    try:
        t1.sqrt(t1)
    except Exception:
        pass
    try:
        jnp.dsplit(t1)
    except Exception:
        pass
    try:
        jnp.dsplit(t1, t1)
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
        getattr(jnp.linalg, "dsplit", lambda *args: None)(t1, t1)
    except Exception:
        pass
    try:
        t1.dsplit()
    except Exception:
        pass
    try:
        t1.dsplit(t1)
    except Exception:
        pass
    try:
        jnp.exp2(t1)
    except Exception:
        pass
    try:
        jnp.exp2(t1, t1)
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
        getattr(jnp.linalg, "exp2", lambda *args: None)(t1, t1)
    except Exception:
        pass
    try:
        t1.exp2()
    except Exception:
        pass
    try:
        t1.exp2(t1)
    except Exception:
        pass
    try:
        jnp.swapaxes(t1)
    except Exception:
        pass
    try:
        jnp.swapaxes(t1, t1)
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
        getattr(jnp.linalg, "swapaxes", lambda *args: None)(t1, t1)
    except Exception:
        pass
    try:
        t1.swapaxes()
    except Exception:
        pass
    try:
        t1.swapaxes(t1)
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
        jnp.tan(t1, t1)
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
        getattr(jnp.linalg, "tan", lambda *args: None)(t1, t1)
    except Exception:
        pass
    try:
        t1.tan()
    except Exception:
        pass
    try:
        t1.tan(t1)
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
        jnp.floor(t1, t1)
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
        getattr(jnp.linalg, "floor", lambda *args: None)(t1, t1)
    except Exception:
        pass
    try:
        t1.floor()
    except Exception:
        pass
    try:
        t1.floor(t1)
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
        jnp.log1p(t1, t1)
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
        getattr(jnp.linalg, "log1p", lambda *args: None)(t1, t1)
    except Exception:
        pass
    try:
        t1.log1p()
    except Exception:
        pass
    try:
        t1.log1p(t1)
    except Exception:
        pass
    try:
        jnp.remainder(t1)
    except Exception:
        pass
    try:
        jnp.remainder(t1, t1)
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
        getattr(jnp.linalg, "remainder", lambda *args: None)(t1, t1)
    except Exception:
        pass
    try:
        t1.remainder()
    except Exception:
        pass
    try:
        t1.remainder(t1)
    except Exception:
        pass
    try:
        jnp.arcsinh(t1)
    except Exception:
        pass
    try:
        jnp.arcsinh(t1, t1)
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
        getattr(jnp.linalg, "arcsinh", lambda *args: None)(t1, t1)
    except Exception:
        pass
    try:
        t1.arcsinh()
    except Exception:
        pass
    try:
        t1.arcsinh(t1)
    except Exception:
        pass
    try:
        jnp.arctanh(t1)
    except Exception:
        pass
    try:
        jnp.arctanh(t1, t1)
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
        getattr(jnp.linalg, "arctanh", lambda *args: None)(t1, t1)
    except Exception:
        pass
    try:
        t1.arctanh()
    except Exception:
        pass
    try:
        t1.arctanh(t1)
    except Exception:
        pass
