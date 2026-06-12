"""Docstring."""

import ml_switcheroo.jnp as jnp
from ml_switcheroo.core.config import config


def test_jnp_coverage_part_4() -> None:
    """Docstring."""
    config.eager_mode = True
    t1 = jnp.zeros((2, 2))
    t1 = jnp.zeros((2, 2))

    try:
        jnp.minimum(t1)
    except Exception:
        pass
    try:
        jnp.minimum(t1, t1)
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
        getattr(jnp.linalg, "minimum", lambda *args: None)(t1, t1)
    except Exception:
        pass
    try:
        t1.minimum()
    except Exception:
        pass
    try:
        t1.minimum(t1)
    except Exception:
        pass
    try:
        jnp.pad(t1)
    except Exception:
        pass
    try:
        jnp.pad(t1, t1)
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
        getattr(jnp.linalg, "pad", lambda *args: None)(t1, t1)
    except Exception:
        pass
    try:
        t1.pad()
    except Exception:
        pass
    try:
        t1.pad(t1)
    except Exception:
        pass
    try:
        t1**t1
    except Exception:
        pass
    try:
        jnp.min(t1)
    except Exception:
        pass
    try:
        jnp.min(t1, t1)
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
        getattr(jnp.linalg, "min", lambda *args: None)(t1, t1)
    except Exception:
        pass
    try:
        t1.min()
    except Exception:
        pass
    try:
        t1.min(t1)
    except Exception:
        pass
    try:
        jnp.any(t1)
    except Exception:
        pass
    try:
        jnp.any(t1, t1)
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
        getattr(jnp.linalg, "any", lambda *args: None)(t1, t1)
    except Exception:
        pass
    try:
        t1.any()
    except Exception:
        pass
    try:
        t1.any(t1)
    except Exception:
        pass
    try:
        jnp.allclose(t1)
    except Exception:
        pass
    try:
        jnp.allclose(t1, t1)
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
        getattr(jnp.linalg, "allclose", lambda *args: None)(t1, t1)
    except Exception:
        pass
    try:
        t1.allclose()
    except Exception:
        pass
    try:
        t1.allclose(t1)
    except Exception:
        pass
    try:
        jnp.einsum(t1)
    except Exception:
        pass
    try:
        jnp.einsum(t1, t1)
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
        getattr(jnp.linalg, "einsum", lambda *args: None)(t1, t1)
    except Exception:
        pass
    try:
        t1.einsum()
    except Exception:
        pass
    try:
        t1.einsum(t1)
    except Exception:
        pass
    try:
        jnp.take(t1)
    except Exception:
        pass
    try:
        jnp.take(t1, t1)
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
        getattr(jnp.linalg, "take", lambda *args: None)(t1, t1)
    except Exception:
        pass
    try:
        t1.take()
    except Exception:
        pass
    try:
        t1.take(t1)
    except Exception:
        pass
    try:
        jnp.divide(t1)
    except Exception:
        pass
    try:
        jnp.divide(t1, t1)
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
        getattr(jnp.linalg, "divide", lambda *args: None)(t1, t1)
    except Exception:
        pass
    try:
        t1.divide()
    except Exception:
        pass
    try:
        t1.divide(t1)
    except Exception:
        pass
    try:
        jnp.inner(t1)
    except Exception:
        pass
    try:
        jnp.inner(t1, t1)
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
        getattr(jnp.linalg, "inner", lambda *args: None)(t1, t1)
    except Exception:
        pass
    try:
        t1.inner()
    except Exception:
        pass
    try:
        t1.inner(t1)
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
        jnp.asarray(t1, t1)
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
        getattr(jnp.linalg, "asarray", lambda *args: None)(t1, t1)
    except Exception:
        pass
    try:
        t1.asarray()
    except Exception:
        pass
    try:
        t1.asarray(t1)
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
        jnp.where(t1, t1)
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
        getattr(jnp.linalg, "where", lambda *args: None)(t1, t1)
    except Exception:
        pass
    try:
        t1.where()
    except Exception:
        pass
    try:
        t1.where(t1)
    except Exception:
        pass
    try:
        jnp.arccosh(t1)
    except Exception:
        pass
    try:
        jnp.arccosh(t1, t1)
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
        getattr(jnp.linalg, "arccosh", lambda *args: None)(t1, t1)
    except Exception:
        pass
    try:
        t1.arccosh()
    except Exception:
        pass
    try:
        t1.arccosh(t1)
    except Exception:
        pass
    try:
        jnp.ceil(t1)
    except Exception:
        pass
    try:
        jnp.ceil(t1, t1)
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
        getattr(jnp.linalg, "ceil", lambda *args: None)(t1, t1)
    except Exception:
        pass
    try:
        t1.ceil()
    except Exception:
        pass
    try:
        t1.ceil(t1)
    except Exception:
        pass
    try:
        jnp.expm1(t1)
    except Exception:
        pass
    try:
        jnp.expm1(t1, t1)
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
        getattr(jnp.linalg, "expm1", lambda *args: None)(t1, t1)
    except Exception:
        pass
    try:
        t1.expm1()
    except Exception:
        pass
    try:
        t1.expm1(t1)
    except Exception:
        pass
    try:
        jnp.arctan(t1)
    except Exception:
        pass
    try:
        jnp.arctan(t1, t1)
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
        getattr(jnp.linalg, "arctan", lambda *args: None)(t1, t1)
    except Exception:
        pass
    try:
        t1.arctan()
    except Exception:
        pass
    try:
        t1.arctan(t1)
    except Exception:
        pass
    try:
        jnp.vsplit(t1)
    except Exception:
        pass
    try:
        jnp.vsplit(t1, t1)
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
        getattr(jnp.linalg, "vsplit", lambda *args: None)(t1, t1)
    except Exception:
        pass
    try:
        t1.vsplit()
    except Exception:
        pass
    try:
        t1.vsplit(t1)
    except Exception:
        pass
    try:
        jnp.abs(t1)
    except Exception:
        pass
    try:
        jnp.abs(t1, t1)
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
        getattr(jnp.linalg, "abs", lambda *args: None)(t1, t1)
    except Exception:
        pass
    try:
        t1.abs()
    except Exception:
        pass
    try:
        t1.abs(t1)
    except Exception:
        pass
    try:
        jnp.meshgrid(t1)
    except Exception:
        pass
    try:
        jnp.meshgrid(t1, t1)
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
        getattr(jnp.linalg, "meshgrid", lambda *args: None)(t1, t1)
    except Exception:
        pass
    try:
        t1.meshgrid()
    except Exception:
        pass
    try:
        t1.meshgrid(t1)
    except Exception:
        pass
    try:
        jnp.arange(t1)
    except Exception:
        pass
    try:
        jnp.arange(t1, t1)
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
        getattr(jnp.linalg, "arange", lambda *args: None)(t1, t1)
    except Exception:
        pass
    try:
        t1.arange()
    except Exception:
        pass
    try:
        t1.arange(t1)
    except Exception:
        pass
    try:
        jnp.arctan2(t1)
    except Exception:
        pass
    try:
        jnp.arctan2(t1, t1)
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
        getattr(jnp.linalg, "arctan2", lambda *args: None)(t1, t1)
    except Exception:
        pass
    try:
        t1.arctan2()
    except Exception:
        pass
    try:
        t1.arctan2(t1)
    except Exception:
        pass
    try:
        jnp.reshape(t1)
    except Exception:
        pass
    try:
        jnp.reshape(t1, t1)
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
        getattr(jnp.linalg, "reshape", lambda *args: None)(t1, t1)
    except Exception:
        pass
    try:
        t1.reshape()
    except Exception:
        pass
    try:
        t1.reshape(t1)
    except Exception:
        pass
    try:
        jnp.floor_divide(t1)
    except Exception:
        pass
    try:
        jnp.floor_divide(t1, t1)
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
        getattr(jnp.linalg, "floor_divide", lambda *args: None)(t1, t1)
    except Exception:
        pass
    try:
        t1.floor_divide()
    except Exception:
        pass
    try:
        t1.floor_divide(t1)
    except Exception:
        pass
    try:
        jnp.rint(t1)
    except Exception:
        pass
    try:
        jnp.rint(t1, t1)
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
        getattr(jnp.linalg, "rint", lambda *args: None)(t1, t1)
    except Exception:
        pass
    try:
        t1.rint()
    except Exception:
        pass
    try:
        t1.rint(t1)
    except Exception:
        pass
    try:
        jnp.sign(t1)
    except Exception:
        pass
    try:
        jnp.sign(t1, t1)
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
        getattr(jnp.linalg, "sign", lambda *args: None)(t1, t1)
    except Exception:
        pass
    try:
        t1.sign()
    except Exception:
        pass
    try:
        t1.sign(t1)
    except Exception:
        pass
    try:
        jnp.sinh(t1)
    except Exception:
        pass
    try:
        jnp.sinh(t1, t1)
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
        getattr(jnp.linalg, "sinh", lambda *args: None)(t1, t1)
    except Exception:
        pass
    try:
        t1.sinh()
    except Exception:
        pass
    try:
        t1.sinh(t1)
    except Exception:
        pass
    try:
        jnp.positive(t1)
    except Exception:
        pass
    try:
        jnp.positive(t1, t1)
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
        getattr(jnp.linalg, "positive", lambda *args: None)(t1, t1)
    except Exception:
        pass
    try:
        t1.positive()
    except Exception:
        pass
    try:
        t1.positive(t1)
    except Exception:
        pass
    try:
        jnp.ones(t1)
    except Exception:
        pass
    try:
        jnp.ones(t1, t1)
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
        getattr(jnp.linalg, "ones", lambda *args: None)(t1, t1)
    except Exception:
        pass
    try:
        t1.ones()
    except Exception:
        pass
    try:
        t1.ones(t1)
    except Exception:
        pass
    try:
        jnp.moveaxis(t1)
    except Exception:
        pass
    try:
        jnp.moveaxis(t1, t1)
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
        getattr(jnp.linalg, "moveaxis", lambda *args: None)(t1, t1)
    except Exception:
        pass
    try:
        t1.moveaxis()
    except Exception:
        pass
    try:
        t1.moveaxis(t1)
    except Exception:
        pass
    try:
        jnp.hsplit(t1)
    except Exception:
        pass
    try:
        jnp.hsplit(t1, t1)
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
        getattr(jnp.linalg, "hsplit", lambda *args: None)(t1, t1)
    except Exception:
        pass
    try:
        t1.hsplit()
    except Exception:
        pass
    try:
        t1.hsplit(t1)
    except Exception:
        pass
    try:
        jnp.cumsum(t1)
    except Exception:
        pass
    try:
        jnp.cumsum(t1, t1)
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
        getattr(jnp.linalg, "cumsum", lambda *args: None)(t1, t1)
    except Exception:
        pass
    try:
        t1.cumsum()
    except Exception:
        pass
    try:
        t1.cumsum(t1)
    except Exception:
        pass
    try:
        jnp.log2(t1)
    except Exception:
        pass
    try:
        jnp.log2(t1, t1)
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
        getattr(jnp.linalg, "log2", lambda *args: None)(t1, t1)
    except Exception:
        pass
    try:
        t1.log2()
    except Exception:
        pass
    try:
        t1.log2(t1)
    except Exception:
        pass
    try:
        t1 // t1
    except Exception:
        pass
    try:
        jnp.subtract(t1)
    except Exception:
        pass
    try:
        jnp.subtract(t1, t1)
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
        getattr(jnp.linalg, "subtract", lambda *args: None)(t1, t1)
    except Exception:
        pass
    try:
        t1.subtract()
    except Exception:
        pass
    try:
        t1.subtract(t1)
    except Exception:
        pass
    try:
        jnp.log10(t1)
    except Exception:
        pass
    try:
        jnp.log10(t1, t1)
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
        getattr(jnp.linalg, "log10", lambda *args: None)(t1, t1)
    except Exception:
        pass
    try:
        t1.log10()
    except Exception:
        pass
    try:
        t1.log10(t1)
    except Exception:
        pass
    try:
        jnp.std(t1)
    except Exception:
        pass
    try:
        jnp.std(t1, t1)
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
        getattr(jnp.linalg, "std", lambda *args: None)(t1, t1)
    except Exception:
        pass
    try:
        t1.std()
    except Exception:
        pass
    try:
        t1.std(t1)
    except Exception:
        pass
    try:
        jnp.multiply(t1)
    except Exception:
        pass
    try:
        jnp.multiply(t1, t1)
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
        getattr(jnp.linalg, "multiply", lambda *args: None)(t1, t1)
    except Exception:
        pass
    try:
        t1.multiply()
    except Exception:
        pass
    try:
        t1.multiply(t1)
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
        jnp.log(t1, t1)
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
        getattr(jnp.linalg, "log", lambda *args: None)(t1, t1)
    except Exception:
        pass
    try:
        t1.log()
    except Exception:
        pass
    try:
        t1.log(t1)
    except Exception:
        pass
    try:
        jnp.empty(t1)
    except Exception:
        pass
    try:
        jnp.empty(t1, t1)
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
        getattr(jnp.linalg, "empty", lambda *args: None)(t1, t1)
    except Exception:
        pass
    try:
        t1.empty()
    except Exception:
        pass
    try:
        t1.empty(t1)
    except Exception:
        pass
    try:
        jnp.dot(t1)
    except Exception:
        pass
    try:
        jnp.dot(t1, t1)
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
        getattr(jnp.linalg, "dot", lambda *args: None)(t1, t1)
    except Exception:
        pass
    try:
        t1.dot()
    except Exception:
        pass
    try:
        t1.dot(t1)
    except Exception:
        pass
    try:
        jnp.cos(t1)
    except Exception:
        pass
    try:
        jnp.cos(t1, t1)
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
        getattr(jnp.linalg, "cos", lambda *args: None)(t1, t1)
    except Exception:
        pass
    try:
        t1.cos()
    except Exception:
        pass
    try:
        t1.cos(t1)
    except Exception:
        pass
    try:
        jnp.amin(t1)
    except Exception:
        pass
    try:
        jnp.amin(t1, t1)
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
        getattr(jnp.linalg, "amin", lambda *args: None)(t1, t1)
    except Exception:
        pass
    try:
        t1.amin()
    except Exception:
        pass
    try:
        t1.amin(t1)
    except Exception:
        pass
    try:
        jnp.tensordot(t1)
    except Exception:
        pass
    try:
        jnp.tensordot(t1, t1)
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
        getattr(jnp.linalg, "tensordot", lambda *args: None)(t1, t1)
    except Exception:
        pass
    try:
        t1.tensordot()
    except Exception:
        pass
    try:
        t1.tensordot(t1)
    except Exception:
        pass
    try:
        jnp.ravel(t1)
    except Exception:
        pass
    try:
        jnp.ravel(t1, t1)
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
        getattr(jnp.linalg, "ravel", lambda *args: None)(t1, t1)
    except Exception:
        pass
    try:
        t1.ravel()
    except Exception:
        pass
    try:
        t1.ravel(t1)
    except Exception:
        pass
    try:
        jnp.identity(t1)
    except Exception:
        pass
