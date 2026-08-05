"""Generate random ops module."""

from __future__ import annotations

from ml_switcheroo_compiler.core import dtype as dtypes
from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.random.state import _dispatch_random, _emit_random_node


def randint(key: object, shape: object, minval: object, maxval: object, dtype: object = None) -> object:
    """Sample uniform random integers from a given key.

    Args:
        key (object): The PRNG key.
        shape (object): The target shape.
        minval (object): The minval parameter for the operation.
        maxval (object): The maxval parameter for the operation.
        dtype (object): The target data type.

    Returns:
        object: The computed result.
    """
    dtype = dtype or dtypes.DType.Int32
    return _emit_random_node("RandomRandint", [key], shape, dtype, {"minval": minval, "maxval": maxval})


def bernoulli(key: object, p: object = 0.5, shape: object = None) -> object:
    """Sample Bernoulli random variables from a given key.

    Args:
        key (object): The PRNG key.
        p (object): The p parameter for the operation.
        shape (object): The target shape.

    Returns:
        object: The computed result.
    """
    if shape is None:
        shape = ()
    return _emit_random_node("RandomBernoulli", [key], shape, dtypes.DType.Bool, {"p": p})


def categorical(key: object, logits: object, axis: object = -1, shape: object = None) -> object:
    """Sample categorical random variables from a given key.

    Args:
        key (object): The PRNG key.
        logits (object): The logits parameter for the operation.
        axis (object): The axis along which to perform the operation.
        shape (object): The target shape.

    Returns:
        object: The computed result.
    """
    out_shape = shape or ()
    inputs = [key]
    if isinstance(logits, Tensor):
        inputs.append(logits)
    return _emit_random_node("RandomCategorical", inputs, out_shape, dtypes.DType.Int32, {"axis": axis})


def permutation(key: object, x: object, axis: object = 0, independent: object = False) -> object:
    """Generate random permutation of a sequence.

    Args:
        key (object): The key parameter.
        x (object): The x parameter.
        axis (object): The axis parameter.
        independent (object): The independent parameter.

    Returns:
        object: Result.
    """
    shape = getattr(x, "shape", ())
    dtype = getattr(x, "dtype", None)
    return _emit_random_node("RandomPermutation", [key, x], shape, dtype, {"axis": axis, "independent": independent})


def choice(key: object, a: object, **kwargs: object) -> object:
    """Generate a random sample from a given 1-D array.

    Args:
        key (object): The PRNG key.
        a (object): The input a tensor.
        **kwargs (object): Optional arguments shape, replace, p, axis.

    Returns:
        object: The computed result.
    """
    shape = kwargs.get("shape", ())
    replace = kwargs.get("replace", True)
    p = kwargs.get("p", None)
    axis = kwargs.get("axis", 0)
    inputs = [key, a]
    if isinstance(p, Tensor):
        inputs.append(p)
    return _emit_random_node("RandomChoice", inputs, shape, a.dtype, {"replace": replace, "axis": axis})


def binomial(key: object, n: object, p: object, shape: object = None, dtype: object = None) -> object:
    """Sample binomial random values from a given key.

    Args:
        key (object): The key parameter.
        n (object): The n parameter.
        p (object): The p parameter.
        shape (object): The shape parameter.
        dtype (object): The dtype parameter.

    Returns:
        object: Result.
    """
    if shape is None:
        shape = ()
    dtype = dtype or dtypes.DType.Int32
    return _emit_random_node("RandomBinomial", [key], shape, dtype)


def geometric(*args: object, **kwargs: object) -> object:
    """Evaluate geometric operation.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
        object: Result.
    """
    return _dispatch_random("geometric", *args, **kwargs)


def poisson(key: object, lam: object, shape: object = None, dtype: object = None) -> object:
    """Sample poisson random values from a given key.

    Args:
        key (object): The key parameter.
        lam (object): The lam parameter.
        shape (object): The shape parameter.
        dtype (object): The dtype parameter.

    Returns:
        object: Result.
    """
    if shape is None:
        shape = ()
    dtype = dtype or dtypes.DType.Int32
    return _emit_random_node("RandomPoisson", [key, lam], shape, dtype)


def rademacher(*args: object, **kwargs: object) -> object:
    """Evaluate rademacher operation.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
        object: Result.
    """
    return _dispatch_random("rademacher", *args, **kwargs)


def multinomial(key: object, n: int, pvals: object, shape: object = None) -> object:
    """Sample from multinomial distribution.

    Args:
        key (object): The PRNG key.
        n (int): Number of experiments.
        pvals (object): Probabilities of each of the p different outcomes.
        shape (object): Target shape.

    Returns:
        object: The computed result.
    """
    if shape is None:
        shape = ()
    return _emit_random_node("RandomMultinomial", [key, pvals], shape, dtypes.DType.Int32, {"n": n})
