"""Implementation of distributions discrete operations and utilities for the random subsystem (`ml_switcheroo_compiler.random.distributions_discrete`)."""

from __future__ import annotations

import typing
from collections.abc import Sequence

from ml_switcheroo_compiler.backends.eager_registry import ShapeProtocol
from ml_switcheroo_compiler.core import dtype as dtypes
from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.random.state import RandomArg, _dispatch_random, _emit_random_node


@typing.runtime_checkable
class _RandomTensorProtocol(typing.Protocol):
    """Protocol for tensor-like objects exposing `.shape` and `.dtype`."""

    @property
    def shape(self) -> Sequence[int | str]:
        """Return tensor shape.

        Returns:
            Sequence[int | str]: Shape sequence.
        """
        ...

    @property
    def dtype(self) -> dtypes.DType | str | None:
        """Return tensor dtype.

        Returns:
            dtypes.DType | str | None: Tensor dtype.
        """
        ...


def _extract_int_shape(seq: Sequence[int | str]) -> tuple[int, ...]:
    """Convert a sequence of shape dimensions into a concrete integer tuple.

    Args:
        seq (Sequence[int | str]): Shape sequence.

    Returns:
        tuple[int, ...]: Concrete integer shape tuple.
    """
    return tuple(int(d) for d in seq if isinstance(d, int))


def randint(
    key: Tensor | str | int,
    shape: Sequence[int] | int,
    minval: int | float,
    maxval: int | float,
    dtype: dtypes.DType | str | None = None,
) -> Tensor:
    """Sample uniform random integers from a given key.

    Args:
        key (Tensor | str | int): The PRNG key.
        shape (Sequence[int] | int): The target shape.
        minval (int | float): Inclusive minimum value.
        maxval (int | float): Exclusive maximum value.
        dtype (dtypes.DType | str | None): The target data type.

    Returns:
        Tensor: The computed result.
    """
    dtype = dtype or dtypes.DType.Int32
    return _emit_random_node("RandomRandint", [key], shape, dtype, {"minval": minval, "maxval": maxval})


def bernoulli(
    key: Tensor | str | int,
    p: float | int = 0.5,
    shape: Sequence[int] | int | None = None,
) -> Tensor:
    """Sample Bernoulli random variables from a given key.

    Args:
        key (Tensor | str | int): The PRNG key.
        p (float | int): Success probability parameter.
        shape (Sequence[int] | int | None): The target shape.

    Returns:
        Tensor: The computed result.
    """
    if shape is None:
        shape = ()
    return _emit_random_node("RandomBernoulli", [key], shape, dtypes.DType.Bool, {"p": p})


def categorical(
    key: Tensor | str | int,
    logits: Tensor | Sequence[float],
    axis: int = -1,
    shape: Sequence[int] | int | None = None,
) -> Tensor:
    """Sample categorical random variables from a given key.

    Args:
        key (Tensor | str | int): The PRNG key.
        logits (Tensor | Sequence[float]): Unnormalized log probabilities.
        axis (int): The axis along which to perform the operation.
        shape (Sequence[int] | int | None): The target shape.

    Returns:
        Tensor: The computed result.
    """
    out_shape = shape or ()
    inputs: list[Tensor | str | float | int] = [key]
    if isinstance(logits, Tensor):
        inputs.append(logits)
    return _emit_random_node("RandomCategorical", inputs, out_shape, dtypes.DType.Int32, {"axis": axis})


def permutation(
    key: Tensor | str | int,
    x: Tensor | int,
    axis: int = 0,
    independent: bool = False,
) -> Tensor:
    """Generate random permutation of a sequence.

    Args:
        key (Tensor | str | int): Input `key` argument (key).
        x (Tensor | int): The input tensor or integer upper bound.
        axis (int): Input `axis` argument (axis).
        independent (bool): Whether permutations are independent across slices.

    Returns:
        Tensor: Permuted tensor.
    """
    shape: tuple[int, ...] = _extract_int_shape(x.shape) if isinstance(x, (Tensor, ShapeProtocol, _RandomTensorProtocol)) else ()
    dtype: dtypes.DType | str | None = x.dtype if isinstance(x, (Tensor, _RandomTensorProtocol)) else None
    return _emit_random_node("RandomPermutation", [key, x], shape, dtype, {"axis": axis, "independent": independent})


def choice(
    key: Tensor | str | int,
    a: Tensor,
    **kwargs: Tensor | Sequence[int] | int | bool | None,
) -> Tensor:
    """Generate a random sample from a given 1-D array.

    Args:
        key (Tensor | str | int): The PRNG key.
        a (Tensor): The input tensor.
        **kwargs (Tensor | Sequence[int] | int | bool | None): Optional arguments shape, replace, p, axis.

    Returns:
        Tensor: The computed result.
    """
    raw_shape = kwargs.get("shape", ())
    shape: Sequence[int] | int = raw_shape if isinstance(raw_shape, (Sequence, int)) and not isinstance(raw_shape, (str, bool)) else ()
    replace = bool(kwargs.get("replace", True))
    p = kwargs.get("p", None)
    raw_axis = kwargs.get("axis", 0)
    axis = int(raw_axis) if isinstance(raw_axis, int) else 0
    inputs: list[Tensor | str | float | int] = [key, a]
    if isinstance(p, Tensor):
        inputs.append(p)
    return _emit_random_node("RandomChoice", inputs, shape, a.dtype, {"replace": replace, "axis": axis})


def binomial(
    key: Tensor | str | int,
    n: int | float | Tensor,
    p: float | Tensor,
    shape: Sequence[int] | int | None = None,
    dtype: dtypes.DType | str | None = None,
) -> Tensor:
    """Sample binomial random values from a given key.

    Args:
        key (Tensor | str | int): Input `key` argument (key).
        n (int | float | Tensor): Number of trials.
        p (float | Tensor): Probability of success.
        shape (Sequence[int] | int | None): Input `shape` argument (shape).
        dtype (dtypes.DType | str | None): Input `dtype` argument (dtype).

    Returns:
        Tensor: Sampled tensor.
    """
    _ = (n, p)
    if shape is None:
        shape = ()
    dtype = dtype or dtypes.DType.Int32
    return _emit_random_node("RandomBinomial", [key], shape, dtype)


def geometric(*args: RandomArg, **kwargs: RandomArg) -> Tensor:
    """Execute geometric for compiler processing.

    Args:
        *args (RandomArg): Positional args.
        **kwargs (RandomArg): Keyword args.

    Returns:
        Tensor: Sampled tensor.
    """
    return _dispatch_random("geometric", *args, **kwargs)


def poisson(
    key: Tensor | str | int,
    lam: float | int | Tensor,
    shape: Sequence[int] | int | None = None,
    dtype: dtypes.DType | str | None = None,
) -> Tensor:
    """Sample poisson random values from a given key.

    Args:
        key (Tensor | str | int): Input `key` argument (key).
        lam (float | int | Tensor): Rate parameter lambda.
        shape (Sequence[int] | int | None): Input `shape` argument (shape).
        dtype (dtypes.DType | str | None): Input `dtype` argument (dtype).

    Returns:
        Tensor: Sampled tensor.
    """
    if shape is None:
        shape = ()
    dtype = dtype or dtypes.DType.Int32
    return _emit_random_node("RandomPoisson", [key, lam], shape, dtype)


def rademacher(*args: RandomArg, **kwargs: RandomArg) -> Tensor:
    """Execute rademacher for compiler processing.

    Args:
        *args (RandomArg): Positional args.
        **kwargs (RandomArg): Keyword args.

    Returns:
        Tensor: Sampled tensor.
    """
    return _dispatch_random("rademacher", *args, **kwargs)


def multinomial(
    key: Tensor | str | int,
    n: int,
    pvals: Tensor | float | int,
    shape: Sequence[int] | int | None = None,
) -> Tensor:
    """Sample from multinomial distribution.

    Args:
        key (Tensor | str | int): The PRNG key.
        n (int): Number of experiments.
        pvals (Tensor | float | int): Probabilities of each of the p different outcomes.
        shape (Sequence[int] | int | None): Target shape.

    Returns:
        Tensor: The computed result.
    """
    if shape is None:
        shape = ()
    return _emit_random_node("RandomMultinomial", [key, pvals], shape, dtypes.DType.Int32, {"n": n})
