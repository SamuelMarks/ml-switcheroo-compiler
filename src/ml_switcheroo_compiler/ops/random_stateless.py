"""Stateless random operations."""

from typing import Union  # pragma: no cover
from collections.abc import Sequence  # pragma: no cover
from ml_switcheroo_compiler.core.tensor import Tensor  # pragma: no cover
from ml_switcheroo_compiler.core.dtype import DType  # pragma: no cover


def stateless_random_uniform(  # pragma: no cover
    shape: Sequence[int],
    seed: Tensor,
    minval: float = 0.0,
    maxval: float = 1.0,
    dtype: str = "float32",
) -> Tensor:
    """Stateless random uniform distribution.

    Args:
        shape (Sequence[int]): The shape of the output tensor.
        seed (Tensor): The seed tensor.
        minval (float): The lower bound.
        maxval (float): The upper bound.
        dtype (str): The data type.

    Returns:
        Tensor: The generated tensor.
    """
    from ml_switcheroo_compiler.random.distributions_continuous import uniform  # pragma: no cover
    from ml_switcheroo_compiler.ops.binary import multiply, add  # pragma: no cover
    from ml_switcheroo_compiler.ops.creation import full  # pragma: no cover

    dtype_enum = DType(dtype) if isinstance(dtype, str) else dtype  # pragma: no cover
    res = uniform(seed, tuple(shape), dtype_enum)  # pragma: no cover
    res = multiply(res, full(shape, maxval - minval, dtype_enum))  # pragma: no cover
    res = add(res, full(shape, minval, dtype_enum))  # pragma: no cover
    return res  # pragma: no cover


def stateless_random_normal(  # pragma: no cover
    shape: Sequence[int],
    seed: Tensor,
    mean: float = 0.0,
    stddev: float = 1.0,
    dtype: str = "float32",
) -> Tensor:
    """Stateless random normal distribution.

    Args:
        shape (Sequence[int]): The shape of the output tensor.
        seed (Tensor): The seed tensor.
        mean (float): The mean.
        stddev (float): The standard deviation.
        dtype (str): The data type.

    Returns:
        Tensor: The generated tensor.
    """
    from ml_switcheroo_compiler.random.distributions_continuous import normal  # pragma: no cover
    from ml_switcheroo_compiler.ops.binary import multiply, add  # pragma: no cover
    from ml_switcheroo_compiler.ops.creation import full  # pragma: no cover

    dtype_enum = DType(dtype) if isinstance(dtype, str) else dtype  # pragma: no cover
    res = normal(seed, tuple(shape), dtype_enum)  # pragma: no cover
    res = multiply(res, full(shape, stddev, dtype_enum))  # pragma: no cover
    res = add(res, full(shape, mean, dtype_enum))  # pragma: no cover
    return res  # pragma: no cover


def stateless_random_binomial(  # pragma: no cover
    shape: Sequence[int],
    seed: Tensor,
    counts: Union[float, Tensor],
    probabilities: Union[float, Tensor],
    dtype: str = "int32",
) -> Tensor:
    """Stateless random binomial distribution.

    Args:
        shape (Sequence[int]): The shape of the output tensor.
        seed (Tensor): The seed tensor.
        counts (Union[float, Tensor]): Number of trials.
        probabilities (Union[float, Tensor]): Probability of success.
        dtype (str): The data type.

    Returns:
        Tensor: The generated tensor.
    """
    from ml_switcheroo_compiler.random.distributions_discrete import binomial  # pragma: no cover

    dtype_enum = DType(dtype) if isinstance(dtype, str) else dtype  # pragma: no cover
    res = binomial(seed, counts, probabilities, tuple(shape), dtype_enum)  # pragma: no cover
    return res  # pragma: no cover


def stateless_truncated_normal(  # pragma: no cover
    shape: Sequence[int],
    seed: Tensor,
    mean: float = 0.0,
    stddev: float = 1.0,
    dtype: str = "float32",
) -> Tensor:
    """Stateless random truncated normal distribution.

    Args:
        shape (Sequence[int]): The shape of the output tensor.
        seed (Tensor): The seed tensor.
        mean (float): The mean.
        stddev (float): The standard deviation.
        dtype (str): The data type.

    Returns:
        Tensor: The generated tensor.
    """
    from ml_switcheroo_compiler.random.distributions_continuous import (
        truncated_normal,
    )  # pragma: no cover
    from ml_switcheroo_compiler.ops.binary import multiply, add  # pragma: no cover
    from ml_switcheroo_compiler.ops.creation import full  # pragma: no cover

    dtype_enum = DType(dtype) if isinstance(dtype, str) else dtype  # pragma: no cover
    lower = -2.0  # pragma: no cover
    upper = 2.0  # pragma: no cover
    res = truncated_normal(seed, lower, upper, tuple(shape), dtype_enum)  # pragma: no cover
    res = multiply(res, full(shape, stddev, dtype_enum))  # pragma: no cover
    res = add(res, full(shape, mean, dtype_enum))  # pragma: no cover
    return res  # pragma: no cover


def stateless_categorical(  # pragma: no cover
    logits: Tensor,
    num_samples: int,
    seed: Tensor,
    dtype: str = "int32",
) -> Tensor:
    """Stateless random categorical distribution.

    Args:
        logits (Tensor): The logits tensor (batch_size, num_classes).
        num_samples (int): Number of samples to draw.
        seed (Tensor): The seed tensor.
        dtype (str): The data type.

    Returns:
        Tensor: The generated tensor (batch_size, num_samples).
    """
    from ml_switcheroo_compiler.random.distributions_discrete import categorical  # pragma: no cover
    from ml_switcheroo_compiler.ops import cast  # pragma: no cover

    dtype_enum = DType(dtype) if isinstance(dtype, str) else dtype  # pragma: no cover

    batch_shape = logits.shape[:-1] if logits.shape else ()  # pragma: no cover
    shape = tuple(batch_shape) + (num_samples,)  # pragma: no cover

    res = categorical(seed, logits, axis=-1, shape=shape)  # pragma: no cover
    res = cast(res, dtype_enum)  # pragma: no cover
    return res  # pragma: no cover


def stateless_gamma(  # pragma: no cover
    shape: Sequence[int],
    seed: Tensor,
    alpha: Tensor,
    dtype: str = "float32",
) -> Tensor:
    """Stateless random gamma distribution.

    Args:
        shape (Sequence[int]): The shape of the output tensor.
        seed (Tensor): The seed tensor.
        alpha (Tensor): The alpha parameter.
        dtype (str): The data type.

    Returns:
        Tensor: The generated tensor.
    """
    from ml_switcheroo_compiler.random.distributions_continuous import gamma  # pragma: no cover

    dtype_enum = DType(dtype) if isinstance(dtype, str) else dtype  # pragma: no cover
    res = gamma(seed, alpha, tuple(shape), dtype_enum)  # pragma: no cover
    return res  # pragma: no cover
