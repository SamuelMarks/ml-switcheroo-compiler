"""Stateless random operations."""

from typing import Union
from collections.abc import Sequence
from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.core.dtype import DType


def stateless_random_uniform(
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
    from ml_switcheroo_compiler.random.distributions_continuous import uniform
    from ml_switcheroo_compiler.ops.binary import multiply, add
    from ml_switcheroo_compiler.ops.creation import full

    dtype_enum = DType(dtype) if isinstance(dtype, str) else dtype
    res = uniform(seed, tuple(shape), dtype_enum)
    res = multiply(res, full(shape, maxval - minval, dtype_enum))
    res = add(res, full(shape, minval, dtype_enum))
    return res


def stateless_random_normal(
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
    from ml_switcheroo_compiler.random.distributions_continuous import normal
    from ml_switcheroo_compiler.ops.binary import multiply, add
    from ml_switcheroo_compiler.ops.creation import full

    dtype_enum = DType(dtype) if isinstance(dtype, str) else dtype
    res = normal(seed, tuple(shape), dtype_enum)
    res = multiply(res, full(shape, stddev, dtype_enum))
    res = add(res, full(shape, mean, dtype_enum))
    return res


def stateless_random_binomial(
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
    from ml_switcheroo_compiler.random.distributions_discrete import binomial

    dtype_enum = DType(dtype) if isinstance(dtype, str) else dtype
    res = binomial(seed, counts, probabilities, tuple(shape), dtype_enum)
    return res


def stateless_truncated_normal(
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
    from ml_switcheroo_compiler.random.distributions_continuous import truncated_normal
    from ml_switcheroo_compiler.ops.binary import multiply, add
    from ml_switcheroo_compiler.ops.creation import full

    dtype_enum = DType(dtype) if isinstance(dtype, str) else dtype
    lower = -2.0
    upper = 2.0
    res = truncated_normal(seed, lower, upper, tuple(shape), dtype_enum)
    res = multiply(res, full(shape, stddev, dtype_enum))
    res = add(res, full(shape, mean, dtype_enum))
    return res


def stateless_categorical(
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
    from ml_switcheroo_compiler.random.distributions_discrete import categorical
    from ml_switcheroo_compiler.ops import cast

    dtype_enum = DType(dtype) if isinstance(dtype, str) else dtype

    batch_shape = logits.shape[:-1] if logits.shape else ()
    shape = tuple(batch_shape) + (num_samples,)

    res = categorical(seed, logits, axis=-1, shape=shape)
    res = cast(res, dtype_enum)
    return res


def stateless_gamma(
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
    from ml_switcheroo_compiler.random.distributions_continuous import gamma

    dtype_enum = DType(dtype) if isinstance(dtype, str) else dtype
    res = gamma(seed, alpha, tuple(shape), dtype_enum)
    return res
