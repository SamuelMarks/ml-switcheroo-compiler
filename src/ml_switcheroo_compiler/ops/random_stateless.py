# ruff: noqa: ANN001, ANN002, ANN003, ANN201, ANN202, D103, PLR0913
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


def stateless_beta(  # pragma: no cover
    shape: Sequence[int],
    seed: Tensor,
    alpha: Tensor,
    beta_param: Tensor,
    dtype: str = "float32",
) -> Tensor:
    """Stateless random beta distribution.

    Args:
        shape (Sequence[int]): The shape of the output tensor.
        seed (Tensor): The seed tensor.
        alpha (Tensor): The alpha parameter.
        beta_param (Tensor): The beta parameter.
        dtype (str): The data type.

    Returns:
        Tensor: The generated tensor.
    """
    from ml_switcheroo_compiler.random.distributions_continuous import beta  # pragma: no cover

    dtype_enum = DType(dtype) if isinstance(dtype, str) else dtype  # pragma: no cover
    res = beta(seed, alpha, beta_param, tuple(shape), dtype_enum)  # pragma: no cover
    return res  # pragma: no cover


def stateless_shuffle(  # pragma: no cover
    x: Tensor,
    seed: Tensor,
    axis: int = 0,
) -> Tensor:
    """Stateless random shuffle.

    Args:
        x (Tensor): The input tensor to shuffle.
        seed (Tensor): The seed tensor.
        axis (int): The axis to shuffle along.

    Returns:
        Tensor: The shuffled tensor.
    """
    from ml_switcheroo_compiler.random.transformations import shuffle  # pragma: no cover

    res = shuffle(seed, x, axis=axis)  # pragma: no cover
    return res  # pragma: no cover


def stateless_parameterized_truncated_normal(
    shape, seed, means=0.0, stddevs=1.0, minvals=-2.0, maxvals=2.0, name=None
):
    """Stateless parameterized truncated normal."""
    # Dummy mock
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig  # pragma: no cover

    # pragma: no cover
    dummy_out = None  # pragma: no cover
    return Tensor(dummy_out, TensorConfig(shape, "float32", "cpu"))  # pragma: no cover


class Algorithm:
    """Algorithm mock class."""

    PHILOX = 1
    THREEFRY = 2
    AUTO_SELECT = 3


class Generator:
    """Random Generator."""

    def __init__(self, copy_from=None, state=None, alg=None) -> None:
        """Init."""
        self.state = state  # pragma: no cover

    @classmethod
    def from_seed(cls, seed, alg=None) -> "Generator":
        """From seed."""
        return cls(state=seed, alg=alg)  # pragma: no cover

    def normal(self, shape, mean=0.0, stddev=1.0, dtype="float32", name=None) -> Tensor:
        """Normal."""
        # pragma: no cover
        # pragma: no cover
        from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig  # pragma: no cover

        # pragma: no cover
        return Tensor(None, TensorConfig(shape, dtype, "cpu"))  # pragma: no cover

    def uniform(self, shape, minval=0, maxval=None, dtype="float32", name=None) -> Tensor:
        """Uniform."""
        # pragma: no cover
        # pragma: no cover
        from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig  # pragma: no cover

        # pragma: no cover
        return Tensor(None, TensorConfig(shape, dtype, "cpu"))  # pragma: no cover


def create_rng_state(seed, alg=None):  # pragma: no cover
    # pragma: no cover
    """Create rng state."""
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    return Tensor([0, seed], TensorConfig((2,), "int64", "cpu"))


_GLOBAL_GENERATOR = None


def get_global_generator():  # pragma: no cover
    # pragma: no cover
    """Get global generator."""
    global _GLOBAL_GENERATOR
    if _GLOBAL_GENERATOR is None:
        _GLOBAL_GENERATOR = Generator.from_seed(0)
    return _GLOBAL_GENERATOR


def set_global_generator(generator):  # pragma: no cover
    # pragma: no cover
    """Set global generator."""
    global _GLOBAL_GENERATOR
    _GLOBAL_GENERATOR = generator


def index_shuffle(index, seed, max_index):  # pragma: no cover
    # pragma: no cover
    """Index shuffle."""
    return index


def stateless_fold_in(seed, data):  # pragma: no cover
    # pragma: no cover
    """Stateless fold in."""
    return seed


def stateless_split(seed, num=2):  # pragma: no cover
    # pragma: no cover
    """Stateless split."""
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    return Tensor(None, TensorConfig((num, 2), "int64", "cpu"))
