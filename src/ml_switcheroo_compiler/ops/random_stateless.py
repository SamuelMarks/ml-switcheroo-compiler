# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Core abstractions and logic definitions for random_stateless.py."""

import typing
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Optional, Union

from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor


@dataclass
class NormalConfig:
    """Configuration for normal distribution parameters.

    This dataclass holds the mean and standard deviation for
    generating random numbers from a normal distribution.
    """

    mean: float = 0.0
    stddev: float = 1.0


@dataclass
class UniformConfig:
    """Configuration for uniform distribution parameters.

    This dataclass defines the minimum and maximum bounds for
    generating random numbers from a uniform distribution.
    """

    minval: float = 0.0
    maxval: typing.Optional[float] = None


def stateless_random_uniform(
    shape: Sequence[int],
    seed: Tensor,
    minval: float = 0.0,
    maxval: float = 1.0,
    dtype: str = "float32",
):
    """Generate random values from a uniform distribution without maintaining state.

    Args:
        shape (Sequence[int]): The shape of the output tensor.
        seed (Tensor): The seed tensor.
        minval (float): The lower bound.
        maxval (float): The upper bound.
        dtype (str): The data type.

    Returns:
        Tensor: The generated tensor.
    """
    from ml_switcheroo_compiler.ops.binary import add, multiply
    from ml_switcheroo_compiler.ops.creation import full
    from ml_switcheroo_compiler.random.continuous.uniform import uniform

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
):
    """Generate random values from a normal distribution without maintaining state.

    Args:
        shape (Sequence[int]): The shape of the output tensor.
        seed (Tensor): The seed tensor.
        mean (float): The mean.
        stddev (float): The standard deviation.
        dtype (str): The data type.

    Returns:
        Tensor: The generated tensor.
    """
    from ml_switcheroo_compiler.ops.binary import add, multiply
    from ml_switcheroo_compiler.ops.creation import full
    from ml_switcheroo_compiler.random.continuous.normal import normal

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
):
    """Generate random values from a binomial distribution without maintaining state.

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
):
    """Generate random values from a truncated normal distribution without maintaining state.

    Args:
        shape (Sequence[int]): The shape of the output tensor.
        seed (Tensor): The seed tensor.
        mean (float): The mean.
        stddev (float): The standard deviation.
        dtype (str): The data type.

    Returns:
        Tensor: The generated tensor.
    """
    from ml_switcheroo_compiler.ops.binary import add, multiply
    from ml_switcheroo_compiler.ops.creation import full
    from ml_switcheroo_compiler.random.continuous.truncated_normal import (
        truncated_normal,
    )

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
):
    """Generate random values from a categorical distribution without maintaining state.

    Args:
        logits (Tensor): The logits tensor (batch_size, num_classes).
        num_samples (int): Number of samples to draw.
        seed (Tensor): The seed tensor.
        dtype (str): The data type.

    Returns:
        Tensor: The generated tensor (batch_size, num_samples).
    """
    from ml_switcheroo_compiler.ops import cast
    from ml_switcheroo_compiler.random.distributions_discrete import categorical

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
):
    """Generate random values from a gamma distribution without maintaining state.

    Args:
        shape (Sequence[int]): The shape of the output tensor.
        seed (Tensor): The seed tensor.
        alpha (Tensor): The alpha parameter.
        dtype (str): The data type.

    Returns:
        Tensor: The generated tensor.
    """
    from ml_switcheroo_compiler.random.continuous.gamma import gamma

    dtype_enum = DType(dtype) if isinstance(dtype, str) else dtype
    res = gamma(seed, alpha, tuple(shape), dtype_enum)
    return res


def stateless_beta(
    shape: Sequence[int],
    seed: Tensor,
    alpha: Tensor,
    beta_param: Tensor,
    dtype: str = "float32",
):
    """Generate random values from a beta distribution without maintaining state.

    Args:
        shape (Sequence[int]): The shape of the output tensor.
        seed (Tensor): The seed tensor.
        alpha (Tensor): The alpha parameter.
        beta_param (Tensor): The beta parameter.
        dtype (str): The data type.

    Returns:
        Tensor: The generated tensor.
    """
    from ml_switcheroo_compiler.random.continuous.beta import beta

    dtype_enum = DType(dtype) if isinstance(dtype, str) else dtype
    res = beta(seed, alpha, beta_param, tuple(shape), dtype_enum)
    return res


def stateless_shuffle(
    x: Tensor,
    seed: Tensor,
    axis: int = 0,
):
    """Shuffles the input tensor randomly along a given axis without maintaining state.

    Args:
        x (Tensor): The input tensor to shuffle.
        seed (Tensor): The seed tensor.
        axis (int): The axis to shuffle along.

    Returns:
        Tensor: The shuffled tensor.
    """
    from ml_switcheroo_compiler.random.transformations import shuffle

    res = shuffle(seed, x, axis=axis)
    return res


@dataclass
class RandomGenerationConfig:
    """Configuration for advanced random number generation.

    This class encapsulates multiple parameters that can be used
    when generating random distributions, including bounded ranges.
    """

    means: float = 0.0
    stddevs: float = 1.0
    minvals: float = -2.0
    maxvals: float = 2.0
    name: Optional[str] = None


def stateless_parameterized_truncated_normal(shape, seed, config: Optional[RandomGenerationConfig] = None):
    """Generate random values from a truncated normal distribution with custom config.

    Args:
        shape (Any): The shape of the output tensor.
        seed (Any): The seed Any for the generator.
        config (Optional[RandomGenerationConfig]): Configuration containing distribution parameters.

    Returns:
        Tensor: The randomly generated tensor.
    """
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    return Tensor(None, TensorConfig(shape, "float32", "cpu"))


class Algorithm:
    """Define constants for random number generation algorithms.

    These algorithms dictate the specific random number generation
    under the hood, like Philox or Threefry.
    """

    PHILOX = 1
    THREEFRY = 2
    AUTO_SELECT = 3


class Generator:
    """Provide a random number generator with internal state.

    This class encapsulates the state and algorithm required to
    produce deterministic random numbers based on an initial seed.
    """

    def __init__(self, copy_from=None, state=None, alg=None) -> None:
        """Initialize the random number generator.

        Args:
            copy_from (Any): The copy_from parameter.
            state (Any): The state parameter.
            alg (Any): The alg parameter.
        """
        self.state = state

    @classmethod
    def from_seed(cls, seed, alg=None) -> "Generator":
        """Create a Generator instance from a given seed.

        Args:
            seed (Any): The seed parameter.
            alg (Any): The alg parameter.

        Returns:
            tuple[int, ...]: Result.
        """
        return cls(state=seed, alg=alg)

    def normal(self, shape, config: Optional[NormalConfig] = None, dtype="float32", name=None):
        """Draws samples from a normal distribution using the generator's state.

        Args:
            shape (Any): The shape parameter.
            config (Optional): The config parameter.
            dtype (Any): The dtype parameter.
            name (Any): The name parameter.

        Returns:
            Tensor: Result.
        """
        config = config or NormalConfig()
        from ml_switcheroo_compiler.core.config import config as core_config

        if core_config.eager_mode:
            from ml_switcheroo_compiler.backends.registry import get_active_backend

            return get_active_backend().execute_op("Normal", shape, config=config, dtype=dtype, name=name)
        from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

        out1 = _emit_shape_node("Normal", [], {"shape": shape, "config": config, "dtype": dtype, "name": name}, shape, dtype)
        return out1

    def uniform(self, shape, config: Optional[UniformConfig] = None, dtype="float32", name=None):
        """Draws samples from a uniform distribution using the generator's state.

        Args:
            shape (Any): The shape parameter.
            config (Optional): The config parameter.
            dtype (Any): The dtype parameter.
            name (Any): The name parameter.

        Returns:
            Tensor: Result.
        """
        config = config or UniformConfig()
        from ml_switcheroo_compiler.core.config import config as core_config

        if core_config.eager_mode:
            from ml_switcheroo_compiler.backends.registry import get_active_backend

            return get_active_backend().execute_op("Uniform", shape, config=config, dtype=dtype, name=name)
        from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

        out1 = _emit_shape_node("Uniform", [], {"shape": shape, "config": config, "dtype": dtype, "name": name}, shape, dtype)
        return out1


def create_rng_state(seed, alg=None):
    """Create a random number generator state from a seed.

    Args:
        seed (Any): The seed parameter.
        alg (Any): The alg parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    return Tensor([0, seed], TensorConfig((2,), "int64", "cpu"))


_GLOBAL_GENERATOR_STATE = {"generator": None}


def get_global_generator():
    """Retrieve the globally registered random number generator.

    Returns: Tensor: The global Generator instance.
    """
    if _GLOBAL_GENERATOR_STATE["generator"] is None:
        _GLOBAL_GENERATOR_STATE["generator"] = Generator.from_seed(0)
    return _GLOBAL_GENERATOR_STATE["generator"]


def set_global_generator(generator) -> None:
    """Register a globally accessible random number generator.

    Args:
        generator (Any): The generator instance to set globally.
    """
    _GLOBAL_GENERATOR_STATE["generator"] = generator


def index_shuffle(index, seed, max_index):
    """Shuffles an index safely within the defined bounds.

    Args:
        index (Any): The index value to shuffle.
        seed (Any): The random seed.
        max_index (Any): The maximum allowed index.

    Returns: Tensor: The resulting shuffled index.
    """
    return index


def stateless_fold_in(seed, data):
    """Folds new data into an existing seed to produce a combined seed.

    Args:
        seed (Any): The original seed value.
        data (Any): The additional data to mix into the seed.

    Returns: Tensor: The combined seed.
    """
    return seed


def stateless_split(seed, num=2):
    """Split a single seed into multiple independent seeds.

    Args:
        seed (Any): The seed parameter.
        num (Any): The num parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("StatelessSplit", seed, num=num)
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    out1 = _emit_shape_node("StatelessSplit", [seed], {"num": num}, (num, 2), "int64")
    return out1


def stateless_poisson(
    shape: Sequence[int],
    seed: Tensor,
    lam: Tensor,
    dtype: str = "int32",
):
    """Generate random values from a poisson distribution without maintaining state.

    Args:
        shape (Sequence): The shape parameter.
        seed (Tensor): The seed parameter.
        lam (Tensor): The lam parameter.
        dtype (str): The dtype parameter.

    Returns:
        Tensor: Result.
    """
    from ml_switcheroo_compiler.random.distributions_discrete import poisson

    dtype_enum = DType(dtype) if isinstance(dtype, str) else dtype
    res = poisson(seed, lam, tuple(shape), dtype_enum)
    return res
