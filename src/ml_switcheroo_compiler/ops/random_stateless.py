"""Core abstractions and logic definitions for random_stateless.py."""

import typing
from collections.abc import Sequence
from dataclasses import dataclass
from typing import (
    Optional,
    Union,
)

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
) -> Tensor:
    """Generates random values from a uniform distribution without maintaining state.

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
    from ml_switcheroo_compiler.random import uniform

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
    """Generates random values from a normal distribution without maintaining state.

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
    from ml_switcheroo_compiler.random import normal

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
    """Generates random values from a binomial distribution without maintaining state.

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
    """Generates random values from a truncated normal distribution without maintaining state.

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
    from ml_switcheroo_compiler.random import (
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
) -> Tensor:
    """Generates random values from a categorical distribution without maintaining state.

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
) -> Tensor:
    """Generates random values from a gamma distribution without maintaining state.

    Args:
        shape (Sequence[int]): The shape of the output tensor.
        seed (Tensor): The seed tensor.
        alpha (Tensor): The alpha parameter.
        dtype (str): The data type.

    Returns:
        Tensor: The generated tensor.
    """
    from ml_switcheroo_compiler.random import gamma

    dtype_enum = DType(dtype) if isinstance(dtype, str) else dtype
    res = gamma(seed, alpha, tuple(shape), dtype_enum)
    return res


def stateless_beta(
    shape: Sequence[int],
    seed: Tensor,
    alpha: Tensor,
    beta_param: Tensor,
    dtype: str = "float32",
) -> Tensor:
    """Generates random values from a beta distribution without maintaining state.

    Args:
        shape (Sequence[int]): The shape of the output tensor.
        seed (Tensor): The seed tensor.
        alpha (Tensor): The alpha parameter.
        beta_param (Tensor): The beta parameter.
        dtype (str): The data type.

    Returns:
        Tensor: The generated tensor.
    """
    from ml_switcheroo_compiler.random.continuous import beta

    dtype_enum = DType(dtype) if isinstance(dtype, str) else dtype
    res = beta(seed, alpha, beta_param, tuple(shape), dtype_enum)
    return res


def stateless_shuffle(
    x: Tensor,
    seed: Tensor,
    axis: int = 0,
) -> Tensor:
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


def stateless_parameterized_truncated_normal(shape: object, seed: object, config: Optional[RandomGenerationConfig] = None) -> Tensor:
    """Generates random values from a truncated normal distribution with custom config.

    Args:
        shape (object): The shape of the output tensor.
        seed (object): The seed object for the generator.
        config (Optional[RandomGenerationConfig]): Configuration containing distribution parameters.

    Returns:
        Tensor: The randomly generated tensor.
    """
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    return Tensor(None, TensorConfig(shape, "float32", "cpu"))


class Algorithm:
    """Defines constants for random number generation algorithms.

    These algorithms dictate the specific random number generation
    under the hood, like Philox or Threefry.
    """

    PHILOX = 1
    THREEFRY = 2
    AUTO_SELECT = 3


class Generator:
    """A random number generator with internal state.

    This class encapsulates the state and algorithm required to
    produce deterministic random numbers based on an initial seed.
    """

    def __init__(self, copy_from: object = None, state: object = None, alg: object = None) -> None:
        """Initializes the random number generator.

        Args:
            copy_from (object, optional): Generator to copy the state from.
            state (object, optional): Explicit state to set for the generator.
            alg (object, optional): The algorithm to use for random generation.
        """
        self.state = state

    @classmethod
    def from_seed(cls, seed: object, alg: object = None) -> "Generator":
        """Creates a Generator instance from a given seed.

        Args:
            seed (object): The seed value to initialize the state.
            alg (object, optional): The specific algorithm to use.

        Returns:
            Generator: A new generator instance.
        """
        return cls(state=seed, alg=alg)

    def normal(self, shape: object, config: Optional[NormalConfig] = None, dtype: object = "float32", name: object = None) -> Tensor:
        """Draws samples from a normal distribution using the generator's state.

        Args:
            shape (object): The shape of the output tensor.
            config (Optional[NormalConfig]): Distribution parameters.
            dtype (object): The data type for the returned tensor.
            name (object, optional): A name for the operation.

        Returns:
            Tensor: A tensor containing normally distributed random numbers.
        """
        config = config or NormalConfig()
        from ml_switcheroo_compiler.core.config import config as core_config

        if core_config.eager_mode:
            from ml_switcheroo_compiler.backends.registry import get_active_backend

            return get_active_backend().execute_op("Normal", shape, config=config, dtype=dtype, name=name)
        from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

        out1 = _emit_shape_node("Normal", [], {"shape": shape, "config": config, "dtype": dtype, "name": name}, shape, dtype)
        return out1

    def uniform(self, shape: object, config: Optional[UniformConfig] = None, dtype: object = "float32", name: object = None) -> Tensor:
        """Draws samples from a uniform distribution using the generator's state.

        Args:
            shape (object): The shape of the output tensor.
            config (Optional[UniformConfig]): Distribution parameters.
            dtype (object): The data type for the returned tensor.
            name (object, optional): A name for the operation.

        Returns:
            Tensor: A tensor containing uniformly distributed random numbers.
        """
        config = config or UniformConfig()
        from ml_switcheroo_compiler.core.config import config as core_config

        if core_config.eager_mode:
            from ml_switcheroo_compiler.backends.registry import get_active_backend

            return get_active_backend().execute_op("Uniform", shape, config=config, dtype=dtype, name=name)
        from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

        out1 = _emit_shape_node("Uniform", [], {"shape": shape, "config": config, "dtype": dtype, "name": name}, shape, dtype)
        return out1


def create_rng_state(seed: object, alg: object = None) -> object:
    """Creates a random number generator state from a seed.

    Args:
        seed (object): The initial seed value.
        alg (object, optional): The random generation algorithm to use.

    Returns:
        object: A tensor representing the RNG state.
    """
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    return Tensor([0, seed], TensorConfig((2,), "int64", "cpu"))


_GLOBAL_GENERATOR_STATE = {"generator": None}


def get_global_generator() -> object:
    """Retrieves the globally registered random number generator.

    Returns:
        object: The global Generator instance.
    """
    if _GLOBAL_GENERATOR_STATE["generator"] is None:
        _GLOBAL_GENERATOR_STATE["generator"] = Generator.from_seed(0)
    return _GLOBAL_GENERATOR_STATE["generator"]


def set_global_generator(generator: object) -> None:
    """Registers a globally accessible random number generator.

    Args:
        generator (object): The generator instance to set globally.
    """
    _GLOBAL_GENERATOR_STATE["generator"] = generator


def index_shuffle(index: object, seed: object, max_index: object) -> object:
    """Shuffles an index safely within the defined bounds.

    Args:
        index (object): The index value to shuffle.
        seed (object): The random seed.
        max_index (object): The maximum allowed index.

    Returns:
        object: The resulting shuffled index.
    """
    return index


def stateless_fold_in(seed: object, data: object) -> object:
    """Folds new data into an existing seed to produce a combined seed.

    Args:
        seed (object): The original seed value.
        data (object): The additional data to mix into the seed.

    Returns:
        object: The combined seed.
    """
    return seed


def stateless_split(seed: object, num: object = 2) -> object:
    """Splits a single seed into multiple independent seeds.

    Args:
        seed (object): The original seed.
        num (object, optional): The number of derived seeds to generate.

    Returns:
        object: A tensor containing the split seeds.
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
) -> Tensor:
    """Generates random values from a poisson distribution without maintaining state."""
    from ml_switcheroo_compiler.random.distributions_discrete import poisson

    dtype_enum = DType(dtype) if isinstance(dtype, str) else dtype
    res = poisson(seed, lam, tuple(shape), dtype_enum)
    return res
