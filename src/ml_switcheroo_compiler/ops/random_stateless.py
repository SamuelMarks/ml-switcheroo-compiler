"""Stateless functional pseudo-random number generation (PRNG) operations."""

from __future__ import annotations

import typing
from collections.abc import Sequence

import pydantic

from ml_switcheroo_compiler.backends.registry import get_active_backend
from ml_switcheroo_compiler.core.config import config as core_config
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.shape import TensorOrScalar
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node


class NormalConfig(pydantic.BaseModel):
    """Configuration for normal distribution parameters.

    Attributes:
        mean (float): Mean of the normal distribution.
        stddev (float): Standard deviation of the normal distribution.
    """

    model_config = pydantic.ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    mean: float = 0.0
    stddev: float = 1.0


class UniformConfig(pydantic.BaseModel):
    """Configuration for uniform distribution parameters.

    Attributes:
        minval (float): Lower bound of the uniform interval.
        maxval (Optional[float]): Upper bound of the uniform interval.
    """

    model_config = pydantic.ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    minval: float = 0.0
    maxval: float | None = None


class RandomGenerationConfig(pydantic.BaseModel):
    """Configuration for parameterized truncated normal random number generation.

    Attributes:
        means (float): Mean of the distribution prior to truncation.
        stddevs (float): Standard deviation prior to truncation.
        minvals (float): Minimum truncation bound in standard deviations.
        maxvals (float): Maximum truncation bound in standard deviations.
        name (Optional[str]): Optional operation name.
    """

    model_config = pydantic.ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    means: float = 0.0
    stddevs: float = 1.0
    minvals: float = -2.0
    maxvals: float = 2.0
    name: str | None = None


class Algorithm:
    """Constants for counter-based stateless PRNG algorithms."""

    PHILOX = 1
    THREEFRY = 2
    AUTO_SELECT = 3


def _philox_round(v0: int, v1: int, k0: int) -> tuple[int, int]:
    """Execute a single 32-bit Philox-2x32 mixing round.

    Args:
        v0 (int): First 32-bit counter word.
        v1 (int): Second 32-bit counter word.
        k0 (int): 32-bit key word.

    Returns:
        tuple[int, int]: Mixed 32-bit counter pair (v0_next, v1_next).
    """
    product = (v0 & 0xFFFFFFFF) * 0xD256D193
    hi = (product >> 32) & 0xFFFFFFFF
    lo = product & 0xFFFFFFFF
    return (hi ^ (k0 & 0xFFFFFFFF) ^ (v1 & 0xFFFFFFFF)) & 0xFFFFFFFF, lo


def _flatten_seed_sequence(seq: Sequence[TensorOrScalar]) -> list[int]:
    """Flatten a 1D or 2D sequence of seed items into 32-bit integers.

    Args:
        seq (Sequence[TensorOrScalar]): Input seed sequence.

    Returns:
        list[int]: Flattened integer seed words.
    """
    flat: list[int] = []
    for item in seq:
        if isinstance(item, (list, tuple)):
            flat.extend(int(x) if isinstance(x, (int, float)) else hash(str(x)) for x in item)
        elif isinstance(item, (int, float)):
            flat.append(int(item))
        else:
            flat.append(hash(str(item)))
    return flat


def _extract_seed_pair(seed: TensorOrScalar) -> tuple[int, int]:
    """Extract a deterministic 2-tuple of 32-bit unsigned integers from a seed input.

    Args:
        seed (TensorOrScalar): Input seed tensor, sequence, integer, or string.

    Returns:
        tuple[int, int]: Pair of 32-bit integer state words.
    """
    raw = getattr(seed, "data", seed)
    if isinstance(raw, (int, float)):
        s0 = int(raw) & 0xFFFFFFFF
        return s0, (s0 ^ 0x9E3779B9) & 0xFFFFFFFF
    if isinstance(raw, str):
        h = 0x811C9DC5
        for ch in raw:
            h = ((h ^ ord(ch)) * 0x01000193) & 0xFFFFFFFF
        return h, (h ^ 0x9E3779B9) & 0xFFFFFFFF
    tolist_fn = getattr(raw, "tolist", None)
    if callable(tolist_fn):
        raw = tolist_fn()
    if isinstance(raw, (list, tuple)):
        flat = _flatten_seed_sequence(raw)
        if len(flat) >= 2:
            return flat[0] & 0xFFFFFFFF, flat[1] & 0xFFFFFFFF
        if len(flat) == 1:
            return flat[0] & 0xFFFFFFFF, (flat[0] ^ 0x9E3779B9) & 0xFFFFFFFF
    return 0, 0x9E3779B9


def stateless_random_uniform(
    shape: Sequence[int],
    seed: Tensor,
    minval: float = 0.0,
    maxval: float = 1.0,
    dtype: str | DType = "float32",
) -> TensorOrScalar:
    """Generate random values from a uniform distribution without maintaining state.

    Args:
        shape (Sequence[int]): The shape of the output tensor.
        seed (Tensor): The stateless PRNG seed tensor.
        minval (float): The lower bound.
        maxval (float): The upper bound.
        dtype (str | DType): The output data type.

    Returns:
        TensorOrScalar: The generated uniform random tensor.
    """
    from ml_switcheroo_compiler.ops.binary import add, multiply
    from ml_switcheroo_compiler.ops.creation import full
    from ml_switcheroo_compiler.random.continuous import uniform

    dtype_enum = DType(dtype)
    res = uniform(seed, tuple(shape), dtype_enum)
    res = multiply(res, full(shape, maxval - minval, dtype_enum))
    res = add(res, full(shape, minval, dtype_enum))
    return res


def stateless_random_normal(
    shape: Sequence[int],
    seed: Tensor,
    mean: float = 0.0,
    stddev: float = 1.0,
    dtype: str | DType = "float32",
) -> TensorOrScalar:
    """Generate random values from a normal distribution without maintaining state.

    Args:
        shape (Sequence[int]): The shape of the output tensor.
        seed (Tensor): The stateless PRNG seed tensor.
        mean (float): The mean of the normal distribution.
        stddev (float): The standard deviation of the normal distribution.
        dtype (str | DType): The output data type.

    Returns:
        TensorOrScalar: The generated normal random tensor.
    """
    from ml_switcheroo_compiler.ops.binary import add, multiply
    from ml_switcheroo_compiler.ops.creation import full
    from ml_switcheroo_compiler.random.continuous import normal

    dtype_enum = DType(dtype)
    res = normal(seed, tuple(shape), dtype_enum)
    res = multiply(res, full(shape, stddev, dtype_enum))
    res = add(res, full(shape, mean, dtype_enum))
    return res


def stateless_random_binomial(
    shape: Sequence[int],
    seed: Tensor,
    counts: float | Tensor,
    probabilities: float | Tensor,
    dtype: str | DType = "int32",
) -> TensorOrScalar:
    """Generate random values from a binomial distribution without maintaining state.

    Args:
        shape (Sequence[int]): The shape of the output tensor.
        seed (Tensor): The stateless PRNG seed tensor.
        counts (Union[float, Tensor]): Number of trials.
        probabilities (Union[float, Tensor]): Probability of success.
        dtype (str | DType): The output data type.

    Returns:
        TensorOrScalar: The generated binomial random tensor.
    """
    from ml_switcheroo_compiler.random.distributions_discrete import binomial

    dtype_enum = DType(dtype)
    return binomial(seed, counts, probabilities, tuple(shape), dtype_enum)


def stateless_truncated_normal(
    shape: Sequence[int],
    seed: Tensor,
    mean: float = 0.0,
    stddev: float = 1.0,
    dtype: str | DType = "float32",
) -> TensorOrScalar:
    """Generate random values from a truncated normal distribution without maintaining state.

    Args:
        shape (Sequence[int]): The shape of the output tensor.
        seed (Tensor): The stateless PRNG seed tensor.
        mean (float): The mean of the distribution.
        stddev (float): The standard deviation of the distribution.
        dtype (str | DType): The output data type.

    Returns:
        TensorOrScalar: The generated truncated normal tensor.
    """
    from ml_switcheroo_compiler.ops.binary import add, multiply
    from ml_switcheroo_compiler.ops.creation import full
    from ml_switcheroo_compiler.random.continuous import truncated_normal

    dtype_enum = DType(dtype)
    res = truncated_normal(seed, -2.0, 2.0, tuple(shape), dtype_enum)
    res = multiply(res, full(shape, stddev, dtype_enum))
    res = add(res, full(shape, mean, dtype_enum))
    return res


def stateless_categorical(
    logits: Tensor,
    num_samples: int,
    seed: Tensor,
    dtype: str | DType = "int32",
) -> TensorOrScalar:
    """Generate random samples from a categorical distribution without maintaining state.

    Args:
        logits (Tensor): Logits tensor of shape (batch_size, num_classes).
        num_samples (int): Number of samples to draw.
        seed (Tensor): The stateless PRNG seed tensor.
        dtype (str | DType): The output integer data type.

    Returns:
        TensorOrScalar: Sampled class indices of shape (batch_size, num_samples).
    """
    from ml_switcheroo_compiler.ops import cast as op_cast
    from ml_switcheroo_compiler.random.distributions_discrete import categorical

    dtype_enum = DType(dtype)
    batch_shape = logits.shape[:-1] if logits.shape else ()
    shape = tuple(batch_shape) + (num_samples,)

    res = categorical(seed, logits, axis=-1, shape=shape)
    res = op_cast(res, dtype_enum)
    return typing.cast(TensorOrScalar, res)


def stateless_gamma(
    shape: Sequence[int],
    seed: Tensor,
    alpha: Tensor,
    dtype: str | DType = "float32",
) -> TensorOrScalar:
    """Generate random values from a gamma distribution without maintaining state.

    Args:
        shape (Sequence[int]): The shape of the output tensor.
        seed (Tensor): The stateless PRNG seed tensor.
        alpha (Tensor): Shape parameter alpha.
        dtype (str | DType): The output data type.

    Returns:
        TensorOrScalar: The generated gamma random tensor.
    """
    from ml_switcheroo_compiler.random.continuous import gamma

    dtype_enum = DType(dtype)
    return gamma(seed, alpha, tuple(shape), dtype_enum)


def stateless_beta(
    shape: Sequence[int],
    seed: Tensor,
    alpha: Tensor,
    beta_param: Tensor,
    dtype: str | DType = "float32",
) -> TensorOrScalar:
    """Generate random values from a beta distribution without maintaining state.

    Args:
        shape (Sequence[int]): The shape of the output tensor.
        seed (Tensor): The stateless PRNG seed tensor.
        alpha (Tensor): Alpha concentration parameter.
        beta_param (Tensor): Beta concentration parameter.
        dtype (str | DType): The output data type.

    Returns:
        TensorOrScalar: The generated beta random tensor.
    """
    from ml_switcheroo_compiler.random.continuous import beta

    dtype_enum = DType(dtype)
    return beta(seed, alpha, beta_param, tuple(shape), dtype_enum)


def stateless_shuffle(
    x: Tensor,
    seed: Tensor,
    axis: int = 0,
) -> TensorOrScalar:
    """Shuffle the input tensor randomly along a given axis without maintaining state.

    Args:
        x (Tensor): The input tensor to shuffle.
        seed (Tensor): The stateless PRNG seed tensor.
        axis (int): The axis along which to shuffle.

    Returns:
        TensorOrScalar: The shuffled tensor.
    """
    from ml_switcheroo_compiler.random.transformations import shuffle

    return shuffle(seed, x, axis=axis)


def stateless_parameterized_truncated_normal(
    shape: TensorOrScalar,
    seed: TensorOrScalar,
    config: RandomGenerationConfig | None = None,
) -> TensorOrScalar:
    """Generate random values from a parameterized truncated normal distribution.

    Args:
        shape (TensorOrScalar): Output tensor shape tuple.
        seed (TensorOrScalar): Stateless PRNG seed tensor.
        config (Optional[RandomGenerationConfig]): Distribution configuration.

    Returns:
        TensorOrScalar: Sampled truncated normal tensor.
    """
    cfg = config or RandomGenerationConfig()
    shape_tuple = (int(shape),) if isinstance(shape, int) else tuple(typing.cast("Sequence[int]", shape))
    if core_config.eager_mode:
        k0, k1 = _extract_seed_pair(seed)
        num_elems = 1
        for d in shape_tuple:
            num_elems *= max(1, int(d))
        values: list[float] = []
        c0, c1 = k0, k1
        for idx in range(num_elems):
            c0, c1 = _philox_round(c0 ^ idx, c1, k0 + idx)
            c0, c1 = _philox_round(c0, c1, k1 + idx)
            u = (c0 & 0xFFFFFF) / float(0x1000000)
            z = cfg.minvals + u * (cfg.maxvals - cfg.minvals)
            values.append(cfg.means + z * cfg.stddevs)
        backend = get_active_backend()
        arr = backend.array(values, dtype="float32")
        reshape_fn = getattr(arr, "reshape", None)
        if callable(reshape_fn) and shape_tuple:
            arr = reshape_fn(shape_tuple)
        return Tensor(arr, TensorConfig(shape_tuple, "float32", "cpu"))
    return _emit_shape_node(
        "ParameterizedTruncatedNormal",
        [seed],
        {"means": cfg.means, "stddevs": cfg.stddevs, "minvals": cfg.minvals, "maxvals": cfg.maxvals},
        shape_tuple,
        "float32",
    )


def index_shuffle(index: TensorOrScalar, seed: TensorOrScalar, max_index: TensorOrScalar) -> TensorOrScalar:
    """Compute a stateless Feistel/Philox bijective permutation of `index` within `[0, max_index]`.

    Args:
        index (TensorOrScalar): Input index scalar or tensor in `[0, max_index]`.
        seed (TensorOrScalar): Stateless PRNG seed.
        max_index (TensorOrScalar): Inclusive maximum index bound.

    Returns:
        TensorOrScalar: Deterministically permuted index in `[0, max_index]`.
    """
    if not core_config.eager_mode:
        idx_shape = getattr(index, "shape", ()) if hasattr(index, "shape") else ()
        return _emit_shape_node("IndexShuffle", [index, seed, max_index], {}, idx_shape, "int32")

    raw_max = getattr(max_index, "data", max_index)
    item_fn = getattr(raw_max, "item", None)
    if callable(item_fn):
        raw_max = item_fn()
    elif isinstance(raw_max, (list, tuple)) and raw_max:
        raw_max = raw_max[0]
    m = int(raw_max) if isinstance(raw_max, (int, float)) else 0
    if m <= 0:
        return index

    k0, k1 = _extract_seed_pair(seed)
    modulus = m + 1

    def _permute_scalar(val: int) -> int:
        """Compute deterministic Philox-based modular permutation for a single integer index.

        Args:
            val (int): Input index in `[0, max_index]`.

        Returns:
            int: Permuted index in `[0, max_index]`.
        """
        v0, _ = _philox_round(val, k1, k0)
        offset = v0 % modulus
        return (val + offset) % modulus

    if isinstance(index, int):
        return _permute_scalar(index)

    raw_idx = getattr(index, "data", index)
    tolist_idx = getattr(raw_idx, "tolist", None)
    raw_list = tolist_idx() if callable(tolist_idx) else raw_idx
    if not isinstance(raw_list, (list, tuple)):
        return index
    permuted = [_permute_scalar(int(x)) for x in raw_list]
    backend = get_active_backend()
    arr = backend.array(permuted, dtype="int32")
    return Tensor(arr, TensorConfig(getattr(index, "shape", ()), getattr(index, "dtype", "int32"), getattr(index, "device", "cpu")))


def stateless_fold_in(seed: TensorOrScalar, data: TensorOrScalar) -> TensorOrScalar:
    """Fold integer data into a stateless PRNG seed using Philox mixing.

    Args:
        seed (TensorOrScalar): Input stateless seed tensor or value.
        data (TensorOrScalar): Data value to mix into the seed.

    Returns:
        TensorOrScalar: New mixed stateless PRNG seed.
    """
    if not core_config.eager_mode:
        seed_shape = getattr(seed, "shape", (2,)) if hasattr(seed, "shape") else (2,)
        return _emit_shape_node("StatelessFoldIn", [seed, data], {}, seed_shape, "int64")

    s0, s1 = _extract_seed_pair(seed)
    d0, d1 = _extract_seed_pair(data)
    m0, m1 = _philox_round(s0 ^ d0, s1 ^ d1, 0x9E3779B9)
    if isinstance(seed, Tensor):
        backend = get_active_backend()
        arr = backend.array([m0, m1], dtype="int64")
        return Tensor(arr, TensorConfig((2,), seed.dtype, seed.device))
    if isinstance(seed, str):
        return f"{seed}_{m0:08x}"
    return [m0, m1]


def stateless_split(seed: TensorOrScalar, num: TensorOrScalar = 2) -> TensorOrScalar:
    """Split a stateless PRNG seed into `num` independent seeds.

    Args:
        seed (TensorOrScalar): Input stateless PRNG seed.
        num (TensorOrScalar): Number of sub-seeds to generate.

    Returns:
        TensorOrScalar: Tensor of shape `(num, 2)` containing split seeds.
    """
    if core_config.eager_mode:
        return get_active_backend().execute_op("StatelessSplit", seed, num=num)
    return _emit_shape_node("StatelessSplit", [seed], {"num": num}, (int(typing.cast(int, num)), 2), "int64")


def stateless_poisson(
    shape: Sequence[int],
    seed: Tensor,
    lam: Tensor,
    dtype: str | DType = "int32",
) -> TensorOrScalar:
    """Generate random values from a Poisson distribution without maintaining state.

    Args:
        shape (Sequence[int]): Output tensor shape.
        seed (Tensor): Stateless PRNG seed tensor.
        lam (Tensor): Rate parameter lambda.
        dtype (str | DType): Output data type.

    Returns:
        TensorOrScalar: Sampled Poisson tensor.
    """
    from ml_switcheroo_compiler.random.distributions_discrete import poisson

    dtype_enum = DType(dtype)
    return poisson(seed, lam, tuple(shape), dtype_enum)


__all__ = [
    "Algorithm",
    "NormalConfig",
    "RandomGenerationConfig",
    "UniformConfig",
    "index_shuffle",
    "stateless_beta",
    "stateless_categorical",
    "stateless_fold_in",
    "stateless_gamma",
    "stateless_parameterized_truncated_normal",
    "stateless_poisson",
    "stateless_random_binomial",
    "stateless_random_normal",
    "stateless_random_uniform",
    "stateless_shuffle",
    "stateless_split",
    "stateless_truncated_normal",
]
