"""Random Operations (Functional & Stateful)."""

from typing import Optional, Union
from collections.abc import Sequence
import uuid
import numpy as np
from ml_switcheroo.core.tensor import Tensor
from ml_switcheroo.core.dtype import DType
from ml_switcheroo.core.config import config
from ml_switcheroo.tracing import _tracer, ProxyTensor
from ml_switcheroo_ir import LogicalNode

_GLOBAL_SEED = 0
_GLOBAL_RNG = np.random.default_rng(_GLOBAL_SEED)


def seed(seed: int) -> None:
    """Sets the global random seed. Used primarily as a stateful fallback."""
    global _GLOBAL_SEED, _GLOBAL_RNG
    _GLOBAL_SEED = seed
    _GLOBAL_RNG = np.random.default_rng(seed)


def PRNGKey(seed: int) -> Tensor:
    """Creates a pseudo-random number generator (PRNG) key given an integer seed."""
    shape = (2,)
    dtype = (
        DType.UInt8
    )  # usually represented as uint32 array, but plan uses Int64 or similar

    if config.eager_mode:
        data = np.array([seed, 0], dtype=np.uint32)
        return Tensor(data, shape, dtype, config.default_device)
    else:
        if not _tracer.is_tracing:
            raise RuntimeError("Cannot emit PRNGKey node outside of tracing context.")
        out_id = str(uuid.uuid4())
        node = LogicalNode(
            id=out_id,
            op_type="PRNGKey",
            inputs=[],
            attributes={"seed": seed},
            shape_metadata=shape,
        )
        _tracer.add_node(node)
        proxy = ProxyTensor(id=out_id, shape=shape, dtype=dtype.value)
        return Tensor(
            data=proxy, shape=shape, dtype=dtype, device=config.default_device
        )


def split(key: Tensor, num: int = 2) -> Tensor:
    """Splits a PRNG key into `num` new keys. Returns an array of keys."""
    shape = (num, 2)
    dtype = key.dtype

    if config.eager_mode:
        # Simple simulated split for eager
        rng = np.random.default_rng(
            key.data.item() if key.data.ndim == 0 else key.data[0]
        )
        data = rng.integers(0, 2**32 - 1, size=(num, 2), dtype=np.uint32)
        return Tensor(data, shape, dtype, key.device)
    else:
        if not _tracer.is_tracing:
            raise RuntimeError("Cannot emit PRNGSplit node outside of tracing context.")
        out_id = str(uuid.uuid4())
        node = LogicalNode(
            id=out_id,
            op_type="PRNGSplit",
            inputs=[key.data.id],
            attributes={"num": num},
            shape_metadata=shape,
        )
        _tracer.add_node(node)
        proxy = ProxyTensor(id=out_id, shape=shape, dtype=dtype.value)
        return Tensor(data=proxy, shape=shape, dtype=dtype, device=key.device)


def fold_in(key: Tensor, data: int) -> Tensor:
    """Folds in an integer value into a PRNG key to derive a new PRNG key."""
    shape = (2,)
    dtype = key.dtype

    if config.eager_mode:
        rng = np.random.default_rng(
            key.data.item() if key.data.ndim == 0 else key.data[0] + data
        )
        new_data = rng.integers(0, 2**32 - 1, size=(2,), dtype=np.uint32)
        return Tensor(new_data, shape, dtype, key.device)
    else:
        if not _tracer.is_tracing:
            raise RuntimeError(
                "Cannot emit PRNGFoldIn node outside of tracing context."
            )
        out_id = str(uuid.uuid4())
        node = LogicalNode(
            id=out_id,
            op_type="PRNGFoldIn",
            inputs=[key.data.id],
            attributes={"data": data},
            shape_metadata=shape,
        )
        _tracer.add_node(node)
        proxy = ProxyTensor(id=out_id, shape=shape, dtype=dtype.value)
        return Tensor(data=proxy, shape=shape, dtype=dtype, device=key.device)


def uniform(
    key: Tensor,
    shape: Sequence[int],
    dtype: DType = DType.Float32,
    minval: float = 0.0,
    maxval: float = 1.0,
) -> Tensor:
    """Samples uniformly from the interval `[minval, maxval)`."""
    if config.eager_mode:
        rng = np.random.default_rng(
            key.data.item() if key.data.ndim == 0 else key.data[0]
        )
        data = rng.uniform(minval, maxval, size=shape).astype(dtype.value)
        return Tensor(data, shape, dtype, key.device)
    else:
        if not _tracer.is_tracing:
            raise RuntimeError(
                "Cannot emit RandomUniform node outside of tracing context."
            )
        out_id = str(uuid.uuid4())
        node = LogicalNode(
            id=out_id,
            op_type="RandomUniform",
            inputs=[key.data.id],
            attributes={"minval": minval, "maxval": maxval},
            shape_metadata=shape,
        )
        _tracer.add_node(node)
        proxy = ProxyTensor(id=out_id, shape=shape, dtype=dtype.value)
        return Tensor(data=proxy, shape=shape, dtype=dtype, device=key.device)


def normal(key: Tensor, shape: Sequence[int], dtype: DType = DType.Float32) -> Tensor:
    """Samples from a standard normal distribution (mean 0, variance 1)."""
    if config.eager_mode:
        rng = np.random.default_rng(
            key.data.item() if key.data.ndim == 0 else key.data[0]
        )
        data = rng.standard_normal(size=shape).astype(dtype.value)
        return Tensor(data, shape, dtype, key.device)
    else:
        if not _tracer.is_tracing:
            raise RuntimeError(
                "Cannot emit RandomNormal node outside of tracing context."
            )
        out_id = str(uuid.uuid4())
        node = LogicalNode(
            id=out_id,
            op_type="RandomNormal",
            inputs=[key.data.id],
            attributes={},
            shape_metadata=shape,
        )
        _tracer.add_node(node)
        proxy = ProxyTensor(id=out_id, shape=shape, dtype=dtype.value)
        return Tensor(data=proxy, shape=shape, dtype=dtype, device=key.device)


def bernoulli(
    key: Tensor, p: Union[float, Tensor], shape: Optional[Sequence[int]] = None
) -> Tensor:
    """Samples from a Bernoulli distribution with probability `p`."""
    p_val = p.data if isinstance(p, Tensor) else p
    if shape is None:
        shape = p.shape if isinstance(p, Tensor) else ()

    dtype = DType.Bool
    if config.eager_mode:
        rng = np.random.default_rng(
            key.data.item() if key.data.ndim == 0 else key.data[0]
        )
        data = rng.random(size=shape) < p_val
        return Tensor(data, shape, dtype, key.device)
    else:
        if not _tracer.is_tracing:
            raise RuntimeError(
                "Cannot emit RandomBernoulli node outside of tracing context."
            )
        out_id = str(uuid.uuid4())
        inputs = [key.data.id]
        if isinstance(p, Tensor):
            inputs.append(p.data.id)
        node = LogicalNode(
            id=out_id,
            op_type="RandomBernoulli",
            inputs=inputs,
            attributes={"p": p if not isinstance(p, Tensor) else None},
            shape_metadata=shape,
        )
        _tracer.add_node(node)
        proxy = ProxyTensor(id=out_id, shape=shape, dtype=dtype.value)
        return Tensor(data=proxy, shape=shape, dtype=dtype, device=key.device)


def truncated_normal(
    key: Tensor,
    lower: float,
    upper: float,
    shape: Sequence[int],
    dtype: DType = DType.Float32,
) -> Tensor:
    """Samples from a truncated normal distribution bounded by `lower` and `upper`."""
    if config.eager_mode:
        rng = np.random.default_rng(
            key.data.item() if key.data.ndim == 0 else key.data[0]
        )
        # Rejection sampling for truncated normal
        data = np.empty(shape, dtype=dtype.value)
        flat_data = data.reshape(-1)
        needed = len(flat_data)
        idx = 0
        while needed > 0:
            samples = rng.standard_normal(size=needed)
            valid = samples[(samples >= lower) & (samples <= upper)]
            count = len(valid)
            flat_data[idx : idx + count] = valid
            idx += count
            needed -= count
        return Tensor(data, shape, dtype, key.device)
    else:
        if not _tracer.is_tracing:
            raise RuntimeError(
                "Cannot emit RandomTruncatedNormal node outside of tracing context."
            )
        out_id = str(uuid.uuid4())
        node = LogicalNode(
            id=out_id,
            op_type="RandomTruncatedNormal",
            inputs=[key.data.id],
            attributes={"lower": lower, "upper": upper},
            shape_metadata=shape,
        )
        _tracer.add_node(node)
        proxy = ProxyTensor(id=out_id, shape=shape, dtype=dtype.value)
        return Tensor(data=proxy, shape=shape, dtype=dtype, device=key.device)


def randint(
    key: Tensor,
    shape: Sequence[int],
    minval: int,
    maxval: int,
    dtype: DType = DType.Int32,
) -> Tensor:
    """Samples uniform random integers in the interval `[minval, maxval)`."""
    if config.eager_mode:
        rng = np.random.default_rng(
            key.data.item() if key.data.ndim == 0 else key.data[0]
        )
        data = rng.integers(minval, maxval, size=shape, dtype=dtype.value)
        return Tensor(data, shape, dtype, key.device)
    else:
        if not _tracer.is_tracing:
            raise RuntimeError("Cannot emit RandomInt node outside of tracing context.")
        out_id = str(uuid.uuid4())
        node = LogicalNode(
            id=out_id,
            op_type="RandomInt",
            inputs=[key.data.id],
            attributes={"minval": minval, "maxval": maxval},
            shape_metadata=shape,
        )
        _tracer.add_node(node)
        proxy = ProxyTensor(id=out_id, shape=shape, dtype=dtype.value)
        return Tensor(data=proxy, shape=shape, dtype=dtype, device=key.device)
