"""Random number generation and state management."""

from __future__ import annotations

import uuid

import numpy as np


from ml_switcheroo_compiler.core import dtype as dtypes
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_ir import LogicalNode
from ml_switcheroo_compiler.tracing.tracer import ProxyTensor, _tracer


def _emit_random_node(
    op_type: str,
    inputs: list[Tensor],
    shape: tuple[int, ...],
    dtype: dtypes.DType,
    attributes: dict | None = None,
) -> Tensor:
    """Execute _emit_random_node.

    Args:
        op_type (str): The op_type parameter for the operation.
        inputs (list[Tensor]): The inputs parameter for the operation.
        shape (tuple[int, ...]): The target shape.
        dtype (dtypes.DType): The target data type.
        attributes (dict | None): The attributes parameter for the operation.

    Returns:
        Tensor: The result.
    """
    if config.eager_mode:
        raise NotImplementedError(f"{op_type} not implemented in eager mode via _emit_random_node")

    out_id = str(uuid.uuid4())
    node = LogicalNode(
        id=out_id,
        op_type=op_type,
        inputs=[inp.data.id for inp in inputs],
        attributes=attributes or {},
        shape_metadata=shape,
    )
    _tracer.add_node(node)
    proxy = ProxyTensor(id=out_id, shape=shape, dtype=dtype.value)
    return Tensor(data=proxy, shape=shape, dtype=dtype, device=config.default_device)


def PRNGKey(seed: int) -> Tensor:
    """Creates a PRNG key given an integer seed.

    Args:
        seed (int): The random seed.

    Returns:
        Tensor: A tensor containing the result of the operation.
    """
    if config.eager_mode:
        return Tensor(
            np.array([0, seed], dtype=np.uint32), (2,), dtypes.DType.UInt32, config.default_device
        )

    # Trace as a creation node
    out_id = str(uuid.uuid4())
    node = LogicalNode(
        id=out_id,
        op_type="PRNGKey",
        inputs=[],
        attributes={"seed": seed},
        shape_metadata=(2,),
    )
    if _tracer.is_tracing:
        _tracer.add_node(node)
    proxy = ProxyTensor(id=out_id, shape=(2,), dtype="uint32")
    return Tensor(data=proxy, shape=(2,), dtype=dtypes.DType.UInt32, device=config.default_device)


def split(key: Tensor, num: int = 2) -> Tensor:
    """Splits a PRNG key into num new keys.

    Args:
        key (Tensor): The PRNG key.
        num (int): The num parameter for the operation.

    Returns:
        Tensor: A tensor containing the result of the operation.
    """
    if config.eager_mode:
        return Tensor(
            np.random.randint(0, 2**32, size=(num, 2), dtype=np.uint32),
            (num, 2),
            dtypes.DType.UInt32,
            config.default_device,
        )
    return _emit_random_node("RandomSplit", [key], (num, 2), dtypes.DType.UInt32, {"num": num})


def fold_in(key: Tensor, data: int) -> Tensor:
    """Folds in data to a PRNG key to derive a new key.

    Args:
        key (Tensor): The PRNG key.
        data (int): The data parameter for the operation.

    Returns:
        Tensor: A tensor containing the result of the operation.
    """
    if config.eager_mode:
        return Tensor(
            np.array([key.data[0] + data, key.data[1]], dtype=np.uint32),
            (2,),
            dtypes.DType.UInt32,
            config.default_device,
        )
    return _emit_random_node("RandomFoldIn", [key], (2,), dtypes.DType.UInt32, {"data": data})


def uniform(
    key: object,
    shape: object = (),
    dtype: object = None,
    minval: object = 0.0,
    maxval: object = 1.0,
) -> object:
    """Samples uniform random values from a given key.

    Args:
        key (object): The PRNG key.
        shape (object): The target shape.
        dtype (object): The target data type.
        minval (object): The minval parameter for the operation.
        maxval (object): The maxval parameter for the operation.

    Returns:
        object: The evaluated output resulting from this operation.
    """
    dtype = dtype or dtypes.DType.Float32
    if config.eager_mode:
        np_dtype = np.dtype(dtype.value)
        minv = getattr(minval, "data", minval)
        maxv = getattr(maxval, "data", maxval)

        # Broadcast minv and maxv to the target shape if they are arrays
        minv = np.broadcast_to(minv, shape) if np.ndim(minv) > 0 else minv
        maxv = np.broadcast_to(maxv, shape) if np.ndim(maxv) > 0 else maxv

        # We can seed using the key for deterministic eager mode
        key_data = getattr(key, "data", key)
        seed = [int(x) for x in np.asarray(key_data).ravel()] if np.ndim(key_data) > 0 else None
        rng = np.random.default_rng(seed)

        return Tensor(
            rng.uniform(minv, maxv, size=shape).astype(np_dtype),
            shape,
            dtype,
            config.default_device,
        )
    return _emit_random_node(
        "RandomUniform", [key], shape, dtype, {"minval": minval, "maxval": maxval}
    )


def normal(key: object, shape: object = (), dtype: object = None) -> object:
    """Samples standard normal random values from a given key.

    Args:
        key (object): The PRNG key.
        shape (object): The target shape.
        dtype (object): The target data type.

    Returns:
        object: The evaluated output resulting from this operation.
    """
    dtype = dtype or dtypes.DType.Float32
    if config.eager_mode:
        np_dtype = np.dtype(dtype.value)
        key_data = getattr(key, "data", key)
        seed = [int(x) for x in np.asarray(key_data).ravel()] if np.ndim(key_data) > 0 else None
        rng = np.random.default_rng(seed)
        return Tensor(rng.normal(size=shape).astype(np_dtype), shape, dtype, config.default_device)
    return _emit_random_node("RandomNormal", [key], shape, dtype)


def randint(
    key: object,
    shape: object,
    minval: object,
    maxval: object,
    dtype: object = None,
) -> object:
    """Samples uniform random integers from a given key.

    Args:
        key (object): The PRNG key.
        shape (object): The target shape.
        minval (object): The minval parameter for the operation.
        maxval (object): The maxval parameter for the operation.
        dtype (object): The target data type.

    Returns:
        object: The evaluated output resulting from this operation.
    """
    dtype = dtype or dtypes.DType.Int32
    if config.eager_mode:
        np_dtype = np.dtype(dtype.value)
        minv = getattr(minval, "data", minval)
        maxv = getattr(maxval, "data", maxval)
        minv = np.broadcast_to(minv, shape) if np.ndim(minv) > 0 else minv
        maxv = np.broadcast_to(maxv, shape) if np.ndim(maxv) > 0 else maxv

        key_data = getattr(key, "data", key)
        seed = [int(x) for x in np.asarray(key_data).ravel()] if np.ndim(key_data) > 0 else None
        rng = np.random.default_rng(seed)

        return Tensor(
            rng.integers(minv, maxv, size=shape, dtype=np_dtype),
            shape,
            dtype,
            config.default_device,
        )
    return _emit_random_node(
        "RandomRandint", [key], shape, dtype, {"minval": minval, "maxval": maxval}
    )


def bernoulli(key: object, p: object = 0.5, shape: object = None) -> object:
    """Samples Bernoulli random variables from a given key.

    Args:
        key (object): The PRNG key.
        p (object): The p parameter for the operation.
        shape (object): The target shape.

    Returns:
        object: The evaluated output resulting from this operation.
    """
    if shape is None:
        shape = ()
    if config.eager_mode:
        pv = getattr(p, "data", p)
        pv = np.broadcast_to(pv, shape) if np.ndim(pv) > 0 else pv

        key_data = getattr(key, "data", key)
        seed = [int(x) for x in np.asarray(key_data).ravel()] if np.ndim(key_data) > 0 else None
        rng = np.random.default_rng(seed)

        return Tensor(
            rng.binomial(1, pv, size=shape).astype(bool),
            shape,
            dtypes.DType.Bool,
            config.default_device,
        )
    return _emit_random_node("RandomBernoulli", [key], shape, dtypes.DType.Bool, {"p": p})


def categorical(key: object, logits: object, axis: object = -1, shape: object = None) -> object:
    """Samples categorical random variables from a given key.

    Args:
        key (object): The PRNG key.
        logits (object): The logits parameter for the operation.
        axis (object): The axis along which to perform the operation.
        shape (object): The target shape.

    Returns:
        object: The evaluated output resulting from this operation.
    """
    out_shape = shape or ()
    if config.eager_mode:
        logits_arr = getattr(logits, "data", logits)
        if logits_arr is None:
            logits_arr = np.zeros(out_shape)
        probs = np.exp(logits_arr - np.max(logits_arr, axis=-1, keepdims=True))
        probs /= np.sum(probs, axis=-1, keepdims=True)

        # Simplified sampling using numpy
        def sample(p: object) -> object:
            """Sample."""
            return np.random.choice(len(p), p=p)

        if probs.ndim > 1:
            res = np.apply_along_axis(sample, -1, probs)
        elif probs.ndim == 1:
            res = sample(probs)
        else:
            res = np.zeros(out_shape, dtype=np.int32)
        return Tensor(res, out_shape, dtypes.DType.Int32, config.default_device)
    inputs = [key]
    if isinstance(logits, Tensor):
        inputs.append(logits)
    return _emit_random_node(
        "RandomCategorical", inputs, out_shape, dtypes.DType.Int32, {"axis": axis}
    )


def permutation(key: object, x: object, axis: object = 0, independent: object = False) -> object:
    """Randomly permutes a sequence or array.

    Args:
        key (object): The PRNG key.
        x (object): The input x tensor.
        axis (object): The axis along which to perform the operation.
        independent (object): The independent parameter for the operation.

    Returns:
        object: The evaluated output resulting from this operation.
    """
    if config.eager_mode:
        arr = getattr(x, "data", x)
        res = np.random.permutation(arr)
        return Tensor(
            res,
            getattr(res, "shape", ()),
            getattr(x, "dtype", dtypes.DType.Float32),
            config.default_device,
        )
    return _emit_random_node(
        "RandomPermutation", [key, x], x.shape, x.dtype, {"axis": axis, "independent": independent}
    )


def choice(
    key: object,
    a: object,
    **kwargs: object,
) -> object:
    """Generates a random sample from a given 1-D array.

    Args:
        key (object): The PRNG key.
        a (object): The input a tensor.
        **kwargs (object): Optional arguments shape, replace, p, axis.

    Returns:
        object: The evaluated output resulting from this operation.
    """
    shape = kwargs.get("shape", ())
    replace = kwargs.get("replace", True)
    p = kwargs.get("p", None)
    axis = kwargs.get("axis", 0)

    if config.eager_mode:
        arr = getattr(a, "data", a)
        p_arr = getattr(p, "data", p) if p is not None else None
        res = np.random.choice(arr, size=shape, replace=replace, p=p_arr)
        return Tensor(res, shape, getattr(a, "dtype", dtypes.DType.Float32), config.default_device)
    inputs = [key, a]
    if isinstance(p, Tensor):
        inputs.append(p)
    return _emit_random_node(
        "RandomChoice", inputs, shape, a.dtype, {"replace": replace, "axis": axis}
    )


def truncated_normal(
    key: object,
    lower: object,
    upper: object,
    shape: object = (),
    dtype: object = None,
) -> object:
    """Returns an initializer that generates arrays from a truncated normal distribution.

    Args:
        key (object): The PRNG key.
        lower (object): The lower parameter for the operation.
        upper (object): The upper parameter for the operation.
        shape (object): The target shape.
        dtype (object): The target data type.

    Returns:
        object: The evaluated output resulting from this operation.
    """
    dtype = dtype or dtypes.DType.Float32
    if config.eager_mode:
        np_dtype = np.dtype(dtype.value)
        key_data = getattr(key, "data", key)
        seed = [int(x) for x in np.asarray(key_data).ravel()] if np.ndim(key_data) > 0 else None
        rng = np.random.default_rng(seed)

        # A simple truncated normal using accept-reject or just clip (clip is inaccurate but works for tests)
        # We use a crude loop for accept-reject since it's eager mode fallback
        arr = rng.normal(size=shape)
        low = getattr(lower, "data", lower)
        up = getattr(upper, "data", upper)
        low = np.broadcast_to(low, shape) if np.ndim(low) > 0 else low
        up = np.broadcast_to(up, shape) if np.ndim(up) > 0 else up

        mask = (arr < low) | (arr > up)
        while np.any(mask):
            arr[mask] = rng.normal(size=np.count_nonzero(mask))
            mask = (arr < low) | (arr > up)

        return Tensor(arr.astype(np_dtype), shape, dtype, config.default_device)
    return _emit_random_node(
        "RandomTruncatedNormal", [key], shape, dtype, {"lower": lower, "upper": upper}
    )
