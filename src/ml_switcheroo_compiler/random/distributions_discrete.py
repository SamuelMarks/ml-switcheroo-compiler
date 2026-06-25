"""Random operations."""

from __future__ import annotations

from __future__ import annotations
from ml_switcheroo_compiler.backends.registry import get_active_backend
import numpy as np
from ml_switcheroo_compiler.core import dtype as dtypes
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

from ml_switcheroo_compiler.random.state import _emit_random_node


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
            TensorConfig(shape, dtype, config.default_device),
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
            TensorConfig(shape, dtypes.DType.Bool, config.default_device),
        )
    return _emit_random_node("RandomBernoulli", [key], shape, dtypes.DType.Bool, {"p": p})


def _validate_categorical_shapes(
    logits_arr: object, out_shape: tuple[int, ...]
) -> tuple[object, int, object]:
    """Validates shapes and computes probabilities for categorical distribution."""
    if logits_arr is None:
        logits_arr = np.zeros(out_shape)
    probs = np.exp(logits_arr - np.max(logits_arr, axis=-1, keepdims=True))
    probs /= np.sum(probs, axis=-1, keepdims=True)
    num_classes = probs.shape[-1] if probs.ndim > 0 else 1

    if len(out_shape) > probs.ndim - 1:
        logits_expanded = (
            np.expand_dims(logits_arr, axis=-2)
            if probs.ndim > 0
            else np.expand_dims(logits_arr, axis=-1)
        )
    else:
        logits_expanded = logits_arr

    return logits_expanded, num_classes, probs


def _sample_gumbel_max(
    rng: np.random.Generator, logits_expanded: object, out_shape: tuple[int, ...], num_classes: int
) -> object:
    """Samples from categorical using the Gumbel-max trick."""
    gumbel_noise = rng.gumbel(size=out_shape + (num_classes,))
    return np.argmax(logits_expanded + gumbel_noise, axis=-1).astype(np.int32)


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
        logits_expanded, num_classes, probs = _validate_categorical_shapes(logits_arr, out_shape)

        if isinstance(key, Tensor):  # pragma: no branch
            seed_val = int(key.data[1])
        else:
            seed_val = 0  # pragma: no cover
        rng = np.random.default_rng(seed_val)

        res = _sample_gumbel_max(rng, logits_expanded, out_shape, num_classes)

        return Tensor(res, TensorConfig(out_shape, dtypes.DType.Int32, config.default_device))
    inputs = [key]
    if isinstance(logits, Tensor):  # pragma: no branch
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
            TensorConfig(
                getattr(res, "shape", ()),
                getattr(x, "dtype", dtypes.DType.Float32),
                config.default_device,
            ),
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
        return Tensor(
            res,
            TensorConfig(shape, getattr(a, "dtype", dtypes.DType.Float32), config.default_device),
        )
    inputs = [key, a]
    if isinstance(p, Tensor):
        inputs.append(p)
    return _emit_random_node(
        "RandomChoice", inputs, shape, a.dtype, {"replace": replace, "axis": axis}
    )


def binomial(
    key: object, n: object, p: object, shape: object = None, dtype: object = None
) -> object:
    """Samples binomial random values from a given key."""
    if shape is None:
        shape = ()
    dtype = dtype or dtypes.DType.Int32  # pragma: no cover
    if config.eager_mode:  # pragma: no cover
        np_dtype = np.dtype(dtype.value)  # pragma: no cover
        n_val = getattr(n, "data", n)  # pragma: no cover
        p_val = getattr(p, "data", p)  # pragma: no cover
        if isinstance(key, Tensor):  # pragma: no cover
            seed_val = int(key.data[1])  # pragma: no cover
        else:
            seed_val = 0  # pragma: no cover
        rng = np.random.default_rng(seed_val)  # pragma: no cover
        if shape is None:  # pragma: no cover
            if hasattr(n_val, "shape"):  # pragma: no cover
                shape = n_val.shape  # pragma: no cover
            else:
                shape = ()  # pragma: no cover
        res = rng.binomial(n_val, p_val, size=shape).astype(np_dtype)  # pragma: no cover
        return Tensor(res, TensorConfig(shape, dtype, config.default_device))  # pragma: no cover
    return _emit_random_node("RandomBinomial", [key], shape, dtype)  # pragma: no cover


def geometric(*args: object, **kwargs: object) -> object:
    """Execute geometric."""
    if config.eager_mode:
        backend = get_active_backend()
        if hasattr(backend.module, "random") and hasattr(
            backend.module.random, "geometric"
        ):  # pragma: no branch
            return backend.module.random.geometric(*args, **kwargs)  # pragma: no cover
        raise NotImplementedError(  # pragma: no cover
            "geometric is not supported in eager mode without backend support."
        )
    raise NotImplementedError("geometric is not fully supported in tracing mode.")


def poisson(key: object, lam: object, shape: object = None, dtype: object = None) -> object:
    """Samples poisson random values from a given key."""
    if shape is None:
        shape = ()
    dtype = dtype or dtypes.DType.Int32
    if config.eager_mode:
        np_dtype = np.dtype(dtype.value)
        lam_val = getattr(lam, "data", lam)
        if isinstance(key, Tensor):
            seed_val = int(key.data[1])
        else:
            seed_val = 0
        rng = np.random.default_rng(seed_val)
        res = rng.poisson(lam_val, size=shape).astype(np_dtype)
        return Tensor(res, TensorConfig(getattr(res, "shape", ()), dtype, config.default_device))
    return _emit_random_node("RandomPoisson", [key, lam], shape, dtype)


def rademacher(*args: object, **kwargs: object) -> object:
    """Execute rademacher."""
    if config.eager_mode:
        backend = get_active_backend()
        if hasattr(backend.module, "random") and hasattr(
            backend.module.random, "rademacher"
        ):  # pragma: no branch
            return backend.module.random.rademacher(*args, **kwargs)  # pragma: no cover
        raise NotImplementedError(  # pragma: no cover
            "rademacher is not supported in eager mode without backend support."
        )
    raise NotImplementedError("rademacher is not fully supported in tracing mode.")


def multinomial(key: object, n: int, pvals: object, shape: object = None) -> object:
    """Samples from multinomial distribution.

    Args:
        key (object): The PRNG key.
        n (int): Number of experiments.
        pvals (object): Probabilities of each of the p different outcomes.
        shape (object): Target shape.

    Returns:
        object: The evaluated output resulting from this operation.
    """
    if shape is None:
        shape = ()
    if config.eager_mode:
        p_arr = getattr(pvals, "data", pvals)
        if isinstance(key, Tensor):
            seed_val = int(key.data[1])
        else:
            seed_val = 0
        rng = np.random.default_rng(seed_val)
        res = rng.multinomial(n, p_arr, size=shape)
        return Tensor(
            res, TensorConfig(getattr(res, "shape", ()), dtypes.DType.Int32, config.default_device)
        )
    return _emit_random_node("RandomMultinomial", [key, pvals], shape, dtypes.DType.Int32, {"n": n})
