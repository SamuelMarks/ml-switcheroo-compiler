"""Random operations."""

from __future__ import annotations

from __future__ import annotations
from ml_switcheroo_compiler.backends.registry import get_active_backend
import numpy as np
from ml_switcheroo_compiler.core import dtype as dtypes
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

from ml_switcheroo_compiler.random.state import _emit_random_node, _dispatch_random


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
            TensorConfig(shape, dtype, config.default_device),
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
        return Tensor(
            rng.normal(size=shape).astype(np_dtype),
            TensorConfig(shape, dtype, config.default_device),
        )
    return _emit_random_node("RandomNormal", [key], shape, dtype)


def _eager_truncated_normal(
    rng: object, shape: object, dtype: object, lower: object, upper: object
) -> object:
    import numpy as np
    from ml_switcheroo_compiler.core.tensor import Tensor
    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.core.tensor import TensorConfig

    np_dtype = np.dtype(dtype.value)
    arr = rng.normal(size=shape)

    low = getattr(lower, "data", lower)
    up = getattr(upper, "data", upper)
    low = np.broadcast_to(low, shape) if np.ndim(low) > 0 else low
    up = np.broadcast_to(up, shape) if np.ndim(up) > 0 else up

    mask = (arr < low) | (arr > up)
    while np.any(mask):
        arr[mask] = rng.normal(size=np.count_nonzero(mask))
        mask = (arr < low) | (arr > up)

    return Tensor(arr.astype(np_dtype), TensorConfig(shape, dtype, config.default_device))


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
        key_data = getattr(key, "data", key)
        seed = [int(x) for x in np.asarray(key_data).ravel()] if np.ndim(key_data) > 0 else None
        rng = np.random.default_rng(seed)

        return _eager_truncated_normal(rng, shape, dtype, lower, upper)
    return _emit_random_node(
        "RandomTruncatedNormal", [key], shape, dtype, {"lower": lower, "upper": upper}
    )


def ball(*args: object, **kwargs: object) -> object:
    """Execute ball."""
    return _dispatch_random("ball", *args, **kwargs)


def beta(*args: object, **kwargs: object) -> object:
    """Execute beta."""
    return _dispatch_random("beta", *args, **kwargs)


def cauchy(*args: object, **kwargs: object) -> object:
    """Execute cauchy."""
    return _dispatch_random("cauchy", *args, **kwargs)


def chisquare(*args: object, **kwargs: object) -> object:
    """Execute chisquare."""
    if config.eager_mode:
        backend = get_active_backend()
        if hasattr(backend.module, "random") and hasattr(backend.module.random, "chisquare"):
            return backend.module.random.chisquare(*args, **kwargs)
        raise NotImplementedError(
            "chisquare is not supported in eager mode without backend support."
        )
    raise NotImplementedError("chisquare is not fully supported in tracing mode.")


def dirichlet(*args: object, **kwargs: object) -> object:
    """Execute dirichlet."""
    if config.eager_mode:
        backend = get_active_backend()
        if hasattr(backend.module, "random") and hasattr(backend.module.random, "dirichlet"):
            return backend.module.random.dirichlet(*args, **kwargs)
        raise NotImplementedError(
            "dirichlet is not supported in eager mode without backend support."
        )
    raise NotImplementedError("dirichlet is not fully supported in tracing mode.")


def double_sided_maxwell(*args: object, **kwargs: object) -> object:
    """Execute double_sided_maxwell."""
    if config.eager_mode:
        backend = get_active_backend()
        if hasattr(backend.module, "random") and hasattr(
            backend.module.random, "double_sided_maxwell"
        ):
            return backend.module.random.double_sided_maxwell(*args, **kwargs)
        raise NotImplementedError(
            "double_sided_maxwell is not supported in eager mode without backend support."
        )
    raise NotImplementedError("double_sided_maxwell is not fully supported in tracing mode.")


def exponential(*args: object, **kwargs: object) -> object:
    """Execute exponential."""
    if config.eager_mode:
        backend = get_active_backend()
        if hasattr(backend.module, "random") and hasattr(backend.module.random, "exponential"):
            return backend.module.random.exponential(*args, **kwargs)
        raise NotImplementedError(
            "exponential is not supported in eager mode without backend support."
        )
    raise NotImplementedError("exponential is not fully supported in tracing mode.")


def f(*args: object, **kwargs: object) -> object:
    """Execute f."""
    return _dispatch_random("f", *args, **kwargs)


def gamma(*args: object, **kwargs: object) -> object:
    """Execute gamma."""
    return _dispatch_random("gamma", *args, **kwargs)


def generalized_normal(*args: object, **kwargs: object) -> object:
    """Execute generalized_normal."""
    if config.eager_mode:
        backend = get_active_backend()
        if hasattr(backend.module, "random") and hasattr(
            backend.module.random, "generalized_normal"
        ):
            return backend.module.random.generalized_normal(*args, **kwargs)
        raise NotImplementedError(
            "generalized_normal is not supported in eager mode without backend support."
        )
    raise NotImplementedError("generalized_normal is not fully supported in tracing mode.")


def gumbel(*args: object, **kwargs: object) -> object:
    """Execute gumbel."""
    return _dispatch_random("gumbel", *args, **kwargs)


def laplace(*args: object, **kwargs: object) -> object:
    """Execute laplace."""
    return _dispatch_random("laplace", *args, **kwargs)


def loggamma(*args: object, **kwargs: object) -> object:
    """Execute loggamma."""
    if config.eager_mode:
        backend = get_active_backend()
        if hasattr(backend.module, "random") and hasattr(backend.module.random, "loggamma"):
            return backend.module.random.loggamma(*args, **kwargs)
        raise NotImplementedError(
            "loggamma is not supported in eager mode without backend support."
        )
    raise NotImplementedError("loggamma is not fully supported in tracing mode.")


def logistic(*args: object, **kwargs: object) -> object:
    """Execute logistic."""
    if config.eager_mode:
        backend = get_active_backend()
        if hasattr(backend.module, "random") and hasattr(backend.module.random, "logistic"):
            return backend.module.random.logistic(*args, **kwargs)
        raise NotImplementedError(
            "logistic is not supported in eager mode without backend support."
        )
    raise NotImplementedError("logistic is not fully supported in tracing mode.")


def lognormal(*args: object, **kwargs: object) -> object:
    """Execute lognormal."""
    if config.eager_mode:
        backend = get_active_backend()
        if hasattr(backend.module, "random") and hasattr(backend.module.random, "lognormal"):
            return backend.module.random.lognormal(*args, **kwargs)
        raise NotImplementedError(
            "lognormal is not supported in eager mode without backend support."
        )
    raise NotImplementedError("lognormal is not fully supported in tracing mode.")


def maxwell(*args: object, **kwargs: object) -> object:
    """Execute maxwell."""
    return _dispatch_random("maxwell", *args, **kwargs)


def multivariate_normal(*args: object, **kwargs: object) -> object:
    """Execute multivariate_normal."""
    if config.eager_mode:
        backend = get_active_backend()
        if hasattr(backend.module, "random") and hasattr(
            backend.module.random, "multivariate_normal"
        ):
            return backend.module.random.multivariate_normal(*args, **kwargs)
        raise NotImplementedError(
            "multivariate_normal is not supported in eager mode without backend support."
        )
    raise NotImplementedError("multivariate_normal is not fully supported in tracing mode.")


def orthogonal(*args: object, **kwargs: object) -> object:
    """Execute orthogonal."""
    if config.eager_mode:
        backend = get_active_backend()
        if hasattr(backend.module, "random") and hasattr(backend.module.random, "orthogonal"):
            return backend.module.random.orthogonal(*args, **kwargs)
        raise NotImplementedError(
            "orthogonal is not supported in eager mode without backend support."
        )
    raise NotImplementedError("orthogonal is not fully supported in tracing mode.")


def pareto(*args: object, **kwargs: object) -> object:
    """Execute pareto."""
    return _dispatch_random("pareto", *args, **kwargs)


def random_gamma_p(*args: object, **kwargs: object) -> object:
    """Execute random_gamma_p."""
    if config.eager_mode:
        backend = get_active_backend()
        if hasattr(backend.module, "random") and hasattr(backend.module.random, "random_gamma_p"):
            return backend.module.random.random_gamma_p(*args, **kwargs)
        raise NotImplementedError(
            "random_gamma_p is not supported in eager mode without backend support."
        )
    raise NotImplementedError("random_gamma_p is not fully supported in tracing mode.")


def rayleigh(*args: object, **kwargs: object) -> object:
    """Execute rayleigh."""
    if config.eager_mode:
        backend = get_active_backend()
        if hasattr(backend.module, "random") and hasattr(backend.module.random, "rayleigh"):
            return backend.module.random.rayleigh(*args, **kwargs)
        raise NotImplementedError(
            "rayleigh is not supported in eager mode without backend support."
        )
    raise NotImplementedError("rayleigh is not fully supported in tracing mode.")


def t(*args: object, **kwargs: object) -> object:
    """Execute t."""
    return _dispatch_random("t", *args, **kwargs)


def triangular(*args: object, **kwargs: object) -> object:
    """Execute triangular."""
    if config.eager_mode:
        backend = get_active_backend()
        if hasattr(backend.module, "random") and hasattr(backend.module.random, "triangular"):
            return backend.module.random.triangular(*args, **kwargs)
        raise NotImplementedError(
            "triangular is not supported in eager mode without backend support."
        )
    raise NotImplementedError("triangular is not fully supported in tracing mode.")


def wald(*args: object, **kwargs: object) -> object:
    """Execute wald."""
    return _dispatch_random("wald", *args, **kwargs)


def weibull_min(*args: object, **kwargs: object) -> object:
    """Execute weibull_min."""
    if config.eager_mode:
        backend = get_active_backend()
        if hasattr(backend.module, "random") and hasattr(backend.module.random, "weibull_min"):
            return backend.module.random.weibull_min(*args, **kwargs)
        raise NotImplementedError(
            "weibull_min is not supported in eager mode without backend support."
        )
    raise NotImplementedError("weibull_min is not fully supported in tracing mode.")
