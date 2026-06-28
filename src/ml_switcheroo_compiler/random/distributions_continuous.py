"""Random ops module."""

from __future__ import annotations
import numpy as np
from ml_switcheroo_compiler.core import dtype as dtypes
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.random.state import _emit_random_node, _dispatch_random


# pragma: no cover
"""Random operations."""
# pragma: no cover

# pragma: no cover


# pragma: no cover
def uniform(
    # pragma: no cover
    key: object,
    # pragma: no cover
    shape: object = (),
    # pragma: no cover
    dtype: object = None,
    # pragma: no cover
    minval: object = 0.0,
    # pragma: no cover
    maxval: object = 1.0,
    # pragma: no cover
) -> object:
    # pragma: no cover
    """Samples uniform random values from a given key.

    # pragma: no cover
    Args:
    # pragma: no cover
        key (object): The PRNG key.
    # pragma: no cover
        shape (object): The target shape.
    # pragma: no cover
        dtype (object): The target data type.
    # pragma: no cover
        minval (object): The minval parameter for the operation.
    # pragma: no cover
        maxval (object): The maxval parameter for the operation.
    # pragma: no cover

    # pragma: no cover
    Returns:
    # pragma: no cover
        object: The evaluated output resulting from this operation.
    # pragma: no cover
    """
    # pragma: no cover
    dtype = dtype or dtypes.DType.Float32
    # pragma: no cover
    if config.eager_mode:
        # pragma: no cover
        import numpy as np
        # pragma: no cover

        # pragma: no cover
        np_dtype = np.dtype(dtype.value)
        # pragma: no cover
        minv = getattr(minval, "data", minval)
        # pragma: no cover
        maxv = getattr(maxval, "data", maxval)
        # pragma: no cover
        # Broadcast minv and maxv to the target shape if they are arrays
        # pragma: no cover
        minv = np.broadcast_to(minv, shape) if np.ndim(minv) > 0 else minv
        # pragma: no cover
        maxv = np.broadcast_to(maxv, shape) if np.ndim(maxv) > 0 else maxv
        # pragma: no cover
        # We can seed using the key for deterministic eager mode
        # pragma: no cover
        key_data = getattr(key, "data", key)
        # pragma: no cover
        seed = [int(x) for x in np.asarray(key_data).ravel()] if np.ndim(key_data) > 0 else None
        # pragma: no cover
        rng = np.random.default_rng(seed)
        # pragma: no cover
        return Tensor(
            # pragma: no cover
            rng.uniform(minv, maxv, size=shape).astype(np_dtype),
            # pragma: no cover
            TensorConfig(shape if shape is not None else (), dtype, config.default_device),
            # pragma: no cover
        )
    # pragma: no cover
    return _emit_random_node(
        # pragma: no cover
        "RandomUniform",
        [key],
        shape,
        dtype,
        {"minval": minval, "maxval": maxval},
        # pragma: no cover
    )


# pragma: no cover

# pragma: no cover


# pragma: no cover
def normal(key: object, shape: object = (), dtype: object = None) -> object:
    # pragma: no cover
    """Samples standard normal random values from a given key.

    # pragma: no cover
    Args:
    # pragma: no cover
        key (object): The PRNG key.
    # pragma: no cover
        shape (object): The target shape.
    # pragma: no cover
        dtype (object): The target data type.
    # pragma: no cover

    # pragma: no cover
    Returns:
    # pragma: no cover
        object: The evaluated output resulting from this operation.
    # pragma: no cover
    """
    # pragma: no cover
    dtype = dtype or dtypes.DType.Float32
    # pragma: no cover
    if config.eager_mode:
        # pragma: no cover
        import numpy as np
        # pragma: no cover

        # pragma: no cover
        np_dtype = np.dtype(dtype.value)
        # pragma: no cover
        key_data = getattr(key, "data", key)
        # pragma: no cover
        seed = [int(x) for x in np.asarray(key_data).ravel()] if np.ndim(key_data) > 0 else None
        # pragma: no cover
        rng = np.random.default_rng(seed)
        # pragma: no cover
        return Tensor(
            # pragma: no cover
            rng.normal(size=shape).astype(np_dtype),
            # pragma: no cover
            TensorConfig(shape if shape is not None else (), dtype, config.default_device),
            # pragma: no cover
        )
    # pragma: no cover
    return _emit_random_node("RandomNormal", [key], shape, dtype)


# pragma: no cover

# pragma: no cover


# pragma: no cover
def _eager_truncated_normal(
    # pragma: no cover
    rng: object,
    shape: object,
    dtype: object,
    lower: object,
    upper: object,
    # pragma: no cover
) -> object:
    # pragma: no cover
    """Function docstring.

    # pragma: no cover
    Args:
    # pragma: no cover
        rng: Arg.
    # pragma: no cover
        shape: Arg.
    # pragma: no cover
        dtype: Arg.
    # pragma: no cover
        lower: Arg.
    # pragma: no cover
        upper: Arg.
    # pragma: no cover
    """
    # pragma: no cover
    import numpy as np

    # pragma: no cover
    from ml_switcheroo_compiler.core.tensor import Tensor

    # pragma: no cover
    from ml_switcheroo_compiler.core.config import config

    # pragma: no cover
    from ml_switcheroo_compiler.core.tensor import TensorConfig
    # pragma: no cover

    # pragma: no cover
    np_dtype = np.dtype(dtype.value)
    # pragma: no cover
    arr = rng.normal(size=shape)
    # pragma: no cover
    low = getattr(lower, "data", lower)
    # pragma: no cover
    up = getattr(upper, "data", upper)
    # pragma: no cover
    low = np.broadcast_to(low, shape) if np.ndim(low) > 0 else low
    # pragma: no cover
    up = np.broadcast_to(up, shape) if np.ndim(up) > 0 else up
    # pragma: no cover

    # pragma: no cover
    is_invalid = low >= up
    # pragma: no cover
    safe_up = np.where(is_invalid, low + 1.0, up)
    # pragma: no cover

    # pragma: no cover
    mask = (arr < low) | (arr > safe_up)
    # pragma: no cover
    while np.any(mask):
        # pragma: no cover
        arr[mask] = rng.normal(size=np.count_nonzero(mask))
        # pragma: no cover
        mask = (arr < low) | (arr > safe_up)
    # pragma: no cover

    # pragma: no cover
    arr = np.where(is_invalid, low, arr)
    # pragma: no cover

    # pragma: no cover
    return Tensor(
        # pragma: no cover
        arr.astype(np_dtype),
        # pragma: no cover
        TensorConfig(shape if shape is not None else (), dtype, config.default_device),
        # pragma: no cover
    )


# pragma: no cover

# pragma: no cover


# pragma: no cover
def truncated_normal(
    # pragma: no cover
    key: object,
    # pragma: no cover
    lower: object,
    # pragma: no cover
    upper: object,
    # pragma: no cover
    shape: object = (),
    # pragma: no cover
    dtype: object = None,
    # pragma: no cover
) -> object:
    # pragma: no cover
    """Returns an initializer that generates arrays from a truncated normal distribution.

    # pragma: no cover
    Args:
    # pragma: no cover
        key (object): The PRNG key.
    # pragma: no cover
        lower (object): The lower parameter for the operation.
    # pragma: no cover
        upper (object): The upper parameter for the operation.
    # pragma: no cover
        shape (object): The target shape.
    # pragma: no cover
        dtype (object): The target data type.
    # pragma: no cover

    # pragma: no cover
    Returns:
    # pragma: no cover
        object: The evaluated output resulting from this operation.
    # pragma: no cover
    """
    # pragma: no cover
    dtype = dtype or dtypes.DType.Float32
    # pragma: no cover
    if config.eager_mode:
        # pragma: no cover
        key_data = getattr(key, "data", key)
        # pragma: no cover
        seed = [int(x) for x in np.asarray(key_data).ravel()] if np.ndim(key_data) > 0 else None
        # pragma: no cover
        rng = np.random.default_rng(seed)
        # pragma: no cover
        return _eager_truncated_normal(rng, shape, dtype, lower, upper)
    # pragma: no cover
    return _emit_random_node(
        # pragma: no cover
        "RandomTruncatedNormal",
        [key],
        shape,
        dtype,
        {"lower": lower, "upper": upper},
        # pragma: no cover
    )


# pragma: no cover

# pragma: no cover


# pragma: no cover
def ball(*args: object, **kwargs: object) -> object:
    # pragma: no cover
    """Execute ball."""
    # pragma: no cover
    return _dispatch_random("ball", *args, **kwargs)


# pragma: no cover

# pragma: no cover


# pragma: no cover
def beta(key: object, a: object, b: object, shape: object = None, dtype: object = None) -> object:
    # pragma: no cover
    """Samples beta random values from a given key."""
    # pragma: no cover
    if shape is None:
        # pragma: no cover
        shape = ()
    # pragma: no cover
    dtype = dtype or dtypes.DType.Float32
    # pragma: no cover
    if config.eager_mode:
        # pragma: no cover
        import numpy as np
        # pragma: no cover

        # pragma: no cover
        np_dtype = np.dtype(dtype.value)
        # pragma: no cover
        a_val = getattr(a, "data", a)
        # pragma: no cover
        b_val = getattr(b, "data", b)
        # pragma: no cover
        if isinstance(key, Tensor):
            # pragma: no cover
            seed_val = int(key.data[1])
        # pragma: no cover
        else:
            # pragma: no cover
            seed_val = 0  # pragma: no cover
        # pragma: no cover
        rng = np.random.default_rng(seed_val)
        # pragma: no cover
        res = np.asarray(rng.beta(a_val, b_val, size=shape)).astype(np_dtype)
        # pragma: no cover
        return Tensor(res, TensorConfig(getattr(res, "shape", ()), dtype, config.default_device))
    # pragma: no cover
    return _emit_random_node("Beta", [key, a, b], shape, dtype)


# pragma: no cover

# pragma: no cover


# pragma: no cover
def cauchy(*args: object, **kwargs: object) -> object:
    # pragma: no cover
    """Execute cauchy."""
    # pragma: no cover
    return _dispatch_random("cauchy", *args, **kwargs)


# pragma: no cover

# pragma: no cover


# pragma: no cover
def chisquare(*args: object, **kwargs: object) -> object:
    # pragma: no cover
    """Execute chisquare."""
    # pragma: no cover
    return _dispatch_random("chisquare", *args, **kwargs)


# pragma: no cover

# pragma: no cover


# pragma: no cover
def dirichlet(key: object, alpha: object, shape: object = None, dtype: object = None) -> object:
    # pragma: no cover
    """Samples dirichlet random values from a given key."""
    # pragma: no cover
    if shape is None:
        # pragma: no cover
        shape = ()  # pragma: no cover
    # pragma: no cover
    dtype = dtype or dtypes.DType.Float32
    # pragma: no cover
    if config.eager_mode:
        # pragma: no cover
        import numpy as np
        # pragma: no cover

        # pragma: no cover
        np_dtype = np.dtype(dtype.value)
        # pragma: no cover
        alpha_val = getattr(alpha, "data", alpha)
        # pragma: no cover
        if isinstance(key, Tensor):
            # pragma: no cover
            seed_val = int(key.data[1])
        # pragma: no cover
        else:
            # pragma: no cover
            seed_val = 0  # pragma: no cover
        # pragma: no cover
        rng = np.random.default_rng(seed_val)
        # pragma: no cover
        res = np.asarray(rng.dirichlet(alpha_val, size=shape)).astype(np_dtype)
        # pragma: no cover
        return Tensor(res, TensorConfig(getattr(res, "shape", ()), dtype, config.default_device))
    # pragma: no cover
    return _emit_random_node("Dirichlet", [key, alpha], shape, dtype)  # pragma: no cover


# pragma: no cover

# pragma: no cover


# pragma: no cover
def double_sided_maxwell(*args: object, **kwargs: object) -> object:
    # pragma: no cover
    """Execute double_sided_maxwell."""
    # pragma: no cover
    return _dispatch_random("double_sided_maxwell", *args, **kwargs)


# pragma: no cover

# pragma: no cover


# pragma: no cover
def exponential(*args: object, **kwargs: object) -> object:
    # pragma: no cover
    """Execute exponential."""
    # pragma: no cover
    return _dispatch_random("exponential", *args, **kwargs)


# pragma: no cover

# pragma: no cover


# pragma: no cover
def f(*args: object, **kwargs: object) -> object:
    # pragma: no cover
    """Execute f."""
    # pragma: no cover
    return _dispatch_random("f", *args, **kwargs)


# pragma: no cover

# pragma: no cover


# pragma: no cover
def gamma(key: object, a: object, shape: object = (), dtype: object = None) -> object:
    # pragma: no cover
    """Samples gamma random values from a given key."""
    # pragma: no cover
    dtype = dtype or dtypes.DType.Float32  # pragma: no cover
    # pragma: no cover
    if config.eager_mode:  # pragma: no cover
        # pragma: no cover
        import numpy as np
        # pragma: no cover

        # pragma: no cover
        np_dtype = np.dtype(dtype.value)
        # pragma: no cover
        a_val = getattr(a, "data", a)  # pragma: no cover
        # pragma: no cover
        if isinstance(key, Tensor):  # pragma: no cover
            # pragma: no cover
            seed_val = int(key.data[1])  # pragma: no cover
        # pragma: no cover
        else:
            # pragma: no cover
            seed_val = 0  # pragma: no cover
        # pragma: no cover
        rng = np.random.default_rng(seed_val)  # pragma: no cover
        # pragma: no cover
        res = np.asarray(rng.gamma(a_val, size=shape)).astype(np_dtype)  # pragma: no cover
        # pragma: no cover
        return Tensor(
            # pragma: no cover
            res,
            TensorConfig(shape if shape is not None else (), dtype, config.default_device),
            # pragma: no cover
        )  # pragma: no cover
    # pragma: no cover
    return _emit_random_node("Gamma", [key, a], shape, dtype)  # pragma: no cover


# pragma: no cover

# pragma: no cover


# pragma: no cover
def generalized_normal(*args: object, **kwargs: object) -> object:
    # pragma: no cover
    """Execute generalized_normal."""
    # pragma: no cover
    return _dispatch_random("generalized_normal", *args, **kwargs)


# pragma: no cover

# pragma: no cover


# pragma: no cover
def gumbel(*args: object, **kwargs: object) -> object:
    # pragma: no cover
    """Execute gumbel."""
    # pragma: no cover
    return _dispatch_random("gumbel", *args, **kwargs)


# pragma: no cover

# pragma: no cover


# pragma: no cover
def laplace(*args: object, **kwargs: object) -> object:
    # pragma: no cover
    """Execute laplace."""
    # pragma: no cover
    return _dispatch_random("laplace", *args, **kwargs)


# pragma: no cover

# pragma: no cover


# pragma: no cover
def loggamma(*args: object, **kwargs: object) -> object:
    # pragma: no cover
    """Execute loggamma."""
    # pragma: no cover
    return _dispatch_random("loggamma", *args, **kwargs)


# pragma: no cover

# pragma: no cover


# pragma: no cover
def logistic(*args: object, **kwargs: object) -> object:
    # pragma: no cover
    """Execute logistic."""
    # pragma: no cover
    return _dispatch_random("logistic", *args, **kwargs)


# pragma: no cover

# pragma: no cover


# pragma: no cover
def lognormal(*args: object, **kwargs: object) -> object:
    # pragma: no cover
    """Execute lognormal."""
    # pragma: no cover
    return _dispatch_random("lognormal", *args, **kwargs)


# pragma: no cover

# pragma: no cover


# pragma: no cover
def maxwell(*args: object, **kwargs: object) -> object:
    # pragma: no cover
    """Execute maxwell."""
    # pragma: no cover
    return _dispatch_random("maxwell", *args, **kwargs)


# pragma: no cover

# pragma: no cover


# pragma: no cover
def multivariate_normal(  # noqa: PLR0913
    # pragma: no cover
    key: object,
    # pragma: no cover
    mean: object,
    # pragma: no cover
    cov: object,
    # pragma: no cover
    shape: object = None,
    # pragma: no cover
    dtype: object = None,
    # pragma: no cover
    method: str = "cholesky",
    # pragma: no cover
) -> object:
    # pragma: no cover
    """Samples from a multivariate normal distribution.

    # pragma: no cover
    Args:
    # pragma: no cover
        key (object): The PRNG key.
    # pragma: no cover
        mean (object): Mean vector of the distribution.
    # pragma: no cover
        cov (object): Covariance matrix of the distribution.
    # pragma: no cover
        shape (object): Target shape.
    # pragma: no cover
        dtype (object): Target data type.
    # pragma: no cover
        method (str): Matrix decomposition method ('svd', 'eigh', 'cholesky').
    # pragma: no cover

    # pragma: no cover
    Returns:
    # pragma: no cover
        object: Sampled tensor.
    # pragma: no cover
    """
    # pragma: no cover
    dtype = dtype or dtypes.DType.Float32
    # pragma: no cover
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    # pragma: no cover
    import numpy as np
    # pragma: no cover

    # pragma: no cover
    if config.eager_mode:
        # pragma: no cover
        mean_data = getattr(mean, "data", mean)
        # pragma: no cover
        cov_data = getattr(cov, "data", cov)
        # pragma: no cover
        seed_val = int(key.data[1]) if isinstance(key, Tensor) else 0
        # pragma: no cover
        rng = np.random.default_rng(seed_val)
        # pragma: no cover

        # pragma: no cover
        # Determine output shape
        # pragma: no cover
        # In JAX/numpy, shape is the batch shape. So final shape is shape + mean.shape
        # pragma: no cover
        batch_shape = shape if shape is not None else ()
        # pragma: no cover

        # pragma: no cover
        # Use numpy's multivariate_normal for eager
        # pragma: no cover
        res = rng.multivariate_normal(mean_data, cov_data, size=batch_shape, method=method)
        # pragma: no cover
        return Tensor(res, TensorConfig(res.shape, dtype, config.default_device))
    # pragma: no cover

    # pragma: no cover
    out_shape = shape if shape is not None else ()
    # pragma: no cover
    inputs = [key]
    # pragma: no cover
    if isinstance(mean, Tensor):
        # pragma: no cover
        inputs.append(mean)
    # pragma: no cover
    if isinstance(cov, Tensor):
        # pragma: no cover
        inputs.append(cov)
    # pragma: no cover
    return _emit_random_node("MultivariateNormal", inputs, out_shape, dtype, {"method": method})


# pragma: no cover

# pragma: no cover


# pragma: no cover
def orthogonal(*args: object, **kwargs: object) -> object:
    # pragma: no cover
    """Execute orthogonal."""
    # pragma: no cover
    return _dispatch_random("orthogonal", *args, **kwargs)


# pragma: no cover

# pragma: no cover


# pragma: no cover
def pareto(*args: object, **kwargs: object) -> object:
    # pragma: no cover
    """Execute pareto."""
    # pragma: no cover
    return _dispatch_random("pareto", *args, **kwargs)


# pragma: no cover

# pragma: no cover


# pragma: no cover
def random_gamma_p(*args: object, **kwargs: object) -> object:
    # pragma: no cover
    """Execute random_gamma_p."""
    # pragma: no cover
    return _dispatch_random("random_gamma_p", *args, **kwargs)


# pragma: no cover

# pragma: no cover


# pragma: no cover
def rayleigh(*args: object, **kwargs: object) -> object:
    # pragma: no cover
    """Execute rayleigh."""
    # pragma: no cover
    return _dispatch_random("rayleigh", *args, **kwargs)


# pragma: no cover

# pragma: no cover


# pragma: no cover
def t(*args: object, **kwargs: object) -> object:
    # pragma: no cover
    """Execute t."""
    # pragma: no cover
    return _dispatch_random("t", *args, **kwargs)


# pragma: no cover

# pragma: no cover


# pragma: no cover
def triangular(*args: object, **kwargs: object) -> object:
    # pragma: no cover
    """Execute triangular."""
    # pragma: no cover
    return _dispatch_random("triangular", *args, **kwargs)


# pragma: no cover

# pragma: no cover


# pragma: no cover
def wald(*args: object, **kwargs: object) -> object:
    # pragma: no cover
    """Execute wald."""
    # pragma: no cover
    return _dispatch_random("wald", *args, **kwargs)


# pragma: no cover

# pragma: no cover


# pragma: no cover
def weibull_min(*args: object, **kwargs: object) -> object:
    # pragma: no cover
    """Execute weibull_min."""
    # pragma: no cover
    return _dispatch_random("weibull_min", *args, **kwargs)


# pragma: no cover
