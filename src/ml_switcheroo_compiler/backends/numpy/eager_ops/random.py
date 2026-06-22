"""Module docstring."""

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry


def _randint(*args: object, **kwargs: object) -> object:
    r"""Execute _randint.\n\n    Args:\n        cls (Any): The class.\n        *args (Any): Argument *args.\n        **kwargs (Any): Argument **kwargs.\n\n    Returns:\n    Any: The result.\n."""
    size = kwargs.get("size")
    if (size is None) and (len(args) > 2):
        size = args[2]
    if size is None:
        res = np.random.randint(*(args[:2] if (len(args) > 1) else args[:1]))
    else:
        res = np.random.randint(*(args[:2] if (len(args) > 1) else args[:1]), size=size)
    dt = getattr(kwargs.get("dtype", np.int64), "value", kwargs.get("dtype", np.int64))
    if dt is None:
        dt = np.int64
    return np.asarray(res).astype(dt)


@numpy_eager_registry.register("Rand")
def _np_rand(backend_module: object, *args: object, **kwargs: object) -> object:
    dtype = kwargs.get("dtype", getattr(backend_module, "float32", None))
    dtype_str = str(dtype).split(".")[(-1)]
    dt = getattr(backend_module, dtype_str, dtype)
    return backend_module.random.rand(*args).astype(dt)


@numpy_eager_registry.register("Randn")
def _np_randn(backend_module: object, *args: object, **kwargs: object) -> object:
    dtype = kwargs.get("dtype", getattr(backend_module, "float32", None))
    dtype_str = str(dtype).split(".")[(-1)]
    dt = getattr(backend_module, dtype_str, dtype)
    return backend_module.random.randn(*args).astype(dt)


@numpy_eager_registry.register("Seed")
def _np_seed(backend_module: object, seed: object) -> object:
    backend_module.random.seed(seed)
    return seed


@numpy_eager_registry.register("ManualSeed")
def _np_manual_seed(backend_module: object, seed: object) -> object:
    backend_module.random.seed(seed)
    return seed


@numpy_eager_registry.register("Randint")
def _np_randint(backend_module: object, *args: object, **kwargs: object) -> object:
    dtype = kwargs.pop("dtype", None)
    res = backend_module.random.randint(*args, **kwargs)
    if dtype is not None:
        dtype_str = str(dtype).split(".")[(-1)]
        dt = getattr(backend_module, dtype_str, dtype)
        res = res.astype(dt)
    return res


@numpy_eager_registry.register("Dropout")
def dropout(
    np_mod: object,
    x: object,
    rate: object,
    noise_shape: object = None,
    training: object = True,
    seed: object = None,
) -> object:
    """Execute Dropout eager op.

    Args:
        np_mod: Numpy module
        x: Input array
        rate: Dropout rate
        noise_shape: Noise shape
        training: Training boolean
        seed: Random seed

    Returns:
        Resulting array
    """
    if rate == 0.0 or not training:
        return x
    keep_prob = 1.0 - rate
    if seed is not None:
        rng = np_mod.random.RandomState(seed)
    else:
        rng = np_mod.random
    shape = noise_shape if noise_shape is not None else getattr(x, "shape", ())
    mask = rng.binomial(1, keep_prob, size=shape)
    if hasattr(x, "dtype"):
        mask = mask.astype(x.dtype)
    return (x * mask) / keep_prob
