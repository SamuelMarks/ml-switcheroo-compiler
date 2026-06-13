"""Random number generation and state management."""

import numpy as np

from ml_switcheroo_compiler.core.tensor import Tensor


def PRNGKey(seed: int) -> Tensor:
    """Creates a PRNG key given an integer seed.

    Args:
        seed (int): The seed.

    Returns:
        Tensor: The computed result.
    """
    from ml_switcheroo_compiler.core import dtype
    from ml_switcheroo_compiler.core.device import Device

    return Tensor(np.array([0, seed], dtype=np.uint32), (2,), dtype.DType.Int32, Device("cpu"))


def split(key: Tensor, num: int = 2) -> tuple[object, ...]:
    """Splits a PRNG key into num new keys.

    Args:
        key (Tensor): The key.
        num (int): The num.

    Returns:
        tuple[object, ...]: The computed result.
    """
    from ml_switcheroo_compiler.core import dtype
    from ml_switcheroo_compiler.core.device import Device

    # dummy implementation
    return Tensor(np.zeros((num, 2), dtype=np.uint32), (num, 2), dtype.DType.Int32, Device("cpu"))


def fold_in(key: Tensor, data: int) -> Tensor:
    """Folds in data to a PRNG key to derive a new key.

    Args:
        key (Tensor): The key.
        data (int): The data.

    Returns:
        Tensor: The computed result.
    """
    return PRNGKey(int(key.data[1] if hasattr(key.data, "__getitem__") else 0) + data)


def uniform(
    key: object,
    shape: object = (),
    dtype: object = None,
    minval: object = 0.0,
    maxval: object = 1.0,
) -> object:
    """Samples uniform random values from a given key.

    Args:
        key (object): The key.
        shape (object): The shape.
        dtype (object): The dtype.
        minval (object): The minval.
        maxval (object): The maxval.

    Returns:
        object: The computed result.
    """
    import ml_switcheroo_compiler.core.dtype as dtypes
    from ml_switcheroo_compiler.core.device import Device

    dtype = dtype or dtypes.DType.Float32
    # Mock eager impl
    return Tensor(np.random.uniform(minval, maxval, shape), shape, dtype, Device("cpu"))


def normal(key: object, shape: object = (), dtype: object = None) -> object:
    """Samples standard normal random values from a given key.

    Args:
        key (object): The key.
        shape (object): The shape.
        dtype (object): The dtype.

    Returns:
        object: The computed result.
    """
    import ml_switcheroo_compiler.core.dtype as dtypes
    from ml_switcheroo_compiler.core.device import Device

    dtype = dtype or dtypes.DType.Float32
    return Tensor(np.random.normal(0, 1, shape), shape, dtype, Device("cpu"))


def randint(
    key: object,
    shape: object,
    minval: object,
    maxval: object,
    dtype: object = None,
) -> object:
    """Samples uniform random integers from a given key.

    Args:
        key (object): The key.
        shape (object): The shape.
        minval (object): The minval.
        maxval (object): The maxval.
        dtype (object): The dtype.

    Returns:
        object: The computed result.
    """
    import ml_switcheroo_compiler.core.dtype as dtypes
    from ml_switcheroo_compiler.core.device import Device

    dtype = dtype or dtypes.DType.Int32
    return Tensor(np.random.randint(minval, maxval, shape), shape, dtype, Device("cpu"))


def bernoulli(key: object, p: object = 0.5, shape: object = None) -> object:
    """Samples Bernoulli random variables from a given key.

    Args:
        key (object): The key.
        p (object): The p.
        shape (object): The shape.

    Returns:
        object: The computed result.
    """
    import ml_switcheroo_compiler.core.dtype as dtypes
    from ml_switcheroo_compiler.core.device import Device

    if shape is None:
        shape = ()
    return Tensor(np.random.binomial(1, p, shape), shape, dtypes.DType.Bool, Device("cpu"))


def categorical(key: object, logits: object, axis: object = -1, shape: object = None) -> object:
    """Samples categorical random variables from a given key.

    Args:
        key (object): The key.
        logits (object): The logits.
        axis (object): The axis.
        shape (object): The shape.

    Returns:
        object: The computed result.
    """
    # Dummy mock
    import ml_switcheroo_compiler.core.dtype as dtypes
    from ml_switcheroo_compiler.core.device import Device

    return Tensor(np.zeros(shape or ()), shape or (), dtypes.DType.Int32, Device("cpu"))


def permutation(key: object, x: object, axis: object = 0, independent: object = False) -> object:
    """Randomly permutes a sequence or array.

    Args:
        key (object): The key.
        x (object): The x.
        axis (object): The axis.
        independent (object): The independent.

    Returns:
        object: The computed result.
    """
    # Mock
    return x


def choice(
    key: object,
    a: object,
    shape: object = (),
    replace: object = True,
    p: object = None,
    axis: object = 0,
) -> object:
    """Generates a random sample from a given 1-D array.

    Args:
        key (object): The key.
        a (object): The a.
        shape (object): The shape.
        replace (object): The replace.
        p (object): The p.
        axis (object): The axis.

    Returns:
        object: The computed result.
    """
    return a


def truncated_normal(
    key: object,
    lower: object,
    upper: object,
    shape: object = (),
    dtype: object = None,
) -> object:
    """Returns an initializer that generates arrays from a truncated normal distribution.

    Args:
        key (object): The key.
        lower (object): The lower.
        upper (object): The upper.
        shape (object): The shape.
        dtype (object): The dtype.

    Returns:
        object: The computed result.
    """
    import ml_switcheroo_compiler.core.dtype as dtypes
    from ml_switcheroo_compiler.core.device import Device

    dtype = dtype or dtypes.DType.Float32
    from scipy.stats import truncnorm

    out = truncnorm.rvs(lower, upper, size=shape)
    return Tensor(out, shape, dtype, Device("cpu"))
