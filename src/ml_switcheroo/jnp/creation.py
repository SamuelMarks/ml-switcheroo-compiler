"""Docstring."""

import ml_switcheroo.ops as ops
from ml_switcheroo.jnp.array import ndarray, _to_tensor, _wrap
from ml_switcheroo.jnp.math_ops import power


def zeros_like(x: object, dtype: object = None) -> object:
    """Return an array of zeros with the same shape and type as a given array.

    Args:
        x (Any): Argument x.
        dtype (Any): Argument dtype.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.zeros_like(_to_tensor(x), dtype=dtype))


def zeros(shape: object, dtype: object = None) -> object:
    """Return a new array of given shape and type, filled with zeros.

    Args:
        shape (Any): Argument shape.
        dtype (Any): Argument dtype.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.zeros(shape=shape, dtype=dtype))


def array(x: object, dtype: object = None) -> object:
    """Create an array.

    Args:
        x (Any): Argument x.
        dtype (Any): Argument dtype.

    Returns:
        Any: The result of the operation.
    """
    if isinstance(x, ndarray):
        return x
    return _wrap(_to_tensor(x))


def ones(shape: object, dtype: object = None) -> object:
    """Return a new array of given shape and type, filled with ones.

    Args:
        shape (Any): Argument shape.
        dtype (Any): Argument dtype.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.ones(shape=shape, dtype=dtype))


def empty(shape: object, dtype: object = None) -> object:
    """Return a new array of given shape and type, without initializing entries.

    Args:
        shape (Any): Argument shape.
        dtype (Any): Argument dtype.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.empty(shape=shape, dtype=dtype))


def full(shape: object, fill_value: object, dtype: object = None) -> object:
    """Return a new array of given shape and type, filled with fill_value.

    Args:
        shape (Any): Argument shape.
        fill_value (Any): Argument fill_value.
        dtype (Any): Argument dtype.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.full(shape=shape, fill_value=fill_value, dtype=dtype))


def ones_like(x: object, dtype: object = None) -> object:
    """Return an array of ones with the same shape and type as a given array.

    Args:
        x (Any): Argument x.
        dtype (Any): Argument dtype.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.ones_like(_to_tensor(x), dtype=dtype))


def empty_like(x: object, dtype: object = None) -> object:
    """Return a new array with the same shape and type as a given array.

    Args:
        x (Any): Argument x.
        dtype (Any): Argument dtype.

    Returns:
        Any: The result of the operation.
    """
    t = _to_tensor(x)
    return _wrap(ops.empty(shape=t.shape, dtype=dtype if dtype else t.dtype))


def full_like(x: object, fill_value: object, dtype: object = None) -> object:
    """Return a full array with the same shape and type as a given array.

    Args:
        x (Any): Argument x.
        fill_value (Any): Argument fill_value.
        dtype (Any): Argument dtype.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.full_like(_to_tensor(x), fill_value=fill_value, dtype=dtype))


def asarray(x: object, dtype: object = None) -> object:
    """Convert the input to an array.

    Args:
        x (Any): Argument x.
        dtype (Any): Argument dtype.

    Returns:
        Any: The result of the operation.
    """
    return array(x, dtype=dtype)


def arange(
    start: object, stop: object = None, step: object = 1, dtype: object = None
) -> object:
    """Return evenly spaced values within a given interval.

    Args:
        start (Any): Argument start.
        stop (Any): Argument stop.
        step (Any): Argument step.
        dtype (Any): Argument dtype.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.arange(start=start, stop=stop, step=step, dtype=dtype))


def linspace(
    start: object,
    stop: object,
    num: int = 50,
    endpoint: bool = True,
    retstep: bool = False,
    dtype: object = None,
    axis: int = 0,
) -> object:
    """Return evenly spaced numbers over a specified interval.

    Args:
        start (Any): Argument start.
        stop (Any): Argument stop.
        num (Any): Argument num.
        endpoint (Any): Argument endpoint.
        retstep (Any): Argument retstep.
        dtype (Any): Argument dtype.
        axis (Any): Argument axis.

    Returns:
        Any: The result of the operation.
    """
    if retstep or axis != 0 or not endpoint:
        raise NotImplementedError("linspace currently only supports basic usage")
    return _wrap(ops.linspace(start=start, stop=stop, steps=num, dtype=dtype))


def logspace(
    start: object,
    stop: object,
    num: int = 50,
    endpoint: bool = True,
    base: float = 10.0,
    dtype: object = None,
    axis: int = 0,
) -> object:
    """Return numbers spaced evenly on a log scale.

    Args:
        start (Any): Argument start.
        stop (Any): Argument stop.
        num (Any): Argument num.
        endpoint (Any): Argument endpoint.
        base (Any): Argument base.
        dtype (Any): Argument dtype.
        axis (Any): Argument axis.

    Returns:
        Any: The result of the operation.
    """
    lin = linspace(start, stop, num, endpoint=endpoint, dtype=dtype, axis=axis)
    return power(base, lin)


def eye(N: int, M: int = None, k: int = 0, dtype: object = None) -> object:
    """Return a 2-D array with ones on the diagonal and zeros elsewhere.

    Args:
        N (Any): Argument N.
        M (Any): Argument M.
        k (Any): Argument k.
        dtype (Any): Argument dtype.

    Returns:
        Any: The result of the operation.
    """
    if k != 0:
        raise NotImplementedError()
    return _wrap(ops.eye(n=N, m=M, dtype=dtype))


def identity(n: int, dtype: object = None) -> object:
    """Return the identity array.

    Args:
        n (Any): Argument n.
        dtype (Any): Argument dtype.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.identity(n=n, dtype=dtype))


def meshgrid(
    *xi: object, copy: object = True, sparse: object = False, indexing: object = "xy"
) -> object:
    """Return coordinate matrices from coordinate vectors.

    Returns:
        Any: The result of the operation.
    """
    if sparse or not copy:
        raise NotImplementedError()

    tensors = [_to_tensor(x) for x in xi]

    ndim = len(tensors)
    if indexing == "xy" and ndim > 1:
        tensors[0], tensors[1] = tensors[1], tensors[0]

    s0 = (1,) * ndim
    output = []
    for i, t in enumerate(tensors):
        shape = list(s0)
        shape[i] = -1
        reshaped = ops.reshape(t, shape=tuple(shape))
        output.append(reshaped)

    broadcast_shape = tuple(t.shape[0] for t in tensors)
    output = [ops.broadcast_to(t, size=broadcast_shape) for t in output]

    if indexing == "xy" and ndim > 1:
        output[0], output[1] = output[1], output[0]

    return tuple(_wrap(t) for t in output)
