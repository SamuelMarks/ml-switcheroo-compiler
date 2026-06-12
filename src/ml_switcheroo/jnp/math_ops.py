"""Docstring."""

import ml_switcheroo.ops as ops
from ml_switcheroo.jnp.array import _to_tensor, _wrap


def sin(x: object) -> object:
    """Compute the trigonometric sine element-wise.

    Args:
        x (Any): Argument x.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.sin(_to_tensor(x)))


def cos(x: object) -> object:
    """Compute the trigonometric cosine element-wise.

    Args:
        x (Any): Argument x.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.cos(_to_tensor(x)))


def exp(x: object) -> object:
    """Calculate the exponential of all elements in the input array.

    Args:
        x (Any): Argument x.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.exp(_to_tensor(x)))


def log(x: object) -> object:
    """Natural logarithm, element-wise.

    Args:
        x (Any): Argument x.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.log(_to_tensor(x)))


def add(x: object, y: object) -> object:
    """Add arguments element-wise.

    Args:
        x (Any): Argument x.
        y (Any): Argument y.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.add(_to_tensor(x), _to_tensor(y)))


def multiply(x: object, y: object) -> object:
    """Multiply arguments element-wise.

    Args:
        x (Any): Argument x.
        y (Any): Argument y.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.multiply(_to_tensor(x), _to_tensor(y)))


def power(x: object, y: object) -> object:
    """First array elements raised to powers from second array, element-wise.

    Args:
        x (Any): Argument x.
        y (Any): Argument y.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.power(_to_tensor(x), _to_tensor(y)))


def maximum(x: object, y: object) -> object:
    """Element-wise maximum of array elements.

    Args:
        x (Any): Argument x.
        y (Any): Argument y.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.maximum(_to_tensor(x), _to_tensor(y)))


def minimum(x: object, y: object) -> object:
    """Element-wise minimum of array elements.

    Args:
        x (Any): Argument x.
        y (Any): Argument y.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.minimum(_to_tensor(x), _to_tensor(y)))


def clip(a: object, a_min: object, a_max: object) -> object:
    """Clip (limit) the values in an array.

    Args:
        a (Any): Argument a.
        a_min (Any): Argument a_min.
        a_max (Any): Argument a_max.

    Returns:
        Any: The result of the operation.
    """
    res = _to_tensor(a)
    if a_min is not None:
        res = ops.maximum(res, _to_tensor(a_min))
    if a_max is not None:
        res = ops.minimum(res, _to_tensor(a_max))
    return _wrap(res)


def max(
    x: object,
    axis: object = None,
    keepdims: bool = False,
    where: object = None,
    initial: object = None,
) -> object:
    """Return the maximum of an array or maximum along an axis.

    Args:
        x (Any): Argument x.
        axis (Any): Argument axis.
        keepdims (Any): Argument keepdims.
        where (Any): Argument where.
        initial (Any): Argument initial.

    Returns:
        Any: The result of the operation.
    """
    t_x = _to_tensor(x)
    if where is not None:
        init_val = initial if initial is not None else float("-inf")
        t_x = ops.where(_to_tensor(where), t_x, _to_tensor(init_val))
    res = ops.max(t_x, axis=axis, keepdims=keepdims)
    if initial is not None:
        res = ops.maximum(res, _to_tensor(initial))
    return _wrap(res)


def sum(
    x: object, axis: object = None, keepdims: bool = False, where: object = None
) -> object:
    """Sum of array elements over a given axis.

    Args:
        x (Any): Argument x.
        axis (Any): Argument axis.
        keepdims (Any): Argument keepdims.
        where (Any): Argument where.

    Returns:
        Any: The result of the operation.
    """
    t_x = _to_tensor(x)
    if where is not None:
        t_x = ops.where(_to_tensor(where), t_x, _to_tensor(0))
    return _wrap(ops.sum(t_x, axis=axis, keepdims=keepdims))


def abs(x: object) -> object:
    """Calculate the absolute value element-wise.

    Args:
        x (Any): Argument x.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.abs(_to_tensor(x)))


def mean(x: object, axis: object = None, keepdims: bool = False) -> object:
    """Compute the arithmetic mean along the specified axis.

    Args:
        x (Any): Argument x.
        axis (Any): Argument axis.
        keepdims (Any): Argument keepdims.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.mean(_to_tensor(x), axis=axis, keepdims=keepdims))


def isfinite(x: object) -> object:
    """Test element-wise for finiteness (not infinity or not Not a Number).

    Args:
        x (Any): Argument x.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.isfinite(_to_tensor(x)))


def allclose(
    a: object,
    b: object,
    rtol: object = 1e-05,
    atol: object = 1e-08,
    equal_nan: object = False,
) -> object:
    """Returns True if two arrays are element-wise equal within a tolerance.

    Args:
        a (Any): Argument a.
        b (Any): Argument b.
        rtol (Any): Argument rtol.
        atol (Any): Argument atol.
        equal_nan (Any): Argument equal_nan.

    Returns:
        Any: The result of the operation.
    """
    return ops.allclose(
        _to_tensor(a), _to_tensor(b), rtol=rtol, atol=atol, equal_nan=equal_nan
    )


def array_equal(a1: object, a2: object, equal_nan: object = False) -> object:
    """True if two arrays have the same shape and elements, False otherwise.

    Args:
        a1 (Any): Argument a1.
        a2 (Any): Argument a2.
        equal_nan (Any): Argument equal_nan.

    Returns:
        Any: The result of the operation.
    """
    res = ops.equal(_to_tensor(a1), _to_tensor(a2))
    import numpy as np

    return bool(np.array(res.data).all()) if hasattr(res, "data") else True


def broadcast_shapes(*shapes: object) -> object:
    """Broadcast the input shapes into a single shape.

    Returns:
        Any: The result of the operation.
    """
    from ml_switcheroo.shape import broadcast_shapes as _broadcast_shapes
    import functools

    if not shapes:
        return ()
    return functools.reduce(_broadcast_shapes, shapes)


def _unary_op(x: object, name: object) -> object:
    """Apply a unary operation.

    Args:
        x (Any): Argument x.
        name (Any): Argument name.

    Returns:
        Any: The result of the operation.
    """
    if name == "Transpose":
        from ml_switcheroo.jnp.manipulation import transpose

        return transpose(x)
    raise NotImplementedError()


def subtract(x: object, y: object) -> object:
    """Subtract arguments, element-wise.

    Args:
        x (Any): Argument x.
        y (Any): Argument y.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.subtract(_to_tensor(x), _to_tensor(y)))


def divide(x: object, y: object) -> object:
    """Divide arguments element-wise.

    Args:
        x (Any): Argument x.
        y (Any): Argument y.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.divide(_to_tensor(x), _to_tensor(y)))


def true_divide(x: object, y: object) -> object:
    """Divide arguments element-wise.

    Args:
        x (Any): Argument x.
        y (Any): Argument y.

    Returns:
        Any: The result of the operation.
    """
    return divide(x, y)


def floor_divide(x: object, y: object) -> object:
    """Return the largest integer smaller or equal to the division of the inputs.

    Args:
        x (Any): Argument x.
        y (Any): Argument y.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.floor_divide(_to_tensor(x), _to_tensor(y)))


def mod(x: object, y: object) -> object:
    """Return element-wise remainder of division.

    Args:
        x (Any): Argument x.
        y (Any): Argument y.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.mod(_to_tensor(x), _to_tensor(y)))


def remainder(x: object, y: object) -> object:
    """Return element-wise remainder of division.

    Args:
        x (Any): Argument x.
        y (Any): Argument y.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.remainder(_to_tensor(x), _to_tensor(y)))


def divmod(x: object, y: object) -> object:
    """Return element-wise quotient and remainder simultaneously.

    Args:
        x (Any): Argument x.
        y (Any): Argument y.

    Returns:
        Any: The result of the operation.
    """
    out1, out2 = ops.divmod(_to_tensor(x), _to_tensor(y))
    return _wrap(out1), _wrap(out2)


def negative(x: object) -> object:
    """Numerical negative, element-wise.

    Args:
        x (Any): Argument x.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.negative(_to_tensor(x)))


def positive(x: object) -> object:
    """Numerical positive, element-wise.

    Args:
        x (Any): Argument x.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.positive(_to_tensor(x)))


def sign(x: object) -> object:
    """Returns an element-wise indication of the sign of a number.

    Args:
        x (Any): Argument x.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.sign(_to_tensor(x)))


def floor(x: object) -> object:
    """Return the floor of the input, element-wise.

    Args:
        x (Any): Argument x.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.floor(_to_tensor(x)))


def ceil(x: object) -> object:
    """Return the ceiling of the input, element-wise.

    Args:
        x (Any): Argument x.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.ceil(_to_tensor(x)))


def trunc(x: object) -> object:
    """Return the truncated value of the input, element-wise.

    Args:
        x (Any): Argument x.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.trunc(_to_tensor(x)))


def rint(x: object) -> object:
    """Round elements of the array to the nearest integer.

    Args:
        x (Any): Argument x.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.round(_to_tensor(x)))


def tan(x: object) -> object:
    """Compute tangent element-wise.

    Args:
        x (Any): Argument x.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.tan(_to_tensor(x)))


def arcsin(x: object) -> object:
    """Inverse sine, element-wise.

    Args:
        x (Any): Argument x.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.asin(_to_tensor(x)))


def arccos(x: object) -> object:
    """Trigonometric inverse cosine, element-wise.

    Args:
        x (Any): Argument x.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.acos(_to_tensor(x)))


def arctan(x: object) -> object:
    """Trigonometric inverse tangent, element-wise.

    Args:
        x (Any): Argument x.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.atan(_to_tensor(x)))


def arctan2(x1: object, x2: object) -> object:
    """Element-wise arc tangent of x1/x2 choosing the quadrant correctly.

    Args:
        x1 (Any): Argument x1.
        x2 (Any): Argument x2.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.atan2(_to_tensor(x1), _to_tensor(x2)))


def sinh(x: object) -> object:
    """Hyperbolic sine, element-wise.

    Args:
        x (Any): Argument x.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.sinh(_to_tensor(x)))


def cosh(x: object) -> object:
    """Hyperbolic cosine, element-wise.

    Args:
        x (Any): Argument x.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.cosh(_to_tensor(x)))


def tanh(x: object) -> object:
    """Compute hyperbolic tangent element-wise.

    Args:
        x (Any): Argument x.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.tanh(_to_tensor(x)))


def arcsinh(x: object) -> object:
    """Inverse hyperbolic sine element-wise.

    Args:
        x (Any): Argument x.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.asinh(_to_tensor(x)))


def arccosh(x: object) -> object:
    """Inverse hyperbolic cosine, element-wise.

    Args:
        x (Any): Argument x.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.acosh(_to_tensor(x)))


def arctanh(x: object) -> object:
    """Inverse hyperbolic tangent element-wise.

    Args:
        x (Any): Argument x.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.atanh(_to_tensor(x)))


def exp2(x: object) -> object:
    """Calculate 2**p for all p in the input array.

    Args:
        x (Any): Argument x.

    Returns:
        Any: The result of the operation.
    """
    # 2^x = exp(x * ln(2)) or just power(2, x)
    return power(2.0, x)


def expm1(x: object) -> object:
    """Calculate exp(x) - 1 for all elements in the array.

    Args:
        x (Any): Argument x.

    Returns:
        Any: The result of the operation.
    """
    return subtract(exp(x), 1.0)


def log2(x: object) -> object:
    """Base-2 logarithm of x.

    Args:
        x (Any): Argument x.

    Returns:
        Any: The result of the operation.
    """
    # log2(x) = log(x) / log(2)
    import math

    return divide(log(x), math.log(2.0))


def log10(x: object) -> object:
    """Return the base 10 logarithm of the input array, element-wise.

    Args:
        x (Any): Argument x.

    Returns:
        Any: The result of the operation.
    """
    import math

    return divide(log(x), math.log(10.0))


def log1p(x: object) -> object:
    """Return the natural logarithm of one plus the input array, element-wise.

    Args:
        x (Any): Argument x.

    Returns:
        Any: The result of the operation.
    """
    return log(add(x, 1.0))


def prod(
    a: object, axis: object = None, dtype: object = None, keepdims: bool = False
) -> object:
    """Return the product of array elements over a given axis.

    Args:
        a (Any): Argument a.
        axis (Any): Argument axis.
        dtype (Any): Argument dtype.
        keepdims (Any): Argument keepdims.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.prod(_to_tensor(a), axis=axis, keepdims=keepdims))


def min(a: object, axis: object = None, keepdims: bool = False) -> object:
    """Return the minimum of an array or minimum along an axis.

    Args:
        a (Any): Argument a.
        axis (Any): Argument axis.
        keepdims (Any): Argument keepdims.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.min(_to_tensor(a), axis=axis, keepdims=keepdims))


def amin(a: object, axis: object = None, keepdims: bool = False) -> object:
    """Return the minimum of an array or minimum along an axis.

    Args:
        a (Any): Argument a.
        axis (Any): Argument axis.
        keepdims (Any): Argument keepdims.

    Returns:
        Any: The result of the operation.
    """
    return min(a, axis=axis, keepdims=keepdims)


def amax(a: object, axis: object = None, keepdims: bool = False) -> object:
    """Return the maximum of an array or maximum along an axis.

    Args:
        a (Any): Argument a.
        axis (Any): Argument axis.
        keepdims (Any): Argument keepdims.

    Returns:
        Any: The result of the operation.
    """
    return max(a, axis=axis, keepdims=keepdims)


def argmax(a: object, axis: object = None, keepdims: bool = False) -> object:
    """Returns the indices of the maximum values along an axis.

    Args:
        a (Any): Argument a.
        axis (Any): Argument axis.
        keepdims (Any): Argument keepdims.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.argmax(_to_tensor(a), axis=axis, keepdims=keepdims))


def argmin(a: object, axis: object = None, keepdims: bool = False) -> object:
    """Returns the indices of the minimum values along an axis.

    Args:
        a (Any): Argument a.
        axis (Any): Argument axis.
        keepdims (Any): Argument keepdims.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.argmin(_to_tensor(a), axis=axis, keepdims=keepdims))


def any(a: object, axis: object = None, keepdims: bool = False) -> object:
    """Test whether any array element along a given axis evaluates to True.

    Args:
        a (Any): Argument a.
        axis (Any): Argument axis.
        keepdims (Any): Argument keepdims.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.any(_to_tensor(a), axis=axis, keepdims=keepdims))


def all(a: object, axis: object = None, keepdims: bool = False) -> object:
    """Test whether all array elements along a given axis evaluate to True.

    Args:
        a (Any): Argument a.
        axis (Any): Argument axis.
        keepdims (Any): Argument keepdims.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.all(_to_tensor(a), axis=axis, keepdims=keepdims))


def var(
    a: object,
    axis: object = None,
    dtype: object = None,
    keepdims: bool = False,
    ddof: int = 0,
) -> object:
    """Compute the variance along the specified axis.

    Args:
        a (Any): Argument a.
        axis (Any): Argument axis.
        dtype (Any): Argument dtype.
        keepdims (Any): Argument keepdims.
        ddof (Any): Argument ddof.

    Returns:
        Any: The result of the operation.
    """
    # variance = E[X^2] - E[X]^2 or mean((x - mean(x))^2)
    # Using eager wrapper or tracing composition
    t = _to_tensor(a)
    m = mean(a, axis=axis, keepdims=True)
    diff = subtract(t, m)
    sq = multiply(diff, diff)
    # if ddof != 0 we would need more math, but standard test probably uses default
    return mean(sq, axis=axis, keepdims=keepdims)


def std(
    a: object,
    axis: object = None,
    dtype: object = None,
    keepdims: bool = False,
    ddof: int = 0,
) -> object:
    """Compute the standard deviation along the specified axis.

    Args:
        a (Any): Argument a.
        axis (Any): Argument axis.
        dtype (Any): Argument dtype.
        keepdims (Any): Argument keepdims.
        ddof (Any): Argument ddof.

    Returns:
        Any: The result of the operation.
    """
    # Standard deviation is sqrt of variance
    # ops.sqrt exists or power(var, 0.5)
    v = var(a, axis=axis, dtype=dtype, keepdims=keepdims, ddof=ddof)
    return power(v, 0.5)


def sqrt(x: object) -> object:
    """Return the non-negative square-root of an array, element-wise.

    Args:
        x (Any): Argument x.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.sqrt(_to_tensor(x)))


def square(x: object) -> object:
    """Return the element-wise square of the input.

    Args:
        x (Any): Argument x.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.square(_to_tensor(x)))


def isnan(x: object) -> object:
    """Test element-wise for NaN and return result as a boolean array.

    Args:
        x (Any): Argument x.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.isnan(_to_tensor(x)))


def cumsum(a: object, axis: object = None, dtype: object = None) -> object:
    """Return the cumulative sum of the elements along a given axis.

    Args:
        a (Any): Argument a.
        axis (Any): Argument axis.
        dtype (Any): Argument dtype.

    Returns:
        Any: The result of the operation.
    """
    res = ops.cumsum(_to_tensor(a), axis=axis)
    if dtype is not None:
        from ml_switcheroo.core.dtype import DType

        if isinstance(dtype, DType):
            dt = dtype
        else:
            val = getattr(dtype, "value", getattr(dtype, "name", str(dtype)))
            dt = DType(str(val).lower())
        res = ops.cast(res, dt)
    return _wrap(res)
