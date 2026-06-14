"""Alias operations."""

import math
from ml_switcheroo_compiler.core.shape import broadcast_shapes as _bs
from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.ops.binary import maximum, minimum, power
from ml_switcheroo_compiler.ops.creation.frontend import linspace
from ml_switcheroo_compiler.ops.shape.frontend import broadcast_to
from ml_switcheroo_compiler.ops.unary import round
from ml_switcheroo_compiler.ops.unary import asin, acos, atan, atan2, asinh, acosh, atanh
from ml_switcheroo_compiler.ops.reductions import min, max, variance
from ml_switcheroo_compiler.ops.shape import unsqueeze

arcsin = asin
arccos = acos
arctan = atan
arctan2 = atan2
arcsinh = asinh
arccosh = acosh
arctanh = atanh
amin = min
amax = max
var = variance
expand_dims = unsqueeze


def clamp(min_val: object, x: object, max_val: object) -> object:
    """Execute clamp.

    Args:
        min_val (Any): Argument min_val.
        x (Any): Argument x.
        max_val (Any): Argument max_val.

    Returns:
    Any: The result.
    """
    if min_val is not None:
        x = maximum(x, min_val)
    if max_val is not None:
        x = minimum(x, max_val)
    return x


def clip(a: object, a_min: object = None, a_max: object = None) -> object:
    """Execute clip.

    Args:
        a (Any): Argument a.
        a_min (Any): Argument a_min.
        a_max (Any): Argument a_max.

    Returns:
    Any: The result.
    """
    return clamp(a_min, a, a_max)


def broadcast_shapes(*shapes: object) -> object:
    """Execute broadcast_shapes.

    Args:
        *shapes (Any): Argument *shapes.

    Returns:
    Any: The result.
    """
    return _bs(*shapes)


def logspace(
    start: object,
    stop: object,
    num: object = 50,
    endpoint: object = True,
    base: object = 10.0,
    dtype: object = None,
    axis: object = 0,
) -> object:
    """Execute logspace.

    Args:
        start (Any): Argument start.
        stop (Any): Argument stop.
        num (Any): Argument num.
        endpoint (Any): Argument endpoint.
        base (Any): Argument base.
        dtype (Any): Argument dtype.
        axis (Any): Argument axis.

    Returns:
    Any: The result.
    """
    y = linspace(start, stop, steps=num, dtype=dtype)
    if base == 10.0:
        return power(10.0, y)
    return power(base, y)


def rint(x: object) -> object:
    """Execute rint.

    Args:
        x (Any): Argument x.

    Returns:
    Any: The result.
    """
    return round(x)


def broadcast(x: object, sizes: object) -> object:
    """Execute broadcast.

    Args:
        x (Any): Argument x.
        sizes (Any): Argument sizes.

    Returns:
    Any: The result.
    """
    return broadcast_to(x, sizes)


pi = math.pi
ndarray = Tensor
