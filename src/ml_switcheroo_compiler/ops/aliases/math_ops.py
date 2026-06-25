"""Aliases for math_ops."""

from ml_switcheroo_compiler.ops.base import get_op
import math

from ml_switcheroo_compiler.ops.binary import left_shift, maximum, minimum, power, right_shift
from ml_switcheroo_compiler.ops.unary import (
    abs,
    acos,
    acosh,
    asin,
    asinh,
    atan,
    atan2,
    atanh,
    bitwise_not,
    rad2deg,
    round,
)

from .common import create_eager_alias

arcsin = asin

arccos = acos

arctan = atan

arctan2 = atan2

arcsinh = asinh

arccosh = acosh

arctanh = atanh


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


def rint(x: object) -> object:
    """Execute rint.

    Args:
        x (Any): Argument x.

    Returns:
    Any: The result.
    """
    return round(x)


pi = math.pi

absolute = abs

around = round

bitwise_invert = bitwise_not

bitwise_left_shift = left_shift

bitwise_right_shift = right_shift

degrees = rad2deg

e = math.e

euler_gamma = 0.5772156649015328606065120900824024310421

fabs = abs


modf = create_eager_alias("modf")


radians = create_eager_alias("radians")


unwrap = create_eager_alias("unwrap")


vecdot = create_eager_alias("vecdot")


invert = bitwise_invert

pow = power

inf = float("inf")

add_n = get_op("AddN")()
accumulate_n = get_op("AccumulateN")()
cumulative_logsumexp = get_op("CumulativeLogsumexp")()
divide_no_nan = get_op("DivideNoNan")()
multiply_no_nan = get_op("MultiplyNoNan")()
squared_difference = get_op("SquaredDifference")()
xdivy = get_op("Xdivy")()
xlog1py = get_op("Xlog1py")()
reciprocal_no_nan = get_op("ReciprocalNoNan")()
reduce_euclidean_norm = get_op("ReduceEuclideanNorm")()
l2_normalize = get_op("L2Normalize")()
zero_fraction = get_op("ZeroFraction")()
is_non_decreasing = get_op("IsNonDecreasing")()
is_strictly_increasing = get_op("IsStrictlyIncreasing")()
bessel_j0 = get_op("BesselJ0")()
bessel_j1 = get_op("BesselJ1")()
bessel_k0 = get_op("BesselK0")()
bessel_k0e = get_op("BesselK0e")()
bessel_k1 = get_op("BesselK1")()
bessel_k1e = get_op("BesselK1e")()
bessel_y0 = get_op("BesselY0")()
bessel_y1 = get_op("BesselY1")()
dawsn = get_op("Dawsn")()
expint = get_op("Expint")()
fresnel_cos = get_op("FresnelCos")()
fresnel_sin = get_op("FresnelSin")()
spence = get_op("Spence")()
scalar_mul = get_op("Multiply")()
reduce_logsumexp = get_op("Logsumexp")()

truediv = get_op("TrueDivide")()
truncatediv = get_op("TruncateDiv")()
truncatemod = get_op("TruncateMod")()
