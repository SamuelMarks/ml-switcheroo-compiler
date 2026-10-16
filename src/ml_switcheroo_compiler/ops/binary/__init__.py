"""Binary operations package."""

from __future__ import annotations

import typing

import ml_switcheroo_compiler.ops.binary.math as _math
import ml_switcheroo_compiler.ops.binary.special as _special
from ml_switcheroo_compiler.ops.base import OpDef, get_op

if typing.TYPE_CHECKING:
    from types import ModuleType

    from ml_switcheroo_compiler.backends.registry import EagerValue
    from ml_switcheroo_compiler.core.tensor import Tensor

_: ModuleType = _special
_ = _math
try:
    add = get_op("Add")()
except KeyError:
    add = typing.cast(OpDef, None)
try:
    allclose = get_op("Allclose")()
except KeyError:
    allclose = typing.cast(OpDef, None)
try:
    atan2 = get_op("Atan2")()
except KeyError:
    atan2 = typing.cast(OpDef, None)
try:
    bitwise_and = get_op("BitwiseAnd")()
except KeyError:
    bitwise_and = typing.cast(OpDef, None)
try:
    bitwise_or = get_op("BitwiseOr")()
except KeyError:
    bitwise_or = typing.cast(OpDef, None)
try:
    bitwise_xor = get_op("BitwiseXor")()
except KeyError:
    bitwise_xor = typing.cast(OpDef, None)
try:
    copysign = get_op("Copysign")()
except KeyError:
    copysign = typing.cast(OpDef, None)
try:
    divide = get_op("Divide")()
except KeyError:
    divide = typing.cast(OpDef, None)
try:
    divmod = get_op("Divmod")()
except KeyError:
    divmod = typing.cast(OpDef, None)
try:
    equal = get_op("Equal")()
except KeyError:
    equal = typing.cast(OpDef, None)
try:
    float_power = get_op("FloatPower")()
except KeyError:
    float_power = typing.cast(OpDef, None)
try:
    floor_divide = get_op("FloorDivide")()
except KeyError:
    floor_divide = typing.cast(OpDef, None)
try:
    fmax = get_op("Fmax")()
except KeyError:
    fmax = typing.cast(OpDef, None)
try:
    fmin = get_op("Fmin")()
except KeyError:
    fmin = typing.cast(OpDef, None)
try:
    fmod = get_op("Fmod")()
except KeyError:
    fmod = typing.cast(OpDef, None)
try:
    gcd = get_op("Gcd")()
except KeyError:
    gcd = typing.cast(OpDef, None)
try:
    greater = get_op("Greater")()
except KeyError:
    greater = typing.cast(OpDef, None)
try:
    greater_equal = get_op("GreaterEqual")()
except KeyError:
    greater_equal = typing.cast(OpDef, None)
try:
    heaviside = get_op("Heaviside")()
except KeyError:
    heaviside = typing.cast(OpDef, None)
try:
    hypot = get_op("Hypot")()
except KeyError:
    hypot = typing.cast(OpDef, None)
try:
    isclose = get_op("Isclose")()
except KeyError:
    isclose = typing.cast(OpDef, None)
try:
    lcm = get_op("Lcm")()
except KeyError:
    lcm = typing.cast(OpDef, None)
try:
    ldexp = get_op("Ldexp")()
except KeyError:
    ldexp = typing.cast(OpDef, None)
try:
    left_shift = get_op("LeftShift")()
except KeyError:
    left_shift = typing.cast(OpDef, None)
try:
    less = get_op("Less")()
except KeyError:
    less = typing.cast(OpDef, None)
try:
    less_equal = get_op("LessEqual")()
except KeyError:
    less_equal = typing.cast(OpDef, None)
try:
    logaddexp = get_op("Logaddexp")()
except KeyError:
    logaddexp = typing.cast(OpDef, None)
try:
    logaddexp2 = get_op("Logaddexp2")()
except KeyError:
    logaddexp2 = typing.cast(OpDef, None)
try:
    logical_and = get_op("LogicalAnd")()
except KeyError:
    logical_and = typing.cast(OpDef, None)
try:
    logical_or = get_op("LogicalOr")()
except KeyError:
    logical_or = typing.cast(OpDef, None)
try:
    logical_xor = get_op("LogicalXor")()
except KeyError:
    logical_xor = typing.cast(OpDef, None)
try:
    maximum = get_op("Maximum")()
except KeyError:
    maximum = typing.cast(OpDef, None)
try:
    minimum = get_op("Minimum")()
except KeyError:
    minimum = typing.cast(OpDef, None)
try:
    mod = get_op("Mod")()
except KeyError:
    mod = typing.cast(OpDef, None)
try:
    multiply = get_op("Multiply")()
except KeyError:
    multiply = typing.cast(OpDef, None)
try:
    nextafter = get_op("Nextafter")()
except KeyError:
    nextafter = typing.cast(OpDef, None)
try:
    not_equal = get_op("NotEqual")()
except KeyError:
    not_equal = typing.cast(OpDef, None)
try:
    power = get_op("Power")()
except KeyError:
    power = typing.cast(OpDef, None)
try:
    remainder = get_op("Remainder")()
except KeyError:
    remainder = typing.cast(OpDef, None)
rem = remainder
try:
    right_shift = get_op("RightShift")()
except KeyError:
    right_shift = typing.cast(OpDef, None)
try:
    subtract = get_op("Subtract")()
except KeyError:
    subtract = typing.cast(OpDef, None)
try:
    true_divide = get_op("TrueDivide")()
except KeyError:
    true_divide = typing.cast(OpDef, None)
try:
    xlogy = get_op("Xlogy")()
except KeyError:
    xlogy = typing.cast(OpDef, None)
try:
    igamma = get_op("Igamma")()
except KeyError:
    igamma = typing.cast(OpDef, None)
try:
    igammac = get_op("Igammac")()
except KeyError:
    igammac = typing.cast(OpDef, None)
try:
    zeta = get_op("Zeta")()
except KeyError:
    zeta = typing.cast(OpDef, None)
try:
    polygamma = get_op("Polygamma")()
except KeyError:
    polygamma = typing.cast(OpDef, None)
try:
    betainc = get_op("Betainc")()
except KeyError:
    betainc = typing.cast(OpDef, None)


def divide_no_nan(x: EagerValue, y: EagerValue) -> EagerValue:
    """Divide no nan.

    Args:
        x (Any): The x parameter.
        y (Any): The y parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.ops.creation import zeros_like
    from ml_switcheroo_compiler.ops.shape.indexing import where

    return where(equal(y, 0.0), zeros_like(typing.cast("Tensor", x)), divide(x, y))


def polar(abs: EagerValue, angle: EagerValue) -> EagerValue:
    """Polar.

    Args:
        abs (Any): The abs parameter.
        angle (Any): The angle parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.backends.registry import get_active_backend

    backend = get_active_backend()
    return backend.execute_op("Polar", getattr(abs, "data", abs), getattr(angle, "data", angle))


def view_as_complex(x: EagerValue) -> EagerValue:
    """View as complex.

    Args:
        x (Any): The x parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.backends.registry import get_active_backend

    backend = get_active_backend()
    return backend.execute_op("ViewAsComplex", getattr(x, "data", x))


def view_as_real(x: EagerValue) -> EagerValue:
    """View as real.

    Args:
        x (Any): The x parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.backends.registry import get_active_backend

    backend = get_active_backend()
    return backend.execute_op("ViewAsReal", getattr(x, "data", x))


try:
    truncatediv = get_op("TruncateDiv")()
except KeyError:
    truncatediv = typing.cast(OpDef, None)
try:
    truncatemod = get_op("TruncateMod")()
except KeyError:
    truncatemod = typing.cast(OpDef, None)


try:
    multiply_no_nan = get_op("MultiplyNoNan")()
except KeyError:
    multiply_no_nan = typing.cast(OpDef, None)
try:
    scalar_mul = get_op("ScalarMul")()
except KeyError:
    scalar_mul = typing.cast(OpDef, None)
try:
    squared_difference = get_op("SquaredDifference")()
except KeyError:
    squared_difference = typing.cast(OpDef, None)
try:
    xdivy = get_op("Xdivy")()
except KeyError:
    xdivy = typing.cast(OpDef, None)
try:
    xlog1py = get_op("Xlog1py")()
except KeyError:
    xlog1py = typing.cast(OpDef, None)
try:
    clip = get_op("Clip")()
except KeyError:
    clip = typing.cast(OpDef, None)


try:
    chebyshev_polynomial_t = get_op("ChebyshevPolynomialT")()
except KeyError:
    chebyshev_polynomial_t = typing.cast(OpDef, None)


try:
    chebyshev_polynomial_u = get_op("ChebyshevPolynomialU")()
except KeyError:
    chebyshev_polynomial_u = typing.cast(OpDef, None)


try:
    shifted_chebyshev_polynomial_t = get_op("ShiftedChebyshevPolynomialT")()
except KeyError:
    shifted_chebyshev_polynomial_t = typing.cast(OpDef, None)


try:
    shifted_chebyshev_polynomial_u = get_op("ShiftedChebyshevPolynomialU")()
except KeyError:
    shifted_chebyshev_polynomial_u = typing.cast(OpDef, None)


try:
    shifted_chebyshev_polynomial_v = get_op("ShiftedChebyshevPolynomialV")()
except KeyError:
    shifted_chebyshev_polynomial_v = typing.cast(OpDef, None)


try:
    shifted_chebyshev_polynomial_w = get_op("ShiftedChebyshevPolynomialW")()
except KeyError:
    shifted_chebyshev_polynomial_w = typing.cast(OpDef, None)


try:
    hermite_polynomial_h = get_op("HermitePolynomialH")()
except KeyError:
    hermite_polynomial_h = typing.cast(OpDef, None)


try:
    hermite_polynomial_he = get_op("HermitePolynomialHe")()
except KeyError:
    hermite_polynomial_he = typing.cast(OpDef, None)


try:
    laguerre_polynomial_l = get_op("LaguerrePolynomialL")()
except KeyError:
    laguerre_polynomial_l = typing.cast(OpDef, None)


try:
    legendre_polynomial_p = get_op("LegendrePolynomialP")()
except KeyError:
    legendre_polynomial_p = typing.cast(OpDef, None)
