# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Binary operations package."""

import ml_switcheroo_compiler.ops.binary.math as _math
import ml_switcheroo_compiler.ops.binary.special as _special

# However we might not have complex type in our dummy compiler
# So we just mock it using a tuple or mock operation
from ml_switcheroo_compiler.ops.base import get_op

_ = _special
_ = _math
try:
    add = get_op("Add")()
except KeyError:
    add = None
try:
    allclose = get_op("Allclose")()
except KeyError:
    allclose = None
try:
    atan2 = get_op("Atan2")()
except KeyError:
    atan2 = None
try:
    bitwise_and = get_op("BitwiseAnd")()
except KeyError:
    bitwise_and = None
try:
    bitwise_or = get_op("BitwiseOr")()
except KeyError:
    bitwise_or = None
try:
    bitwise_xor = get_op("BitwiseXor")()
except KeyError:
    bitwise_xor = None
try:
    copysign = get_op("Copysign")()
except KeyError:
    copysign = None
try:
    divide = get_op("Divide")()
except KeyError:
    divide = None
try:
    divmod = get_op("Divmod")()
except KeyError:
    divmod = None
try:
    equal = get_op("Equal")()
except KeyError:
    equal = None
try:
    float_power = get_op("FloatPower")()
except KeyError:
    float_power = None
try:
    floor_divide = get_op("FloorDivide")()
except KeyError:
    floor_divide = None
try:
    fmax = get_op("Fmax")()
except KeyError:
    fmax = None
try:
    fmin = get_op("Fmin")()
except KeyError:
    fmin = None
try:
    fmod = get_op("Fmod")()
except KeyError:
    fmod = None
try:
    gcd = get_op("Gcd")()
except KeyError:
    gcd = None
try:
    greater = get_op("Greater")()
except KeyError:
    greater = None
try:
    greater_equal = get_op("GreaterEqual")()
except KeyError:
    greater_equal = None
try:
    heaviside = get_op("Heaviside")()
except KeyError:
    heaviside = None
try:
    hypot = get_op("Hypot")()
except KeyError:
    hypot = None
try:
    isclose = get_op("Isclose")()
except KeyError:
    isclose = None
try:
    lcm = get_op("Lcm")()
except KeyError:
    lcm = None
try:
    ldexp = get_op("Ldexp")()
except KeyError:
    ldexp = None
try:
    left_shift = get_op("LeftShift")()
except KeyError:
    left_shift = None
try:
    less = get_op("Less")()
except KeyError:
    less = None
try:
    less_equal = get_op("LessEqual")()
except KeyError:
    less_equal = None
try:
    logaddexp = get_op("Logaddexp")()
except KeyError:
    logaddexp = None
try:
    logaddexp2 = get_op("Logaddexp2")()
except KeyError:
    logaddexp2 = None
try:
    logical_and = get_op("LogicalAnd")()
except KeyError:
    logical_and = None
try:
    logical_or = get_op("LogicalOr")()
except KeyError:
    logical_or = None
try:
    logical_xor = get_op("LogicalXor")()
except KeyError:
    logical_xor = None
try:
    maximum = get_op("Maximum")()
except KeyError:
    maximum = None
try:
    minimum = get_op("Minimum")()
except KeyError:
    minimum = None
try:
    mod = get_op("Mod")()
except KeyError:
    mod = None
try:
    multiply = get_op("Multiply")()
except KeyError:
    multiply = None
try:
    nextafter = get_op("Nextafter")()
except KeyError:
    nextafter = None
try:
    not_equal = get_op("NotEqual")()
except KeyError:
    not_equal = None
try:
    power = get_op("Power")()
except KeyError:
    power = None
try:
    remainder = get_op("Remainder")()
except KeyError:
    remainder = None
rem = remainder
try:
    right_shift = get_op("RightShift")()
except KeyError:
    right_shift = None
try:
    subtract = get_op("Subtract")()
except KeyError:
    subtract = None
try:
    true_divide = get_op("TrueDivide")()
except KeyError:
    true_divide = None
try:
    xlogy = get_op("Xlogy")()
except KeyError:
    xlogy = None
try:
    igamma = get_op("Igamma")()
except KeyError:
    igamma = None
try:
    igammac = get_op("Igammac")()
except KeyError:
    igammac = None
try:
    zeta = get_op("Zeta")()
except KeyError:
    zeta = None
try:
    polygamma = get_op("Polygamma")()
except KeyError:
    polygamma = None
try:
    betainc = get_op("Betainc")()
except KeyError:
    betainc = None


def divide_no_nan(x, y):
    """Divide no nan.

    Args:
        x (object): The x parameter.
        y (object): The y parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.ops.creation import zeros_like
    from ml_switcheroo_compiler.ops.shape.indexing import where

    return where(equal(y, 0.0), zeros_like(x), divide(x, y))


def polar(abs, angle):
    """Polar.

    Args:
        abs (object): The abs parameter.
        angle (object): The angle parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.backends.registry import get_active_backend

    backend = get_active_backend()
    return backend.execute_op("Polar", getattr(abs, "data", abs), getattr(angle, "data", angle))


def view_as_complex(x):
    """View as complex.

    Args:
        x (object): The x parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.backends.registry import get_active_backend

    backend = get_active_backend()
    return backend.execute_op("ViewAsComplex", getattr(x, "data", x))


def view_as_real(x):
    """View as real.

    Args:
        x (object): The x parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.backends.registry import get_active_backend

    backend = get_active_backend()
    return backend.execute_op("ViewAsReal", getattr(x, "data", x))


try:
    truncatediv = get_op("TruncateDiv")()
except KeyError:
    truncatediv = None
try:
    truncatemod = get_op("TruncateMod")()
except KeyError:
    truncatemod = None


try:
    multiply_no_nan = get_op("MultiplyNoNan")()
except KeyError:
    multiply_no_nan = None
try:
    scalar_mul = get_op("ScalarMul")()
except KeyError:
    scalar_mul = None
try:
    squared_difference = get_op("SquaredDifference")()
except KeyError:
    squared_difference = None
try:
    xdivy = get_op("Xdivy")()
except KeyError:
    xdivy = None
try:
    xlog1py = get_op("Xlog1py")()
except KeyError:
    xlog1py = None
try:
    clip = get_op("Clip")()
except KeyError:
    clip = None


try:
    chebyshev_polynomial_t = get_op("ChebyshevPolynomialT")()
except KeyError:
    chebyshev_polynomial_t = None


try:
    chebyshev_polynomial_u = get_op("ChebyshevPolynomialU")()
except KeyError:
    chebyshev_polynomial_u = None


try:
    shifted_chebyshev_polynomial_t = get_op("ShiftedChebyshevPolynomialT")()
except KeyError:
    shifted_chebyshev_polynomial_t = None


try:
    shifted_chebyshev_polynomial_u = get_op("ShiftedChebyshevPolynomialU")()
except KeyError:
    shifted_chebyshev_polynomial_u = None


try:
    shifted_chebyshev_polynomial_v = get_op("ShiftedChebyshevPolynomialV")()
except KeyError:
    shifted_chebyshev_polynomial_v = None


try:
    shifted_chebyshev_polynomial_w = get_op("ShiftedChebyshevPolynomialW")()
except KeyError:
    shifted_chebyshev_polynomial_w = None


try:
    hermite_polynomial_h = get_op("HermitePolynomialH")()
except KeyError:
    hermite_polynomial_h = None


try:
    hermite_polynomial_he = get_op("HermitePolynomialHe")()
except KeyError:
    hermite_polynomial_he = None


try:
    laguerre_polynomial_l = get_op("LaguerrePolynomialL")()
except KeyError:
    laguerre_polynomial_l = None


try:
    legendre_polynomial_p = get_op("LegendrePolynomialP")()
except KeyError:
    legendre_polynomial_p = None
