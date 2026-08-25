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
    add: object = get_op("Add")()
except KeyError:
    add: object = None
try:
    allclose: object = get_op("Allclose")()
except KeyError:
    allclose: object = None
try:
    atan2: object = get_op("Atan2")()
except KeyError:
    atan2: object = None
try:
    bitwise_and: object = get_op("BitwiseAnd")()
except KeyError:
    bitwise_and: object = None
try:
    bitwise_or: object = get_op("BitwiseOr")()
except KeyError:
    bitwise_or: object = None
try:
    bitwise_xor: object = get_op("BitwiseXor")()
except KeyError:
    bitwise_xor: object = None
try:
    copysign: object = get_op("Copysign")()
except KeyError:
    copysign: object = None
try:
    divide: object = get_op("Divide")()
except KeyError:
    divide: object = None
try:
    divmod: object = get_op("Divmod")()
except KeyError:
    divmod: object = None
try:
    equal: object = get_op("Equal")()
except KeyError:
    equal: object = None
try:
    float_power: object = get_op("FloatPower")()
except KeyError:
    float_power: object = None
try:
    floor_divide: object = get_op("FloorDivide")()
except KeyError:
    floor_divide: object = None
try:
    fmax: object = get_op("Fmax")()
except KeyError:
    fmax: object = None
try:
    fmin: object = get_op("Fmin")()
except KeyError:
    fmin: object = None
try:
    fmod: object = get_op("Fmod")()
except KeyError:
    fmod: object = None
try:
    gcd: object = get_op("Gcd")()
except KeyError:
    gcd: object = None
try:
    greater: object = get_op("Greater")()
except KeyError:
    greater: object = None
try:
    greater_equal: object = get_op("GreaterEqual")()
except KeyError:
    greater_equal: object = None
try:
    heaviside: object = get_op("Heaviside")()
except KeyError:
    heaviside: object = None
try:
    hypot: object = get_op("Hypot")()
except KeyError:
    hypot: object = None
try:
    isclose: object = get_op("Isclose")()
except KeyError:
    isclose: object = None
try:
    lcm: object = get_op("Lcm")()
except KeyError:
    lcm: object = None
try:
    ldexp: object = get_op("Ldexp")()
except KeyError:
    ldexp: object = None
try:
    left_shift: object = get_op("LeftShift")()
except KeyError:
    left_shift: object = None
try:
    less: object = get_op("Less")()
except KeyError:
    less: object = None
try:
    less_equal: object = get_op("LessEqual")()
except KeyError:
    less_equal: object = None
try:
    logaddexp: object = get_op("Logaddexp")()
except KeyError:
    logaddexp: object = None
try:
    logaddexp2: object = get_op("Logaddexp2")()
except KeyError:
    logaddexp2: object = None
try:
    logical_and: object = get_op("LogicalAnd")()
except KeyError:
    logical_and: object = None
try:
    logical_or: object = get_op("LogicalOr")()
except KeyError:
    logical_or: object = None
try:
    logical_xor: object = get_op("LogicalXor")()
except KeyError:
    logical_xor: object = None
try:
    maximum: object = get_op("Maximum")()
except KeyError:
    maximum: object = None
try:
    minimum: object = get_op("Minimum")()
except KeyError:
    minimum: object = None
try:
    mod: object = get_op("Mod")()
except KeyError:
    mod: object = None
try:
    multiply: object = get_op("Multiply")()
except KeyError:
    multiply: object = None
try:
    nextafter: object = get_op("Nextafter")()
except KeyError:
    nextafter: object = None
try:
    not_equal: object = get_op("NotEqual")()
except KeyError:
    not_equal: object = None
try:
    power: object = get_op("Power")()
except KeyError:
    power: object = None
try:
    remainder: object = get_op("Remainder")()
except KeyError:
    remainder: object = None
rem: object = remainder
try:
    right_shift: object = get_op("RightShift")()
except KeyError:
    right_shift: object = None
try:
    subtract: object = get_op("Subtract")()
except KeyError:
    subtract: object = None
try:
    true_divide: object = get_op("TrueDivide")()
except KeyError:
    true_divide: object = None
try:
    xlogy: object = get_op("Xlogy")()
except KeyError:
    xlogy: object = None
try:
    igamma: object = get_op("Igamma")()
except KeyError:
    igamma: object = None
try:
    igammac: object = get_op("Igammac")()
except KeyError:
    igammac: object = None
try:
    zeta: object = get_op("Zeta")()
except KeyError:
    zeta: object = None
try:
    polygamma: object = get_op("Polygamma")()
except KeyError:
    polygamma: object = None
try:
    betainc: object = get_op("Betainc")()
except KeyError:
    betainc: object = None


def divide_no_nan(x: object, y: object) -> object:
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


def polar(abs: object, angle: object) -> object:
    """Polar.

    Args:
        abs (object): The abs parameter.
        angle (object): The angle parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.backends.registry import get_active_backend

    backend: object = get_active_backend()
    return backend.execute_op("Polar", getattr(abs, "data", abs), getattr(angle, "data", angle))


def view_as_complex(x: object) -> object:
    """View as complex.

    Args:
        x (object): The x parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.backends.registry import get_active_backend

    backend: object = get_active_backend()
    return backend.execute_op("ViewAsComplex", getattr(x, "data", x))


def view_as_real(x: object) -> object:
    """View as real.

    Args:
        x (object): The x parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.backends.registry import get_active_backend

    backend: object = get_active_backend()
    return backend.execute_op("ViewAsReal", getattr(x, "data", x))


try:
    truncatediv: object = get_op("TruncateDiv")()
except KeyError:
    truncatediv: object = None
try:
    truncatemod: object = get_op("TruncateMod")()
except KeyError:
    truncatemod: object = None


try:
    multiply_no_nan: object = get_op("MultiplyNoNan")()
except KeyError:
    multiply_no_nan: object = None
try:
    scalar_mul: object = get_op("ScalarMul")()
except KeyError:
    scalar_mul: object = None
try:
    squared_difference: object = get_op("SquaredDifference")()
except KeyError:
    squared_difference: object = None
try:
    xdivy: object = get_op("Xdivy")()
except KeyError:
    xdivy: object = None
try:
    xlog1py: object = get_op("Xlog1py")()
except KeyError:
    xlog1py: object = None
try:
    clip: object = get_op("Clip")()
except KeyError:
    clip: object = None


try:
    chebyshev_polynomial_t: object = get_op("ChebyshevPolynomialT")()
except KeyError:
    chebyshev_polynomial_t: object = None


try:
    chebyshev_polynomial_u: object = get_op("ChebyshevPolynomialU")()
except KeyError:
    chebyshev_polynomial_u: object = None


try:
    shifted_chebyshev_polynomial_t: object = get_op("ShiftedChebyshevPolynomialT")()
except KeyError:
    shifted_chebyshev_polynomial_t: object = None


try:
    shifted_chebyshev_polynomial_u: object = get_op("ShiftedChebyshevPolynomialU")()
except KeyError:
    shifted_chebyshev_polynomial_u: object = None


try:
    shifted_chebyshev_polynomial_v: object = get_op("ShiftedChebyshevPolynomialV")()
except KeyError:
    shifted_chebyshev_polynomial_v: object = None


try:
    shifted_chebyshev_polynomial_w: object = get_op("ShiftedChebyshevPolynomialW")()
except KeyError:
    shifted_chebyshev_polynomial_w: object = None


try:
    hermite_polynomial_h: object = get_op("HermitePolynomialH")()
except KeyError:
    hermite_polynomial_h: object = None


try:
    hermite_polynomial_he: object = get_op("HermitePolynomialHe")()
except KeyError:
    hermite_polynomial_he: object = None


try:
    laguerre_polynomial_l: object = get_op("LaguerrePolynomialL")()
except KeyError:
    laguerre_polynomial_l: object = None


try:
    legendre_polynomial_p: object = get_op("LegendrePolynomialP")()
except KeyError:
    legendre_polynomial_p: object = None
