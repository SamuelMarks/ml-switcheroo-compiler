# ruff: noqa: E402, D100, D103, D104, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, D101, D102, D107, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Binary operations package."""

from typing import Any

import ml_switcheroo_compiler.ops.binary.math as _math
import ml_switcheroo_compiler.ops.binary.special as _special

# However we might not have complex type in our dummy compiler
# So we just mock it using a tuple or mock operation
from ml_switcheroo_compiler.ops.base import get_op

_ = _special
_ = _math  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism
try:
    add: Any = get_op("Add")()
except KeyError:
    add = None
try:
    allclose: Any = get_op("Allclose")()
except KeyError:
    allclose = None
try:
    atan2: Any = get_op("Atan2")()
except KeyError:
    atan2 = None
try:
    bitwise_and: Any = get_op("BitwiseAnd")()
except KeyError:
    bitwise_and = None
try:
    bitwise_or: Any = get_op("BitwiseOr")()
except KeyError:
    bitwise_or = None
try:
    bitwise_xor: Any = get_op("BitwiseXor")()
except KeyError:
    bitwise_xor = None
try:
    copysign: Any = get_op("Copysign")()
except KeyError:
    copysign = None
try:
    divide: Any = get_op("Divide")()
except KeyError:
    divide = None
try:
    divmod: Any = get_op("Divmod")()
except KeyError:
    divmod = None
try:
    equal: Any = get_op("Equal")()
except KeyError:
    equal = None
try:
    float_power: Any = get_op("FloatPower")()
except KeyError:
    float_power = None
try:
    floor_divide: Any = get_op("FloorDivide")()
except KeyError:
    floor_divide = None
try:
    fmax: Any = get_op("Fmax")()
except KeyError:
    fmax = None
try:
    fmin: Any = get_op("Fmin")()
except KeyError:
    fmin = None
try:
    fmod: Any = get_op("Fmod")()
except KeyError:
    fmod = None
try:
    gcd: Any = get_op("Gcd")()
except KeyError:
    gcd = None
try:
    greater: Any = get_op("Greater")()
except KeyError:
    greater = None
try:
    greater_equal: Any = get_op("GreaterEqual")()
except KeyError:
    greater_equal = None
try:
    heaviside: Any = get_op("Heaviside")()
except KeyError:
    heaviside = None
try:
    hypot: Any = get_op("Hypot")()
except KeyError:
    hypot = None
try:
    isclose: Any = get_op("Isclose")()
except KeyError:
    isclose = None
try:
    lcm: Any = get_op("Lcm")()
except KeyError:
    lcm = None
try:
    ldexp: Any = get_op("Ldexp")()
except KeyError:
    ldexp = None
try:
    left_shift: Any = get_op("LeftShift")()
except KeyError:
    left_shift = None
try:
    less: Any = get_op("Less")()
except KeyError:
    less = None
try:
    less_equal: Any = get_op("LessEqual")()
except KeyError:
    less_equal = None
try:
    logaddexp: Any = get_op("Logaddexp")()
except KeyError:
    logaddexp = None
try:
    logaddexp2: Any = get_op("Logaddexp2")()
except KeyError:
    logaddexp2 = None
try:
    logical_and: Any = get_op("LogicalAnd")()
except KeyError:
    logical_and = None
try:
    logical_or: Any = get_op("LogicalOr")()
except KeyError:
    logical_or = None
try:
    logical_xor: Any = get_op("LogicalXor")()
except KeyError:
    logical_xor = None
try:
    maximum: Any = get_op("Maximum")()
except KeyError:
    maximum = None
try:
    minimum: Any = get_op("Minimum")()
except KeyError:
    minimum = None
try:
    mod: Any = get_op("Mod")()
except KeyError:
    mod = None
try:
    multiply: Any = get_op("Multiply")()
except KeyError:
    multiply = None
try:
    nextafter: Any = get_op("Nextafter")()
except KeyError:
    nextafter = None
try:
    not_equal: Any = get_op("NotEqual")()
except KeyError:
    not_equal = None
try:
    power: Any = get_op("Power")()
except KeyError:
    power = None
try:
    remainder: Any = get_op("Remainder")()
except KeyError:
    remainder = None
rem = remainder
try:
    right_shift: Any = get_op("RightShift")()
except KeyError:
    right_shift = None
try:
    subtract: Any = get_op("Subtract")()
except KeyError:
    subtract = None
try:
    true_divide: Any = get_op("TrueDivide")()
except KeyError:
    true_divide = None
try:
    xlogy: Any = get_op("Xlogy")()
except KeyError:
    xlogy = None
try:
    igamma: Any = get_op("Igamma")()
except KeyError:
    igamma = None
try:
    igammac: Any = get_op("Igammac")()
except KeyError:
    igammac = None
try:
    zeta: Any = get_op("Zeta")()
except KeyError:
    zeta = None
try:
    polygamma: Any = get_op("Polygamma")()
except KeyError:
    polygamma = None
try:
    betainc: Any = get_op("Betainc")()
except KeyError:
    betainc = None


def divide_no_nan(x: Any, y: Any) -> Any:
    """Divide no nan.

    Args:
        x (object): The x parameter.
        y (object): The y parameter.

    Returns: Any: Result.
    """
    from ml_switcheroo_compiler.ops.creation import zeros_like
    from ml_switcheroo_compiler.ops.shape.indexing import where

    return where(equal(y, 0.0), zeros_like(x), divide(x, y))


def polar(abs: Any, angle: Any) -> Any:
    """Polar.

    Args:
        abs (object): The abs parameter.
        angle (object): The angle parameter.

    Returns: Any: Result.
    """
    from ml_switcheroo_compiler.backends.registry import get_active_backend

    backend = get_active_backend()
    return backend.execute_op("Polar", getattr(abs, "data", abs), getattr(angle, "data", angle))


def view_as_complex(x: Any) -> Any:
    """View as complex.

    Args:
        x (object): The x parameter.

    Returns: Any: Result.
    """
    from ml_switcheroo_compiler.backends.registry import get_active_backend

    backend = get_active_backend()
    return backend.execute_op("ViewAsComplex", getattr(x, "data", x))


def view_as_real(x: Any) -> Any:
    """View as real.

    Args:
        x (object): The x parameter.

    Returns: Any: Result.
    """
    from ml_switcheroo_compiler.backends.registry import get_active_backend

    backend = get_active_backend()
    return backend.execute_op("ViewAsReal", getattr(x, "data", x))


try:
    truncatediv: Any = get_op("TruncateDiv")()
except KeyError:
    truncatediv = None
try:
    truncatemod: Any = get_op("TruncateMod")()
except KeyError:
    truncatemod = None


try:
    multiply_no_nan: Any = get_op("MultiplyNoNan")()
except KeyError:
    multiply_no_nan = None
try:
    scalar_mul: Any = get_op("ScalarMul")()
except KeyError:
    scalar_mul = None
try:
    squared_difference: Any = get_op("SquaredDifference")()
except KeyError:
    squared_difference = None
try:
    xdivy: Any = get_op("Xdivy")()
except KeyError:
    xdivy = None
try:
    xlog1py: Any = get_op("Xlog1py")()
except KeyError:
    xlog1py = None
try:
    clip: Any = get_op("Clip")()
except KeyError:
    clip = None


try:
    chebyshev_polynomial_t: Any = get_op("ChebyshevPolynomialT")()
except KeyError:
    chebyshev_polynomial_t = None


try:
    chebyshev_polynomial_u: Any = get_op("ChebyshevPolynomialU")()
except KeyError:
    chebyshev_polynomial_u = None


try:
    shifted_chebyshev_polynomial_t: Any = get_op("ShiftedChebyshevPolynomialT")()
except KeyError:
    shifted_chebyshev_polynomial_t = None


try:
    shifted_chebyshev_polynomial_u: Any = get_op("ShiftedChebyshevPolynomialU")()
except KeyError:
    shifted_chebyshev_polynomial_u = None


try:
    shifted_chebyshev_polynomial_v: Any = get_op("ShiftedChebyshevPolynomialV")()
except KeyError:
    shifted_chebyshev_polynomial_v = None


try:
    shifted_chebyshev_polynomial_w: Any = get_op("ShiftedChebyshevPolynomialW")()
except KeyError:
    shifted_chebyshev_polynomial_w = None


try:
    hermite_polynomial_h: Any = get_op("HermitePolynomialH")()
except KeyError:
    hermite_polynomial_h = None


try:
    hermite_polynomial_he: Any = get_op("HermitePolynomialHe")()
except KeyError:
    hermite_polynomial_he = None


try:
    laguerre_polynomial_l: Any = get_op("LaguerrePolynomialL")()
except KeyError:
    laguerre_polynomial_l = None


try:
    legendre_polynomial_p: Any = get_op("LegendrePolynomialP")()
except KeyError:
    legendre_polynomial_p = None
