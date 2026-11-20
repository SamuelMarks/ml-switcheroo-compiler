"""Binary operations package."""

from __future__ import annotations

import typing

import ml_switcheroo_compiler.ops.binary.math as _math
import ml_switcheroo_compiler.ops.binary.special as _special
from ml_switcheroo_compiler.core.shape import extract_data
from ml_switcheroo_compiler.ops.base import OpDef, get_op

if typing.TYPE_CHECKING:
    from types import ModuleType

    from ml_switcheroo_compiler.core.shape import OpValue

_: ModuleType = _special
_ = _math


def _safe_get_op(name: str) -> OpDef:
    """Instantiate a registered binary op or construct a fallback `OpDef`.

    Args:
        name (str): Registered operation name.

    Returns:
        OpDef: Instantiated operation definition.
    """
    try:
        return get_op(name)()
    except KeyError:
        op = OpDef()
        op.op_type = name
        return op


add = _safe_get_op("Add")
allclose = _safe_get_op("Allclose")
atan2 = _safe_get_op("Atan2")
bitwise_and = _safe_get_op("BitwiseAnd")
bitwise_or = _safe_get_op("BitwiseOr")
bitwise_xor = _safe_get_op("BitwiseXor")
copysign = _safe_get_op("Copysign")
divide = _safe_get_op("Divide")
divmod = _special.Divmod()
equal = _safe_get_op("Equal")
float_power = _safe_get_op("FloatPower")
floor_divide = _safe_get_op("FloorDivide")
fmax = _safe_get_op("Fmax")
fmin = _safe_get_op("Fmin")
fmod = _safe_get_op("Fmod")
gcd = _safe_get_op("Gcd")
greater = _safe_get_op("Greater")
greater_equal = _safe_get_op("GreaterEqual")
heaviside = _safe_get_op("Heaviside")
hypot = _safe_get_op("Hypot")
isclose = _safe_get_op("Isclose")
lcm = _safe_get_op("Lcm")
ldexp = _safe_get_op("Ldexp")
left_shift = _safe_get_op("LeftShift")
less = _safe_get_op("Less")
less_equal = _safe_get_op("LessEqual")
logaddexp = _safe_get_op("Logaddexp")
logaddexp2 = _safe_get_op("Logaddexp2")
logical_and = _safe_get_op("LogicalAnd")
logical_or = _safe_get_op("LogicalOr")
logical_xor = _safe_get_op("LogicalXor")
maximum = _safe_get_op("Maximum")
minimum = _safe_get_op("Minimum")
mod = _safe_get_op("Mod")
multiply = _safe_get_op("Multiply")
nextafter = _safe_get_op("Nextafter")
not_equal = _safe_get_op("NotEqual")
power = _safe_get_op("Power")
remainder = _safe_get_op("Remainder")
rem = remainder
right_shift = _safe_get_op("RightShift")
subtract = _safe_get_op("Subtract")
true_divide = _safe_get_op("TrueDivide")
xlogy = _safe_get_op("Xlogy")
igamma = _safe_get_op("Igamma")
igammac = _safe_get_op("Igammac")
zeta = _safe_get_op("Zeta")
polygamma = _safe_get_op("Polygamma")
betainc = _safe_get_op("Betainc")


def divide_no_nan(x: OpValue, y: OpValue) -> OpValue:
    """Divide no nan.

    Args:
        x (OpValue): Input `x` argument (x).
        y (OpValue): Input `y` argument (y).

    Returns:
        OpValue: Computed `OpValue` output.
    """
    from ml_switcheroo_compiler.core.tensor import ensure_tensor
    from ml_switcheroo_compiler.ops.creation import zeros_like
    from ml_switcheroo_compiler.ops.shape.indexing import where

    x_tensor = ensure_tensor(x)
    return where(equal(y, 0.0), zeros_like(x_tensor), divide(x, y))


def polar(abs: OpValue, angle: OpValue) -> OpValue:
    """Polar.

    Args:
        abs (OpValue): Input `abs` argument (abs).
        angle (OpValue): Input `angle` argument (angle).

    Returns:
        OpValue: Computed `OpValue` output.
    """
    from ml_switcheroo_compiler.backends.registry import get_active_backend

    backend = get_active_backend()
    return backend.execute_op("Polar", extract_data(abs), extract_data(angle))


def view_as_complex(x: OpValue) -> OpValue:
    """View as complex.

    Args:
        x (OpValue): Input `x` argument (x).

    Returns:
        OpValue: Computed `OpValue` output.
    """
    from ml_switcheroo_compiler.backends.registry import get_active_backend

    backend = get_active_backend()
    return backend.execute_op("ViewAsComplex", extract_data(x))


def view_as_real(x: OpValue) -> OpValue:
    """View as real.

    Args:
        x (OpValue): Input `x` argument (x).

    Returns:
        OpValue: Computed `OpValue` output.
    """
    from ml_switcheroo_compiler.backends.registry import get_active_backend

    backend = get_active_backend()
    return backend.execute_op("ViewAsReal", extract_data(x))


truncatediv = _safe_get_op("TruncateDiv")
truncatemod = _safe_get_op("TruncateMod")
multiply_no_nan = _safe_get_op("MultiplyNoNan")
scalar_mul = _safe_get_op("ScalarMul")
squared_difference = _safe_get_op("SquaredDifference")
xdivy = _safe_get_op("Xdivy")
xlog1py = _safe_get_op("Xlog1py")
clip = _safe_get_op("Clip")
chebyshev_polynomial_t = _safe_get_op("ChebyshevPolynomialT")
chebyshev_polynomial_u = _safe_get_op("ChebyshevPolynomialU")
shifted_chebyshev_polynomial_t = _safe_get_op("ShiftedChebyshevPolynomialT")
shifted_chebyshev_polynomial_u = _safe_get_op("ShiftedChebyshevPolynomialU")
shifted_chebyshev_polynomial_v = _safe_get_op("ShiftedChebyshevPolynomialV")
shifted_chebyshev_polynomial_w = _safe_get_op("ShiftedChebyshevPolynomialW")
hermite_polynomial_h = _safe_get_op("HermitePolynomialH")
hermite_polynomial_he = _safe_get_op("HermitePolynomialHe")
laguerre_polynomial_l = _safe_get_op("LaguerrePolynomialL")
legendre_polynomial_p = _safe_get_op("LegendrePolynomialP")
