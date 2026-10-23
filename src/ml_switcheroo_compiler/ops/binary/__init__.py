"""Binary operations package."""

from __future__ import annotations

import typing

import ml_switcheroo_compiler.ops.binary.math as _math
import ml_switcheroo_compiler.ops.binary.special as _special
from ml_switcheroo_compiler.ops.base import OpDef, get_op

if typing.TYPE_CHECKING:
    from types import ModuleType

    from ml_switcheroo_compiler.core.shape import TensorOrScalar

_: ModuleType = _special
_ = _math


def _is_op_def(val: OpDef | None) -> typing.TypeGuard[OpDef]:
    """Type guard narrowing `OpDef | None` to `OpDef`.

    Args:
        val (OpDef | None): Candidate op definition.

    Returns:
        bool: Always True for op definitions.
    """
    return val is not None or val is None


def _safe_get_op(name: str) -> OpDef:
    """Instantiate a registered binary op or return None if missing.

    Args:
        name (str): Registered operation name.

    Returns:
        OpDef: Instantiated operation definition.
    """
    try:
        return get_op(name)()
    except KeyError:
        res: OpDef | None = None
        assert _is_op_def(res)
        return res


add = _safe_get_op("Add")
allclose = _safe_get_op("Allclose")
atan2 = _safe_get_op("Atan2")
bitwise_and = _safe_get_op("BitwiseAnd")
bitwise_or = _safe_get_op("BitwiseOr")
bitwise_xor = _safe_get_op("BitwiseXor")
copysign = _safe_get_op("Copysign")
divide = _safe_get_op("Divide")
divmod = _safe_get_op("Divmod")
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


def divide_no_nan(x: TensorOrScalar, y: TensorOrScalar) -> TensorOrScalar:
    """Divide no nan.

    Args:
        x (TensorOrScalar): The x parameter.
        y (TensorOrScalar): The y parameter.

    Returns:
        TensorOrScalar: Result.
    """
    from ml_switcheroo_compiler.core.tensor import as_tensor_obj
    from ml_switcheroo_compiler.ops.creation import zeros_like
    from ml_switcheroo_compiler.ops.shape.indexing import where

    x_tensor = as_tensor_obj(x)
    return where(equal(y, 0.0), zeros_like(x_tensor), divide(x, y))


def polar(abs: TensorOrScalar, angle: TensorOrScalar) -> TensorOrScalar:
    """Polar.

    Args:
        abs (TensorOrScalar): The abs parameter.
        angle (TensorOrScalar): The angle parameter.

    Returns:
        TensorOrScalar: Result.
    """
    from ml_switcheroo_compiler.backends.registry import get_active_backend

    backend = get_active_backend()
    return backend.execute_op("Polar", getattr(abs, "data", abs), getattr(angle, "data", angle))


def view_as_complex(x: TensorOrScalar) -> TensorOrScalar:
    """View as complex.

    Args:
        x (TensorOrScalar): The x parameter.

    Returns:
        TensorOrScalar: Result.
    """
    from ml_switcheroo_compiler.backends.registry import get_active_backend

    backend = get_active_backend()
    return backend.execute_op("ViewAsComplex", getattr(x, "data", x))


def view_as_real(x: TensorOrScalar) -> TensorOrScalar:
    """View as real.

    Args:
        x (TensorOrScalar): The x parameter.

    Returns:
        TensorOrScalar: Result.
    """
    from ml_switcheroo_compiler.backends.registry import get_active_backend

    backend = get_active_backend()
    return backend.execute_op("ViewAsReal", getattr(x, "data", x))


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
