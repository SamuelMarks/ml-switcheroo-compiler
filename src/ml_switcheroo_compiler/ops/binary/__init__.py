"""Binary operations package."""

import ml_switcheroo_compiler.ops.binary.math as _math
import ml_switcheroo_compiler.ops.binary.special as _special

# However we might not have complex type in our dummy compiler
# So we just mock it using a tuple or mock operation
from ml_switcheroo_compiler.ops.base import get_op

_ = _math
_ = _special
add = get_op("Add")()
allclose = get_op("Allclose")()
atan2 = get_op("Atan2")()
bitwise_and = get_op("BitwiseAnd")()
bitwise_or = get_op("BitwiseOr")()
bitwise_xor = get_op("BitwiseXor")()
copysign = get_op("Copysign")()
divide = get_op("Divide")()
divmod = get_op("Divmod")()
equal = get_op("Equal")()
float_power = get_op("FloatPower")()
floor_divide = get_op("FloorDivide")()
fmax = get_op("Fmax")()
fmin = get_op("Fmin")()
fmod = get_op("Fmod")()
gcd = get_op("Gcd")()
greater = get_op("Greater")()
greater_equal = get_op("GreaterEqual")()
heaviside = get_op("Heaviside")()
hypot = get_op("Hypot")()
isclose = get_op("Isclose")()
lcm = get_op("Lcm")()
ldexp = get_op("Ldexp")()
left_shift = get_op("LeftShift")()
less = get_op("Less")()
less_equal = get_op("LessEqual")()
logaddexp = get_op("Logaddexp")()
logaddexp2 = get_op("Logaddexp2")()
logical_and = get_op("LogicalAnd")()
logical_or = get_op("LogicalOr")()
logical_xor = get_op("LogicalXor")()
maximum = get_op("Maximum")()
minimum = get_op("Minimum")()
mod = get_op("Mod")()
multiply = get_op("Multiply")()
nextafter = get_op("Nextafter")()
not_equal = get_op("NotEqual")()
power = get_op("Power")()
remainder = get_op("Remainder")()
right_shift = get_op("RightShift")()
subtract = get_op("Subtract")()
true_divide = get_op("TrueDivide")()
xlogy = get_op("Xlogy")()
igamma = get_op("Igamma")()
igammac = get_op("Igammac")()
zeta = get_op("Zeta")()
polygamma = get_op("Polygamma")()
betainc = get_op("Betainc")()


def divide_no_nan(x: object, y: object) -> object:
    """Safe division."""
    from ml_switcheroo_compiler.ops.creation import zeros_like
    from ml_switcheroo_compiler.ops.shape.frontend import where

    return where(equal(y, 0.0), zeros_like(x), divide(x, y))


def polar(abs: object, angle: object) -> object:
    """Converts polar coordinates to a complex tensor."""
    from ml_switcheroo_compiler.backends.registry import get_active_backend

    backend = get_active_backend()
    return backend.execute_op("Polar", getattr(abs, "data", abs), getattr(angle, "data", angle))


def view_as_complex(x: object) -> object:
    """Views a real tensor as a complex tensor."""
    from ml_switcheroo_compiler.backends.registry import get_active_backend

    backend = get_active_backend()
    return backend.execute_op("ViewAsComplex", getattr(x, "data", x))


def view_as_real(x: object) -> object:
    """Views a complex tensor as a real tensor."""
    from ml_switcheroo_compiler.backends.registry import get_active_backend

    backend = get_active_backend()
    return backend.execute_op("ViewAsReal", getattr(x, "data", x))
