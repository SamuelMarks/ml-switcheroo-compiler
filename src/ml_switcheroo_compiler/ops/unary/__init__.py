"""Module docstring."""

import ml_switcheroo_compiler.ops.binary.special as _binary_special
import ml_switcheroo_compiler.ops.unary.arithmetic as _arith
import ml_switcheroo_compiler.ops.unary.base as _unary_base
import ml_switcheroo_compiler.ops.unary.complex_types as _cmplx
import ml_switcheroo_compiler.ops.unary.exponential as _exp
import ml_switcheroo_compiler.ops.unary.hyperbolic as _hyp
import ml_switcheroo_compiler.ops.unary.logical as _log
import ml_switcheroo_compiler.ops.unary.normalization as _norm
import ml_switcheroo_compiler.ops.unary.sets as _sets
import ml_switcheroo_compiler.ops.unary.special as _special
import ml_switcheroo_compiler.ops.unary.trigonometric as _trig
from ml_switcheroo_compiler.ops.base import get_op

_ = (_binary_special, _special, _norm, _sets, _cmplx, _log, _exp, _arith, _hyp, _trig, _unary_base)

abs = get_op("Abs")()
angle = get_op("Angle")()

acos = get_op("Acos")()
acosh = get_op("Acosh")()
asin = get_op("Asin")()
asinh = get_op("Asinh")()
atan = get_op("Atan")()
atanh = get_op("Atanh")()
bitwise_not = get_op("BitwiseNot")()
bitwise_count = get_op("BitwiseCount")()

cbrt = get_op("Cbrt")()
ceil = get_op("Ceil")()
conj = get_op("Conj")()
cos = get_op("Cos")()
cosh = get_op("Cosh")()
deg2rad = get_op("Deg2Rad")()
digamma = get_op("Digamma")()
erf = get_op("Erf")()
erfc = get_op("Erfc")()
erfinv = get_op("Erfinv")()
bessel_i0e = get_op("BesselI0e")()
bessel_i1e = get_op("BesselI1e")()
exp = get_op("Exp")()
exp2 = get_op("Exp2")()
expm1 = get_op("Expm1")()
fix = get_op("Fix")()
floor = get_op("Floor")()
imag = get_op("Imag")()
isfinite = get_op("Isfinite")()
isinf = get_op("Isinf")()
isnan = get_op("Isnan")()
lgamma = get_op("Lgamma")()
log = get_op("Log")()
log10 = get_op("Log10")()
log1p = get_op("Log1P")()
log2 = get_op("Log2")()
logical_not = get_op("LogicalNot")()
negative = get_op("Negative")()
positive = get_op("Positive")()
rad2deg = get_op("Rad2Deg")()
real = get_op("Real")()
reciprocal = get_op("Reciprocal")()
round = get_op("Round")()
rsqrt = get_op("Rsqrt")()
sign = get_op("Sign")()
sin = get_op("Sin")()
sinc = get_op("Sinc")()
sinh = get_op("Sinh")()
sqrt = get_op("Sqrt")()
square = get_op("Square")()
tan = get_op("Tan")()
tanh = get_op("Tanh")()
trunc = get_op("Trunc")()

cast = get_op("Cast")()
bitcast = get_op("Bitcast")()
frexp = get_op("Frexp")()

atan2 = get_op("Atan2")()

_ = _special
logit = get_op("Logit")()
mvlgamma = get_op("Mvlgamma")()
nan_to_num = get_op("NanToNum")()
signbit = get_op("Signbit")()
bessel_i0 = get_op("BesselI0")()
bessel_i1 = get_op("BesselI1")()
erfcinv = get_op("Erfcinv")()
ndtri = get_op("Ndtri")()
lbeta = get_op("Lbeta")()
