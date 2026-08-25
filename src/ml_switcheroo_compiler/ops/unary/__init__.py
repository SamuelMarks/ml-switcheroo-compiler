# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Module __init__.py."""

"""Core abstractions and logic definitions for __init__.py."""

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

abs: object = get_op("Abs")()
angle: object = get_op("Angle")()

acos: object = get_op("Acos")()
acosh: object = get_op("Acosh")()
asin: object = get_op("Asin")()
asinh: object = get_op("Asinh")()
atan: object = get_op("Atan")()
atanh: object = get_op("Atanh")()
bitwise_not: object = get_op("BitwiseNot")()
bitwise_count: object = get_op("BitwiseCount")()

cbrt: object = get_op("Cbrt")()
ceil: object = get_op("Ceil")()
conj: object = get_op("Conj")()
cos: object = get_op("Cos")()
cosh: object = get_op("Cosh")()
deg2rad: object = get_op("Deg2Rad")()
degrees: object = get_op("Degrees")()
radians: object = get_op("Radians")()
digamma: object = get_op("Digamma")()
erf: object = get_op("Erf")()
erfc: object = get_op("Erfc")()
erfinv: object = get_op("Erfinv")()
erf_inv: object = erfinv
bessel_i0e: object = get_op("BesselI0e")()
bessel_i1e: object = get_op("BesselI1e")()
exp: object = get_op("Exp")()
exp2: object = get_op("Exp2")()
expm1: object = get_op("Expm1")()
fix: object = get_op("Fix")()
floor: object = get_op("Floor")()
imag: object = get_op("Imag")()
isfinite: object = get_op("Isfinite")()
isinf: object = get_op("Isinf")()
isneginf: object = get_op("Isneginf")()
isposinf: object = get_op("Isposinf")()
isnan: object = get_op("Isnan")()
lgamma: object = get_op("Lgamma")()
log: object = get_op("Log")()
log10: object = get_op("Log10")()
log1p: object = get_op("Log1P")()
log2: object = get_op("Log2")()
logical_not: object = get_op("LogicalNot")()
negative: object = get_op("Negative")()
positive: object = get_op("Positive")()
rad2deg: object = get_op("Rad2Deg")()
real: object = get_op("Real")()
reciprocal: object = get_op("Reciprocal")()
round: object = get_op("Round")()
rsqrt: object = get_op("Rsqrt")()
sign: object = get_op("Sign")()
sin: object = get_op("Sin")()
sinc: object = get_op("Sinc")()
sinh: object = get_op("Sinh")()
sqrt: object = get_op("Sqrt")()
square: object = get_op("Square")()
tan: object = get_op("Tan")()
tanh: object = get_op("Tanh")()
trunc: object = get_op("Trunc")()

cast: object = get_op("Cast")()
bitcast: object = get_op("Bitcast")()
frexp: object = get_op("Frexp")()

atan2: object = get_op("Atan2")()

_ = _special
logit: object = get_op("Logit")()
mvlgamma: object = get_op("Mvlgamma")()
nan_to_num: object = get_op("NanToNum")()
signbit: object = get_op("Signbit")()
bessel_i0: object = get_op("BesselI0")()
bessel_i1: object = get_op("BesselI1")()
erfcinv: object = get_op("Erfcinv")()
ndtri: object = get_op("Ndtri")()
lbeta: object = get_op("Lbeta")()

rint: object = get_op("Rint")()


reciprocal_no_nan: object = get_op("ReciprocalNoNan")()
zero_fraction: object = get_op("ZeroFraction")()
is_non_decreasing: object = get_op("IsNonDecreasing")()
is_strictly_increasing: object = get_op("IsStrictlyIncreasing")()


modified_bessel_i0: object = get_op("ModifiedBesselI0")()


modified_bessel_i1: object = get_op("ModifiedBesselI1")()


modified_bessel_k0: object = get_op("ModifiedBesselK0")()


modified_bessel_k1: object = get_op("ModifiedBesselK1")()
iscomplex: object = get_op("Iscomplex")()
isreal: object = get_op("Isreal")()
iscomplexobj: object = get_op("Iscomplexobj")()
isrealobj: object = get_op("Isrealobj")()
issubdtype: object = get_op("Issubdtype")()
isin: object = get_op("Isin")()
ediff1d: object = get_op("Ediff1d")()
