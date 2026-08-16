# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Module __init__.py."""

from typing import Any

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

abs: Any = get_op("Abs")()
angle: Any = get_op("Angle")()

acos: Any = get_op("Acos")()
acosh: Any = get_op("Acosh")()
asin: Any = get_op("Asin")()
asinh: Any = get_op("Asinh")()
atan: Any = get_op("Atan")()
atanh: Any = get_op("Atanh")()
bitwise_not: Any = get_op("BitwiseNot")()
bitwise_count: Any = get_op("BitwiseCount")()

cbrt: Any = get_op("Cbrt")()
ceil: Any = get_op("Ceil")()
conj: Any = get_op("Conj")()
cos: Any = get_op("Cos")()
cosh: Any = get_op("Cosh")()
deg2rad: Any = get_op("Deg2Rad")()
degrees: Any = get_op("Degrees")()
radians: Any = get_op("Radians")()
digamma: Any = get_op("Digamma")()
erf: Any = get_op("Erf")()
erfc: Any = get_op("Erfc")()
erfinv: Any = get_op("Erfinv")()
erf_inv = erfinv
bessel_i0e: Any = get_op("BesselI0e")()
bessel_i1e: Any = get_op("BesselI1e")()
exp: Any = get_op("Exp")()
exp2: Any = get_op("Exp2")()
expm1: Any = get_op("Expm1")()
fix: Any = get_op("Fix")()
floor: Any = get_op("Floor")()
imag: Any = get_op("Imag")()
isfinite: Any = get_op("Isfinite")()
isinf: Any = get_op("Isinf")()
isneginf: Any = get_op("Isneginf")()
isposinf: Any = get_op("Isposinf")()
isnan: Any = get_op("Isnan")()
lgamma: Any = get_op("Lgamma")()
log: Any = get_op("Log")()
log10: Any = get_op("Log10")()
log1p: Any = get_op("Log1P")()
log2: Any = get_op("Log2")()
logical_not: Any = get_op("LogicalNot")()
negative: Any = get_op("Negative")()
positive: Any = get_op("Positive")()
rad2deg: Any = get_op("Rad2Deg")()
real: Any = get_op("Real")()
reciprocal: Any = get_op("Reciprocal")()
round: Any = get_op("Round")()
rsqrt: Any = get_op("Rsqrt")()
sign: Any = get_op("Sign")()
sin: Any = get_op("Sin")()
sinc: Any = get_op("Sinc")()
sinh: Any = get_op("Sinh")()
sqrt: Any = get_op("Sqrt")()
square: Any = get_op("Square")()
tan: Any = get_op("Tan")()
tanh: Any = get_op("Tanh")()
trunc: Any = get_op("Trunc")()

cast: Any = get_op("Cast")()
bitcast: Any = get_op("Bitcast")()
frexp: Any = get_op("Frexp")()

atan2: Any = get_op("Atan2")()

_ = _special  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism
logit: Any = get_op("Logit")()
mvlgamma: Any = get_op("Mvlgamma")()
nan_to_num: Any = get_op("NanToNum")()
signbit: Any = get_op("Signbit")()
bessel_i0: Any = get_op("BesselI0")()
bessel_i1: Any = get_op("BesselI1")()
erfcinv: Any = get_op("Erfcinv")()
ndtri: Any = get_op("Ndtri")()
lbeta: Any = get_op("Lbeta")()

rint: Any = get_op("Rint")()


reciprocal_no_nan: Any = get_op("ReciprocalNoNan")()
zero_fraction: Any = get_op("ZeroFraction")()
is_non_decreasing: Any = get_op("IsNonDecreasing")()
is_strictly_increasing: Any = get_op("IsStrictlyIncreasing")()


modified_bessel_i0: Any = get_op("ModifiedBesselI0")()


modified_bessel_i1: Any = get_op("ModifiedBesselI1")()


modified_bessel_k0: Any = get_op("ModifiedBesselK0")()


modified_bessel_k1: Any = get_op("ModifiedBesselK1")()
iscomplex: Any = get_op("Iscomplex")()
isreal: Any = get_op("Isreal")()
iscomplexobj: Any = get_op("Iscomplexobj")()
isrealobj: Any = get_op("Isrealobj")()
issubdtype: Any = get_op("Issubdtype")()
isin: Any = get_op("Isin")()
ediff1d: Any = get_op("Ediff1d")()
