"""Operations library for the ml-switcheroo compiler."""

import numpy as np

from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.ops.base import OpDef, get_op, register_op
from ml_switcheroo_compiler.ops.binary import (
    add,
    allclose,
    bitwise_and,
    bitwise_or,
    bitwise_xor,
    copysign,
    divide,
    divmod,
    equal,
    float_power,
    floor_divide,
    fmax,
    fmin,
    fmod,
    gcd,
    greater,
    greater_equal,
    heaviside,
    hypot,
    isclose,
    lcm,
    ldexp,
    left_shift,
    less,
    less_equal,
    logaddexp,
    logaddexp2,
    logical_and,
    logical_or,
    logical_xor,
    maximum,
    minimum,
    mod,
    multiply,
    nextafter,
    not_equal,
    power,
    remainder,
    right_shift,
    subtract,
    true_divide,
)
from ml_switcheroo_compiler.ops.creation import (
    arange,
    diag,
    empty,
    empty_like,
    eye,
    full,
    full_like,
    identity,
    linspace,
    ones,
    ones_like,
    zeros,
    zeros_like,
)
from ml_switcheroo_compiler.ops.creation.frontend import array, asarray
from ml_switcheroo_compiler.ops.linalg import (
    conv_general_dilated,
    dot,
    dot_general,
    eigvalsh,
    einsum,
    fft,
    inner,
    matmul,
    matrix_power,
    outer,
    rfft,
    slogdet,
    tensordot,
    vdot,
)
from ml_switcheroo_compiler.ops.reductions import (
    all,
    any,
    argmax,
    argmin,
    count_nonzero,
    cumsum,
    logsumexp,
    max,
    mean,
    min,
    norm,
    pmean,
    prod,
    psum,
    reduce_window,
    segment_sum,
    std,
    sum,
    variance,
)
from ml_switcheroo_compiler.ops.shape import (
    array_split,
    broadcast_in_dim,
    broadcast_to,
    concatenate,
    dsplit,
    dstack,
    dynamic_slice,
    dynamic_update_slice,
    expand,
    flatten,
    gather,
    gather_nd,
    hsplit,
    hstack,
    image_resize,
    meshgrid,
    moveaxis,
    pad,
    permute,
    repeat,
    reshape,
    roll,
    scatter,
    scatter_add,
    scatter_nd,
    select,
    slice,
    sort,
    split,
    squeeze,
    stack,
    strided_slice,
    swapaxes,
    take,
    take_along_axis,
    tile,
    top_k,
    transpose,
    tril,
    triu,
    unsqueeze,
    unstack,
    update_slice,
    vsplit,
    vstack,
    where,
)
from ml_switcheroo_compiler.ops.unary import (
    abs,
    acos,
    acosh,
    asin,
    asinh,
    atan,
    atan2,
    atanh,
    bitcast,
    bitwise_not,
    cast,
    cbrt,
    ceil,
    conj,
    cos,
    cosh,
    deg2rad,
    digamma,
    erf,
    erfc,
    exp,
    exp2,
    expm1,
    fix,
    floor,
    frexp,
    imag,
    isfinite,
    isinf,
    isnan,
    lgamma,
    log,
    log1p,
    log2,
    log10,
    logical_not,
    negative,
    positive,
    rad2deg,
    real,
    reciprocal,
    round,
    rsqrt,
    sign,
    sin,
    sinc,
    sinh,
    sqrt,
    square,
    tan,
    tanh,
    trunc,
)

__all__ = [
    "OpDef",
    "abs",
    "acos",
    "acosh",
    "add",
    "all",
    "allclose",
    "amax",
    "amin",
    "any",
    "arange",
    "arccos",
    "arccosh",
    "arcsin",
    "arcsinh",
    "arctan",
    "arctan2",
    "arctanh",
    "argmax",
    "argmin",
    "array",
    "array_equal",
    "array_split",
    "asarray",
    "asin",
    "asinh",
    "atan",
    "atan2",
    "atanh",
    "bitcast",
    "bitwise_and",
    "bitwise_not",
    "bitwise_or",
    "bitwise_xor",
    "broadcast",
    "broadcast_in_dim",
    "broadcast_in_dim",
    "broadcast_shapes",
    "broadcast_to",
    "cast",
    "cbrt",
    "ceil",
    "clamp",
    "clip",
    "concatenate",
    "conj",
    "conv_general_dilated",
    "copysign",
    "cos",
    "cosh",
    "count_nonzero",
    "cumsum",
    "deg2rad",
    "diag",
    "digamma",
    "divide",
    "divmod",
    "dot",
    "dot_general",
    "dsplit",
    "dstack",
    "dynamic_slice",
    "dynamic_update_slice",
    "dynamic_update_slice",
    "eigvalsh",
    "einsum",
    "empty",
    "empty_like",
    "equal",
    "erf",
    "erfc",
    "exp",
    "exp2",
    "expand",
    "expand_dims",
    "expm1",
    "eye",
    "fft",
    "fix",
    "flatten",
    "float_power",
    "floor",
    "floor_divide",
    "fmax",
    "fmin",
    "fmod",
    "frexp",
    "full",
    "full_like",
    "gather",
    "gather_nd",
    "gcd",
    "get_op",
    "greater",
    "greater_equal",
    "heaviside",
    "hsplit",
    "hstack",
    "hypot",
    "identity",
    "imag",
    "image_resize",
    "image_resize",
    "inner",
    "isclose",
    "isfinite",
    "isinf",
    "isnan",
    "lcm",
    "ldexp",
    "left_shift",
    "less",
    "less_equal",
    "lgamma",
    "linspace",
    "log",
    "log1p",
    "log2",
    "log10",
    "logaddexp",
    "logaddexp2",
    "logical_and",
    "logical_not",
    "logical_or",
    "logical_xor",
    "logspace",
    "logsumexp",
    "matmul",
    "matrix_power",
    "max",
    "maximum",
    "mean",
    "meshgrid",
    "min",
    "minimum",
    "mod",
    "moveaxis",
    "multiply",
    "negative",
    "nextafter",
    "norm",
    "not_equal",
    "ones",
    "ones_like",
    "outer",
    "pad",
    "permute",
    "pmean",
    "positive",
    "power",
    "prod",
    "psum",
    "rad2deg",
    "real",
    "reciprocal",
    "reduce",
    "reduce_window",
    "register_op",
    "remainder",
    "repeat",
    "reshape",
    "rfft",
    "right_shift",
    "rint",
    "roll",
    "round",
    "rsqrt",
    "scatter",
    "scatter_add",
    "scatter_nd",
    "segment_sum",
    "select",
    "select",
    "sign",
    "sin",
    "sinc",
    "sinh",
    "slice",
    "slogdet",
    "sort",
    "split",
    "sqrt",
    "square",
    "squeeze",
    "stack",
    "std",
    "strided_slice",
    "subtract",
    "sum",
    "swapaxes",
    "take",
    "take_along_axis",
    "tan",
    "tanh",
    "tensordot",
    "tile",
    "top_k",
    "transpose",
    "tril",
    "triu",
    "true_divide",
    "trunc",
    "unsqueeze",
    "unstack",
    "update_slice",
    "var",
    "variance",
    "vdot",
    "vsplit",
    "vstack",
    "where",
    "zeros",
    "zeros_like",
]

from ml_switcheroo_compiler.ops.state import AssignVariable, ReadVariable

__all__ += ["AssignVariable", "OpDef", "ReadVariable", "get_op", "register_op"]
__all__.extend(["ndarray", "pi"])


# Aliases
arcsin = asin
arccos = acos
arctan = atan
arctan2 = atan2
arcsinh = asinh
arccosh = acosh
arctanh = atanh
amin = min
amax = max
var = variance
expand_dims = unsqueeze


def clamp(min_val: object, x: object, max_val: object) -> object:
    """Docstring.

    Args:
        min_val (object): The min_val.
        x (object): The x.
        max_val (object): The max_val.

    Returns:
        object: The computed result.
    """
    if min_val is not None:
        x = maximum(x, min_val)
    if max_val is not None:
        x = minimum(x, max_val)
    return x


def clip(a: object, a_min: object = None, a_max: object = None) -> object:
    """Docstring.

    Args:
        a (object): The a.
        a_min (object): The a_min.
        a_max (object): The a_max.

    Returns:
        object: The computed result.
    """
    return clamp(a_min, a, a_max)


def broadcast_shapes(*shapes: object) -> object:
    """Docstring.

    Args:
        *shapes: Additional arguments.

    Returns:
        object: The computed result.
    """
    return np.broadcast_shapes(*shapes)


def logspace(
    start: object,
    stop: object,
    num: object = 50,
    endpoint: object = True,
    base: object = 10.0,
    dtype: object = None,
    axis: object = 0,
) -> object:
    """Docstring.

    Args:
        start (object): The start.
        stop (object): The stop.
        num (object): The num.
        endpoint (object): The endpoint.
        base (object): The base.
        dtype (object): The dtype.
        axis (object): The axis.

    Returns:
        object: The computed result.
    """
    from ml_switcheroo_compiler.ops.binary import power
    from ml_switcheroo_compiler.ops.creation.frontend import linspace

    # 10 ** linspace(...)
    y = linspace(start, stop, steps=num, dtype=dtype)
    if base == 10.0:
        return power(10.0, y)
    return power(base, y)


def rint(x: object) -> object:
    """Docstring.

    Args:
        x (object): The x.

    Returns:
        object: The computed result.
    """
    from ml_switcheroo_compiler.ops.unary import round

    return round(x)


def broadcast(x: object, sizes: object) -> object:
    """Docstring.

    Args:
        x (object): The x.
        sizes (object): The sizes.

    Returns:
        object: The computed result.
    """
    from ml_switcheroo_compiler.ops.shape.frontend import broadcast_to

    return broadcast_to(x, sizes)


pi = np.pi
ndarray = Tensor
