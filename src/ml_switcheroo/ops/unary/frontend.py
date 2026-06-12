"""Unary Operations."""

import uuid
import numpy as np
from ml_switcheroo.core.tensor import Tensor
from ml_switcheroo.core.dtype import DType
from ml_switcheroo.core.config import config
from ml_switcheroo.tracing import _tracer, ProxyTensor
from ml_switcheroo_ir import LogicalNode


def _emit_unary_node(input: Tensor, op_type: str, out_dtype: DType = None) -> Tensor:
    """Emit a unary node to the IR graph."""
    if out_dtype is None:
        out_dtype = input.dtype
    if config.eager_mode:
        raise RuntimeError("Cannot emit node in eager mode.")
    if not _tracer.is_tracing:
        raise RuntimeError(f"Cannot emit {op_type} node outside of a tracing context.")

    out_id = str(uuid.uuid4())
    node = LogicalNode(
        id=out_id,
        op_type=op_type,
        inputs=[input.data.id],
        shape_metadata=input.shape,
    )
    _tracer.add_node(node)

    proxy = ProxyTensor(id=out_id, shape=input.shape, dtype=out_dtype.value)
    return Tensor(data=proxy, shape=input.shape, dtype=out_dtype, device=input.device)


def abs(input: Tensor) -> Tensor:
    """Computes abs of each element."""
    if config.eager_mode:
        data = np.abs(input.data)
        # Re-wrap boolean functions explicitly to ensure correct dtype type mapping
        return Tensor(data, input.shape, input.dtype, input.device)
    else:
        return _emit_unary_node(input, "Abs", input.dtype)


def acos(input: Tensor) -> Tensor:
    """Computes acos of each element."""
    if config.eager_mode:
        data = np.arccos(input.data)
        # Re-wrap boolean functions explicitly to ensure correct dtype type mapping
        return Tensor(data, input.shape, input.dtype, input.device)
    else:
        return _emit_unary_node(input, "Acos", input.dtype)


def acosh(input: Tensor) -> Tensor:
    """Computes acosh of each element."""
    if config.eager_mode:
        data = np.arccosh(input.data)
        # Re-wrap boolean functions explicitly to ensure correct dtype type mapping
        return Tensor(data, input.shape, input.dtype, input.device)
    else:
        return _emit_unary_node(input, "Acosh", input.dtype)


def asin(input: Tensor) -> Tensor:
    """Computes asin of each element."""
    if config.eager_mode:
        data = np.arcsin(input.data)
        # Re-wrap boolean functions explicitly to ensure correct dtype type mapping
        return Tensor(data, input.shape, input.dtype, input.device)
    else:
        return _emit_unary_node(input, "Asin", input.dtype)


def asinh(input: Tensor) -> Tensor:
    """Computes asinh of each element."""
    if config.eager_mode:
        data = np.arcsinh(input.data)
        # Re-wrap boolean functions explicitly to ensure correct dtype type mapping
        return Tensor(data, input.shape, input.dtype, input.device)
    else:
        return _emit_unary_node(input, "Asinh", input.dtype)


def atan(input: Tensor) -> Tensor:
    """Computes atan of each element."""
    if config.eager_mode:
        data = np.arctan(input.data)
        # Re-wrap boolean functions explicitly to ensure correct dtype type mapping
        return Tensor(data, input.shape, input.dtype, input.device)
    else:
        return _emit_unary_node(input, "Atan", input.dtype)


def atanh(input: Tensor) -> Tensor:
    """Computes atanh of each element."""
    if config.eager_mode:
        data = np.arctanh(input.data)
        # Re-wrap boolean functions explicitly to ensure correct dtype type mapping
        return Tensor(data, input.shape, input.dtype, input.device)
    else:
        return _emit_unary_node(input, "Atanh", input.dtype)


def bitwise_not(input: Tensor) -> Tensor:
    """Computes bitwise_not of each element."""
    if config.eager_mode:
        data = np.bitwise_not(input.data)
        # Re-wrap boolean functions explicitly to ensure correct dtype type mapping
        return Tensor(data, input.shape, input.dtype, input.device)
    else:
        return _emit_unary_node(input, "BitwiseNot", input.dtype)


def cbrt(input: Tensor) -> Tensor:
    """Computes cbrt of each element."""
    if config.eager_mode:
        data = np.cbrt(input.data)
        # Re-wrap boolean functions explicitly to ensure correct dtype type mapping
        return Tensor(data, input.shape, input.dtype, input.device)
    else:
        return _emit_unary_node(input, "Cbrt", input.dtype)


def ceil(input: Tensor) -> Tensor:
    """Computes ceil of each element."""
    if config.eager_mode:
        data = np.ceil(input.data)
        # Re-wrap boolean functions explicitly to ensure correct dtype type mapping
        return Tensor(data, input.shape, input.dtype, input.device)
    else:
        return _emit_unary_node(input, "Ceil", input.dtype)


def conj(input: Tensor) -> Tensor:
    """Computes conj of each element."""
    if config.eager_mode:
        data = np.conj(input.data)
        # Re-wrap boolean functions explicitly to ensure correct dtype type mapping
        return Tensor(data, input.shape, input.dtype, input.device)
    else:
        return _emit_unary_node(input, "Conj", input.dtype)


def cos(input: Tensor) -> Tensor:
    """Computes cos of each element."""
    if config.eager_mode:
        data = np.cos(input.data)
        # Re-wrap boolean functions explicitly to ensure correct dtype type mapping
        return Tensor(data, input.shape, input.dtype, input.device)
    else:
        return _emit_unary_node(input, "Cos", input.dtype)


def cosh(input: Tensor) -> Tensor:
    """Computes cosh of each element."""
    if config.eager_mode:
        data = np.cosh(input.data)
        # Re-wrap boolean functions explicitly to ensure correct dtype type mapping
        return Tensor(data, input.shape, input.dtype, input.device)
    else:
        return _emit_unary_node(input, "Cosh", input.dtype)


def deg2rad(input: Tensor) -> Tensor:
    """Computes deg2rad of each element."""
    if config.eager_mode:
        data = np.deg2rad(input.data)
        # Re-wrap boolean functions explicitly to ensure correct dtype type mapping
        return Tensor(data, input.shape, input.dtype, input.device)
    else:
        return _emit_unary_node(input, "Deg2Rad", input.dtype)


def erf(input: Tensor) -> Tensor:
    """Computes erf of each element."""
    if config.eager_mode:
        import math
        import numpy as np

        f = np.vectorize(math.erf)
        data = f(input.data).astype(input.data.dtype)
        return Tensor(data, input.shape, input.dtype, input.device)
    else:
        return _emit_unary_node(input, "Erf", input.dtype)


def erfc(input: Tensor) -> Tensor:
    """Computes erfc of each element."""
    if config.eager_mode:
        from ml_switcheroo.core.errors import UnimplementedMathError

        raise UnimplementedMathError("No direct NumPy equivalent for erfc.")
    else:
        return _emit_unary_node(input, "Erfc", input.dtype)


def erfinv(input: Tensor) -> Tensor:
    """Computes erfinv of each element."""
    if config.eager_mode:
        from ml_switcheroo.core.errors import UnimplementedMathError

        raise UnimplementedMathError("No direct NumPy equivalent for erfinv.")
    else:
        return _emit_unary_node(input, "Erfinv", input.dtype)


def exp(input: Tensor) -> Tensor:
    """Computes exp of each element."""
    if config.eager_mode:
        data = np.exp(input.data)
        # Re-wrap boolean functions explicitly to ensure correct dtype type mapping
        return Tensor(data, input.shape, input.dtype, input.device)
    else:
        return _emit_unary_node(input, "Exp", input.dtype)


def exp2(input: Tensor) -> Tensor:
    """Computes exp2 of each element."""
    if config.eager_mode:
        data = np.exp2(input.data)
        # Re-wrap boolean functions explicitly to ensure correct dtype type mapping
        return Tensor(data, input.shape, input.dtype, input.device)
    else:
        return _emit_unary_node(input, "Exp2", input.dtype)


def expm1(input: Tensor) -> Tensor:
    """Computes expm1 of each element."""
    if config.eager_mode:
        data = np.expm1(input.data)
        # Re-wrap boolean functions explicitly to ensure correct dtype type mapping
        return Tensor(data, input.shape, input.dtype, input.device)
    else:
        return _emit_unary_node(input, "Expm1", input.dtype)


def fix(input: Tensor) -> Tensor:
    """Computes fix of each element."""
    if config.eager_mode:
        data = np.fix(input.data)
        # Re-wrap boolean functions explicitly to ensure correct dtype type mapping
        return Tensor(data, input.shape, input.dtype, input.device)
    else:
        return _emit_unary_node(input, "Fix", input.dtype)


def floor(input: Tensor) -> Tensor:
    """Computes floor of each element."""
    if config.eager_mode:
        data = np.floor(input.data)
        # Re-wrap boolean functions explicitly to ensure correct dtype type mapping
        return Tensor(data, input.shape, input.dtype, input.device)
    else:
        return _emit_unary_node(input, "Floor", input.dtype)


def imag(input: Tensor) -> Tensor:
    """Computes imag of each element."""
    if config.eager_mode:
        data = np.imag(input.data)
        # Re-wrap boolean functions explicitly to ensure correct dtype type mapping
        return Tensor(data, input.shape, input.dtype, input.device)
    else:
        return _emit_unary_node(input, "Imag", input.dtype)


def isfinite(input: Tensor) -> Tensor:
    """Computes isfinite of each element."""
    if config.eager_mode:
        data = np.isfinite(input.data)
        # Re-wrap boolean functions explicitly to ensure correct dtype type mapping
        return Tensor(data, input.shape, DType.Bool, input.device)
        return Tensor(data, input.shape, input.dtype, input.device)
    else:
        return _emit_unary_node(input, "Isfinite", DType.Bool)


def isinf(input: Tensor) -> Tensor:
    """Computes isinf of each element."""
    if config.eager_mode:
        data = np.isinf(input.data)
        # Re-wrap boolean functions explicitly to ensure correct dtype type mapping
        return Tensor(data, input.shape, DType.Bool, input.device)
        return Tensor(data, input.shape, input.dtype, input.device)
    else:
        return _emit_unary_node(input, "Isinf", DType.Bool)


def isnan(input: Tensor) -> Tensor:
    """Computes isnan of each element."""
    if config.eager_mode:
        data = np.isnan(input.data)
        # Re-wrap boolean functions explicitly to ensure correct dtype type mapping
        return Tensor(data, input.shape, DType.Bool, input.device)
        return Tensor(data, input.shape, input.dtype, input.device)
    else:
        return _emit_unary_node(input, "Isnan", DType.Bool)


def lgamma(input: Tensor) -> Tensor:
    """Computes lgamma of each element."""
    if config.eager_mode:
        from ml_switcheroo.core.errors import UnimplementedMathError

        raise UnimplementedMathError("No direct NumPy equivalent for lgamma.")
    else:
        return _emit_unary_node(input, "Lgamma", input.dtype)


def log(input: Tensor) -> Tensor:
    """Computes log of each element."""
    if config.eager_mode:
        data = np.log(input.data)
        # Re-wrap boolean functions explicitly to ensure correct dtype type mapping
        return Tensor(data, input.shape, input.dtype, input.device)
    else:
        return _emit_unary_node(input, "Log", input.dtype)


def log10(input: Tensor) -> Tensor:
    """Computes log10 of each element."""
    if config.eager_mode:
        data = np.log10(input.data)
        # Re-wrap boolean functions explicitly to ensure correct dtype type mapping
        return Tensor(data, input.shape, input.dtype, input.device)
    else:
        return _emit_unary_node(input, "Log10", input.dtype)


def log1p(input: Tensor) -> Tensor:
    """Computes log1p of each element."""
    if config.eager_mode:
        data = np.log1p(input.data)
        # Re-wrap boolean functions explicitly to ensure correct dtype type mapping
        return Tensor(data, input.shape, input.dtype, input.device)
    else:
        return _emit_unary_node(input, "Log1P", input.dtype)


def log2(input: Tensor) -> Tensor:
    """Computes log2 of each element."""
    if config.eager_mode:
        data = np.log2(input.data)
        # Re-wrap boolean functions explicitly to ensure correct dtype type mapping
        return Tensor(data, input.shape, input.dtype, input.device)
    else:
        return _emit_unary_node(input, "Log2", input.dtype)


def logical_not(input: Tensor) -> Tensor:
    """Computes logical_not of each element."""
    if config.eager_mode:
        data = np.logical_not(input.data)
        # Re-wrap boolean functions explicitly to ensure correct dtype type mapping
        return Tensor(data, input.shape, DType.Bool, input.device)
        return Tensor(data, input.shape, input.dtype, input.device)
    else:
        return _emit_unary_node(input, "LogicalNot", DType.Bool)


def negative(input: Tensor) -> Tensor:
    """Computes negative of each element."""
    if config.eager_mode:
        data = np.negative(input.data)
        # Re-wrap boolean functions explicitly to ensure correct dtype type mapping
        return Tensor(data, input.shape, input.dtype, input.device)
    else:
        return _emit_unary_node(input, "Negative", input.dtype)


def positive(input: Tensor) -> Tensor:
    """Computes positive of each element."""
    if config.eager_mode:
        data = np.positive(input.data)
        # Re-wrap boolean functions explicitly to ensure correct dtype type mapping
        return Tensor(data, input.shape, input.dtype, input.device)
    else:
        return _emit_unary_node(input, "Positive", input.dtype)


def rad2deg(input: Tensor) -> Tensor:
    """Computes rad2deg of each element."""
    if config.eager_mode:
        data = np.rad2deg(input.data)
        # Re-wrap boolean functions explicitly to ensure correct dtype type mapping
        return Tensor(data, input.shape, input.dtype, input.device)
    else:
        return _emit_unary_node(input, "Rad2Deg", input.dtype)


def real(input: Tensor) -> Tensor:
    """Computes real of each element."""
    if config.eager_mode:
        data = np.real(input.data)
        # Re-wrap boolean functions explicitly to ensure correct dtype type mapping
        return Tensor(data, input.shape, input.dtype, input.device)
    else:
        return _emit_unary_node(input, "Real", input.dtype)


def reciprocal(input: Tensor) -> Tensor:
    """Computes reciprocal of each element."""
    if config.eager_mode:
        data = np.reciprocal(input.data)
        # Re-wrap boolean functions explicitly to ensure correct dtype type mapping
        return Tensor(data, input.shape, input.dtype, input.device)
    else:
        return _emit_unary_node(input, "Reciprocal", input.dtype)


def round(input: Tensor) -> Tensor:
    """Computes round of each element."""
    if config.eager_mode:
        data = np.round(input.data)
        # Re-wrap boolean functions explicitly to ensure correct dtype type mapping
        return Tensor(data, input.shape, input.dtype, input.device)
    else:
        return _emit_unary_node(input, "Round", input.dtype)


def sign(input: Tensor) -> Tensor:
    """Computes sign of each element."""
    if config.eager_mode:
        data = np.sign(input.data)
        # Re-wrap boolean functions explicitly to ensure correct dtype type mapping
        return Tensor(data, input.shape, input.dtype, input.device)
    else:
        return _emit_unary_node(input, "Sign", input.dtype)


def sin(input: Tensor) -> Tensor:
    """Computes sin of each element."""
    if config.eager_mode:
        data = np.sin(input.data)
        # Re-wrap boolean functions explicitly to ensure correct dtype type mapping
        return Tensor(data, input.shape, input.dtype, input.device)
    else:
        return _emit_unary_node(input, "Sin", input.dtype)


def sinc(input: Tensor) -> Tensor:
    """Computes sinc of each element."""
    if config.eager_mode:
        data = np.sinc(input.data)
        # Re-wrap boolean functions explicitly to ensure correct dtype type mapping
        return Tensor(data, input.shape, input.dtype, input.device)
    else:
        return _emit_unary_node(input, "Sinc", input.dtype)


def sinh(input: Tensor) -> Tensor:
    """Computes sinh of each element."""
    if config.eager_mode:
        data = np.sinh(input.data)
        # Re-wrap boolean functions explicitly to ensure correct dtype type mapping
        return Tensor(data, input.shape, input.dtype, input.device)
    else:
        return _emit_unary_node(input, "Sinh", input.dtype)


def sqrt(input: Tensor) -> Tensor:
    """Computes sqrt of each element."""
    if config.eager_mode:
        data = np.sqrt(input.data)
        # Re-wrap boolean functions explicitly to ensure correct dtype type mapping
        return Tensor(data, input.shape, input.dtype, input.device)
    else:
        return _emit_unary_node(input, "Sqrt", input.dtype)


def square(input: Tensor) -> Tensor:
    """Computes square of each element."""
    if config.eager_mode:
        data = np.square(input.data)
        # Re-wrap boolean functions explicitly to ensure correct dtype type mapping
        return Tensor(data, input.shape, input.dtype, input.device)
    else:
        return _emit_unary_node(input, "Square", input.dtype)


def tan(input: Tensor) -> Tensor:
    """Computes tan of each element."""
    if config.eager_mode:
        data = np.tan(input.data)
        # Re-wrap boolean functions explicitly to ensure correct dtype type mapping
        return Tensor(data, input.shape, input.dtype, input.device)
    else:
        return _emit_unary_node(input, "Tan", input.dtype)


def tanh(input: Tensor) -> Tensor:
    """Computes tanh of each element."""
    if config.eager_mode:
        data = np.tanh(input.data)
        # Re-wrap boolean functions explicitly to ensure correct dtype type mapping
        return Tensor(data, input.shape, input.dtype, input.device)
    else:
        return _emit_unary_node(input, "Tanh", input.dtype)


def trunc(input: Tensor) -> Tensor:
    """Computes trunc of each element."""
    if config.eager_mode:
        data = np.trunc(input.data)
        # Re-wrap boolean functions explicitly to ensure correct dtype type mapping
        return Tensor(data, input.shape, input.dtype, input.device)
    else:
        return _emit_unary_node(input, "Trunc", input.dtype)


def digamma(input: Tensor) -> Tensor:
    """Computes the logarithmic derivative of the gamma function."""
    if config.eager_mode:
        from ml_switcheroo.core.errors import UnimplementedMathError

        raise UnimplementedMathError("No direct NumPy equivalent for digamma.")
    else:
        return _emit_unary_node(input, "Digamma")


def rsqrt(input: Tensor) -> Tensor:
    """Computes the reciprocal of the square root of each element."""
    if config.eager_mode:
        data = 1.0 / np.sqrt(input.data)
        return Tensor(data, input.shape, input.dtype, input.device)
    else:
        return _emit_unary_node(input, "Rsqrt")


def atan2(input: Tensor, other: Tensor) -> Tensor:
    """Computes the element-wise arctangent of input / other."""
    if config.eager_mode:
        data = np.arctan2(input.data, other.data)
        return Tensor(data, input.shape, input.dtype, input.device)
    else:
        if not _tracer.is_tracing:
            raise RuntimeError("Cannot emit Atan2 node outside of a tracing context.")
        out_id = str(uuid.uuid4())
        from ml_switcheroo.shape import broadcast_shapes

        out_shape = broadcast_shapes(input.shape, other.shape)
        node = LogicalNode(
            id=out_id,
            op_type="Atan2",
            inputs=[input.data.id, other.data.id],
            shape_metadata=out_shape,
        )
        _tracer.add_node(node)
        proxy = ProxyTensor(id=out_id, shape=out_shape, dtype=input.dtype.value)
        return Tensor(
            data=proxy, shape=out_shape, dtype=input.dtype, device=input.device
        )


def frexp(input: Tensor) -> tuple[Tensor, Tensor]:
    """Decomposes a floating-point tensor into its mantissa and exponent components."""
    if config.eager_mode:
        mantissa, exponent = np.frexp(input.data)
        return (
            Tensor(mantissa, input.shape, input.dtype, input.device),
            Tensor(exponent, input.shape, DType.Int32, input.device),
        )
    else:
        if not _tracer.is_tracing:
            raise RuntimeError("Cannot emit Frexp node outside of a tracing context.")
        out_id_mantissa = str(uuid.uuid4())
        out_id_exponent = str(uuid.uuid4())
        node = LogicalNode(
            id=out_id_mantissa,  # Hack: use multiple outputs in a real IR
            op_type="Frexp",
            inputs=[input.data.id],
            shape_metadata=input.shape,
        )
        _tracer.add_node(node)
        proxy_m = ProxyTensor(
            id=out_id_mantissa, shape=input.shape, dtype=input.dtype.value
        )
        proxy_e = ProxyTensor(
            id=out_id_exponent, shape=input.shape, dtype=DType.Int32.value
        )
        return (
            Tensor(
                data=proxy_m, shape=input.shape, dtype=input.dtype, device=input.device
            ),
            Tensor(
                data=proxy_e, shape=input.shape, dtype=DType.Int32, device=input.device
            ),
        )


def cast(input: Tensor, dtype: DType) -> Tensor:
    """Casts a tensor to the specified data type."""
    if config.eager_mode:
        data = input.data.astype(dtype.value)
        return Tensor(data, input.shape, dtype, input.device)
    else:
        if not _tracer.is_tracing:
            raise RuntimeError("Cannot emit Cast node outside of a tracing context.")
        out_id = str(uuid.uuid4())
        node = LogicalNode(
            id=out_id,
            op_type="Cast",
            inputs=[input.data.id],
            attributes={"to": dtype.value},
            shape_metadata=input.shape,
        )
        _tracer.add_node(node)
        proxy = ProxyTensor(id=out_id, shape=input.shape, dtype=dtype.value)
        return Tensor(data=proxy, shape=input.shape, dtype=dtype, device=input.device)


def bitcast(input: Tensor, dtype: DType) -> Tensor:
    """Bitcasts a tensor without changing the underlying bits."""
    if config.eager_mode:
        data = input.data.view(dtype.value)
        return Tensor(data, input.shape, dtype, input.device)
    else:
        if not _tracer.is_tracing:
            raise RuntimeError("Cannot emit Bitcast node outside of a tracing context.")
        out_id = str(uuid.uuid4())
        node = LogicalNode(
            id=out_id,
            op_type="Bitcast",
            inputs=[input.data.id],
            attributes={"to": dtype.value},
            shape_metadata=input.shape,
        )
        _tracer.add_node(node)
        proxy = ProxyTensor(id=out_id, shape=input.shape, dtype=dtype.value)
        return Tensor(data=proxy, shape=input.shape, dtype=dtype, device=input.device)
