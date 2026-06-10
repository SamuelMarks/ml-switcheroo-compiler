"""Binary Operations."""

import uuid
from typing import Tuple, Union
import numpy as np
from ml_switcheroo.core.tensor import Tensor
from ml_switcheroo.core.dtype import DType
from ml_switcheroo.core.config import config
from ml_switcheroo.tracing import _tracer, ProxyTensor
from ml_switcheroo_ir import LogicalNode
from ml_switcheroo.shape import broadcast_shapes


def _emit_binary_node(
    input: Tensor,
    other: Tensor,
    op_type: str,
    out_dtype: DType = None,
    attributes: dict = None,
) -> Tensor:
    """Emit a binary node to the IR graph."""
    if out_dtype is None:
        out_dtype = input.dtype  # Simplification, actual type promotion should happen
    if config.eager_mode:
        raise RuntimeError("Cannot emit node in eager mode.")
    if not _tracer.is_tracing:
        raise RuntimeError(f"Cannot emit {op_type} node outside of a tracing context.")

    out_id = str(uuid.uuid4())
    out_shape = broadcast_shapes(input.shape, other.shape)
    node = LogicalNode(
        id=out_id,
        op_type=op_type,
        inputs=[input.data.id, other.data.id],
        attributes=attributes or {},
        shape_metadata=out_shape,
    )
    _tracer.add_node(node)

    proxy = ProxyTensor(id=out_id, shape=out_shape, dtype=out_dtype.value)
    return Tensor(data=proxy, shape=out_shape, dtype=out_dtype, device=input.device)


def add(input: Tensor, other: Tensor) -> Tensor:
    """Computes add of input and other."""
    if config.eager_mode:
        data = np.add(input.data, other.data)
        if "add" in [
            "equal",
            "greater",
            "greater_equal",
            "less",
            "less_equal",
            "logical_and",
            "logical_or",
            "logical_xor",
            "not_equal",
            "isclose",
        ]:
            return Tensor(data, data.shape, DType.Bool, input.device)
        return Tensor(data, data.shape, input.dtype, input.device)
    else:
        return _emit_binary_node(input, other, "Add", input.dtype, None)


def bitwise_and(input: Tensor, other: Tensor) -> Tensor:
    """Computes bitwise_and of input and other."""
    if config.eager_mode:
        data = np.bitwise_and(input.data, other.data)
        if "bitwise_and" in [
            "equal",
            "greater",
            "greater_equal",
            "less",
            "less_equal",
            "logical_and",
            "logical_or",
            "logical_xor",
            "not_equal",
            "isclose",
        ]:
            return Tensor(data, data.shape, DType.Bool, input.device)
        return Tensor(data, data.shape, input.dtype, input.device)
    else:
        return _emit_binary_node(input, other, "BitwiseAnd", input.dtype, None)


def bitwise_or(input: Tensor, other: Tensor) -> Tensor:
    """Computes bitwise_or of input and other."""
    if config.eager_mode:
        data = np.bitwise_or(input.data, other.data)
        if "bitwise_or" in [
            "equal",
            "greater",
            "greater_equal",
            "less",
            "less_equal",
            "logical_and",
            "logical_or",
            "logical_xor",
            "not_equal",
            "isclose",
        ]:
            return Tensor(data, data.shape, DType.Bool, input.device)
        return Tensor(data, data.shape, input.dtype, input.device)
    else:
        return _emit_binary_node(input, other, "BitwiseOr", input.dtype, None)


def bitwise_xor(input: Tensor, other: Tensor) -> Tensor:
    """Computes bitwise_xor of input and other."""
    if config.eager_mode:
        data = np.bitwise_xor(input.data, other.data)
        if "bitwise_xor" in [
            "equal",
            "greater",
            "greater_equal",
            "less",
            "less_equal",
            "logical_and",
            "logical_or",
            "logical_xor",
            "not_equal",
            "isclose",
        ]:
            return Tensor(data, data.shape, DType.Bool, input.device)
        return Tensor(data, data.shape, input.dtype, input.device)
    else:
        return _emit_binary_node(input, other, "BitwiseXor", input.dtype, None)


def copysign(input: Tensor, other: Tensor) -> Tensor:
    """Computes copysign of input and other."""
    if config.eager_mode:
        data = np.copysign(input.data, other.data)
        if "copysign" in [
            "equal",
            "greater",
            "greater_equal",
            "less",
            "less_equal",
            "logical_and",
            "logical_or",
            "logical_xor",
            "not_equal",
            "isclose",
        ]:
            return Tensor(data, data.shape, DType.Bool, input.device)
        return Tensor(data, data.shape, input.dtype, input.device)
    else:
        return _emit_binary_node(input, other, "Copysign", input.dtype, None)


def divide(input: Tensor, other: Tensor) -> Tensor:
    """Computes divide of input and other."""
    if config.eager_mode:
        data = np.true_divide(input.data, other.data)
        if "divide" in [
            "equal",
            "greater",
            "greater_equal",
            "less",
            "less_equal",
            "logical_and",
            "logical_or",
            "logical_xor",
            "not_equal",
            "isclose",
        ]:
            return Tensor(data, data.shape, DType.Bool, input.device)
        return Tensor(data, data.shape, input.dtype, input.device)
    else:
        return _emit_binary_node(input, other, "Divide", input.dtype, None)


def equal(input: Tensor, other: Tensor) -> Tensor:
    """Computes equal of input and other."""
    if config.eager_mode:
        data = np.equal(input.data, other.data)
        if "equal" in [
            "equal",
            "greater",
            "greater_equal",
            "less",
            "less_equal",
            "logical_and",
            "logical_or",
            "logical_xor",
            "not_equal",
            "isclose",
        ]:
            return Tensor(data, data.shape, DType.Bool, input.device)
        return Tensor(data, data.shape, input.dtype, input.device)
    else:
        return _emit_binary_node(input, other, "Equal", DType.Bool, None)


def float_power(input: Tensor, other: Tensor) -> Tensor:
    """Computes float_power of input and other."""
    if config.eager_mode:
        data = np.float_power(input.data, other.data)
        if "float_power" in [
            "equal",
            "greater",
            "greater_equal",
            "less",
            "less_equal",
            "logical_and",
            "logical_or",
            "logical_xor",
            "not_equal",
            "isclose",
        ]:
            return Tensor(data, data.shape, DType.Bool, input.device)
        return Tensor(data, data.shape, input.dtype, input.device)
    else:
        return _emit_binary_node(input, other, "FloatPower", input.dtype, None)


def floor_divide(input: Tensor, other: Tensor) -> Tensor:
    """Computes floor_divide of input and other."""
    if config.eager_mode:
        data = np.floor_divide(input.data, other.data)
        if "floor_divide" in [
            "equal",
            "greater",
            "greater_equal",
            "less",
            "less_equal",
            "logical_and",
            "logical_or",
            "logical_xor",
            "not_equal",
            "isclose",
        ]:
            return Tensor(data, data.shape, DType.Bool, input.device)
        return Tensor(data, data.shape, input.dtype, input.device)
    else:
        return _emit_binary_node(input, other, "FloorDivide", input.dtype, None)


def fmax(input: Tensor, other: Tensor) -> Tensor:
    """Computes fmax of input and other."""
    if config.eager_mode:
        data = np.fmax(input.data, other.data)
        if "fmax" in [
            "equal",
            "greater",
            "greater_equal",
            "less",
            "less_equal",
            "logical_and",
            "logical_or",
            "logical_xor",
            "not_equal",
            "isclose",
        ]:
            return Tensor(data, data.shape, DType.Bool, input.device)
        return Tensor(data, data.shape, input.dtype, input.device)
    else:
        return _emit_binary_node(input, other, "Fmax", input.dtype, None)


def fmin(input: Tensor, other: Tensor) -> Tensor:
    """Computes fmin of input and other."""
    if config.eager_mode:
        data = np.fmin(input.data, other.data)
        if "fmin" in [
            "equal",
            "greater",
            "greater_equal",
            "less",
            "less_equal",
            "logical_and",
            "logical_or",
            "logical_xor",
            "not_equal",
            "isclose",
        ]:
            return Tensor(data, data.shape, DType.Bool, input.device)
        return Tensor(data, data.shape, input.dtype, input.device)
    else:
        return _emit_binary_node(input, other, "Fmin", input.dtype, None)


def fmod(input: Tensor, other: Tensor) -> Tensor:
    """Computes fmod of input and other."""
    if config.eager_mode:
        data = np.fmod(input.data, other.data)
        if "fmod" in [
            "equal",
            "greater",
            "greater_equal",
            "less",
            "less_equal",
            "logical_and",
            "logical_or",
            "logical_xor",
            "not_equal",
            "isclose",
        ]:
            return Tensor(data, data.shape, DType.Bool, input.device)
        return Tensor(data, data.shape, input.dtype, input.device)
    else:
        return _emit_binary_node(input, other, "Fmod", input.dtype, None)


def gcd(input: Tensor, other: Tensor) -> Tensor:
    """Computes gcd of input and other."""
    if config.eager_mode:
        data = np.gcd(input.data, other.data)
        if "gcd" in [
            "equal",
            "greater",
            "greater_equal",
            "less",
            "less_equal",
            "logical_and",
            "logical_or",
            "logical_xor",
            "not_equal",
            "isclose",
        ]:
            return Tensor(data, data.shape, DType.Bool, input.device)
        return Tensor(data, data.shape, input.dtype, input.device)
    else:
        return _emit_binary_node(input, other, "Gcd", input.dtype, None)


def greater(input: Tensor, other: Tensor) -> Tensor:
    """Computes greater of input and other."""
    if config.eager_mode:
        data = np.greater(input.data, other.data)
        if "greater" in [
            "equal",
            "greater",
            "greater_equal",
            "less",
            "less_equal",
            "logical_and",
            "logical_or",
            "logical_xor",
            "not_equal",
            "isclose",
        ]:
            return Tensor(data, data.shape, DType.Bool, input.device)
        return Tensor(data, data.shape, input.dtype, input.device)
    else:
        return _emit_binary_node(input, other, "Greater", DType.Bool, None)


def greater_equal(input: Tensor, other: Tensor) -> Tensor:
    """Computes greater_equal of input and other."""
    if config.eager_mode:
        data = np.greater_equal(input.data, other.data)
        if "greater_equal" in [
            "equal",
            "greater",
            "greater_equal",
            "less",
            "less_equal",
            "logical_and",
            "logical_or",
            "logical_xor",
            "not_equal",
            "isclose",
        ]:
            return Tensor(data, data.shape, DType.Bool, input.device)
        return Tensor(data, data.shape, input.dtype, input.device)
    else:
        return _emit_binary_node(input, other, "GreaterEqual", DType.Bool, None)


def heaviside(input: Tensor, other: Tensor) -> Tensor:
    """Computes heaviside of input and other."""
    if config.eager_mode:
        data = np.heaviside(input.data, other.data)
        if "heaviside" in [
            "equal",
            "greater",
            "greater_equal",
            "less",
            "less_equal",
            "logical_and",
            "logical_or",
            "logical_xor",
            "not_equal",
            "isclose",
        ]:
            return Tensor(data, data.shape, DType.Bool, input.device)
        return Tensor(data, data.shape, input.dtype, input.device)
    else:
        return _emit_binary_node(input, other, "Heaviside", input.dtype, None)


def hypot(input: Tensor, other: Tensor) -> Tensor:
    """Computes hypot of input and other."""
    if config.eager_mode:
        data = np.hypot(input.data, other.data)
        if "hypot" in [
            "equal",
            "greater",
            "greater_equal",
            "less",
            "less_equal",
            "logical_and",
            "logical_or",
            "logical_xor",
            "not_equal",
            "isclose",
        ]:
            return Tensor(data, data.shape, DType.Bool, input.device)
        return Tensor(data, data.shape, input.dtype, input.device)
    else:
        return _emit_binary_node(input, other, "Hypot", input.dtype, None)


def lcm(input: Tensor, other: Tensor) -> Tensor:
    """Computes lcm of input and other."""
    if config.eager_mode:
        data = np.lcm(input.data, other.data)
        if "lcm" in [
            "equal",
            "greater",
            "greater_equal",
            "less",
            "less_equal",
            "logical_and",
            "logical_or",
            "logical_xor",
            "not_equal",
            "isclose",
        ]:
            return Tensor(data, data.shape, DType.Bool, input.device)
        return Tensor(data, data.shape, input.dtype, input.device)
    else:
        return _emit_binary_node(input, other, "Lcm", input.dtype, None)


def ldexp(input: Tensor, other: Tensor) -> Tensor:
    """Computes ldexp of input and other."""
    if config.eager_mode:
        data = np.ldexp(input.data, other.data)
        if "ldexp" in [
            "equal",
            "greater",
            "greater_equal",
            "less",
            "less_equal",
            "logical_and",
            "logical_or",
            "logical_xor",
            "not_equal",
            "isclose",
        ]:
            return Tensor(data, data.shape, DType.Bool, input.device)
        return Tensor(data, data.shape, input.dtype, input.device)
    else:
        return _emit_binary_node(input, other, "Ldexp", input.dtype, None)


def left_shift(input: Tensor, other: Tensor) -> Tensor:
    """Computes left_shift of input and other."""
    if config.eager_mode:
        data = np.left_shift(input.data, other.data)
        if "left_shift" in [
            "equal",
            "greater",
            "greater_equal",
            "less",
            "less_equal",
            "logical_and",
            "logical_or",
            "logical_xor",
            "not_equal",
            "isclose",
        ]:
            return Tensor(data, data.shape, DType.Bool, input.device)
        return Tensor(data, data.shape, input.dtype, input.device)
    else:
        return _emit_binary_node(input, other, "LeftShift", input.dtype, None)


def less(input: Tensor, other: Tensor) -> Tensor:
    """Computes less of input and other."""
    if config.eager_mode:
        data = np.less(input.data, other.data)
        if "less" in [
            "equal",
            "greater",
            "greater_equal",
            "less",
            "less_equal",
            "logical_and",
            "logical_or",
            "logical_xor",
            "not_equal",
            "isclose",
        ]:
            return Tensor(data, data.shape, DType.Bool, input.device)
        return Tensor(data, data.shape, input.dtype, input.device)
    else:
        return _emit_binary_node(input, other, "Less", DType.Bool, None)


def less_equal(input: Tensor, other: Tensor) -> Tensor:
    """Computes less_equal of input and other."""
    if config.eager_mode:
        data = np.less_equal(input.data, other.data)
        if "less_equal" in [
            "equal",
            "greater",
            "greater_equal",
            "less",
            "less_equal",
            "logical_and",
            "logical_or",
            "logical_xor",
            "not_equal",
            "isclose",
        ]:
            return Tensor(data, data.shape, DType.Bool, input.device)
        return Tensor(data, data.shape, input.dtype, input.device)
    else:
        return _emit_binary_node(input, other, "LessEqual", DType.Bool, None)


def logaddexp(input: Tensor, other: Tensor) -> Tensor:
    """Computes logaddexp of input and other."""
    if config.eager_mode:
        data = np.logaddexp(input.data, other.data)
        if "logaddexp" in [
            "equal",
            "greater",
            "greater_equal",
            "less",
            "less_equal",
            "logical_and",
            "logical_or",
            "logical_xor",
            "not_equal",
            "isclose",
        ]:
            return Tensor(data, data.shape, DType.Bool, input.device)
        return Tensor(data, data.shape, input.dtype, input.device)
    else:
        return _emit_binary_node(input, other, "Logaddexp", input.dtype, None)


def logaddexp2(input: Tensor, other: Tensor) -> Tensor:
    """Computes logaddexp2 of input and other."""
    if config.eager_mode:
        data = np.logaddexp2(input.data, other.data)
        if "logaddexp2" in [
            "equal",
            "greater",
            "greater_equal",
            "less",
            "less_equal",
            "logical_and",
            "logical_or",
            "logical_xor",
            "not_equal",
            "isclose",
        ]:
            return Tensor(data, data.shape, DType.Bool, input.device)
        return Tensor(data, data.shape, input.dtype, input.device)
    else:
        return _emit_binary_node(input, other, "Logaddexp2", input.dtype, None)


def logical_and(input: Tensor, other: Tensor) -> Tensor:
    """Computes logical_and of input and other."""
    if config.eager_mode:
        data = np.logical_and(input.data, other.data)
        if "logical_and" in [
            "equal",
            "greater",
            "greater_equal",
            "less",
            "less_equal",
            "logical_and",
            "logical_or",
            "logical_xor",
            "not_equal",
            "isclose",
        ]:
            return Tensor(data, data.shape, DType.Bool, input.device)
        return Tensor(data, data.shape, input.dtype, input.device)
    else:
        return _emit_binary_node(input, other, "LogicalAnd", DType.Bool, None)


def logical_or(input: Tensor, other: Tensor) -> Tensor:
    """Computes logical_or of input and other."""
    if config.eager_mode:
        data = np.logical_or(input.data, other.data)
        if "logical_or" in [
            "equal",
            "greater",
            "greater_equal",
            "less",
            "less_equal",
            "logical_and",
            "logical_or",
            "logical_xor",
            "not_equal",
            "isclose",
        ]:
            return Tensor(data, data.shape, DType.Bool, input.device)
        return Tensor(data, data.shape, input.dtype, input.device)
    else:
        return _emit_binary_node(input, other, "LogicalOr", DType.Bool, None)


def logical_xor(input: Tensor, other: Tensor) -> Tensor:
    """Computes logical_xor of input and other."""
    if config.eager_mode:
        data = np.logical_xor(input.data, other.data)
        if "logical_xor" in [
            "equal",
            "greater",
            "greater_equal",
            "less",
            "less_equal",
            "logical_and",
            "logical_or",
            "logical_xor",
            "not_equal",
            "isclose",
        ]:
            return Tensor(data, data.shape, DType.Bool, input.device)
        return Tensor(data, data.shape, input.dtype, input.device)
    else:
        return _emit_binary_node(input, other, "LogicalXor", DType.Bool, None)


def maximum(input: Tensor, other: Tensor) -> Tensor:
    """Computes maximum of input and other."""
    if config.eager_mode:
        data = np.maximum(input.data, other.data)
        if "maximum" in [
            "equal",
            "greater",
            "greater_equal",
            "less",
            "less_equal",
            "logical_and",
            "logical_or",
            "logical_xor",
            "not_equal",
            "isclose",
        ]:
            return Tensor(data, data.shape, DType.Bool, input.device)
        return Tensor(data, data.shape, input.dtype, input.device)
    else:
        return _emit_binary_node(input, other, "Maximum", input.dtype, None)


def minimum(input: Tensor, other: Tensor) -> Tensor:
    """Computes minimum of input and other."""
    if config.eager_mode:
        data = np.minimum(input.data, other.data)
        if "minimum" in [
            "equal",
            "greater",
            "greater_equal",
            "less",
            "less_equal",
            "logical_and",
            "logical_or",
            "logical_xor",
            "not_equal",
            "isclose",
        ]:
            return Tensor(data, data.shape, DType.Bool, input.device)
        return Tensor(data, data.shape, input.dtype, input.device)
    else:
        return _emit_binary_node(input, other, "Minimum", input.dtype, None)


def mod(input: Tensor, other: Tensor) -> Tensor:
    """Computes mod of input and other."""
    if config.eager_mode:
        data = np.mod(input.data, other.data)
        if "mod" in [
            "equal",
            "greater",
            "greater_equal",
            "less",
            "less_equal",
            "logical_and",
            "logical_or",
            "logical_xor",
            "not_equal",
            "isclose",
        ]:
            return Tensor(data, data.shape, DType.Bool, input.device)
        return Tensor(data, data.shape, input.dtype, input.device)
    else:
        return _emit_binary_node(input, other, "Mod", input.dtype, None)


def multiply(input: Tensor, other: Tensor) -> Tensor:
    """Computes multiply of input and other."""
    if config.eager_mode:
        data = np.multiply(input.data, other.data)
        if "multiply" in [
            "equal",
            "greater",
            "greater_equal",
            "less",
            "less_equal",
            "logical_and",
            "logical_or",
            "logical_xor",
            "not_equal",
            "isclose",
        ]:
            return Tensor(data, data.shape, DType.Bool, input.device)
        return Tensor(data, data.shape, input.dtype, input.device)
    else:
        return _emit_binary_node(input, other, "Mul", input.dtype, None)


def nextafter(input: Tensor, other: Tensor) -> Tensor:
    """Computes nextafter of input and other."""
    if config.eager_mode:
        data = np.nextafter(input.data, other.data)
        if "nextafter" in [
            "equal",
            "greater",
            "greater_equal",
            "less",
            "less_equal",
            "logical_and",
            "logical_or",
            "logical_xor",
            "not_equal",
            "isclose",
        ]:
            return Tensor(data, data.shape, DType.Bool, input.device)
        return Tensor(data, data.shape, input.dtype, input.device)
    else:
        return _emit_binary_node(input, other, "Nextafter", input.dtype, None)


def not_equal(input: Tensor, other: Tensor) -> Tensor:
    """Computes not_equal of input and other."""
    if config.eager_mode:
        data = np.not_equal(input.data, other.data)
        if "not_equal" in [
            "equal",
            "greater",
            "greater_equal",
            "less",
            "less_equal",
            "logical_and",
            "logical_or",
            "logical_xor",
            "not_equal",
            "isclose",
        ]:
            return Tensor(data, data.shape, DType.Bool, input.device)
        return Tensor(data, data.shape, input.dtype, input.device)
    else:
        return _emit_binary_node(input, other, "NotEqual", DType.Bool, None)


def power(input: Tensor, other: Tensor) -> Tensor:
    """Computes power of input and other."""
    if config.eager_mode:
        data = np.power(input.data, other.data)
        if "power" in [
            "equal",
            "greater",
            "greater_equal",
            "less",
            "less_equal",
            "logical_and",
            "logical_or",
            "logical_xor",
            "not_equal",
            "isclose",
        ]:
            return Tensor(data, data.shape, DType.Bool, input.device)
        return Tensor(data, data.shape, input.dtype, input.device)
    else:
        return _emit_binary_node(input, other, "Power", input.dtype, None)


def remainder(input: Tensor, other: Tensor) -> Tensor:
    """Computes remainder of input and other."""
    if config.eager_mode:
        data = np.remainder(input.data, other.data)
        if "remainder" in [
            "equal",
            "greater",
            "greater_equal",
            "less",
            "less_equal",
            "logical_and",
            "logical_or",
            "logical_xor",
            "not_equal",
            "isclose",
        ]:
            return Tensor(data, data.shape, DType.Bool, input.device)
        return Tensor(data, data.shape, input.dtype, input.device)
    else:
        return _emit_binary_node(input, other, "Remainder", input.dtype, None)


def right_shift(input: Tensor, other: Tensor) -> Tensor:
    """Computes right_shift of input and other."""
    if config.eager_mode:
        data = np.right_shift(input.data, other.data)
        if "right_shift" in [
            "equal",
            "greater",
            "greater_equal",
            "less",
            "less_equal",
            "logical_and",
            "logical_or",
            "logical_xor",
            "not_equal",
            "isclose",
        ]:
            return Tensor(data, data.shape, DType.Bool, input.device)
        return Tensor(data, data.shape, input.dtype, input.device)
    else:
        return _emit_binary_node(input, other, "RightShift", input.dtype, None)


def subtract(input: Tensor, other: Tensor) -> Tensor:
    """Computes subtract of input and other."""
    if config.eager_mode:
        data = np.subtract(input.data, other.data)
        if "subtract" in [
            "equal",
            "greater",
            "greater_equal",
            "less",
            "less_equal",
            "logical_and",
            "logical_or",
            "logical_xor",
            "not_equal",
            "isclose",
        ]:
            return Tensor(data, data.shape, DType.Bool, input.device)
        return Tensor(data, data.shape, input.dtype, input.device)
    else:
        return _emit_binary_node(input, other, "Subtract", input.dtype, None)


def isclose(
    input: Tensor,
    other: Tensor,
    rtol: float = 1e-05,
    atol: float = 1e-08,
    equal_nan: bool = False,
) -> Tensor:
    """Computes isclose of input and other."""
    if config.eager_mode:
        data = np.isclose(
            input.data, other.data, rtol=rtol, atol=atol, equal_nan=equal_nan
        )
        if "isclose" in [
            "equal",
            "greater",
            "greater_equal",
            "less",
            "less_equal",
            "logical_and",
            "logical_or",
            "logical_xor",
            "not_equal",
            "isclose",
        ]:
            return Tensor(data, data.shape, DType.Bool, input.device)
        return Tensor(data, data.shape, input.dtype, input.device)
    else:
        return _emit_binary_node(
            input,
            other,
            "Isclose",
            DType.Bool,
            attributes={rtol: rtol, atol: atol, equal_nan: equal_nan},
        )


def divmod(input: Tensor, other: Tensor) -> Tuple[Tensor, Tensor]:
    """Returns the quotient and remainder of division element-wise."""
    if config.eager_mode:
        q, r = np.divmod(input.data, other.data)
        out_shape = np.broadcast_shapes(input.shape, other.shape)
        return (
            Tensor(q, out_shape, input.dtype, input.device),
            Tensor(r, out_shape, input.dtype, input.device),
        )
    else:
        if not _tracer.is_tracing:
            raise RuntimeError("Cannot emit Divmod node outside of a tracing context.")
        out_shape = broadcast_shapes(input.shape, other.shape)
        out_id_q = str(uuid.uuid4())
        out_id_r = str(uuid.uuid4())
        node = LogicalNode(
            id=out_id_q,  # Hack: use multiple outputs in a real IR
            op_type="DivMod",
            inputs=[input.data.id, other.data.id],
            shape_metadata=out_shape,
        )
        _tracer.add_node(node)
        proxy_q = ProxyTensor(id=out_id_q, shape=out_shape, dtype=input.dtype.value)
        proxy_r = ProxyTensor(id=out_id_r, shape=out_shape, dtype=input.dtype.value)
        return (
            Tensor(
                data=proxy_q, shape=out_shape, dtype=input.dtype, device=input.device
            ),
            Tensor(
                data=proxy_r, shape=out_shape, dtype=input.dtype, device=input.device
            ),
        )


def allclose(
    input: Tensor,
    other: Tensor,
    rtol: float = 1e-05,
    atol: float = 1e-08,
    equal_nan: bool = False,
) -> Union[bool, Tensor]:
    """Returns True if two tensors are element-wise equal within a tolerance."""
    if config.eager_mode:
        return bool(
            np.allclose(
                input.data, other.data, rtol=rtol, atol=atol, equal_nan=equal_nan
            )
        )
    else:
        if not _tracer.is_tracing:
            raise RuntimeError(
                "Cannot emit Allclose node outside of a tracing context."
            )
        out_id = str(uuid.uuid4())
        node = LogicalNode(
            id=out_id,
            op_type="AllClose",
            inputs=[input.data.id, other.data.id],
            attributes={rtol: rtol, atol: atol, equal_nan: equal_nan},
            shape_metadata=(),
        )
        _tracer.add_node(node)
        proxy = ProxyTensor(id=out_id, shape=(), dtype=DType.Bool.value)
        return Tensor(data=proxy, shape=(), dtype=DType.Bool, device=input.device)
