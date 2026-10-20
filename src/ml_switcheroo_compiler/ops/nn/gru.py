"""RNN operations."""

from __future__ import annotations

import typing  # noqa: F401
from collections.abc import Sequence

from ml_switcheroo_compiler.backends.registry import get_active_backend
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.shape import TensorOrScalar, TypedVariadicCallable
from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.ops.base import OpDef, register_op
from ml_switcheroo_compiler.ops.binary import add, multiply, subtract
from ml_switcheroo_compiler.ops.linalg.matmul import matmul
from ml_switcheroo_compiler.ops.shape.splitting import split
from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node
from ml_switcheroo_compiler.ops.unary import tanh


def _compute_gru_gates(x_parts: TensorOrScalar, r_parts: TensorOrScalar, state: Tensor) -> TensorOrScalar:
    """Compute the compute gru gates tensor operation.

    Args:
        x_parts (tuple): The x_parts parameter.
        r_parts (tuple): The r_parts parameter.
        state (Tensor): The state parameter.

    Returns:
        Tensor: Result.
    """
    x_z, x_r, x_h = getattr(x_parts, "__iter__")()
    recurrent_z, recurrent_r, recurrent_h = getattr(r_parts, "__iter__")()
    z = _sigmoid(add(x_z, recurrent_z))
    r = _sigmoid(add(x_r, recurrent_r))
    hh = tanh(add(x_h, multiply(r, recurrent_h)))
    return add(multiply(z, state), multiply(subtract(1.0, z), hh))


def gru_cell(
    inputs: Tensor,
    state: Tensor,
    kernel: Tensor,
    recurrent_kernel: Tensor,
    bias: Tensor | None = None,
) -> TensorOrScalar:
    """Fused GRU cell math.

    Args:
        inputs (Tensor): The inputs parameter.
        state (Tensor): The state parameter.
        kernel (Tensor): The kernel parameter.
        recurrent_kernel (Tensor): The recurrent_kernel parameter.
        bias (Optional): The bias parameter.

    Returns:
        tuple: Result.
    """
    matrix_x = matmul(inputs, kernel)
    if bias is not None:
        matrix_x = add(matrix_x, bias)

    matrix_inner = matmul(state, recurrent_kernel)

    x_parts = typing.cast(TypedVariadicCallable[Sequence[Tensor]], split)(matrix_x, 3, axis=-1)
    r_parts = typing.cast(TypedVariadicCallable[Sequence[Tensor]], split)(matrix_inner, 3, axis=-1)

    h_new = _compute_gru_gates(x_parts, r_parts, state)
    return h_new, h_new


@register_op("Gru")
class Gru(OpDef):
    """Gru operation."""

    op_name = "Gru"

    def infer_shape(self, *args: TensorOrScalar, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Infer the output shape for the infer_shape operation.

        Args:
        *args (TensorOrScalar): Positional args.
        **kwargs (TensorOrScalar): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        return args[0] if args else ()


def gru(*args: TensorOrScalar, **kwargs: TensorOrScalar) -> TensorOrScalar:
    """GRU layer.

    Args:
        *args (TensorOrScalar): Positional args.
        **kwargs (TensorOrScalar): Keyword args.

    Returns:
        Tensor: Result.
    """
    if config.eager_mode:
        backend = get_active_backend()
        return backend.execute_op("Gru", *[getattr(a, "data", a) for a in args], **kwargs)

    t_args = [a for a in args if isinstance(a, Tensor)]
    out_shape = getattr(t_args[0], "shape", ()) if t_args else ()
    out_dtype = getattr(t_args[0], "dtype", DType.Float32) if t_args else DType.Float32
    return _emit_shape_node("Gru", list(args), kwargs, out_shape, out_dtype)


def _sigmoid(x: Tensor) -> Tensor:
    """Compute the sigmoid tensor operation.

    Args:
        x (Tensor): The x parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.ops.nn.activations import sigmoid as s

    return s(x)
