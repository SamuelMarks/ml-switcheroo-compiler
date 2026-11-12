"""RNN operations."""

from __future__ import annotations

import typing  # noqa: F401

from ml_switcheroo_compiler.backends.registry import ensure_eager_list, get_active_backend
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.shape import (
    TensorOrScalar,
    extract_data,
    extract_dtype,
    extract_shape,
)
from ml_switcheroo_compiler.core.tensor import Tensor, ensure_tensor
from ml_switcheroo_compiler.ops.base import OpDef, register_op
from ml_switcheroo_compiler.ops.binary import add, multiply, subtract
from ml_switcheroo_compiler.ops.linalg.matmul import matmul
from ml_switcheroo_compiler.ops.shape.splitting import split
from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node
from ml_switcheroo_compiler.ops.unary import tanh


def _compute_gru_gates(x_parts: TensorOrScalar, r_parts: TensorOrScalar, state: Tensor) -> TensorOrScalar:
    """Compute the compute gru gates tensor operation.

    Args:
        x_parts (tuple): Input `x_parts` argument (x parts).
        r_parts (tuple): Input `r_parts` argument (r parts).
        state (Tensor): Input `state` argument (state).

    Returns:
        Tensor: Computed `Tensor` output.
    """
    x_seq = [ensure_tensor(t) for t in ensure_eager_list(x_parts)]
    r_seq = [ensure_tensor(t) for t in ensure_eager_list(r_parts)]
    x_z, x_r, x_h = x_seq[0], x_seq[1], x_seq[2]
    recurrent_z, recurrent_r, recurrent_h = r_seq[0], r_seq[1], r_seq[2]
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
        inputs (Tensor): Input `inputs` argument (inputs).
        state (Tensor): Input `state` argument (state).
        kernel (Tensor): Input `kernel` argument (kernel).
        recurrent_kernel (Tensor): Input `recurrent_kernel` argument (recurrent kernel).
        bias (Optional): Input `bias` argument (bias).

    Returns:
        tuple: Computed `tuple` output.
    """
    matrix_x = matmul(inputs, kernel)
    if bias is not None:
        matrix_x = add(matrix_x, bias)

    matrix_inner = matmul(state, recurrent_kernel)

    x_parts = split(matrix_x, 3, axis=-1)
    r_parts = split(matrix_inner, 3, axis=-1)

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
            tuple[int, ...]: Computed `tuple[int, ...]` output.
        """
        return args[0] if args else ()


def gru(*args: TensorOrScalar, **kwargs: TensorOrScalar) -> TensorOrScalar:
    """GRU layer.

    Args:
        *args (TensorOrScalar): Positional args.
        **kwargs (TensorOrScalar): Keyword args.

    Returns:
        Tensor: Computed `Tensor` output.
    """
    if config.eager_mode:
        backend = get_active_backend()
        return backend.execute_op("Gru", *[extract_data(a) for a in args], **kwargs)

    t_args = [a for a in args if isinstance(a, Tensor)]
    out_shape = extract_shape(t_args[0]) if t_args else ()
    out_dtype = extract_dtype(t_args[0], DType.Float32) if t_args else DType.Float32
    return _emit_shape_node("Gru", list(args), kwargs, out_shape, out_dtype)


def _sigmoid(x: Tensor) -> Tensor:
    """Compute the sigmoid tensor operation.

    Args:
        x (Tensor): Input `x` argument (x).

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    from ml_switcheroo_compiler.ops.nn.activations import sigmoid as s

    return s(x)
