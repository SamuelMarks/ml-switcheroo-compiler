"""RNN operations."""

from typing import Optional

from ml_switcheroo_compiler.backends.registry import get_active_backend
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.nn.activations import sigmoid
from ml_switcheroo_compiler.ops.base import OpDef, register_op
from ml_switcheroo_compiler.ops.binary import add, multiply, subtract
from ml_switcheroo_compiler.ops.linalg import matmul
from ml_switcheroo_compiler.ops.shape import split
from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node
from ml_switcheroo_compiler.ops.unary import tanh


def _compute_gru_gates(x_parts: tuple, r_parts: tuple, state: Tensor) -> Tensor:
    """Function docstring.

    Args:
        x_parts: Arg.
        r_parts: Arg.
        state: Arg.
    """
    x_z, x_r, x_h = x_parts
    recurrent_z, recurrent_r, recurrent_h = r_parts
    z = sigmoid(add(x_z, recurrent_z))
    r = sigmoid(add(x_r, recurrent_r))
    hh = tanh(add(x_h, multiply(r, recurrent_h)))
    return add(multiply(z, state), multiply(subtract(1.0, z), hh))


def gru_cell(
    inputs: Tensor,
    state: Tensor,
    kernel: Tensor,
    recurrent_kernel: Tensor,
    bias: Optional[Tensor] = None,
) -> tuple[Tensor, Tensor]:
    """Fused GRU cell math."""
    matrix_x = matmul(inputs, kernel)
    if bias is not None:
        matrix_x = add(matrix_x, bias)

    matrix_inner = matmul(state, recurrent_kernel)

    x_parts = split(matrix_x, 3, dim=-1)
    r_parts = split(matrix_inner, 3, dim=-1)

    h_new = _compute_gru_gates(x_parts, r_parts, state)
    return h_new, h_new


@register_op("Gru")
class Gru(OpDef):
    """Gru operation."""

    op_name = "Gru"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        return args[0] if args else ()


def gru(*args: object, **kwargs: object) -> Tensor:
    """GRU layer."""
    if config.eager_mode:
        backend = get_active_backend()
        return backend.execute_op("Gru", *[getattr(a, "data", a) for a in args], **kwargs)

    t_args = [a for a in args if isinstance(a, Tensor)]
    out_shape = getattr(t_args[0], "shape", ()) if t_args else ()
    out_dtype = getattr(t_args[0], "dtype", DType.Float32) if t_args else DType.Float32
    return _emit_shape_node("Gru", list(args), kwargs, out_shape, out_dtype)
