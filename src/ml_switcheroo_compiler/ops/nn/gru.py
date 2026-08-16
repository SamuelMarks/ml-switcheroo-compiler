# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""RNN operations."""

from typing import Any, Optional

from ml_switcheroo_compiler.backends.registry import get_active_backend
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.ops.base import OpDef, register_op
from ml_switcheroo_compiler.ops.binary import add, multiply, subtract
from ml_switcheroo_compiler.ops.linalg import matmul
from ml_switcheroo_compiler.ops.shape.splitting import split
from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node
from ml_switcheroo_compiler.ops.unary import tanh


def _compute_gru_gates(x_parts: tuple[Any, ...], r_parts: tuple[Any, ...], state: Tensor) -> Any:  # type: ignore
    """Evaluate _compute_gru_gates operation.

    Args:
        x_parts (tuple): The x_parts parameter.
        r_parts (tuple): The r_parts parameter.
        state (Tensor): The state parameter.

    Returns:
        Tensor: Result.
    """
    x_z, x_r, x_h = x_parts
    recurrent_z, recurrent_r, recurrent_h = r_parts
    z = _sigmoid(add(x_z, recurrent_z))
    r = _sigmoid(add(x_r, recurrent_r))
    hh = tanh(add(x_h, multiply(r, recurrent_h)))
    return add(multiply(z, state), multiply(subtract(1.0, z), hh))


def gru_cell(
    inputs: Tensor,  # type: ignore
    state: Tensor,  # type: ignore
    kernel: Tensor,  # type: ignore
    recurrent_kernel: Tensor,  # type: ignore
    bias: Optional[Tensor] = None,  # type: ignore
) -> Any:
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
    matrix_x = matmul(inputs, kernel)  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism
    if bias is not None:
        matrix_x = add(matrix_x, bias)

    matrix_inner = matmul(state, recurrent_kernel)  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism

    x_parts = split(matrix_x, 3, axis=-1)
    r_parts = split(matrix_inner, 3, axis=-1)

    h_new = _compute_gru_gates(x_parts, r_parts, state)  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism
    return h_new, h_new


@register_op("Gru")
class Gru(OpDef):
    """Gru operation."""

    op_name = "Gru"

    def infer_shape(self, *args: Any, **kwargs: Any) -> Any:
        """Infer the output shape for the infer_shape operation.

        Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

        Returns: Any: Result.
        """
        return args[0] if args else ()


def gru(*args: Any, **kwargs: Any) -> Any:
    """GRU layer.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

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


def _sigmoid(x: Any) -> Any:
    """Evaluate _sigmoid operation.

    Args:
        x (object): The x parameter.

    Returns: Any: Result.
    """
    from ml_switcheroo_compiler.ops.nn.activations import sigmoid as s

    return s(x)
