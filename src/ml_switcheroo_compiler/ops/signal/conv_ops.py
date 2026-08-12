from .common_ops import _calculate_padding, _emit_signal_node

# ruff: noqa: E402, D100, D103, D104, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, D101, D102, D107, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Signal processing operations."""

from dataclasses import dataclass
from typing import Any, Optional

from ml_switcheroo_compiler.backends.registry import get_active_backend
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.base import OpDef, register_op
from ml_switcheroo_compiler.ops.linalg.utils import _emit_linalg_node
from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node


@register_op("Convolve2d")
class Convolve2d(OpDef):
    """Convolve2d."""

    def infer_shape(self, *args: Any, **kwargs: Any) -> Any:
        """Infer shape.

        Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

        Returns: Any: Result.
        """
        return args[0].shape


def _validate_conv2d_args(in1: Tensor, in2: Tensor) -> None:
    """Validate arguments for convolve2d.

    Args:
        in1 (Tensor): First input.
        in2 (Tensor): Second input.

    Raises:
        ValueError: If shapes are not statically known.
    """
    if in1.shape is None or in2.shape is None:
        raise ValueError("Inputs to convolve2d must have statically known shapes.")


def convolve2d(
    in1: Tensor,
    in2: Tensor,
    mode: str = "full",
    boundary: str = "fill",
    fillvalue: float = 0.0,
) -> Any:
    """Evaluate convolve2d operation.

    Args:
        in1 (Tensor): The in1 parameter.
        in2 (Tensor): The in2 parameter.
        mode (str): The mode parameter.
        boundary (str): The boundary parameter.
        fillvalue (float): The fillvalue parameter.

    Returns:
        Tensor: Result.
    """
    _validate_conv2d_args(in1, in2)
    kwargs = _calculate_padding(mode, boundary, fillvalue)

    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("Convolve2d", in1.data, in2.data, **kwargs)
        return Tensor(data, TensorConfig(data.shape, in1.dtype, in1.device))

    return _emit_signal_node(
        "Convolve2d",
        [in1, in2],
        kwargs,
        in1.shape,  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism
        in1.dtype,  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism
    )
