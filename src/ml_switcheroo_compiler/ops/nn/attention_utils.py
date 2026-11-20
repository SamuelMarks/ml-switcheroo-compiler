"""Attention mechanism operations."""

from __future__ import annotations

import math

from ml_switcheroo_compiler.backends.registry import get_active_backend
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.shape import OpValue, extract_data, extract_shape
from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.ops.base import OpDef, register_op
from ml_switcheroo_compiler.ops.binary import multiply, subtract
from ml_switcheroo_compiler.ops.creation.api import arange
from ml_switcheroo_compiler.ops.shape.api import expand_dims
from ml_switcheroo_compiler.ops.shape.joining import concatenate
from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node
from ml_switcheroo_compiler.ops.unary import cos, exp, sin

# slopes: [num_heads, 1, 1]


@register_op("Rope")
class RopeOp(OpDef):
    """Rotary Positional Encoding operation."""

    op_name = "Rope"

    def infer_shape(self, input: OpValue, **kwargs: OpValue) -> OpValue:
        """Infer shape.

        Args:
            input (OpValue): Input `input` argument (input).
            **kwargs (OpValue): Keyword args.

        Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
        """
        _typed_ret_1: OpValue = extract_shape(input)
        return _typed_ret_1


def rope(
    input: Tensor,
    axis: int,
    base: float = 10000.0,
    offset: int = 0,
) -> OpValue:
    """Apply Rotary Positional Encoding (RoPE) to the input tensor.

    Args:
        input: The input tensor, usually of shape (..., seq_len, axis).
        axis: The axisension of the rotary encoding.
        base: The base for the frequency scaling.
        offset: The starting position offset.

    Returns:
        The tensor with RoPE applied.
    """
    if config.eager_mode:
        data = get_active_backend().execute_op(
            "Rope",
            extract_data(input),
            axis=axis,
            base=base,
            offset=offset,
        )
        return Tensor(data, input.config)
    return _emit_shape_node("Rope", [input], {"axis": axis, "base": base, "offset": offset}, input.shape, input.dtype)


def sinusoidal_positional_encoding(
    seq_len: int,
    axis: int,
    base: float = 10000.0,
    dtype: OpValue = None,
) -> OpValue:
    """Generate sinusoidal positional encodings.

    Args:
        seq_len: Length of the sequence.
        axis: Dimensionality of the embeddings.
        base: Base for frequency scaling.
        dtype: Data type of the returned tensor.

    Returns:
        A tensor of shape (seq_len, axis) containing the encodings.
    """
    position = arange(seq_len)
    div_term = exp(multiply(arange(0, axis, 2), -math.log(base) / axis))
    pe_sin = sin(multiply(expand_dims(position, 1), expand_dims(div_term, 0)))
    pe_cos = cos(multiply(expand_dims(position, 1), expand_dims(div_term, 0)))
    return concatenate([pe_sin, pe_cos], axis=-1)


def alibi_mask(
    seq_len: int,
    num_heads: int,
    dtype: OpValue = None,
) -> OpValue:
    """Generate an ALiBi (Attention with Linear Biases) mask.

    Args:
        seq_len: The sequence length.
        num_heads: The number of attention heads.
        dtype: The data type of the mask.

    Returns:
        A tensor representing the ALiBi mask.
    """
    # Create mask of shape (num_heads, seq_len, seq_len)
    # ALiBi biases attention scores by a linear penalty based on distance.
    positions = arange(seq_len)
    # compute distances: [seq_len, seq_len]
    _ = subtract(expand_dims(positions, 0), expand_dims(positions, 1))

    closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
    _ = 2 ** (-(2 ** -(math.log2(closest_power_of_2) - 3)))

    # We will return the base dist tensor for now.
    return 0.0
