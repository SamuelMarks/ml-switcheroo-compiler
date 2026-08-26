"""Module einsum_frontend.py."""

from __future__ import annotations

# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915

"""Core abstractions and logic definitions for einsum_frontend.py."""


from collections.abc import Sequence

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.constants import MAGIC_VAL_2
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

from .utils import _emit_linalg_node


def _validate_tensordot_axes(
    axes: tuple[Sequence[int], Sequence[int]],
) -> tuple[Sequence[int], Sequence[int]]:
    """Validate and extracts tensordot axes.

    Args:
        axes (tuple): The axes parameter.

    Returns:
        tuple: Result.
    """
    return axes[0], axes[1]


def _get_tensordot_letters(len_a: int, len_b: int) -> tuple[list[str], list[str]]:
    """Map tensor dimensions to alphabetic characters.

    Args:
        len_a (int): The len_a parameter.
        len_b (int): The len_b parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    a_letters = [alphabet[i] for i in range(len_a)]
    b_letters = [alphabet[i + len_a] for i in range(len_b)]
    return a_letters, b_letters


def _get_tensordot_output_string(a_letters: list[str], b_letters: list[str], contracted: set[str]) -> str:
    """Generate the output string for tensordot einsum routing.

    Args:
        a_letters (object): The a_letters parameter.
        b_letters (object): The b_letters parameter.
        contracted (object): The contracted parameter.

    Returns:
        str: Result.
    """
    out_a = "".join([let for let in a_letters if let not in contracted])
    out_b = "".join([let for let in b_letters if let not in contracted])
    return out_a + out_b


def _generate_tensordot_einsum_strings(shape_a: Sequence[int], shape_b: Sequence[int], axes_a: Sequence[int], axes_b: Sequence[int]) -> tuple[str, str, str]:
    """Generate einsum notation strings for tensordot routing.

    Args:
        shape_a (object): The shape_a parameter.
        shape_b (object): The shape_b parameter.
        axes_a (object): The axes_a parameter.
        axes_b (object): The axes_b parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    if not shape_a and not shape_b:
        return "", "", ""
    a_letters, b_letters = _get_tensordot_letters(len(shape_a), len(shape_b))
    for idx_a, idx_b in zip(axes_a, axes_b):
        b_letters[idx_b] = a_letters[idx_a]
    a_str = "".join(a_letters)
    b_str = "".join(b_letters)
    contracted = {a_letters[i] for i in axes_a}
    out_str = _get_tensordot_output_string(a_letters, b_letters, contracted)
    return a_str, b_str, out_str


def _tensordot_einsum_routing(a: Tensor, b: Tensor, axes: tuple[Sequence[int], Sequence[int]]):
    """Evaluate _tensordot_einsum_routing operation.

    Args:
        a (Tensor): The a parameter.
        b (Tensor): The b parameter.
        axes (tuple): The axes parameter.

    Returns:
        Tensor: Result.
    """
    axes_a, axes_b = _validate_tensordot_axes(axes)
    a_str, b_str, out_str = _generate_tensordot_einsum_strings(a.shape, b.shape, axes_a, axes_b)
    eq = f"{a_str},{b_str}->{out_str}"
    return einsum(eq, a, b)


def tensordot(a: Tensor, b: Tensor, axes: (int | tuple[Sequence[int], Sequence[int]]) = 2):
    """Compute the tensor dot product along specified axes.

    Args:
        a (Tensor): The a parameter.
        b (Tensor): The b parameter.
        axes (object): The axes parameter.

    Returns:
        Tensor: Result.
    """
    if isinstance(axes, tuple) and len(a.shape) > MAGIC_VAL_2 and len(b.shape) > MAGIC_VAL_2:
        return _tensordot_einsum_routing(a, b, axes)
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op("Tensordot", a.data, b.data, axes=axes)
        return Tensor(data, TensorConfig(data.shape, a.dtype, a.device))
    return _emit_linalg_node("Tensordot", [a, b], {"axes": axes}, [()], [a.dtype])


def einsum(equation: str, *operands: Tensor):
    """Evaluate the Einstein summation convention on the operands.

    Args:
        equation (str): The equation parameter.
        *operands (Tensor): Positional args.

    Returns:
        Tensor: Result.
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()

        # Extract raw data from operands if they are Tensors
        raw_operands = [getattr(op, "data", op) for op in operands]
        data = backend.execute_op("Einsum", equation, *raw_operands)

        # Infer dtype and device safely
        first_op = operands[0]
        dtype = getattr(first_op, "dtype", getattr(raw_operands[0], "dtype", "float32"))
        device = getattr(first_op, "device", "cpu")

        return Tensor(data, TensorConfig(data.shape, dtype, device))
    return _emit_linalg_node("Einsum", operands, {"equation": equation}, [()], [operands[0].dtype])


def _get_remaining_dims(shape_len: int, contracting: Sequence[int], batch: Sequence[int]) -> list[int]:
    """Evaluate _get_remaining_dims operation.

    Args:
        shape_len (int): The shape_len parameter.
        contracting (object): The contracting parameter.
        batch (object): The batch parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    contract_set = set(contracting)
    batch_set = set(batch)
    return [i for i in range(shape_len) if i not in contract_set and i not in batch_set]


def _infer_dot_general_shape(
    lhs_shape: Sequence[int],
    rhs_shape: Sequence[int],
    dimension_numbers: tuple[tuple[Sequence[int], Sequence[int]], tuple[Sequence[int], Sequence[int]]],
) -> tuple[int, ...]:
    """Evaluate _infer_dot_general_shape operation.

    Args:
        lhs_shape (object): The lhs_shape parameter.
        rhs_shape (object): The rhs_shape parameter.
        dimension_numbers (object): The dimension_numbers parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    contracting, batch = dimension_numbers
    lhs_contracting, rhs_contracting = contracting
    lhs_batch, rhs_batch = batch
    out_shape = []
    for b in lhs_batch:
        out_shape.append(lhs_shape[b])
    lhs_remaining = _get_remaining_dims(len(lhs_shape), lhs_contracting, lhs_batch)
    for r in lhs_remaining:
        out_shape.append(lhs_shape[r])
    rhs_remaining = _get_remaining_dims(len(rhs_shape), rhs_contracting, rhs_batch)
    for r in rhs_remaining:
        out_shape.append(rhs_shape[r])
    return tuple(out_shape)
