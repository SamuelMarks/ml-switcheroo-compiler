"""Module docstring."""

from __future__ import annotations

from collections.abc import Sequence

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.constants import MAGIC_VAL_2
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

from .utils import _emit_linalg_node


def _validate_tensordot_axes(
    axes: tuple[Sequence[int], Sequence[int]],
) -> tuple[Sequence[int], Sequence[int]]:
    """Validates and extracts tensordot axes."""
    return axes[0], axes[1]


def _get_tensordot_letters(len_a: int, len_b: int) -> tuple[list[str], list[str]]:
    """Maps tensor dimensions to alphabetic characters."""
    alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    a_letters = [alphabet[i] for i in range(len_a)]
    b_letters = [alphabet[i + len_a] for i in range(len_b)]
    return a_letters, b_letters


def _get_tensordot_output_string(a_letters: list[str], b_letters: list[str], contracted: set[str]) -> str:
    """Generates the output string for tensordot einsum routing."""
    out_a = "".join([let for let in a_letters if let not in contracted])
    out_b = "".join([let for let in b_letters if let not in contracted])
    return out_a + out_b


def _generate_tensordot_einsum_strings(shape_a: Sequence[int], shape_b: Sequence[int], axes_a: Sequence[int], axes_b: Sequence[int]) -> tuple[str, str, str]:
    """Generates einsum notation strings for tensordot routing."""
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


def _tensordot_einsum_routing(a: Tensor, b: Tensor, axes: tuple[Sequence[int], Sequence[int]]) -> Tensor:
    """Function docstring."""
    axes_a, axes_b = _validate_tensordot_axes(axes)
    a_str, b_str, out_str = _generate_tensordot_einsum_strings(a.shape, b.shape, axes_a, axes_b)
    eq = f"{a_str},{b_str}->{out_str}"
    return einsum(eq, a, b)


def tensordot(a: Tensor, b: Tensor, axes: (int | tuple[Sequence[int], Sequence[int]]) = 2) -> Tensor:
    """Computes the tensor dot product along specified axes.

    Args:
        a (Tensor): The first tensor
        b (Tensor): The second tensor
        axes (int | tuple[Sequence[int], Sequence[int]]): The axes to contract over
        Defaults to 2

    Returns:
    Tensor: The tensor dot product of the inputs
    """
    if isinstance(axes, tuple) and len(a.shape) > MAGIC_VAL_2 and len(b.shape) > MAGIC_VAL_2:
        return _tensordot_einsum_routing(a, b, axes)
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op("Tensordot", a.data, b.data, axes=axes)
        return Tensor(data, TensorConfig(data.shape, a.dtype, a.device))
    return _emit_linalg_node("Tensordot", [a, b], {"axes": axes}, [()], [a.dtype])


def einsum(equation: str, *operands: Tensor) -> Tensor:
    """Evaluates the Einstein summation convention on the operands.

    Args:
        equation (str): The Einstein summation convention string
        *operands (Tensor): The input tensors to contract

    Returns:
    Tensor: The result of the Einstein summation
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op("Einsum", equation, *[op.data for op in operands])
        return Tensor(data, TensorConfig(data.shape, operands[0].dtype, operands[0].device))
    return _emit_linalg_node("Einsum", operands, {"equation": equation}, [()], [operands[0].dtype])


def _get_remaining_dims(shape_len: int, contracting: Sequence[int], batch: Sequence[int]) -> list[int]:
    """Function docstring.

    Args:
        shape_len: Arg.
        contracting: Arg.
        batch: Arg.
    """
    contract_set = set(contracting)
    batch_set = set(batch)
    return [i for i in range(shape_len) if i not in contract_set and i not in batch_set]


def _infer_dot_general_shape(
    lhs_shape: Sequence[int],
    rhs_shape: Sequence[int],
    dimension_numbers: tuple[tuple[Sequence[int], Sequence[int]], tuple[Sequence[int], Sequence[int]]],
) -> tuple[int, ...]:
    """Execute _infer_dot_general_shape.

    Args:
        lhs_shape (Any): Argument lhs_shape.
        rhs_shape (Any): Argument rhs_shape.
        dimension_numbers (Any): Argument dimension_numbers.

    Returns:
    Any: The result.
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
