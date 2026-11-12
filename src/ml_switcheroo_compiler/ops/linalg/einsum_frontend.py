"""Implementation of einsum frontend operations and utilities for the ops subsystem (`ml_switcheroo_compiler.ops.linalg.einsum_frontend`)."""

from __future__ import annotations

from collections.abc import Sequence

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.constants import MATRIX_RANK
from ml_switcheroo_compiler.core.shape import (
    TensorOrScalar,
    extract_data,
    extract_device,
    extract_dtype,
    extract_shape,
)
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

from .utils import _emit_linalg_node


def _validate_tensordot_axes(
    axes: tuple[Sequence[int], Sequence[int]],
) -> tuple[Sequence[int], Sequence[int]]:
    """Validate and extracts tensordot axes.

    Args:
        axes (tuple): Input `axes` argument (axes).

    Returns:
        tuple: Computed `tuple` output.
    """
    return axes[0], axes[1]


def _get_tensordot_letters(len_a: int, len_b: int) -> tuple[list[str], list[str]]:
    """Map tensor dimensions to alphabetic characters.

    Args:
        len_a (int): Input `len_a` argument (len a).
        len_b (int): Input `len_b` argument (len b).

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    a_letters = [alphabet[i] for i in range(len_a)]
    b_letters = [alphabet[i + len_a] for i in range(len_b)]
    return a_letters, b_letters


def _get_tensordot_output_string(a_letters: list[str], b_letters: list[str], contracted: set[str]) -> str:
    """Generate the output string for tensordot einsum routing.

    Args:
        a_letters (list[str]): Input `a_letters` argument (a letters).
        b_letters (list[str]): Input `b_letters` argument (b letters).
        contracted (set[str]): Input `contracted` argument (contracted).

    Returns:
        str: Computed `str` output.
    """
    out_a = "".join([let for let in a_letters if let not in contracted])
    out_b = "".join([let for let in b_letters if let not in contracted])
    return out_a + out_b


def _generate_tensordot_einsum_strings(shape_a: Sequence[int], shape_b: Sequence[int], axes_a: Sequence[int], axes_b: Sequence[int]) -> tuple[str, str, str]:
    """Generate einsum notation strings for tensordot routing.

    Args:
        shape_a (Sequence[int]): Input `shape_a` argument (shape a).
        shape_b (Sequence[int]): Input `shape_b` argument (shape b).
        axes_a (Sequence[int]): Input `axes_a` argument (axes a).
        axes_b (Sequence[int]): Input `axes_b` argument (axes b).

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
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


def _tensordot_einsum_routing(a: Tensor, b: Tensor, axes: tuple[Sequence[int], Sequence[int]]) -> TensorOrScalar:
    """Compute the tensordot einsum routing tensor operation.

    Args:
        a (Tensor): Input `a` argument (a).
        b (Tensor): Input `b` argument (b).
        axes (tuple): Input `axes` argument (axes).

    Returns:
        Tensor: Computed `Tensor` output.
    """
    axes_a, axes_b = _validate_tensordot_axes(axes)
    a_str, b_str, out_str = _generate_tensordot_einsum_strings(a.shape, b.shape, axes_a, axes_b)
    eq = f"{a_str},{b_str}->{out_str}"
    return einsum(eq, a, b)


def tensordot(a: Tensor, b: Tensor, axes: (int | tuple[Sequence[int], Sequence[int]]) = 2) -> TensorOrScalar:
    """Compute the tensor dot product along specified axes.

    Args:
        a (Tensor): Input `a` argument (a).
        b (Tensor): Input `b` argument (b).
        axes (int | tuple[Sequence[int], Sequence[int]]): Input `axes` argument (axes).

    Returns:
        Tensor: Computed `Tensor` output.
    """
    if isinstance(axes, tuple) and len(a.shape) > MATRIX_RANK and len(b.shape) > MATRIX_RANK:
        return _tensordot_einsum_routing(a, b, axes)
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op("Tensordot", a.data, b.data, axes=axes)
        return Tensor(data, TensorConfig(extract_shape(data), a.dtype, a.device))
    return _emit_linalg_node("Tensordot", [a, b], {"axes": axes}, [()], [a.dtype])


def einsum(equation: str, *operands: Tensor) -> TensorOrScalar:
    """Evaluate the Einstein summation convention on the operands.

    Args:
        equation (str): Input `equation` argument (equation).
        *operands (Tensor): Positional args.

    Returns:
        Tensor: Computed `Tensor` output.
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()

        # Extract raw data from operands if they are Tensors
        raw_operands = [extract_data(op) for op in operands]
        data = backend.execute_op("Einsum", equation, *raw_operands)

        # Infer dtype and device safely
        first_op = operands[0]
        dtype = extract_dtype(first_op, extract_dtype(raw_operands[0], "float32"))
        device = extract_device(first_op, "cpu")

        return Tensor(data, TensorConfig(extract_shape(data), dtype, device))
    return _emit_linalg_node("Einsum", operands, {"equation": equation}, [()], [operands[0].dtype])


def _get_remaining_dims(shape_len: int, contracting: Sequence[int], batch: Sequence[int]) -> list[int]:
    """Compute the get remaining dims tensor operation.

    Args:
        shape_len (int): Input `shape_len` argument (shape len).
        contracting (Sequence[int]): Input `contracting` argument (contracting).
        batch (Sequence[int]): Input `batch` argument (batch).

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    contract_set = set(contracting)
    batch_set = set(batch)
    return [i for i in range(shape_len) if i not in contract_set and i not in batch_set]


def _infer_dot_general_shape(
    lhs_shape: Sequence[int],
    rhs_shape: Sequence[int],
    dimension_numbers: tuple[tuple[Sequence[int], Sequence[int]], tuple[Sequence[int], Sequence[int]]],
) -> tuple[int, ...]:
    """Compute the infer dot general shape tensor operation.

    Args:
        lhs_shape (Sequence[int]): Input `lhs_shape` argument (lhs shape).
        rhs_shape (Sequence[int]): Input `rhs_shape` argument (rhs shape).
        dimension_numbers (tuple[tuple[Sequence[int], Sequence[int]], tuple[Sequence[int], Sequence[int]]]): Input `dimension_numbers` argument (dimension numbers).

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
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
