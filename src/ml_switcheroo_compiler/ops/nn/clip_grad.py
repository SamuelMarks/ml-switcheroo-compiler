# ruff: noqa: E402, D100, D103, D104, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, D101, D102, D107, E701, E722, F403, E711, E712, PLR0913, PLR0915
from typing import Any

"""Gradient norm clipping primitive."""

from collections.abc import Iterable
from typing import Union

from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.ops.binary import add, divide, minimum, multiply, power
from ml_switcheroo_compiler.ops.reductions import max as reduce_max
from ml_switcheroo_compiler.ops.reductions import sum as reduce_sum
from ml_switcheroo_compiler.ops.shape.joining import stack
from ml_switcheroo_compiler.ops.unary import abs as abs_op
from ml_switcheroo_compiler.ops.unary import sqrt, square


def _compute_global_norm(parameters: list[Tensor], norm_type: float) -> Tensor:
    """Compute the global norm of an iterable of parameters.

    Args:
        parameters (object): The parameters parameter.
        norm_type (float): The norm_type parameter.

    Returns:
        Tensor: Result.
    """
    norms = []
    for p in parameters:
        if norm_type == float("inf"):
            norms.append(reduce_max(abs_op(p)))
        elif norm_type == 2.0:
            norms.append(reduce_sum(square(p)))
        else:
            norms.append(reduce_sum(power(abs_op(p), norm_type)))

    if norm_type == float("inf"):
        return reduce_max(stack(norms))

    total_norm = reduce_sum(stack(norms))
    if norm_type == 2.0:
        return sqrt(total_norm)

    return power(total_norm, 1.0 / norm_type)


def _scale_gradients(parameters: list[Tensor], max_norm: float, total_norm: Tensor) -> list[Tensor]:
    """Scales gradients in-place based on maximum and total norm.

    Args:
        parameters (list): The parameters parameter.
        max_norm (float): The max_norm parameter.
        total_norm (Tensor): The total_norm parameter.

    Returns:
        list: Result.
    """
    max_norm_t = max_norm
    clip_coef = divide(max_norm_t, add(total_norm, 1e-6))
    clip_coef_clamped = minimum(1.0, clip_coef)

    clipped_params = []
    for p in parameters:
        clipped_params.append(multiply(p, clip_coef_clamped))
    return clipped_params


def clip_grad_norm(
    parameters: Union[Tensor, Iterable[Tensor]],
    max_norm: float,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
) -> tuple[Union[Tensor, list[Tensor]], Tensor]:
    """Clip gradient norm of an iterable of parameters.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place if possible,
    but since this is a functional backend, we return the modified gradients.

    Args:
        parameters: An iterable of Tensors or a single Tensor that will have
            gradients normalized.
        max_norm: Max norm of the gradients.
        norm_type: Type of the used p-norm. Can be 'inf' for infinity norm.
        error_if_nonfinite: If True, an error is thrown if the total norm is
            non-finite. (Currently not strongly enforced in graph mode).

    Returns:
        A tuple of (clipped_parameters, total_norm).
    """
    is_single_tensor = isinstance(parameters, Tensor)
    if is_single_tensor:
        parameters = [parameters]  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism
    else:
        parameters = list(parameters)  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism

    if len(parameters) == 0:
        return [], 0.0  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism

    total_norm = _compute_global_norm(parameters, norm_type)
    clipped_params = _scale_gradients(parameters, max_norm, total_norm)

    if is_single_tensor:
        return clipped_params[0], total_norm

    return clipped_params, total_norm
