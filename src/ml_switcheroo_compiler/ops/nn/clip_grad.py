"""Gradient norm clipping primitive."""

from typing import Union
from collections.abc import Iterable

from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.ops.binary import add, multiply, divide
from ml_switcheroo_compiler.ops.unary import sqrt, square
from ml_switcheroo_compiler.ops.reductions import sum as reduce_sum


def clip_grad_norm(
    parameters: Union[Tensor, Iterable[Tensor]],
    max_norm: float,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
) -> tuple[Union[Tensor, list[Tensor]], Tensor]:
    """Clips gradient norm of an iterable of parameters.

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
        parameters = [parameters]
    else:
        parameters = list(parameters)

    if len(parameters) == 0:
        from ml_switcheroo_compiler.ops.aliases.memory_ops import convert_to_tensor

        # Return empty list and 0.0 norm
        return [], convert_to_tensor(0.0)

    # Calculate the total norm
    norms = []
    for p in parameters:
        if norm_type == float("inf"):
            from ml_switcheroo_compiler.ops.reductions import max as reduce_max
            from ml_switcheroo_compiler.ops.unary import abs as abs_op

            norms.append(reduce_max(abs_op(p)))
        elif norm_type == 2.0:
            norms.append(reduce_sum(square(p)))
        else:
            from ml_switcheroo_compiler.ops.binary import power
            from ml_switcheroo_compiler.ops.unary import abs as abs_op

            norms.append(reduce_sum(power(abs_op(p), norm_type)))

    if norm_type == float("inf"):
        from ml_switcheroo_compiler.ops.shape.frontend import stack
        from ml_switcheroo_compiler.ops.reductions import max as reduce_max

        total_norm = reduce_max(stack(norms))
    else:
        from ml_switcheroo_compiler.ops.shape.frontend import stack

        total_norm = reduce_sum(stack(norms))
        if norm_type == 2.0:
            total_norm = sqrt(total_norm)
        else:
            from ml_switcheroo_compiler.ops.binary import power

            total_norm = power(total_norm, 1.0 / norm_type)

    # Scaling logic
    # clip_coef = max_norm / (total_norm + 1e-6)
    # clip_coef_clamped = min(1.0, clip_coef)
    from ml_switcheroo_compiler.ops.binary import minimum
    from ml_switcheroo_compiler.ops.aliases.memory_ops import convert_to_tensor

    max_norm_t = convert_to_tensor(max_norm)
    clip_coef = divide(max_norm_t, add(total_norm, convert_to_tensor(1e-6)))
    clip_coef_clamped = minimum(convert_to_tensor(1.0), clip_coef)

    clipped_params = []
    for p in parameters:
        clipped_params.append(multiply(p, clip_coef_clamped))

    # If the user passed a single Tensor originally, return a single Tensor
    if is_single_tensor:
        return clipped_params[0], total_norm

    return clipped_params, total_norm
