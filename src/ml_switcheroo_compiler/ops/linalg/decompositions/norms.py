"""Module docstring."""

from __future__ import annotations

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.linalg.utils import _emit_linalg_node


def matrix_power(input: Tensor, n: int) -> Tensor:
    """Raises a square matrix to the integer power `n`.

    Args:
        input (Tensor): The square matrix
        n (int): The exponent

    Returns:
    Tensor: The matrix raised to the power `n`
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op("MatrixPower", input.data, n)
        return Tensor(data, TensorConfig(data.shape, input.dtype, input.device))
    return _emit_linalg_node("MatrixPower", [input], {"n": n}, [input.shape], [input.dtype])


def _norm_out_shape(x_shape: tuple[int, ...], axis: int | tuple[int, ...] | None, keepdims: bool) -> tuple[int, ...]:
    """Function docstring."""
    if axis is None:
        if not keepdims:
            return ()
        return tuple(1 for _ in x_shape)

    axes = (axis,) if isinstance(axis, int) else axis

    out_shape = []
    for i, s in enumerate(x_shape):
        if i in axes:
            if keepdims:
                out_shape.append(1)
        else:
            out_shape.append(s)
    return tuple(out_shape)


def norm(
    x: Tensor,
    ord: int | str | None = None,
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
) -> Tensor:
    """Matrix or vector norm.

    Args:
        x (Tensor): Input tensor.
        ord (int | str | None): Order of the norm.
        axis (int | tuple[int, ...] | None): If axis is an integer, it specifies the axis of x along which to compute the vector norms.
        keepdims (bool): If True, the axes which are reduced are left in the result as dimensions with size one.

    Returns:
    Tensor: Norm of the matrix or vector(s).
    """
    out_shape = _norm_out_shape(x.shape, axis, keepdims)

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op("Norm", x.data, ord=ord, axis=axis, keepdims=keepdims)

        return Tensor(data, TensorConfig(out_shape, x.dtype, x.device))

    return _emit_linalg_node("Norm", [x], {"ord": ord, "axis": axis, "keepdims": keepdims}, [out_shape], [x.dtype])


def matrix_exponential(a: Tensor) -> Tensor:
    """Compute the matrix exponential of a square matrix.

    Args:
        a (Tensor): The input square matrix.

    Returns:
    Tensor: The matrix exponential.
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op("MatrixExponential", a.data)
        return Tensor(data, TensorConfig(data.shape, a.dtype, a.device))
    return _emit_linalg_node("MatrixExponential", [a], {}, [a.shape], [a.dtype])


def _power_iteration_eager(
    input: Tensor,
    num_iters: int,
    u: Tensor | None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Execute power iteration eagerly."""
    from ml_switcheroo_compiler.backends.registry import get_active_backend

    backend = get_active_backend()

    v_data, u_data, sigma_data = backend.execute_op(
        "PowerIteration",
        input.data,
        num_iters=num_iters,
        u=u.data if u is not None else None,
    )

    return (
        Tensor(v_data, TensorConfig(v_data.shape, input.dtype, input.device)),
        Tensor(u_data, TensorConfig(u_data.shape, input.dtype, input.device)),
        Tensor(sigma_data, TensorConfig(sigma_data.shape, input.dtype, input.device)),
    )


def power_iteration(
    input: Tensor,
    num_iters: int = 1,
    u: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Computes the dominant singular value and vectors using power iteration.

    Args:
        input (Tensor): The input matrix of shape (..., M, N)
        num_iters (int): The number of iterations to perform. Defaults to 1
        u (Tensor | None): Optional initial estimate for the left singular vector
            of shape (..., M, 1). If None, a uniform vector of ones is used.

    Returns:
    tuple[Tensor, Tensor, Tensor]: A tuple containing:
        - v (Tensor): Right singular vector estimate
        - u (Tensor): Left singular vector estimate
        - sigma (Tensor): Spectral norm estimate
    """
    if config.eager_mode:
        return _power_iteration_eager(input, num_iters, u)

    inputs = [input]
    if u is not None:
        inputs.append(u)

    in_shape = input.shape
    v_shape = in_shape[:-2] + (in_shape[-1],)
    u_shape = in_shape[:-2] + (in_shape[-2],)
    sigma_shape = in_shape[:-2]

    return _emit_linalg_node(
        "PowerIteration",
        inputs,
        {"num_iters": num_iters},
        [v_shape, u_shape, sigma_shape],
        [input.dtype, input.dtype, input.dtype],
    )
