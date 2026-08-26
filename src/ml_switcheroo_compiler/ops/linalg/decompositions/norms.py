"""Module norms.py."""

from __future__ import annotations

# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915

"""Core abstractions and logic definitions for norms.py."""


from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.linalg.utils import _emit_linalg_node
from ml_switcheroo_compiler.ops.shape.utils import compute_reduction_shape


def matrix_power(input: Tensor, n: int):
    """Raise a square matrix to the integer power `n`.

    Args:
        input (Tensor): The input parameter.
        n (int): The n parameter.

    Returns:
        Tensor: Result.
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op("MatrixPower", (input.data if type(input).__name__ == "Tensor" else input), n)
        return Tensor(data, TensorConfig(data.shape, getattr(input, "dtype", None), getattr(input, "device", None)))
    return _emit_linalg_node("MatrixPower", [input], {"n": n}, [input.shape], [getattr(input, "dtype", None)])


def _norm_out_shape(x_shape: tuple[int, ...], axis: int | tuple[int, ...] | None, keepdims: bool) -> tuple[int, ...]:
    """Evaluate _norm_out_shape operation.

    Args:
        x_shape (object): The x_shape parameter.
        axis (object): The axis parameter.
        keepdims (bool): The keepdims parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    if axis is None:
        return tuple(1 for _ in x_shape) if keepdims else ()
    axes = (axis,) if isinstance(axis, int) else axis
    return compute_reduction_shape(x_shape, axes, keepdims)


def norm(
    x: Tensor,
    ord: int | str | None = None,
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
):
    """Matrix or vector norm.

    Args:
        x (Tensor): The x parameter.
        ord (object): The ord parameter.
        axis (object): The axis parameter.
        keepdims (bool): The keepdims parameter.

    Returns:
        Tensor: Result.
    """
    out_shape = _norm_out_shape(x.shape, axis, keepdims)

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op("Norm", (x.data if type(x).__name__ == "Tensor" else x), ord=ord, axis=axis, keepdims=keepdims)

        return Tensor(data, TensorConfig(out_shape, getattr(x, "dtype", None), getattr(x, "device", None)))

    return _emit_linalg_node("Norm", [x], {"ord": ord, "axis": axis, "keepdims": keepdims}, [out_shape], [getattr(x, "dtype", None)])


def matrix_exponential(a: Tensor):
    """Evaluate matrix_exponential operation.

    Args:
        a (Tensor): The a parameter.

    Returns:
        Tensor: Result.
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op("MatrixExponential", (a.data if type(a).__name__ == "Tensor" else a))
        return Tensor(data, TensorConfig(data.shape, getattr(a, "dtype", None), getattr(a, "device", None)))
    return _emit_linalg_node("MatrixExponential", [a], {}, [a.shape], [getattr(a, "dtype", None)])


def matrix_exp(a: Tensor):
    """Evaluate matrix_exp operation.

    Args:
        a (Tensor): The a parameter.

    Returns:
        Tensor: Result.
    """
    return matrix_exponential(a)


def _power_iteration_eager(
    input: Tensor,
    num_iters: int,
    u: Tensor | None,
):
    """Execute power iteration eagerly.

    Args:
        input (Tensor): The input parameter.
        num_iters (int): The num_iters parameter.
        u (object): The u parameter.

    Returns:
        tuple: Result.
    """
    from ml_switcheroo_compiler.backends.registry import get_active_backend

    backend = get_active_backend()

    v_data, u_data, sigma_data = backend.execute_op(
        "PowerIteration",
        (input.data if type(input).__name__ == "Tensor" else input),
        num_iters=num_iters,
        u=(u.data if type(u).__name__ == "Tensor" else u) if u is not None else None,
    )

    return (
        Tensor(v_data, TensorConfig(v_data.shape, getattr(input, "dtype", None), getattr(input, "device", None))),
        Tensor(u_data, TensorConfig(u_data.shape, getattr(input, "dtype", None), getattr(input, "device", None))),
        Tensor(sigma_data, TensorConfig(sigma_data.shape, getattr(input, "dtype", None), getattr(input, "device", None))),
    )


def power_iteration(
    input: Tensor,
    num_iters: int = 1,
    u: Tensor | None = None,
):
    """Compute the dominant singular value and vectors using power iteration.

    Args:
        input (Tensor): The input parameter.
        num_iters (int): The num_iters parameter.
        u (object): The u parameter.

    Returns:
        tuple: Result.
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
        [getattr(input, "dtype", None), getattr(input, "dtype", None), getattr(input, "dtype", None)],
    )
