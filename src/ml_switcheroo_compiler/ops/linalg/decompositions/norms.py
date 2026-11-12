"""Implementation of norms operations and utilities for the ops subsystem (`ml_switcheroo_compiler.ops.linalg.decompositions.norms`)."""

from __future__ import annotations

import typing

from ml_switcheroo_compiler.backends.registry import ensure_eager_tensor, ensure_tensor_triple
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.shape import TensorOrScalar, extract_device
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.linalg.utils import _emit_linalg_node, _extract_linalg_dtype
from ml_switcheroo_compiler.ops.shape.utils import compute_reduction_shape

if typing.TYPE_CHECKING:
    pass


def matrix_power(input: Tensor, n: int) -> TensorOrScalar:
    """Raise a square matrix to the integer power `n`.

    Args:
        input (Tensor): Input `input` argument (input).
        n (int): Input `n` argument (n).

    Returns:
        Tensor: Computed `Tensor` output.
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = ensure_eager_tensor(backend.execute_op("MatrixPower", (input.data if type(input).__name__ == "Tensor" else input), n))
        return Tensor(data, TensorConfig(data.shape, _extract_linalg_dtype(input, None), extract_device(input, None)))
    return _emit_linalg_node("MatrixPower", [input], {"n": n}, [input.shape], [_extract_linalg_dtype(input, None)])


def _norm_out_shape(x_shape: tuple[int, ...], axis: int | tuple[int, ...] | None, keepdims: bool) -> tuple[int, ...]:
    """Compute the norm out shape tensor operation.

    Args:
        x_shape (tuple[int, ...]): Input `x_shape` argument (x shape).
        axis (int | tuple[int, ...] | None): Input `axis` argument (axis).
        keepdims (bool): Input `keepdims` argument (keepdims).

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
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
) -> TensorOrScalar:
    """Matrix or vector norm.

    Args:
        x (Tensor): Input `x` argument (x).
        ord (int | str | None): Input `ord` argument (ord).
        axis (int | tuple[int, ...] | None): Input `axis` argument (axis).
        keepdims (bool): Input `keepdims` argument (keepdims).

    Returns:
        Tensor: Computed `Tensor` output.
    """
    out_shape = _norm_out_shape(x.shape, axis, keepdims)

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op("Norm", (x.data if type(x).__name__ == "Tensor" else x), ord=ord, axis=axis, keepdims=keepdims)

        return Tensor(data, TensorConfig(out_shape, _extract_linalg_dtype(x, None), extract_device(x, None)))

    return _emit_linalg_node("Norm", [x], {"ord": ord, "axis": axis, "keepdims": keepdims}, [out_shape], [_extract_linalg_dtype(x, None)])


def matrix_exponential(a: Tensor) -> TensorOrScalar:
    """Compute the matrix exponential tensor operation.

    Args:
        a (Tensor): Input `a` argument (a).

    Returns:
        Tensor: Computed `Tensor` output.
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = ensure_eager_tensor(backend.execute_op("MatrixExponential", (a.data if type(a).__name__ == "Tensor" else a)))
        return Tensor(data, TensorConfig(data.shape, _extract_linalg_dtype(a, None), extract_device(a, None)))
    return _emit_linalg_node("MatrixExponential", [a], {}, [a.shape], [_extract_linalg_dtype(a, None)])


def matrix_exp(a: Tensor) -> TensorOrScalar:
    """Compute the matrix exp tensor operation.

    Args:
        a (Tensor): Input `a` argument (a).

    Returns:
        Tensor: Computed `Tensor` output.
    """
    return matrix_exponential(a)


def _power_iteration_eager(
    input: Tensor,
    num_iters: int,
    u: Tensor | None,
) -> TensorOrScalar:
    """Execute power iteration eagerly.

    Args:
        input (Tensor): Input `input` argument (input).
        num_iters (int): Input `num_iters` argument (num iters).
        u (Tensor | None): Input `u` argument (u).

    Returns:
        tuple: Computed `tuple` output.
    """
    from ml_switcheroo_compiler.backends.registry import get_active_backend

    backend = get_active_backend()

    v_data, u_data, sigma_data = ensure_tensor_triple(
        backend.execute_op(
            "PowerIteration",
            (input.data if type(input).__name__ == "Tensor" else input),
            num_iters=num_iters,
            u=(u.data if type(u).__name__ == "Tensor" else u) if u is not None else None,
        )
    )

    dt = _extract_linalg_dtype(input, None)
    dev = extract_device(input, None)
    return (
        Tensor(v_data, TensorConfig(v_data.shape, dt, dev)),
        Tensor(u_data, TensorConfig(u_data.shape, dt, dev)),
        Tensor(sigma_data, TensorConfig(sigma_data.shape, dt, dev)),
    )


def power_iteration(
    input: Tensor,
    num_iters: int = 1,
    u: Tensor | None = None,
) -> TensorOrScalar:
    """Compute the dominant singular value and vectors using power iteration.

    Args:
        input (Tensor): Input `input` argument (input).
        num_iters (int): Input `num_iters` argument (num iters).
        u (Tensor | None): Input `u` argument (u).

    Returns:
        tuple: Computed `tuple` output.
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
    dt = _extract_linalg_dtype(input, None)

    return _emit_linalg_node(
        "PowerIteration",
        inputs,
        {"num_iters": num_iters},
        [v_shape, u_shape, sigma_shape],
        [dt, dt, dt],
    )
