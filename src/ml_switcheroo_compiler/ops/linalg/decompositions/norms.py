"""Operations and utilities for `ml_switcheroo_compiler.ops.linalg.decompositions.norms`."""

from __future__ import annotations

import typing

from ml_switcheroo_compiler.backends.registry import (
    coerce_int,
    unpack_triple,
)
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.shape import OpValue, broadcast_shapes, extract_device, extract_shape
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.base import OpDef, register_op
from ml_switcheroo_compiler.ops.linalg.utils import _emit_linalg_node, _extract_linalg_dtype
from ml_switcheroo_compiler.ops.shape.utils import compute_reduction_shape

if typing.TYPE_CHECKING:
    pass


def matrix_power(input: Tensor, n: int) -> OpValue:
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
        data = backend.execute_op("MatrixPower", (input.data if type(input).__name__ == "Tensor" else input), n)
        return Tensor(
            data, TensorConfig(extract_shape(data), _extract_linalg_dtype(input, None), extract_device(input, None))
        )
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
) -> OpValue:
    """Matrix or vector norm.

    Args:
        x (Tensor): Input `x` argument (x).
        ord (int | str | None): Input `ord` argument (ord).
        axis (int | tuple[int, ...] | None): Input `axis` argument (axis).
        keepdims (bool): Input `keepdims` argument (keepdims).

    Returns:
        Tensor: Computed `Tensor` output.
    """
    out_shape = _norm_out_shape(extract_shape(x), axis, keepdims)

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op(
            "Norm", (x.data if type(x).__name__ == "Tensor" else x), ord=ord, axis=axis, keepdims=keepdims
        )

        return Tensor(data, TensorConfig(out_shape, _extract_linalg_dtype(x, None), extract_device(x, None)))

    return _emit_linalg_node(
        "Norm", [x], {"ord": ord, "axis": axis, "keepdims": keepdims}, [out_shape], [_extract_linalg_dtype(x, None)]
    )


def matrix_exponential(a: Tensor) -> OpValue:
    """Compute the matrix exponential tensor operation.

    Args:
        a (Tensor): Input `a` argument (a).

    Returns:
        Tensor: Computed `Tensor` output.
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op("MatrixExponential", (a.data if type(a).__name__ == "Tensor" else a))
        return Tensor(data, TensorConfig(extract_shape(data), _extract_linalg_dtype(a, None), extract_device(a, None)))
    return _emit_linalg_node("MatrixExponential", [a], {}, [a.shape], [_extract_linalg_dtype(a, None)])


def matrix_exp(a: Tensor) -> OpValue:
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
) -> OpValue:
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

    v_data, u_data, sigma_data = unpack_triple(
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
        Tensor(v_data, TensorConfig(extract_shape(v_data), dt, dev)),
        Tensor(u_data, TensorConfig(extract_shape(u_data), dt, dev)),
        Tensor(sigma_data, TensorConfig(extract_shape(sigma_data), dt, dev)),
    )


def power_iteration(
    input: Tensor,
    num_iters: int = 1,
    u: Tensor | None = None,
) -> OpValue:
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


@register_op("Norm")
class Norm(OpDef):
    """General tensor/matrix norm operator definition."""

    def infer_shape(self, *args: OpValue, **kwargs: OpValue) -> OpValue:
        """Infer scalar default output shape for `Norm`.

        Args:
            *args (OpValue): Positional input tensors.
            **kwargs (OpValue): Optional keyword attributes.

        Returns:
            OpValue: Empty scalar shape tuple `()`.
        """
        return ()


@register_op("MatrixExponential")
class MatrixExponential(OpDef):
    """Matrix exponential operator definition."""

    def infer_shape(self, *args: OpValue, **kwargs: OpValue) -> OpValue:
        """Infer and validate square matrix shape for `MatrixExponential`.

        Args:
            *args (OpValue): Input square matrix tensor.
            **kwargs (OpValue): Optional keyword attributes.

        Returns:
            OpValue: Validated square matrix shape tuple.

        Raises:
            ValueError: If the input has fewer than 2 dimensions or is not square.
        """
        if not args:
            return ()
        shape = extract_shape(args[0])
        if len(shape) < 2:
            raise ValueError("MatrixExponential requires at least 2 dimensions")
        if shape[-1] != shape[-2]:
            raise ValueError("MatrixExponential requires a square matrix")
        return shape


@register_op("Cross")
class Cross(OpDef):
    """3D vector cross-product operator definition."""

    def infer_shape(self, *args: OpValue, **kwargs: OpValue) -> OpValue:
        """Infer broadcasted output shape for 3D vector cross product.

        Args:
            *args (OpValue): Left and right vector operands.
            **kwargs (OpValue): Optional `axis` keyword attribute.

        Returns:
            OpValue: Broadcasted 3D vector shape.

        Raises:
            ValueError: If either operand lacks dimension size 3 along `axis`.
        """
        if not args:
            return ()
        shape_a = extract_shape(args[0])
        shape_b = extract_shape(args[1])
        axis = int(coerce_int(kwargs.get("axis", -1)))
        if shape_a[axis] != 3 or shape_b[axis] != 3:
            raise ValueError("Cross requires dimension size 3")
        return broadcast_shapes(shape_a, shape_b)


@register_op("PowerIteration")
class PowerIteration(OpDef):
    """Dominant singular triplet power-iteration operator definition."""

    def infer_shape(self, *args: OpValue, **kwargs: OpValue) -> OpValue:
        """Infer `((v_shape, u_shape, sigma_shape), dtypes)` for power iteration.

        Args:
            *args (OpValue): Input matrix tensor.
            **kwargs (OpValue): Optional keyword attributes.

        Returns:
            OpValue: Tuple of output shapes and output dtypes.
        """
        from ml_switcheroo_compiler.core.shape import extract_dtype

        in_shape = extract_shape(args[0])
        v_shape = in_shape[:-2] + (in_shape[-1],)
        u_shape = in_shape[:-2] + (in_shape[-2],)
        sigma_shape = in_shape[:-2]
        return (v_shape, u_shape, sigma_shape), (extract_dtype(args[0]),) * 3


def _infer_axis_norm_shape(
    args: tuple[OpValue, ...],
    kwargs: dict[str, OpValue],
    default_axes: tuple[int, ...],
) -> tuple[int, ...]:
    """Compute reduced output shape for `MatrixNorm` and `VectorNorm`.

    Args:
        args (tuple[OpValue, ...]): Positional input arguments.
        kwargs (dict[str, OpValue]): Keyword attributes (`axis`, `keepdims`).
        default_axes (tuple[int, ...]): Default reduction axes when `axis` is omitted.

    Returns:
        tuple[int, ...]: Reduced norm output shape.
    """
    if not args:
        return ()
    shape = extract_shape(args[0])
    axis = kwargs.get("axis", default_axes if len(default_axes) > 1 else default_axes[0])
    keepdims = bool(kwargs.get("keepdims", False))
    axes: tuple[int, ...] = (
        (axis,)
        if isinstance(axis, int)
        else tuple(int(d) for d in axis if isinstance(d, (int, float)))
        if isinstance(axis, (list, tuple))
        else default_axes
    )
    if keepdims:
        out_shape = list(shape)
        for d in axes:
            out_shape[d] = 1
        return tuple(out_shape)
    axis_indices = [d if d >= 0 else len(shape) + d for d in axes]
    return tuple(s for i, s in enumerate(shape) if i not in axis_indices)


@register_op("MatrixNorm")
class MatrixNorm(OpDef):
    """Matrix norm operator definition."""

    op_name = "MatrixNorm"

    def infer_shape(self, *args: OpValue, **kwargs: OpValue) -> OpValue:
        """Infer output shape for `MatrixNorm`.

        Args:
            *args (OpValue): Input matrix tensor.
            **kwargs (OpValue): Optional `axis` and `keepdims` attributes.

        Returns:
            OpValue: Reduced matrix norm shape.
        """
        return _infer_axis_norm_shape(args, kwargs, (-2, -1))


@register_op("VectorNorm")
class VectorNorm(OpDef):
    """Vector norm operator definition."""

    op_name = "VectorNorm"

    def infer_shape(self, *args: OpValue, **kwargs: OpValue) -> OpValue:
        """Infer output shape for `VectorNorm`.

        Args:
            *args (OpValue): Input vector/tensor.
            **kwargs (OpValue): Optional `axis` and `keepdims` attributes.

        Returns:
            OpValue: Reduced vector norm shape.
        """
        return _infer_axis_norm_shape(args, kwargs, (-1,))


@register_op("Svdvals")
class Svdvals(OpDef):
    """Singular values operator definition."""

    op_name = "Svdvals"

    def infer_shape(self, *args: OpValue, **kwargs: OpValue) -> OpValue:
        """Infer singular-value vector shape `(..., K)` for `Svdvals`.

        Args:
            *args (OpValue): Input matrix tensor.
            **kwargs (OpValue): Optional keyword attributes.

        Returns:
            OpValue: Singular values output shape.
        """
        in_shape = extract_shape(args[0])
        return in_shape[:-1] if len(in_shape) > 1 else ()
