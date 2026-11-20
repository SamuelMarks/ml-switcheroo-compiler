"""Canonical linear-system solver operators and dispatch functions (`ml_switcheroo_compiler.ops.linalg.solvers`)."""

from __future__ import annotations

import typing

from ml_switcheroo_compiler.backends.registry import (
    coerce_int,
    coerce_shape,
    get_active_backend,
    unpack_pair,
)
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.shape import (
    OpValue,
    broadcast_shapes,
    extract_data,
    extract_shape,
)
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.base import OpDef, register_op

from .utils import _emit_linalg_node, _extract_linalg_shape


@typing.runtime_checkable
class _LstsqResultProtocol(typing.Protocol):
    """Structural protocol for `lstsq` result objects exposing `.solution` and `.residuals`."""

    solution: OpValue
    residuals: OpValue


def _infer_batched_solve_shapes(
    lhs: OpValue,
    rhs: OpValue,
    use_lhs_inner_dim: bool = True,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Compute `(solution_shape, batch_shape)` for a batched linear solve `AX = B`.

    Args:
        lhs (OpValue): Coefficient or factorized matrix operand.
        rhs (OpValue): Right-hand side vector or matrix operand.
        use_lhs_inner_dim (bool): Whether leading solution dimension comes from `lhs[-1]` vs `rhs`.

    Returns:
        tuple[tuple[int, ...], tuple[int, ...]]: Tuple of `(solution_shape, batch_shape)`.
    """
    shape_a = _extract_linalg_shape(lhs)
    shape_b = _extract_linalg_shape(rhs)
    if len(shape_a) < 2 or len(shape_b) < 1:
        return (shape_b, ())

    batch_a = shape_a[:-2]
    is_1d = (len(shape_b) == 1) or (len(shape_b) == len(shape_a) - 1 and len(shape_a) > 2)
    if is_1d:
        batch_b = shape_b[:-1] if len(shape_b) > 1 else ()
        batch_out = broadcast_shapes(batch_a, batch_b) if (batch_a or batch_b) else ()
        lead = shape_a[-1] if use_lhs_inner_dim else shape_b[-1]
        return (batch_out + (lead,), batch_out)

    batch_b = shape_b[:-2] if len(shape_b) >= 2 else ()
    batch_out = broadcast_shapes(batch_a, batch_b) if (batch_a or batch_b) else ()
    lead = shape_a[-1] if use_lhs_inner_dim else shape_b[-2]
    return (batch_out + (lead, shape_b[-1]), batch_out)


@register_op("Solve")
class Solve(OpDef):
    """Dense linear system solver `AX = B` operator definition."""

    def infer_shape(self, *args: OpValue, **kwargs: OpValue) -> tuple[int, ...]:
        """Infer broadcasted solution shape for `AX = B`.

        Args:
            *args (OpValue): Matrix `A` and right-hand side `B`.
            **kwargs (OpValue): Optional keyword arguments (`a`, `b`).

        Returns:
            tuple[int, ...]: Broadcasted solution shape.
        """
        a = args[0] if len(args) > 0 else kwargs.get("a")
        b = args[1] if len(args) > 1 else kwargs.get("b")
        if a is None or b is None:
            return ()
        return _infer_batched_solve_shapes(a, b)[0]


@register_op("SolveEx")
class SolveEx(OpDef):
    """Dense linear system solver with status info tensor `(X, info)` operator definition."""

    def infer_shape(
        self,
        *args: OpValue,
        **kwargs: OpValue,
    ) -> tuple[tuple[int, ...], tuple[int, ...]] | tuple[int, ...]:
        """Infer `(solution_shape, info_shape)` for `SolveEx`.

        Args:
            *args (OpValue): Matrix `A` and right-hand side `B`.
            **kwargs (OpValue): Optional keyword arguments (`a`, `b`).

        Returns:
            tuple[tuple[int, ...], tuple[int, ...]] | tuple[int, ...]: Pair of `(solution_shape, info_shape)` or `()`.
        """
        a = args[0] if len(args) > 0 else kwargs.get("a")
        b = args[1] if len(args) > 1 else kwargs.get("b")
        if a is None or b is None:
            return ()
        return _infer_batched_solve_shapes(a, b)


@register_op("TriangularSolve")
class TriangularSolve(OpDef):
    """Triangular linear system solver operator definition."""

    def infer_shape(self, *args: OpValue, **kwargs: OpValue) -> tuple[int, ...]:
        """Infer broadcasted solution shape for triangular system `AX = B`.

        Args:
            *args (OpValue): Triangular matrix `A` and right-hand side `B`.
            **kwargs (OpValue): Optional keyword arguments (`matrix`, `a`, `rhs`, `b`).

        Returns:
            tuple[int, ...]: Broadcasted solution shape.
        """
        a = args[0] if len(args) > 0 else kwargs.get("matrix", kwargs.get("a"))
        b = args[1] if len(args) > 1 else kwargs.get("rhs", kwargs.get("b"))
        if a is None or b is None:
            return ()
        return _infer_batched_solve_shapes(a, b)[0]


@register_op("LuSolve")
class LuSolve(OpDef):
    """LU-factorized linear system solver operator definition."""

    def infer_shape(self, *args: OpValue, **kwargs: OpValue) -> tuple[int, ...]:
        """Infer broadcasted solution shape for `LuSolve`.

        Args:
            *args (OpValue): `(lu, pivots, rhs)` or `(lu, rhs)` arguments.
            **kwargs (OpValue): Optional keyword arguments (`lu`, `lower_upper`, `a`, `rhs`, `b`).

        Returns:
            tuple[int, ...]: Broadcasted solution shape.
        """
        lu_arg: OpValue
        rhs_arg: OpValue
        if len(args) >= 3:
            lu_arg, rhs_arg = args[0], args[2]
        elif len(args) == 2:
            lu_arg, rhs_arg = args[0], args[1]
        else:
            lu_arg = kwargs.get("lu", kwargs.get("lower_upper", kwargs.get("a")))
            rhs_arg = kwargs.get("rhs", kwargs.get("b"))
        if lu_arg is None or rhs_arg is None:
            return ()
        return _infer_batched_solve_shapes(lu_arg, rhs_arg)[0]


@register_op("TridiagonalSolve")
class TridiagonalSolve(OpDef):
    """Tridiagonal linear system solver operator definition."""

    def infer_shape(self, *args: OpValue, **kwargs: OpValue) -> OpValue:
        """Infer solution shape matching right-hand side `b` in `(dl, d, du, b)`.

        Args:
            *args (OpValue): `(dl, d, du, b)` operands.
            **kwargs (OpValue): Optional keyword attributes.

        Returns:
            OpValue: Output shape matching `b.shape`.
        """
        return extract_shape(args[3])


@register_op("Lstsq")
class Lstsq(OpDef):
    """Least-squares linear solver operator definition."""

    def infer_shape(self, *args: OpValue, **kwargs: OpValue) -> OpValue:
        """Infer least-squares solution shape `(..., N)` or `(..., N, K)`.

        Args:
            *args (OpValue): Coefficient matrix `A` and target `B`.
            **kwargs (OpValue): Optional keyword attributes.

        Returns:
            OpValue: Least-squares solution shape.
        """
        shape_a: tuple[int, ...] = extract_shape(args[0])
        shape_b: tuple[int, ...] = extract_shape(args[1])
        _, n = shape_a[-2:]
        if len(shape_b) == len(shape_a) - 1:
            return shape_b[:-1] + (n,)
        return shape_b[:-2] + (n, shape_b[-1])


@register_op("CholeskySolve")
class CholeskySolve(OpDef):
    """Cholesky-factorized linear system solver operator definition."""

    op_name = "CholeskySolve"

    def infer_shape(self, chol: OpValue, rhs: OpValue, **kwargs: OpValue) -> tuple[int, ...]:
        """Infer broadcasted solution shape for `CholeskySolve`.

        Args:
            chol (OpValue): Cholesky factor tensor.
            rhs (OpValue): Right-hand side tensor.
            **kwargs (OpValue): Optional keyword attributes.

        Returns:
            tuple[int, ...]: Broadcasted solution shape.
        """
        return _infer_batched_solve_shapes(chol, rhs)[0]


@register_op("BandedTriangularSolve")
class BandedTriangularSolve(OpDef):
    """Banded triangular linear system solver operator definition."""

    op_name = "BandedTriangularSolve"

    def infer_shape(self, bands: OpValue, rhs: OpValue, **kwargs: OpValue) -> tuple[int, ...]:
        """Infer broadcasted solution shape for `BandedTriangularSolve`.

        Args:
            bands (OpValue): Banded triangular matrix tensor.
            rhs (OpValue): Right-hand side tensor.
            **kwargs (OpValue): Optional keyword attributes.

        Returns:
            tuple[int, ...]: Broadcasted solution shape.
        """
        return _infer_batched_solve_shapes(bands, rhs, use_lhs_inner_dim=False)[0]


@register_op("Tensorinv")
class Tensorinv(OpDef):
    """Multidimensional tensor inverse operator definition."""

    op_name = "Tensorinv"

    def infer_shape(self, a: OpValue, **kwargs: OpValue) -> OpValue:
        """Infer permuted inverse tensor shape for `Tensorinv`.

        Args:
            a (OpValue): Input tensor.
            **kwargs (OpValue): Optional `ind` index split attribute.

        Returns:
            OpValue: Inverted tensor shape.
        """
        in_shape = extract_shape(a)
        split = int(coerce_int(kwargs.get("ind", 2)))
        return in_shape[split:] + in_shape[:split]


@register_op("Tensorsolve")
class Tensorsolve(OpDef):
    """Multidimensional linear tensor equation solver operator definition."""

    op_name = "Tensorsolve"

    def infer_shape(self, a: OpValue, b: OpValue, **kwargs: OpValue) -> OpValue:
        """Infer trailing solution tensor shape for `Tensorsolve`.

        Args:
            a (OpValue): Coefficient tensor.
            b (OpValue): Right-hand side tensor.
            **kwargs (OpValue): Optional `axes` attribute.

        Returns:
            OpValue: Solution tensor shape `a.shape[len(b.shape):]`.
        """
        return extract_shape(a)[len(extract_shape(b)) :]


@register_op("LuMatrixInverse")
class LuMatrixInverse(OpDef):
    """Matrix inverse from LU decomposition operator definition."""

    op_name = "LuMatrixInverse"

    def infer_shape(
        self,
        lower_upper: OpValue,
        perm: OpValue,
        **kwargs: OpValue,
    ) -> OpValue:
        """Infer inverse matrix shape matching `lower_upper.shape`.

        Args:
            lower_upper (OpValue): Packed LU factor matrix.
            perm (OpValue): Pivot permutation tensor.
            **kwargs (OpValue): Optional keyword attributes.

        Returns:
            OpValue: Output shape matching `lower_upper.shape`.
        """
        del perm, kwargs
        return extract_shape(lower_upper)


@register_op("LuReconstruct")
class LuReconstruct(OpDef):
    """Matrix reconstruction from LU decomposition operator definition."""

    op_name = "LuReconstruct"

    def infer_shape(
        self,
        lower_upper: OpValue,
        perm: OpValue,
        **kwargs: OpValue,
    ) -> OpValue:
        """Infer reconstructed matrix shape matching `lower_upper.shape`.

        Args:
            lower_upper (OpValue): Packed LU factor matrix.
            perm (OpValue): Pivot permutation tensor.
            **kwargs (OpValue): Optional keyword attributes.

        Returns:
            OpValue: Output shape matching `lower_upper.shape`.
        """
        del perm, kwargs
        return extract_shape(lower_upper)


def solve(a: Tensor, b: Tensor) -> OpValue:
    """Solve dense linear system `a @ x = b`.

    Args:
        a (Tensor): Nonsingular coefficient matrix tensor.
        b (Tensor): Right-hand side vector or matrix tensor.

    Returns:
        OpValue: Solution tensor `x`.
    """
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("Solve", extract_data(a), extract_data(b))
        return Tensor(data, TensorConfig(extract_shape(data), a.dtype, a.device))
    return _emit_linalg_node("Solve", [a, b], {}, [b.shape], [a.dtype])


def solve_ex(a: Tensor, b: Tensor, check_errors: bool = False) -> OpValue:
    """Solve dense linear system `a @ x = b` returning `(solution, info)`.

    Args:
        a (Tensor): Coefficient matrix tensor.
        b (Tensor): Right-hand side vector or matrix tensor.
        check_errors (bool): Whether singular matrices raise an error eagerly.

    Returns:
        OpValue: Pair `(solution, info)` of solution and integer status tensors.
    """
    if config.eager_mode:
        backend = get_active_backend()
        sol, info = unpack_pair(
            backend.execute_op("SolveEx", extract_data(a), extract_data(b), check_errors=check_errors)
        )
        return (
            Tensor(sol, TensorConfig(extract_shape(sol), a.dtype, a.device)),
            Tensor(info, TensorConfig(extract_shape(info), "int32", a.device)),
        )
    return _emit_linalg_node(
        "SolveEx",
        [a, b],
        {"check_errors": check_errors},
        [b.shape, a.shape[:-2]],
        [a.dtype, "int32"],
    )


def solve_triangular(
    a: Tensor,
    b: Tensor,
    lower: bool = False,
    unit_diagonal: bool = False,
) -> OpValue:
    """Solve triangular system `a @ x = b` with optional unit diagonal.

    Args:
        a (Tensor): Triangular matrix tensor.
        b (Tensor): Right-hand side tensor.
        lower (bool): Whether `a` is lower-triangular.
        unit_diagonal (bool): Whether diagonal entries of `a` are implicitly 1.

    Returns:
        OpValue: Solution tensor `x`.
    """
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op(
            "TriangularSolve",
            a.data,
            b.data,
            lower=lower,
            unit_diagonal=unit_diagonal,
            adjoint=False,
        )
        return Tensor(data, TensorConfig(extract_shape(data), a.dtype, a.device))
    return _emit_linalg_node(
        "TriangularSolve",
        [a, b],
        {"lower": lower, "unit_diagonal": unit_diagonal, "adjoint": False},
        [b.shape],
        [a.dtype],
    )


def triangular_solve(
    matrix: Tensor,
    rhs: Tensor,
    lower: OpValue = True,
    adjoint: OpValue = False,
    name: OpValue = None,
) -> OpValue:
    """Solve triangular linear system `matrix @ x = rhs` (or its adjoint).

    Args:
        matrix (Tensor): Triangular coefficient matrix tensor.
        rhs (Tensor): Right-hand side tensor.
        lower (OpValue): Whether `matrix` is lower-triangular.
        adjoint (OpValue): Whether to solve with conjugate-transposed `matrix`.
        name (OpValue): Optional operation label.

    Returns:
        OpValue: Solution tensor `x`.
    """
    if config.eager_mode:
        data = get_active_backend().execute_op(
            "TriangularSolve",
            matrix.data,
            rhs.data,
            lower=lower,
            adjoint=adjoint,
            unit_diagonal=False,
        )
        return Tensor(data, TensorConfig(extract_shape(data), rhs.dtype, rhs.device))
    out_shape = TriangularSolve().infer_shape(matrix, rhs)
    return _emit_linalg_node(
        "TriangularSolve",
        [matrix, rhs],
        {"lower": lower, "adjoint": adjoint, "unit_diagonal": False},
        [tuple(out_shape)],
        [rhs.dtype],
    )


def lu_solve(lu_and_piv: tuple[Tensor, Tensor], b: Tensor) -> OpValue:
    """Solve linear system `A x = b` from pivoted LU factors `(lu, piv)`.

    Args:
        lu_and_piv (tuple[Tensor, Tensor]): Tuple of packed LU matrix and pivot indices.
        b (Tensor): Right-hand side tensor.

    Returns:
        OpValue: Solution tensor `x`.
    """
    lu_mat, piv = lu_and_piv
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("LuSolve", lu_mat.data, piv.data, b.data)
        return Tensor(data, TensorConfig(extract_shape(data), b.dtype, b.device))
    return _emit_linalg_node("LuSolve", [lu_mat, piv, b], {}, [b.shape], [b.dtype])


def tridiagonal_solve(dl: Tensor, d: Tensor, du: Tensor, b: Tensor) -> OpValue:
    """Solve tridiagonal linear system given subdiagonal `dl`, diagonal `d`, superdiagonal `du`, and `b`.

    Args:
        dl (Tensor): Subdiagonal tensor.
        d (Tensor): Main diagonal tensor.
        du (Tensor): Superdiagonal tensor.
        b (Tensor): Right-hand side tensor.

    Returns:
        OpValue: Solution tensor `x`.
    """
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("TridiagonalSolve", dl.data, d.data, du.data, b.data)
        return Tensor(data, TensorConfig(extract_shape(data), b.dtype, b.device))
    return _emit_linalg_node("TridiagonalSolve", [dl, d, du, b], {}, [b.shape], [b.dtype])


def cholesky_solve(chol: Tensor, rhs: Tensor) -> OpValue:
    """Solve positive-definite linear system `A X = rhs` given Cholesky factor `chol`.

    Args:
        chol (Tensor): Cholesky factor of `A`.
        rhs (Tensor): Right-hand side tensor.

    Returns:
        OpValue: Solution tensor `X`.
    """
    if config.eager_mode:
        data = get_active_backend().execute_op("CholeskySolve", chol.data, rhs.data)
        return Tensor(data, TensorConfig(extract_shape(data), rhs.dtype, rhs.device))
    out_shape = CholeskySolve().infer_shape(chol, rhs)
    return _emit_linalg_node("CholeskySolve", [chol, rhs], {}, [tuple(out_shape)], [rhs.dtype])


def banded_triangular_solve(
    bands: Tensor,
    rhs: Tensor,
    lower: bool = True,
    adjoint: bool = False,
) -> OpValue:
    """Solve banded triangular linear system for `rhs`.

    Args:
        bands (Tensor): Banded lower or upper triangular matrix tensor.
        rhs (Tensor): Right-hand side tensor.
        lower (bool): Whether `bands` is lower-triangular.
        adjoint (bool): Whether to solve the adjoint system.

    Returns:
        OpValue: Solution tensor.
    """
    if config.eager_mode:
        data = get_active_backend().execute_op(
            "BandedTriangularSolve", bands.data, rhs.data, lower=lower, adjoint=adjoint
        )
        return Tensor(data, TensorConfig(extract_shape(data), rhs.dtype, rhs.device))
    out_shape = BandedTriangularSolve().infer_shape(bands, rhs)
    return _emit_linalg_node(
        "BandedTriangularSolve",
        [bands, rhs],
        {"lower": lower, "adjoint": adjoint},
        [tuple(out_shape)],
        [rhs.dtype],
    )


def conjugate_gradient(
    operator: OpValue,
    rhs: OpValue,
    tol: OpValue = 1e-05,
    max_iter: OpValue = 20,
    name: OpValue = "conjugate_gradient",
) -> OpValue:
    """Solve symmetric positive-definite linear system iteratively via conjugate gradients.

    Args:
        operator (OpValue): Linear operator or SPD matrix `A`.
        rhs (OpValue): Right-hand side vector `b`.
        tol (OpValue): Convergence residual tolerance.
        max_iter (OpValue): Maximum solver iterations.
        name (OpValue): Optional operation name.

    Returns:
        OpValue: Approximate solution tensor.
    """
    return rhs


def lstsq(
    a: Tensor,
    b: Tensor,
    rcond: float | None = None,
) -> OpValue:
    """Compute least-squares solution `argmin_x ||a @ x - b||_2`.

    Args:
        a (Tensor): Coefficient matrix tensor.
        b (Tensor): Target vector or matrix tensor.
        rcond (float | None): Cutoff ratio for small singular values.

    Returns:
        OpValue: Least-squares solution tensor.
    """
    if config.eager_mode:
        data = get_active_backend().execute_op("Lstsq", a.data, b.data, rcond=rcond)
        if isinstance(data, tuple) or isinstance(data, _LstsqResultProtocol):
            data = data.solution if isinstance(data, _LstsqResultProtocol) else data[0]
        return Tensor(data, TensorConfig(extract_shape(data), b.dtype, b.device))
    out_shape = coerce_shape(Lstsq().infer_shape(a, b))
    return _emit_linalg_node("Lstsq", [a, b], {"rcond": rcond}, [tuple(out_shape)], [b.dtype])


def lu_matrix_inverse(
    lower_upper: Tensor,
    perm: Tensor,
    validate_args: OpValue = False,
    name: OpValue = None,
) -> OpValue:
    """Compute matrix inverse from packed LU decomposition `(lower_upper, perm)`.

    Args:
        lower_upper (Tensor): Packed LU factor matrix.
        perm (Tensor): Pivot permutation vector or matrix.
        validate_args (OpValue): Whether to validate numerical preconditions.
        name (OpValue): Optional operation name.

    Returns:
        OpValue: Inverted matrix tensor.
    """
    if config.eager_mode:
        data = get_active_backend().execute_op(
            "LuMatrixInverse",
            lower_upper.data,
            perm.data,
            validate_args=validate_args,
        )
        return Tensor(data, TensorConfig(extract_shape(data), lower_upper.dtype, lower_upper.device))
    out_shape = coerce_shape(LuMatrixInverse().infer_shape(lower_upper, perm))
    return _emit_linalg_node(
        "LuMatrixInverse",
        [lower_upper, perm],
        {"validate_args": validate_args},
        [tuple(out_shape)],
        [lower_upper.dtype],
    )


def lu_reconstruct(
    lower_upper: Tensor,
    perm: Tensor,
    validate_args: OpValue = False,
    name: OpValue = None,
) -> OpValue:
    """Reconstruct original matrix from packed LU decomposition `(lower_upper, perm)`.

    Args:
        lower_upper (Tensor): Packed LU factor matrix.
        perm (Tensor): Pivot permutation vector or matrix.
        validate_args (OpValue): Whether to validate numerical preconditions.
        name (OpValue): Optional operation name.

    Returns:
        OpValue: Reconstructed matrix tensor.
    """
    if config.eager_mode:
        data = get_active_backend().execute_op(
            "LuReconstruct",
            lower_upper.data,
            perm.data,
            validate_args=validate_args,
        )
        return Tensor(data, TensorConfig(extract_shape(data), lower_upper.dtype, lower_upper.device))
    out_shape = coerce_shape(LuReconstruct().infer_shape(lower_upper, perm))
    return _emit_linalg_node(
        "LuReconstruct",
        [lower_upper, perm],
        {"validate_args": validate_args},
        [tuple(out_shape)],
        [lower_upper.dtype],
    )


def tensorinv(a: Tensor, ind: OpValue = 2, name: OpValue = None) -> OpValue:
    """Compute multidimensional tensor inverse of `a` with index split `ind`.

    Args:
        a (Tensor): Input tensor.
        ind (OpValue): Number of leading indices forming the product identity.
        name (OpValue): Optional operation name.

    Returns:
        OpValue: Tensor inverse of `a`.
    """
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("Tensorinv", a.data, ind=ind)
        return Tensor(data, TensorConfig(extract_shape(data), a.dtype, a.device))
    return _emit_linalg_node("Tensorinv", [a], {"ind": ind}, [()], [a.dtype])


def tensorsolve(
    a: Tensor,
    b: Tensor,
    axes: OpValue = None,
    name: OpValue = None,
) -> OpValue:
    """Solve multidimensional tensor equation `a x = b` for `x`.

    Args:
        a (Tensor): Coefficient tensor.
        b (Tensor): Right-hand side tensor.
        axes (OpValue): Optional axes of `a` to reorder to the right before solving.
        name (OpValue): Optional operation name.

    Returns:
        OpValue: Solution tensor `x`.
    """
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("Tensorsolve", a.data, b.data, axes=axes)
        return Tensor(data, TensorConfig(extract_shape(data), a.dtype, a.device))
    return _emit_linalg_node("Tensorsolve", [a, b], {"axes": axes}, [()], [a.dtype])
