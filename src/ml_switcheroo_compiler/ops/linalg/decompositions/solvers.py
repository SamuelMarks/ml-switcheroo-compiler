"""Implementation of solvers operations and utilities for the ops subsystem (`ml_switcheroo_compiler.ops.linalg.decompositions.solvers`)."""

from __future__ import annotations

import typing

from ml_switcheroo_compiler.backends.registry import EagerTensorProtocol
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.shape import TensorOrScalar
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.base import OpDef, register_op
from ml_switcheroo_compiler.ops.linalg.utils import _emit_linalg_node

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.interpreter.evaluator import TensorLike


@register_op("TriangularSolve")
class TriangularSolve(OpDef):
    """TriangularSolve Operation Definition."""

    def infer_shape(self, *args: EagerTensorProtocol, **kwargs: TensorOrScalar) -> tuple[int, ...]:
        """Infer solution shape for triangular system AX = B.

        Args:
            *args (EagerTensorProtocol): Matrix A and RHS B arguments.
            **kwargs (EagerTensorProtocol): Optional keyword arguments.

        Returns:
            tuple[int, ...]: Broadcasted solution shape.
        """
        a = args[0] if len(args) > 0 else kwargs.get("matrix", kwargs.get("a"))
        b = args[1] if len(args) > 1 else kwargs.get("rhs", kwargs.get("b"))
        if a is None or b is None:
            return ()
        shape_a = tuple(int(d) for d in getattr(a, "shape", getattr(a, "shape_metadata", a if isinstance(a, (list, tuple)) else ())))
        shape_b = tuple(int(d) for d in getattr(b, "shape", getattr(b, "shape_metadata", b if isinstance(b, (list, tuple)) else ())))

        if len(shape_a) < 2 or len(shape_b) < 1:
            return shape_b

        batch_a = shape_a[:-2]
        is_1d = (len(shape_b) == 1) or (len(shape_b) == len(shape_a) - 1 and len(shape_a) > 2)

        from ml_switcheroo_compiler.core.shape import broadcast_shapes

        if is_1d:
            batch_b = shape_b[:-1] if len(shape_b) > 1 else ()
            batch_out = broadcast_shapes(batch_a, batch_b) if (batch_a or batch_b) else ()
            return batch_out + (shape_a[-1],)

        batch_b = shape_b[:-2] if len(shape_b) >= 2 else ()
        batch_out = broadcast_shapes(batch_a, batch_b) if (batch_a or batch_b) else ()
        return batch_out + (shape_a[-1], shape_b[-1])


@register_op("Lu")
class Lu(OpDef):
    """Lu Operation Definition."""

    def infer_shape(self, *args: EagerTensorProtocol, **kwargs: TensorOrScalar) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]] | tuple[int, ...]:
        """Infer P, L, U factor shapes.

        Args:
            *args (EagerTensorProtocol): Input matrix A argument.
            **kwargs (EagerTensorProtocol): Optional keyword arguments.

        Returns:
            tuple: Tuple of (P_shape, L_shape, U_shape).
        """
        inp = args[0] if args else kwargs.get("a", kwargs.get("input"))
        if inp is None:
            return ()
        shape = tuple(int(d) for d in getattr(inp, "shape", getattr(inp, "shape_metadata", inp if isinstance(inp, (list, tuple)) else ())))
        if len(shape) < 2:
            return (shape, shape, shape)
        m, n = shape[-2], shape[-1]
        k = min(m, n)
        p_shape = shape[:-2] + (m, m)
        l_shape = shape[:-2] + (m, k)
        u_shape = shape[:-2] + (k, n)
        return (p_shape, l_shape, u_shape)


@register_op("LuSolve")
class LuSolve(OpDef):
    """LuSolve Operation Definition."""

    def infer_shape(self, *args: EagerTensorProtocol, **kwargs: TensorOrScalar) -> tuple[int, ...]:
        """Infer solution shape for LU solve.

        Args:
            *args (EagerTensorProtocol): LU, pivots, rhs arguments.
            **kwargs (EagerTensorProtocol): Optional keyword arguments.

        Returns:
            tuple[int, ...]: Broadcasted solution shape.
        """
        lu: TensorOrScalar
        rhs: TensorOrScalar
        if len(args) >= 3:
            lu, rhs = args[0], args[2]
        elif len(args) == 2:
            lu, rhs = args[0], args[1]
        else:
            lu = kwargs.get("lu", kwargs.get("lower_upper", kwargs.get("a")))
            rhs = kwargs.get("rhs", kwargs.get("b"))

        if lu is None or rhs is None:
            return ()

        shape_a = tuple(int(d) for d in getattr(lu, "shape", getattr(lu, "shape_metadata", lu if isinstance(lu, (list, tuple)) else ())))
        shape_b = tuple(int(d) for d in getattr(rhs, "shape", getattr(rhs, "shape_metadata", rhs if isinstance(rhs, (list, tuple)) else ())))

        if len(shape_a) < 2 or len(shape_b) < 1:
            return shape_b

        batch_a = shape_a[:-2]
        is_1d = (len(shape_b) == 1) or (len(shape_b) == len(shape_a) - 1 and len(shape_a) > 2)

        from ml_switcheroo_compiler.core.shape import broadcast_shapes

        if is_1d:
            batch_b = shape_b[:-1] if len(shape_b) > 1 else ()
            batch_out = broadcast_shapes(batch_a, batch_b) if (batch_a or batch_b) else ()
            return batch_out + (shape_a[-1],)

        batch_b = shape_b[:-2] if len(shape_b) >= 2 else ()
        batch_out = broadcast_shapes(batch_a, batch_b) if (batch_a or batch_b) else ()
        return batch_out + (shape_a[-1], shape_b[-1])


@register_op("Norm")
class Norm(OpDef):
    """Norm Operation Definition."""

    def infer_shape(self, *args: EagerTensorProtocol, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Infer shape.

        Args:
            *args (EagerTensorProtocol): Positional args.
            **kwargs (EagerTensorProtocol): Keyword args.

        Returns: Tensor: The shape.
        """
        return ()


@register_op("MatrixExponential")
class MatrixExponential(OpDef):
    """MatrixExponential Operation Definition."""

    def infer_shape(self, *args: EagerTensorProtocol, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Infer shape.

        Args:
            *args (EagerTensorProtocol): Positional args.
            **kwargs (EagerTensorProtocol): Keyword args.

        Returns: Tensor: The shape.
        """
        if not args:
            return ()

        a = args[0]
        shape = tuple(int(d) for d in getattr(a, "shape", getattr(a, "shape_metadata", ())))
        if len(shape) < 2:
            raise ValueError("MatrixExponential requires at least 2 dimensions")
        if shape[-1] != shape[-2]:
            raise ValueError("MatrixExponential requires a square matrix")
        return shape


@register_op("Cross")
class Cross(OpDef):
    """Cross Operation Definition."""

    def infer_shape(self, *args: EagerTensorProtocol, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Infer shape.

        Args:
            *args (EagerTensorProtocol): Positional args.
            **kwargs (EagerTensorProtocol): Keyword args.

        Returns: Tensor: The shape.
        """
        if not args:
            return ()

        a = args[0]
        b = args[1]
        shape_a = tuple(int(d) for d in getattr(a, "shape", getattr(a, "shape_metadata", ())))
        shape_b = tuple(int(d) for d in getattr(b, "shape", getattr(b, "shape_metadata", ())))
        axis = int(typing.cast(int, kwargs.get("axis", -1)))
        if shape_a[axis] != 3 or shape_b[axis] != 3:
            raise ValueError("Cross requires dimension size 3")
        from ml_switcheroo_compiler.core.shape import broadcast_shapes

        return broadcast_shapes(shape_a, shape_b)


@register_op("PowerIteration")
class PowerIteration(OpDef):
    """Power Iteration Operation Definition."""

    def infer_shape(self, *args: EagerTensorProtocol, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Infer shape for Power Iteration.

        Args:
            *args (EagerTensorProtocol): The positional arguments.
            **kwargs (EagerTensorProtocol): The keyword arguments.

        Returns: Tensor: The tuple containing output shapes and dtypes.
        """
        in_shape = args[0].shape
        v_shape = in_shape[:-2] + (in_shape[-1],)
        u_shape = in_shape[:-2] + (in_shape[-2],)
        sigma_shape = in_shape[:-2]
        return (v_shape, u_shape, sigma_shape), (args[0].dtype,) * 3


@register_op("Polar")
class Polar(OpDef):
    """Polar Operation Definition."""

    def infer_shape(self, *args: EagerTensorProtocol, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Infer shape.

        Args:
            *args (EagerTensorProtocol): Positional args.
            **kwargs (EagerTensorProtocol): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        return args[0].shape, args[0].shape


@register_op("TridiagonalSolve")
class TridiagonalSolve(OpDef):
    """TridiagonalSolve Operation Definition."""

    def infer_shape(self, *args: EagerTensorProtocol, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Infer shape.

        Args:
            *args (EagerTensorProtocol): Positional args.
            **kwargs (EagerTensorProtocol): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        return args[3].shape


def lu(a: Tensor) -> TensorOrScalar:
    """Compute the LU decomposition of a matrix.

    Args:
        a (Tensor): The a parameter.

    Returns:
        tuple: Result.
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        p, l_mat, u = typing.cast("tuple[TensorLike, TensorLike, TensorLike]", backend.execute_op("Lu", a.data))
        return (
            Tensor(p, TensorConfig(p.shape, a.dtype, a.device)),
            Tensor(l_mat, TensorConfig(l_mat.shape, a.dtype, a.device)),
            Tensor(u, TensorConfig(u.shape, a.dtype, a.device)),
        )
    return _emit_linalg_node("Lu", [a], {}, [a.shape, a.shape, a.shape], [a.dtype] * 3)


def lu_solve(lu_and_piv: tuple[Tensor, Tensor], b: Tensor) -> TensorOrScalar:
    """Solve an equation system, a x = b, given the LU factorization of a.

    Args:
        lu_and_piv (tuple): The lu_and_piv parameter.
        b (Tensor): The b parameter.

    Returns:
        Tensor: Result.
    """
    lu, piv = lu_and_piv
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = typing.cast("TensorLike", backend.execute_op("LuSolve", lu.data, piv.data, b.data))
        return Tensor(data, TensorConfig(data.shape, b.dtype, b.device))
    return _emit_linalg_node("LuSolve", [lu, piv, b], {}, [b.shape], [b.dtype])


def polar(a: Tensor, side: str = "right") -> TensorOrScalar:
    """Compute the polar decomposition of a matrix.

    Args:
        a (Tensor): The a parameter.
        side (str): The side parameter.

    Returns:
        tuple: Result.
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        u, p = typing.cast("tuple[TensorLike, TensorLike]", backend.execute_op("Polar", a.data, side=side))
        return (
            Tensor(u, TensorConfig(u.shape, a.dtype, a.device)),
            Tensor(p, TensorConfig(p.shape, a.dtype, a.device)),
        )
    return _emit_linalg_node("Polar", [a], {"side": side}, [a.shape, a.shape], [a.dtype] * 2)


def tridiagonal_solve(dl: Tensor, d: Tensor, du: Tensor, b: Tensor) -> TensorOrScalar:
    """Solves a tridiagonal linear system.

    Args:
        dl (Tensor): The dl parameter.
        d (Tensor): The d parameter.
        du (Tensor): The du parameter.
        b (Tensor): The b parameter.

    Returns:
        Tensor: Result.
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = typing.cast("TensorLike", backend.execute_op("TridiagonalSolve", dl.data, d.data, du.data, b.data))
        return Tensor(data, TensorConfig(data.shape, b.dtype, b.device))
    return _emit_linalg_node("TridiagonalSolve", [dl, d, du, b], {}, [b.shape], [b.dtype])


@register_op("TridiagonalMatmul")
class TridiagonalMatmul(OpDef):
    """TridiagonalMatmul Operation Definition."""

    op_name = "TridiagonalMatmul"

    def infer_shape(self, dl: EagerTensorProtocol, d: EagerTensorProtocol, du: EagerTensorProtocol, b: EagerTensorProtocol, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Infer shape.

        Args:
            dl (EagerTensorProtocol): The dl parameter.
            d (EagerTensorProtocol): The d parameter.
            du (EagerTensorProtocol): The du parameter.
            b (EagerTensorProtocol): The b parameter.
            **kwargs (TensorOrScalar): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        return b.shape
