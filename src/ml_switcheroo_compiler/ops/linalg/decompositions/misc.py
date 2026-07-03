"""Module docstring."""

from __future__ import annotations

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.base import OpDef, register_op
from ml_switcheroo_compiler.ops.linalg.utils import _emit_linalg_node


@register_op("TriangularSolve")
class TriangularSolve(OpDef):
    """TriangularSolve Operation Definition."""

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            object: The shape.
        """
        return ()


@register_op("Lu")
class Lu(OpDef):
    """Lu Operation Definition."""

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            object: The shape.
        """
        return ()


@register_op("LuSolve")
class LuSolve(OpDef):
    """LuSolve Operation Definition."""

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            object: The shape.
        """
        return ()


@register_op("Norm")
class Norm(OpDef):
    """Norm Operation Definition."""

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            object: The shape.
        """
        return ()


@register_op("MatrixExponential")
class MatrixExponential(OpDef):
    """MatrixExponential Operation Definition."""

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            object: The shape.
        """
        return ()


@register_op("Cross")
class Cross(OpDef):
    """Cross Operation Definition."""

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            object: The shape.
        """
        return ()


@register_op("PowerIteration")
class PowerIteration(OpDef):
    """Power Iteration Operation Definition."""

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape for Power Iteration.

        Args:
            *args (object): The positional arguments.
            **kwargs (object): The keyword arguments.

        Returns:
            object: The tuple containing output shapes and dtypes.
        """
        in_shape = args[0].shape
        v_shape = in_shape[:-2] + (in_shape[-1],)  # pragma: no cover
        u_shape = in_shape[:-2] + (in_shape[-2],)  # pragma: no cover
        sigma_shape = in_shape[:-2]  # pragma: no cover
        return (v_shape, u_shape, sigma_shape), (args[0].dtype,) * 3  # pragma: no cover


@register_op("Polar")
class Polar(OpDef):
    """Polar Operation Definition."""

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        return args[0].shape, args[0].shape


@register_op("TridiagonalSolve")
class TridiagonalSolve(OpDef):
    """TridiagonalSolve Operation Definition."""

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        return args[3].shape


def lu(a: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Computes the LU decomposition of a matrix.

    Args:
        a (Tensor): The input matrix to decompose

    Returns:
    tuple[Tensor, Tensor, Tensor]: The LU decomposition components (P, L, U)
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        p, l_mat, u = backend.execute_op("Lu", a.data)
        return (
            Tensor(p, TensorConfig(p.shape, a.dtype, a.device)),
            Tensor(l_mat, TensorConfig(l_mat.shape, a.dtype, a.device)),
            Tensor(u, TensorConfig(u.shape, a.dtype, a.device)),
        )
    return _emit_linalg_node("Lu", [a], {}, [a.shape, a.shape, a.shape], [a.dtype] * 3)


def lu_solve(lu_and_piv: tuple[Tensor, Tensor], b: Tensor) -> Tensor:
    """Solve an equation system, a x = b, given the LU factorization of a.

    Args:
        lu_and_piv (tuple[Tensor, Tensor]): LU factorization and pivot indices from lu_factor.
        b (Tensor): Right-hand side matrix or vector.

    Returns:
    Tensor: The solution matrix `x`
    """
    lu, piv = lu_and_piv
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op("LuSolve", lu.data, piv.data, b.data)
        return Tensor(data, TensorConfig(data.shape, b.dtype, b.device))
    return _emit_linalg_node("LuSolve", [lu, piv, b], {}, [b.shape], [b.dtype])


def polar(a: Tensor, side: str = "right") -> tuple[Tensor, Tensor]:
    """Computes the polar decomposition of a matrix.

    Args:
        a (Tensor): The input matrix
        side (str): "right" or "left"

    Returns:
    tuple[Tensor, Tensor]: A tuple containing:
        - U (Tensor): The unitary/orthogonal matrix
        - P (Tensor): The positive semi-definite matrix
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        u, p = backend.execute_op("Polar", a.data, side=side)
        return (
            Tensor(u, TensorConfig(u.shape, a.dtype, a.device)),
            Tensor(p, TensorConfig(p.shape, a.dtype, a.device)),
        )
    return _emit_linalg_node("Polar", [a], {"side": side}, [a.shape, a.shape], [a.dtype] * 2)


def tridiagonal_solve(dl: Tensor, d: Tensor, du: Tensor, b: Tensor) -> Tensor:
    """Solves a tridiagonal linear system.

    Args:
        dl (Tensor): The lower diagonal
        d (Tensor): The main diagonal
        du (Tensor): The upper diagonal
        b (Tensor): The right-hand side

    Returns:
    Tensor: The solution matrix `x`
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op("TridiagonalSolve", dl.data, d.data, du.data, b.data)
        return Tensor(data, TensorConfig(data.shape, b.dtype, b.device))
    return _emit_linalg_node("TridiagonalSolve", [dl, d, du, b], {}, [b.shape], [b.dtype])
