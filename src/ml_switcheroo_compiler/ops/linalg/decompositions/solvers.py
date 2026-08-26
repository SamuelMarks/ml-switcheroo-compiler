"""Module solvers.py."""

from __future__ import annotations

# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915

"""Core abstractions and logic definitions for misc.py."""

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.base import OpDef, register_op
from ml_switcheroo_compiler.ops.linalg.utils import _emit_linalg_node


@register_op("TriangularSolve")
class TriangularSolve(OpDef):
    """TriangularSolve Operation Definition."""

    def infer_shape(self, *args, **kwargs):
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        return ()


@register_op("Lu")
class Lu(OpDef):
    """Lu Operation Definition."""

    def infer_shape(self, *args, **kwargs):
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns: Tensor: The shape.
        """
        return ()


@register_op("LuSolve")
class LuSolve(OpDef):
    """LuSolve Operation Definition."""

    def infer_shape(self, *args, **kwargs):
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns: Tensor: The shape.
        """
        return ()


@register_op("Norm")
class Norm(OpDef):
    """Norm Operation Definition."""

    def infer_shape(self, *args, **kwargs):
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns: Tensor: The shape.
        """
        return ()


@register_op("MatrixExponential")
class MatrixExponential(OpDef):
    """MatrixExponential Operation Definition."""

    def infer_shape(self, *args, **kwargs):
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns: Tensor: The shape.
        """
        return ()


@register_op("Cross")
class Cross(OpDef):
    """Cross Operation Definition."""

    def infer_shape(self, *args, **kwargs):
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns: Tensor: The shape.
        """
        return ()


@register_op("PowerIteration")
class PowerIteration(OpDef):
    """Power Iteration Operation Definition."""

    def infer_shape(self, *args, **kwargs):
        """Infer shape for Power Iteration.

        Args:
            *args (object): The positional arguments.
            **kwargs (object): The keyword arguments.

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

    def infer_shape(self, *args, **kwargs):
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        return args[0].shape, args[0].shape


@register_op("TridiagonalSolve")
class TridiagonalSolve(OpDef):
    """TridiagonalSolve Operation Definition."""

    def infer_shape(self, *args, **kwargs):
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        return args[3].shape


def lu(a: Tensor):
    """Compute the LU decomposition of a matrix.

    Args:
        a (Tensor): The a parameter.

    Returns:
        tuple: Result.
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


def lu_solve(lu_and_piv: tuple[Tensor, Tensor], b: Tensor):
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
        data = backend.execute_op("LuSolve", lu.data, piv.data, b.data)
        return Tensor(data, TensorConfig(data.shape, b.dtype, b.device))
    return _emit_linalg_node("LuSolve", [lu, piv, b], {}, [b.shape], [b.dtype])


def polar(a: Tensor, side: str = "right"):
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
        u, p = backend.execute_op("Polar", a.data, side=side)
        return (
            Tensor(u, TensorConfig(u.shape, a.dtype, a.device)),
            Tensor(p, TensorConfig(p.shape, a.dtype, a.device)),
        )
    return _emit_linalg_node("Polar", [a], {"side": side}, [a.shape, a.shape], [a.dtype] * 2)


def tridiagonal_solve(dl: Tensor, d: Tensor, du: Tensor, b: Tensor):
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
        data = backend.execute_op("TridiagonalSolve", dl.data, d.data, du.data, b.data)
        return Tensor(data, TensorConfig(data.shape, b.dtype, b.device))
    return _emit_linalg_node("TridiagonalSolve", [dl, d, du, b], {}, [b.shape], [b.dtype])


@register_op("TridiagonalMatmul")
class TridiagonalMatmul(OpDef):
    """TridiagonalMatmul Operation Definition."""

    op_name = "TridiagonalMatmul"

    def infer_shape(self, dl, d, du, b, **kwargs):
        """Infer shape.

        Args:
            dl (object): The dl parameter.
            d (object): The d parameter.
            du (object): The du parameter.
            b (object): The b parameter.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        return b.shape
