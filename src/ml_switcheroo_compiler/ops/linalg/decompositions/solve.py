"""Core abstractions and logic definitions for solve.py."""

from __future__ import annotations

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.base import OpDef, register_op
from ml_switcheroo_compiler.ops.linalg.utils import _emit_linalg_node


@register_op("Solve")
class Solve(OpDef):
    """Solve Operation Definition."""

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

        Returns:
        object: Result.
        """
        return ()


@register_op("SolveEx")
class SolveEx(OpDef):
    """SolveEx Operation Definition."""

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            object: Result.
        """
        return ()


def solve(a: Tensor, b: Tensor) -> Tensor:
    """Solves a linear matrix equation, or system of linear scalar equations.

    Args:
        a (Tensor): The a parameter.
        b (Tensor): The b parameter.

    Returns:
        Tensor: Result.
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op(
            "Solve",
            (a.data if hasattr(a, "device") else a),
            (b.data if hasattr(b, "device") else b),
        )
        return Tensor(data, TensorConfig(data.shape, a.dtype, a.device))
    return _emit_linalg_node("Solve", [a, b], {}, [b.shape], [a.dtype])


def solve_ex(a: Tensor, b: Tensor, check_errors: bool = False) -> tuple[Tensor, Tensor]:
    """Solves a linear matrix equation with info tensor.

    Args:
        a (Tensor): Coefficient matrix
        b (Tensor): Ordinate or 'dependent variable' values
        check_errors (bool): If True, throws an error if the decomposition fails

    Returns:
        tuple[Tensor, Tensor]: Solution to the system of linear equations and info tensor
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        sol, info = backend.execute_op(
            "SolveEx",
            (a.data if hasattr(a, "device") else a),
            (b.data if hasattr(b, "device") else b),
            check_errors=check_errors,
        )
        return (
            Tensor(sol, TensorConfig(sol.shape, a.dtype, a.device)),
            Tensor(info, TensorConfig(info.shape, "int32", a.device)),
        )
    return _emit_linalg_node("SolveEx", [a, b], {"check_errors": check_errors}, [b.shape, a.shape[:-2]], [a.dtype, "int32"])


def solve_triangular(
    a: Tensor,
    b: Tensor,
    lower: bool = False,
    unit_diagonal: bool = False,
) -> Tensor:
    """Solves the equation `a x = b` for `x`, assuming `a` is a triangular matrix.

    Args:
        a (Tensor): The a parameter.
        b (Tensor): The b parameter.
        lower (bool): The lower parameter.
        unit_diagonal (bool): The unit_diagonal parameter.

    Returns:
        Tensor: Result.
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op(
            "TriangularSolve",
            a.data,
            b.data,
            lower=lower,
            unit_diagonal=unit_diagonal,
            adjoint=False,
        )
        return Tensor(data, TensorConfig(data.shape, a.dtype, a.device))
    return _emit_linalg_node(
        "TriangularSolve",
        [a, b],
        {"lower": lower, "unit_diagonal": unit_diagonal, "adjoint": False},
        [b.shape],
        [a.dtype],
    )
