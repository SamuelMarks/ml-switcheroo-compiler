"""Module docstring."""

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
            object: The shape.
        """
        return ()


def solve(a: Tensor, b: Tensor) -> Tensor:
    """Solves a linear matrix equation, or system of linear scalar equations.

    Args:
        a (Tensor): Coefficient matrix
        b (Tensor): Ordinate or 'dependent variable' values

    Returns:
    Tensor: Solution to the system of linear equations
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


def solve_triangular(
    a: Tensor,
    b: Tensor,
    lower: bool = False,
    unit_diagonal: bool = False,
) -> Tensor:
    """Solves the equation `a x = b` for `x`, assuming `a` is a triangular matrix.

    Args:
        a (Tensor): Triangular coefficient matrix
        b (Tensor): Right-hand side matrix or vector
        lower (bool): Use only data contained in the lower triangle of a. Default is to use upper triangle.
        unit_diagonal (bool): If True, diagonal elements of a are assumed to be 1.

    Returns:
    Tensor: The solution matrix `x`
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
        )
        return Tensor(data, TensorConfig(data.shape, a.dtype, a.device))
    return _emit_linalg_node(
        "TriangularSolve",
        [a, b],
        {"lower": lower, "unit_diagonal": unit_diagonal},
        [b.shape],
        [a.dtype],
    )
