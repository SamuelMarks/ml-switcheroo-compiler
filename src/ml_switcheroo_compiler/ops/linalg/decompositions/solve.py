"""Module solve.py."""

from __future__ import annotations

# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915

"""Core abstractions and logic definitions for solve.py."""

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.base import OpDef, register_op
from ml_switcheroo_compiler.ops.linalg.utils import _emit_linalg_node


@register_op("Solve")
class Solve(OpDef):
    """Solve Operation Definition."""

    def infer_shape(self, *args, **kwargs) -> tuple[int, ...]:
        """Infer solution shape for linear system AX = B.

        Args:
            *args (object): Matrix A and RHS B arguments.
            **kwargs (object): Optional keyword arguments.

        Returns:
            tuple[int, ...]: Broadcasted solution shape.
        """
        a = args[0] if len(args) > 0 else kwargs.get("a")
        b = args[1] if len(args) > 1 else kwargs.get("b")
        if a is None or b is None:
            return ()
        shape_a = tuple(int(d) for d in getattr(a, "shape", getattr(a, "shape_metadata", a if isinstance(a, (list, tuple)) else ())))
        shape_b = tuple(int(d) for d in getattr(b, "shape", getattr(b, "shape_metadata", b if isinstance(b, (list, tuple)) else ())))

        if len(shape_a) < 2 or len(shape_b) < 1:
            return shape_b

        batch_a = shape_a[:-2]
        is_1d = (len(shape_b) == 1) or (len(shape_b) == len(shape_a) - 1 and len(shape_a) > 2)

        if is_1d:
            batch_b = shape_b[:-1] if len(shape_b) > 1 else ()
            from ml_switcheroo_compiler.core.shape import broadcast_shapes

            batch_out = broadcast_shapes(batch_a, batch_b) if (batch_a or batch_b) else ()
            return batch_out + (shape_a[-1],)

        batch_b = shape_b[:-2] if len(shape_b) >= 2 else ()
        from ml_switcheroo_compiler.core.shape import broadcast_shapes

        batch_out = broadcast_shapes(batch_a, batch_b) if (batch_a or batch_b) else ()
        return batch_out + (shape_a[-1], shape_b[-1])


@register_op("SolveEx")
class SolveEx(OpDef):
    """SolveEx Operation Definition."""

    def infer_shape(self, *args, **kwargs) -> tuple[tuple[int, ...], tuple[int, ...]] | tuple[int, ...]:
        """Infer solution and info shapes for SolveEx.

        Args:
            *args (object): Matrix A and RHS B arguments.
            **kwargs (object): Optional keyword arguments.

        Returns:
            tuple: Tuple of (solution_shape, info_shape).
        """
        a = args[0] if len(args) > 0 else kwargs.get("a")
        b = args[1] if len(args) > 1 else kwargs.get("b")
        if a is None or b is None:
            return ()
        shape_a = tuple(int(d) for d in getattr(a, "shape", getattr(a, "shape_metadata", a if isinstance(a, (list, tuple)) else ())))
        shape_b = tuple(int(d) for d in getattr(b, "shape", getattr(b, "shape_metadata", b if isinstance(b, (list, tuple)) else ())))

        if len(shape_a) < 2 or len(shape_b) < 1:
            return (shape_b, ())

        batch_a = shape_a[:-2]
        is_1d = (len(shape_b) == 1) or (len(shape_b) == len(shape_a) - 1 and len(shape_a) > 2)

        from ml_switcheroo_compiler.core.shape import broadcast_shapes

        if is_1d:
            batch_b = shape_b[:-1] if len(shape_b) > 1 else ()
            batch_out = broadcast_shapes(batch_a, batch_b) if (batch_a or batch_b) else ()
            sol_shape = batch_out + (shape_a[-1],)
        else:
            batch_b = shape_b[:-2] if len(shape_b) >= 2 else ()
            batch_out = broadcast_shapes(batch_a, batch_b) if (batch_a or batch_b) else ()
            sol_shape = batch_out + (shape_a[-1], shape_b[-1])

        info_shape = batch_out
        return (sol_shape, info_shape)


def solve(a: Tensor, b: Tensor):
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


def solve_ex(a: Tensor, b: Tensor, check_errors: bool = False):
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
):
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
