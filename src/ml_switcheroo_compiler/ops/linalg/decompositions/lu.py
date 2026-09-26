"""Module lu.py."""

from __future__ import annotations

# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915

"""Core abstractions and logic definitions for lu.py."""

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.base import OpDef, register_op
from ml_switcheroo_compiler.ops.linalg.utils import _emit_linalg_node


@register_op("LuFactor")
class LuFactor(OpDef):
    """LuFactor Operation Definition."""

    def infer_shape(
        self,
        *args: Tensor | tuple[int, ...] | list[int] | None,
        **kwargs: Tensor | tuple[int, ...] | list[int] | str | int | float | None,
    ) -> tuple[tuple[int, ...], tuple[int, ...]] | tuple[int, ...]:
        """Infer LU factorization output shapes (LU, pivots).

        Args:
            *args (Tensor | tuple[int, ...] | list[int] | None): Positional args representing input tensor.
            **kwargs (Tensor | tuple[int, ...] | list[int] | str | int | float | None): Keyword args representing input tensor.

        Returns:
            tuple[tuple[int, ...], tuple[int, ...]] | tuple[int, ...]: Tuple of (lu_shape (..., M, N), pivots_shape (..., min(M, N))).
        """
        inp = args[0] if args else kwargs.get("a", kwargs.get("input"))
        if inp is None:
            return ()
        shape = tuple(int(d) for d in getattr(inp, "shape", getattr(inp, "shape_metadata", inp if isinstance(inp, (list, tuple)) else ())))
        if len(shape) < 2:
            return (shape, shape)
        m, n = shape[-2], shape[-1]
        piv_shape = shape[:-2] + (min(m, n),)
        return (shape, piv_shape)


@register_op("LuPivotsToPermutation")
class LuPivotsToPermutation(OpDef):
    """LuPivotsToPermutation Operation Definition."""

    def infer_shape(
        self,
        *args: Tensor | tuple[int, ...] | list[int] | None,
        **kwargs: int,
    ) -> tuple[int, ...]:
        """Infer shape.

        Args:
            *args (Tensor | tuple[int, ...] | list[int] | None): Positional args representing input pivots.
            **kwargs (int): Keyword args including permutation_size.

        Returns:
            tuple[int, ...]: Inferred output permutation shape.
        """
        if not args or args[0] is None:
            return ()
        return args[0].shape[:-1] + (kwargs.get("permutation_size", 0),)


def lu_factor(a: Tensor) -> tuple[Tensor, Tensor]:
    """Compute pivoted LU decomposition of a matrix for use in `lu_solve`.

    Args:
        a (Tensor): Input tensor matrix.

    Returns:
        tuple[Tensor, Tensor]: Tuple containing LU factored tensor and pivot indices tensor.
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        lu, piv = backend.execute_op("LuFactor", a.data)
        piv_dtype = a.dtype
        return (
            Tensor(lu, TensorConfig(lu.shape, a.dtype, a.device)),
            Tensor(piv, TensorConfig(piv.shape, piv_dtype, a.device)),
        )

    piv_shape = a.shape[:-1]
    res = _emit_linalg_node("LuFactor", [a], {}, [a.shape, piv_shape], [a.dtype, a.dtype])
    if isinstance(res, tuple):
        return (res[0], res[1])
    return (res, res)


def lu_pivots_to_permutation(pivots: Tensor, permutation_size: int) -> Tensor:
    """Convert LU pivots to a permutation matrix or array.

    Args:
        pivots (Tensor): The pivots tensor.
        permutation_size (int): Size of the resulting permutation dimension.

    Returns:
        Tensor: Transformed permutation tensor.
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op("LuPivotsToPermutation", pivots.data, permutation_size)
        return Tensor(data, TensorConfig(data.shape, pivots.dtype, pivots.device))
    out_shape = pivots.shape[:-1] + (permutation_size,)
    res = _emit_linalg_node(
        "LuPivotsToPermutation",
        [pivots],
        {"permutation_size": permutation_size},
        [out_shape],
        [pivots.dtype],
    )
    if isinstance(res, tuple):
        return res[0]
    return res
