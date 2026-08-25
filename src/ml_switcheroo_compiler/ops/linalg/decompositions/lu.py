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

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        return ()


@register_op("LuPivotsToPermutation")
class LuPivotsToPermutation(OpDef):
    """LuPivotsToPermutation Operation Definition."""

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        if not args:
            return ()
        return args[0].shape[:-1] + (kwargs.get("permutation_size", 0),)


def lu_factor(a: Tensor) -> object:
    """Compute pivoted LU decomposition of a matrix for use in `lu_solve`.

    Args:
        a (Tensor): The a parameter.

    Returns:
        tuple: Result.
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend: object = get_active_backend()
        lu, piv = backend.execute_op("LuFactor", a.data)
        piv_dtype: object = a.dtype
        return (
            Tensor(lu, TensorConfig(lu.shape, a.dtype, a.device)),
            Tensor(piv, TensorConfig(piv.shape, piv_dtype, a.device)),
        )

    piv_shape: object = a.shape[:-1]
    return _emit_linalg_node("LuFactor", [a], {}, [a.shape, piv_shape], [a.dtype, a.dtype])


def lu_pivots_to_permutation(pivots: Tensor, permutation_size: int) -> object:
    """Convert LU pivots to a permutation matrix or array.

    Args:
        pivots (Tensor): The pivots parameter.
        permutation_size (int): The permutation_size parameter.

    Returns:
        Tensor: Result.
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend: object = get_active_backend()
        data: object = backend.execute_op("LuPivotsToPermutation", pivots.data, permutation_size)
        return Tensor(data, TensorConfig(data.shape, pivots.dtype, pivots.device))
    out_shape: object = pivots.shape[:-1] + (permutation_size,)
    return _emit_linalg_node(
        "LuPivotsToPermutation",
        [pivots],
        {"permutation_size": permutation_size},
        [out_shape],
        [pivots.dtype],
    )
