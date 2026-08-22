"""Module svd.py."""

from __future__ import annotations

# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915

"""Core abstractions and logic definitions for svd.py."""
from typing import Any

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.base import OpDef, register_op
from ml_switcheroo_compiler.ops.linalg.utils import _emit_linalg_node


@register_op("Svd")
class Svd(OpDef):
    """Svd Operation Definition."""

    def infer_shape(self, *args: Any, **kwargs: Any) -> Any:
        """Infer the output shape for the infer_shape operation.

        Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        if not args:
            return ()
        a_shape = args[0].shape
        full_matrices = kwargs.get("full_matrices", True)
        m, n = a_shape[-2], a_shape[-1]
        k = min(m, n)
        s_shape = a_shape[:-2] + (k,)
        if full_matrices:
            u_shape = a_shape[:-2] + (m, m)
            vh_shape = a_shape[:-2] + (n, n)
        else:
            u_shape = a_shape[:-2] + (m, k)
            vh_shape = a_shape[:-2] + (k, n)
        if kwargs.get("compute_uv", True):
            return u_shape, s_shape, vh_shape
        return (s_shape,)


def svd(
    input: Tensor,  # type: ignore
    full_matrices: bool = True,
    compute_uv: bool = True,
) -> Any:
    """Compute the Singular Value Decomposition (SVD) of a matrix.

    Args:
        input (Tensor): The input parameter.
        full_matrices (bool): The full_matrices parameter.
        compute_uv (bool): The compute_uv parameter.

    Returns:
        tuple: Result.
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        res = backend.execute_op(
            "Svd",
            input.data,
            full_matrices=full_matrices,
            compute_uv=compute_uv,
        )
        if not compute_uv:
            return Tensor(res, TensorConfig(res.shape, input.dtype, input.device))
        u, s, vh = res
        return (
            Tensor(u, TensorConfig(u.shape, input.dtype, input.device)),
            Tensor(s, TensorConfig(s.shape, input.dtype, input.device)),
            Tensor(vh, TensorConfig(vh.shape, input.dtype, input.device)),
        )
    m, n = input.shape[-2], input.shape[-1]
    k = min(m, n)
    s_shape = input.shape[:-2] + (k,)  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism
    if full_matrices:
        u_shape = input.shape[:-2] + (m, m)  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism
        vh_shape = input.shape[:-2] + (n, n)  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism
    else:
        u_shape = input.shape[:-2] + (m, k)  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism
        vh_shape = input.shape[:-2] + (k, n)  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism
    if not compute_uv:
        return _emit_linalg_node(
            "Svd",
            [input],
            {"full_matrices": full_matrices, "compute_uv": compute_uv},
            [s_shape],
            [input.dtype],
        )
    return _emit_linalg_node(
        "Svd",
        [input],
        {"full_matrices": full_matrices, "compute_uv": compute_uv},
        [u_shape, s_shape, vh_shape],
        [input.dtype] * 3,
    )
