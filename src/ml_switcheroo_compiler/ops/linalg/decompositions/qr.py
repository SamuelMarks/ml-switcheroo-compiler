"""Operations and utilities for `ml_switcheroo_compiler.ops.linalg.decompositions.qr`."""

from __future__ import annotations

import typing

from ml_switcheroo_compiler.backends.registry import (
    coerce_operand_list,
    unpack_pair,
    unpack_triple,
)
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.shape import OpValue, extract_shape
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.base import OpDef, register_op
from ml_switcheroo_compiler.ops.linalg.utils import _emit_linalg_node, _extract_linalg_shape

if typing.TYPE_CHECKING:
    pass


@register_op("Qr")
class Qr(OpDef):
    """Qr Operation Definition."""

    def infer_shape(self, *args: OpValue, **kwargs: OpValue) -> OpValue:
        """Infer shape.

        Args:
        *args (OpValue): Positional args.
        **kwargs (OpValue): Keyword args.

        Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
        """
        if not args:
            return ()
        a_shape = extract_shape(args[0])
        mode = kwargs.get("mode", "reduced")
        m, n = a_shape[-2], a_shape[-1]
        k = min(m, n)
        if mode == "complete":
            return a_shape[:-2] + (m, m), a_shape[:-2] + (m, n)
        if mode == "r":
            return (a_shape[:-2] + (k, n),)
        return a_shape[:-2] + (m, k), a_shape[:-2] + (k, n)


@register_op("Hessenberg")
class Hessenberg(OpDef):
    """Hessenberg Operation Definition."""

    def infer_shape(self, *args: OpValue, **kwargs: OpValue) -> OpValue:
        """Infer shape.

        Args:
            *args (OpValue): Positional args.
            **kwargs (OpValue): Keyword args.

        Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
        """
        return extract_shape(args[0]), extract_shape(args[0])


@register_op("HouseholderProduct")
class HouseholderProduct(OpDef):
    """HouseholderProduct Operation Definition."""

    def infer_shape(self, *args: OpValue, **kwargs: OpValue) -> tuple[int, ...]:
        """Infer shape for HouseholderProduct.

        Args:
            *args (OpValue): Positional args.
            **kwargs (OpValue): Keyword args.

        Returns:
            tuple[int, ...]: Result shape (..., M, N).
        """
        if not args:
            return ()
        return _extract_linalg_shape(args[0])


@register_op("Schur")
class Schur(OpDef):
    """Schur Operation Definition."""

    def infer_shape(self, *args: OpValue, **kwargs: OpValue) -> OpValue:
        """Infer shape.

        Args:
            *args (OpValue): Positional args.
            **kwargs (OpValue): Keyword args.

        Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
        """
        return extract_shape(args[0]), extract_shape(args[0])


@register_op("Tridiagonal")
class Tridiagonal(OpDef):
    """Tridiagonal Operation Definition."""

    def infer_shape(self, *args: OpValue, **kwargs: OpValue) -> OpValue:
        """Infer shape.

        Args:
            *args (OpValue): Positional args.
            **kwargs (OpValue): Keyword args.

        Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
        """
        a_shape = extract_shape(args[0])
        diag_shape = a_shape[:-1]
        off_diag_shape = a_shape[:-2] + (a_shape[-1] - 1,) if a_shape[-1] > 0 else a_shape[:-1]
        return diag_shape, off_diag_shape, a_shape


def qr(input: Tensor, mode: str = "reduced") -> OpValue:
    """Compute the QR decomposition of a matrix.

    Args:
        input (Tensor): Input `input` argument (input).
        mode (str): Input `mode` argument (mode).

    Returns:
        tuple: Computed `tuple` output.
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        res = backend.execute_op("Qr", input.data, mode=mode)
        if mode == "r":
            res_t = res
            return Tensor(res_t, TensorConfig(extract_shape(res_t), input.dtype, input.device))
        q, r = unpack_pair(res)
        return (
            Tensor(q, TensorConfig(extract_shape(q), input.dtype, input.device)),
            Tensor(r, TensorConfig(extract_shape(r), input.dtype, input.device)),
        )
    in_shape = extract_shape(input)
    m, n = in_shape[-2], in_shape[-1]
    k = min(m, n)
    if mode == "complete":
        q_shape, r_shape = in_shape[:-2] + (m, m), in_shape[:-2] + (m, n)
    elif mode == "r":
        r_shape = in_shape[:-2] + (k, n)
        return _emit_linalg_node("Qr", [input], {"mode": mode}, [r_shape], [input.dtype])
    else:
        q_shape, r_shape = in_shape[:-2] + (m, k), in_shape[:-2] + (k, n)
    return _emit_linalg_node("Qr", [input], {"mode": mode}, [q_shape, r_shape], [input.dtype] * 2)


def hessenberg(a: Tensor) -> OpValue:
    """Compute the Hessenberg decomposition of a matrix.

    Args:
        a (Tensor): Input `a` argument (a).

    Returns:
        tuple: Computed `tuple` output.
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        h, q = unpack_pair(backend.execute_op("Hessenberg", a.data))
        return (
            Tensor(h, TensorConfig(extract_shape(h), a.dtype, a.device)),
            Tensor(q, TensorConfig(extract_shape(q), a.dtype, a.device)),
        )
    return _emit_linalg_node("Hessenberg", [a], {}, [a.shape, a.shape], [a.dtype] * 2)


def householder_product(a: Tensor, tau: Tensor) -> OpValue:
    """Compute the product of Householder reflectors.

    Args:
        a (Tensor): Input `a` argument (a).
        tau (Tensor): Input `tau` argument (tau).

    Returns:
        Tensor: Computed `Tensor` output.
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op("HouseholderProduct", a.data, tau.data)
        return Tensor(data, TensorConfig(extract_shape(data), a.dtype, a.device))
    return _emit_linalg_node("HouseholderProduct", [a, tau], {}, [a.shape], [a.dtype])


def schur(a: Tensor) -> OpValue:
    """Compute the Schur decomposition of a matrix.

    Args:
        a (Tensor): Input `a` argument (a).

    Returns:
        tuple: Computed `tuple` output.
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        t, z = unpack_pair(backend.execute_op("Schur", a.data))
        return (
            Tensor(t, TensorConfig(extract_shape(t), a.dtype, a.device)),
            Tensor(z, TensorConfig(extract_shape(z), a.dtype, a.device)),
        )
    return _emit_linalg_node("Schur", [a], {}, [a.shape, a.shape], [a.dtype] * 2)


def tridiagonal(a: Tensor) -> OpValue:
    """Compute the tridiagonal decomposition of a symmetric matrix.

    Args:
        a (Tensor): Input `a` argument (a).

    Returns:
        tuple: Computed `tuple` output.
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        diag, off_diag, q = unpack_triple(backend.execute_op("Tridiagonal", a.data))
        return (
            Tensor(diag, TensorConfig(extract_shape(diag), a.dtype, a.device)),
            Tensor(off_diag, TensorConfig(extract_shape(off_diag), a.dtype, a.device)),
            Tensor(q, TensorConfig(extract_shape(q), a.dtype, a.device)),
        )
    a_shape = extract_shape(a)
    diag_shape = a_shape[:-1]
    off_diag_shape = a_shape[:-2] + (a_shape[-1] - 1,) if a_shape[-1] > 0 else a_shape[:-1]
    return _emit_linalg_node("Tridiagonal", [a], {}, [diag_shape, off_diag_shape, a.shape], [a.dtype] * 3)


@register_op("Qdwh")
class Qdwh(OpDef):
    """Qdwh Operation Definition."""

    def infer_shape(self, *args: OpValue, **kwargs: OpValue) -> OpValue:
        """Infer shape.

        Args:
            *args (OpValue): Positional args.
            **kwargs (OpValue): Keyword args.

        Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
        """
        return extract_shape(args[0]), extract_shape(args[0]), extract_shape(args[0])[:-2], extract_shape(args[0])[:-2]


def qdwh(a: Tensor, is_hermitian: bool = False, max_iterations: int = 10) -> OpValue:
    """Compute the QR-based dynamically weighted Halley iteration.

    Args:
        a (Tensor): Input `a` argument (a).
        is_hermitian (bool): Input `is_hermitian` argument (is hermitian).
        max_iterations (int): Input `max_iterations` argument (max iterations).

    Returns:
        tuple: Computed `tuple` output.
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        parts = coerce_operand_list(
            backend.execute_op("Qdwh", a.data, is_hermitian=is_hermitian, max_iterations=max_iterations)
        )
        q, h, num_iters, is_converged = (
            parts[0],
            parts[1],
            parts[2],
            parts[3],
        )
        return (
            Tensor(q, TensorConfig(extract_shape(q), a.dtype, a.device)),
            Tensor(h, TensorConfig(extract_shape(h), a.dtype, a.device)),
            Tensor(num_iters, TensorConfig(extract_shape(num_iters), "int32", a.device)),
            Tensor(is_converged, TensorConfig(extract_shape(is_converged), "bool", a.device)),
        )
    return _emit_linalg_node(
        "Qdwh",
        [a],
        {"is_hermitian": is_hermitian, "max_iterations": max_iterations},
        [a.shape, a.shape, a.shape[:-2], a.shape[:-2]],
        [a.dtype, a.dtype, "int32", "bool"],
    )
