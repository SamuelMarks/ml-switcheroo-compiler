"""Module qr.py."""

from __future__ import annotations

# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915

"""Core abstractions and logic definitions for qr.py."""

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.base import OpDef, register_op
from ml_switcheroo_compiler.ops.linalg.utils import _emit_linalg_node


@register_op("Qr")
class Qr(OpDef):
    """Qr Operation Definition."""

    def infer_shape(self, *args, **kwargs):
        """Infer shape.

        Args:
        *args (Any): Positional args.
        **kwargs (Any): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        if not args:
            return ()
        a_shape = args[0].shape
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

    def infer_shape(self, *args, **kwargs):
        """Infer shape.

        Args:
            *args (Any): Positional args.
            **kwargs (Any): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        return args[0].shape, args[0].shape


@register_op("HouseholderProduct")
class HouseholderProduct(OpDef):
    """HouseholderProduct Operation Definition."""

    def infer_shape(self, *args, **kwargs):
        """Infer shape.

        Args:
            *args (Any): Positional args.
            **kwargs (Any): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        return ()


@register_op("Schur")
class Schur(OpDef):
    """Schur Operation Definition."""

    def infer_shape(self, *args, **kwargs):
        """Infer shape.

        Args:
            *args (Any): Positional args.
            **kwargs (Any): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        return args[0].shape, args[0].shape


@register_op("Tridiagonal")
class Tridiagonal(OpDef):
    """Tridiagonal Operation Definition."""

    def infer_shape(self, *args, **kwargs):
        """Infer shape.

        Args:
            *args (Any): Positional args.
            **kwargs (Any): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        a_shape = args[0].shape
        diag_shape = a_shape[:-1]
        off_diag_shape = a_shape[:-2] + (a_shape[-1] - 1,) if a_shape[-1] > 0 else a_shape[:-1]
        return diag_shape, off_diag_shape, a_shape


def qr(input: Tensor, mode: str = "reduced"):
    """Compute the QR decomposition of a matrix.

    Args:
        input (Tensor): The input parameter.
        mode (str): The mode parameter.

    Returns:
        tuple: Result.
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        res = backend.execute_op("Qr", input.data, mode=mode)
        if mode == "r":
            return Tensor(res, TensorConfig(res.shape, input.dtype, input.device))
        q, r = res
        return (
            Tensor(q, TensorConfig(q.shape, input.dtype, input.device)),
            Tensor(r, TensorConfig(r.shape, input.dtype, input.device)),
        )
    m, n = input.shape[-2], input.shape[-1]
    k = min(m, n)
    if mode == "complete":
        q_shape, r_shape = input.shape[:-2] + (m, m), input.shape[:-2] + (m, n)
    elif mode == "r":
        r_shape = (input.shape[:-2] + (k, n),)
        return _emit_linalg_node("Qr", [input], {"mode": mode}, [r_shape[0]], [input.dtype])
    else:
        q_shape, r_shape = input.shape[:-2] + (m, k), input.shape[:-2] + (k, n)
    return _emit_linalg_node("Qr", [input], {"mode": mode}, [q_shape, r_shape], [input.dtype] * 2)


def hessenberg(a: Tensor):
    """Compute the Hessenberg decomposition of a matrix.

    Args:
        a (Tensor): The a parameter.

    Returns:
        tuple: Result.
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        h, q = backend.execute_op("Hessenberg", a.data)
        return (
            Tensor(h, TensorConfig(h.shape, a.dtype, a.device)),
            Tensor(q, TensorConfig(q.shape, a.dtype, a.device)),
        )
    return _emit_linalg_node("Hessenberg", [a], {}, [a.shape, a.shape], [a.dtype] * 2)


def householder_product(a: Tensor, tau: Tensor):
    """Compute the product of Householder reflectors.

    Args:
        a (Tensor): The a parameter.
        tau (Tensor): The tau parameter.

    Returns:
        Tensor: Result.
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op("HouseholderProduct", a.data, tau.data)
        return Tensor(data, TensorConfig(data.shape, a.dtype, a.device))
    return _emit_linalg_node("HouseholderProduct", [a, tau], {}, [a.shape], [a.dtype])


def schur(a: Tensor):
    """Compute the Schur decomposition of a matrix.

    Args:
        a (Tensor): The a parameter.

    Returns:
        tuple: Result.
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        t, z = backend.execute_op("Schur", a.data)
        return (
            Tensor(t, TensorConfig(t.shape, a.dtype, a.device)),
            Tensor(z, TensorConfig(z.shape, a.dtype, a.device)),
        )
    return _emit_linalg_node("Schur", [a], {}, [a.shape, a.shape], [a.dtype] * 2)


def tridiagonal(a: Tensor):
    """Compute the tridiagonal decomposition of a symmetric matrix.

    Args:
        a (Tensor): The a parameter.

    Returns:
        tuple: Result.
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        diag, off_diag, q = backend.execute_op("Tridiagonal", a.data)
        return (
            Tensor(diag, TensorConfig(diag.shape, a.dtype, a.device)),
            Tensor(off_diag, TensorConfig(off_diag.shape, a.dtype, a.device)),
            Tensor(q, TensorConfig(q.shape, a.dtype, a.device)),
        )
    diag_shape = a.shape[:-1]
    off_diag_shape = a.shape[:-2] + (a.shape[-1] - 1,) if a.shape[-1] > 0 else a.shape[:-1]
    return _emit_linalg_node("Tridiagonal", [a], {}, [diag_shape, off_diag_shape, a.shape], [a.dtype] * 3)


@register_op("Qdwh")
class Qdwh(OpDef):
    """Qdwh Operation Definition."""

    def infer_shape(self, *args, **kwargs):
        """Infer shape.

        Args:
            *args (Any): Positional args.
            **kwargs (Any): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        return args[0].shape, args[0].shape, args[0].shape[:-2], args[0].shape[:-2]


def qdwh(a: Tensor, is_hermitian: bool = False, max_iterations: int = 10):
    """Compute the QR-based dynamically weighted Halley iteration.

    Args:
        a (Tensor): The a parameter.
        is_hermitian (bool): The is_hermitian parameter.
        max_iterations (int): The max_iterations parameter.

    Returns:
        tuple: Result.
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        q, h, num_iters, is_converged = backend.execute_op("Qdwh", a.data, is_hermitian=is_hermitian, max_iterations=max_iterations)
        return (
            Tensor(q, TensorConfig(q.shape, a.dtype, a.device)),
            Tensor(h, TensorConfig(h.shape, a.dtype, a.device)),
            Tensor(num_iters, TensorConfig(num_iters.shape, "int32", a.device)),
            Tensor(is_converged, TensorConfig(is_converged.shape, "bool", a.device)),
        )
    return _emit_linalg_node(
        "Qdwh",
        [a],
        {"is_hermitian": is_hermitian, "max_iterations": max_iterations},
        [a.shape, a.shape, a.shape[:-2], a.shape[:-2]],
        [a.dtype, a.dtype, "int32", "bool"],
    )
