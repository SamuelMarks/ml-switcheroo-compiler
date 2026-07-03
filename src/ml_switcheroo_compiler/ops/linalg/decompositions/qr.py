"""Module docstring."""

from __future__ import annotations

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.base import OpDef, register_op
from ml_switcheroo_compiler.ops.linalg.utils import _emit_linalg_node


@register_op("Qr")
class Qr(OpDef):
    """Qr Operation Definition."""

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            object: The shape.
        """
        return ()


@register_op("Hessenberg")
class Hessenberg(OpDef):
    """Hessenberg Operation Definition."""

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        return args[0].shape, args[0].shape


@register_op("HouseholderProduct")
class HouseholderProduct(OpDef):
    """HouseholderProduct Operation Definition."""

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        return ()


@register_op("Schur")
class Schur(OpDef):
    """Schur Operation Definition."""

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        return args[0].shape, args[0].shape


@register_op("Tridiagonal")
class Tridiagonal(OpDef):
    """Tridiagonal Operation Definition."""

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        a_shape = args[0].shape
        diag_shape = a_shape[:-1]
        off_diag_shape = a_shape[:-2] + (a_shape[-1] - 1,) if a_shape[-1] > 0 else a_shape[:-1]
        return diag_shape, off_diag_shape, a_shape


def qr(input: Tensor, mode: str = "reduced") -> tuple[Tensor, Tensor]:
    """Computes the QR decomposition of a matrix.

    Args:
        input (Tensor): The input matrix
        mode (str): Specifies the mode of decomposition ('reduced', 'complete',
        'r', or 'raw'). Defaults to 'reduced'

    Returns:
    tuple[Tensor, Tensor]: A tuple containing:
        - Q (Tensor): The orthonormal matrix
        - R (Tensor): The upper-triangular matrix
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        q, r = backend.execute_op("Qr", input.data, mode=mode)
        return (
            Tensor(q, TensorConfig(q.shape, input.dtype, input.device)),
            Tensor(r, TensorConfig(r.shape, input.dtype, input.device)),
        )
    return _emit_linalg_node("Qr", [input], {"mode": mode}, [input.shape, input.shape], [input.dtype] * 2)


def hessenberg(a: Tensor) -> tuple[Tensor, Tensor]:
    """Computes the Hessenberg decomposition of a matrix.

    Args:
        a (Tensor): The input matrix

    Returns:
    tuple[Tensor, Tensor]: A tuple containing:
        - H (Tensor): The upper Hessenberg matrix
        - Q (Tensor): The unitary/orthogonal matrix
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


def householder_product(a: Tensor, tau: Tensor) -> Tensor:
    """Computes the product of Householder reflectors.

    Args:
        a (Tensor): Vectors with Householder reflectors
        tau (Tensor): Scalar factors

    Returns:
    Tensor: The orthogonal/unitary matrix
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op("HouseholderProduct", a.data, tau.data)
        return Tensor(data, TensorConfig(data.shape, a.dtype, a.device))
    return _emit_linalg_node("HouseholderProduct", [a, tau], {}, [a.shape], [a.dtype])


def schur(a: Tensor) -> tuple[Tensor, Tensor]:
    """Computes the Schur decomposition of a matrix.

    Args:
        a (Tensor): The input matrix

    Returns:
    tuple[Tensor, Tensor]: A tuple containing:
        - T (Tensor): The Schur form
        - Z (Tensor): The unitary/orthogonal matrix
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


def tridiagonal(a: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Computes the tridiagonal decomposition of a symmetric matrix.

    Args:
        a (Tensor): The input symmetric matrix

    Returns:
    tuple[Tensor, Tensor, Tensor]: A tuple containing:
        - diag (Tensor): The main diagonal
        - off_diag (Tensor): The off-diagonal
        - q (Tensor): The unitary/orthogonal matrix
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
