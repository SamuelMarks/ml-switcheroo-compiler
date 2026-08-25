"""Module eig.py."""

from __future__ import annotations

# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915

"""Core abstractions and logic definitions for eig.py."""

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.base import OpDef, register_op
from ml_switcheroo_compiler.ops.linalg.utils import _emit_linalg_node


@register_op("Eig")
class Eig(OpDef):
    """Eig Operation Definition."""

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer the output shape for the infer_shape operation.

        Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        return ()


@register_op("Eigh")
class Eigh(OpDef):
    """Eigh Operation Definition."""

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns: object: The shape.
        """
        return ()


@register_op("Eigvalsh")
class Eigvalsh(OpDef):
    """Eigvalsh Operation Definition."""

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns: object: The shape.
        """
        return ()


def eigh(input: Tensor, UPLO: str = "L") -> object:
    """Compute the eigenvalues and eigenvectors of a complex Hermitian or real symmetric.

    Args:
        input (Tensor): The input parameter.
        UPLO (str): The UPLO parameter.

    Returns:
        tuple: Result.
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend: object = get_active_backend()
        w, v = backend.execute_op("Eigh", (input.data if type(input).__name__ == "Tensor" else input), UPLO=UPLO)
        return (
            Tensor(w, TensorConfig(w.shape, getattr(input, "dtype", None), getattr(input, "device", None))),
            Tensor(v, TensorConfig(v.shape, getattr(input, "dtype", None), getattr(input, "device", None))),
        )
    return _emit_linalg_node("Eigh", [input], {"UPLO": UPLO}, [input.shape[:-1], input.shape], [getattr(input, "dtype", None)] * 2)


def eigvalsh(input: Tensor, UPLO: str = "L") -> object:
    """Compute the eigenvalues of a complex Hermitian or real symmetric matrix.

    Args:
        input (Tensor): The input parameter.
        UPLO (str): The UPLO parameter.

    Returns:
        Tensor: Result.
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend: object = get_active_backend()
        data: object = backend.execute_op("Eigvalsh", (input.data if type(input).__name__ == "Tensor" else input), UPLO=UPLO)
        return Tensor(
            backend.array(data),
            TensorConfig(backend.array(data).shape, getattr(input, "dtype", None), getattr(input, "device", None)),
        )
    return _emit_linalg_node("Eigvalsh", [input], {"UPLO": UPLO}, [input.shape[:-1]], [getattr(input, "dtype", None)])


@register_op("Eigvals")
class Eigvals(OpDef):
    """Eigvals Operation Definition."""

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns: object: The shape.
        """
        return ()


def eigvals(input: Tensor) -> object:
    """Compute the eigenvalues of a general matrix.

    Args:
        input (Tensor): The input parameter.

    Returns:
        Tensor: Result.
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend: object = get_active_backend()
        data: object = backend.execute_op("Eigvals", (input.data if type(input).__name__ == "Tensor" else input))
        return Tensor(
            backend.array(data),
            TensorConfig(backend.array(data).shape, getattr(input, "dtype", None), getattr(input, "device", None)),
        )
    return _emit_linalg_node("Eigvals", [input], {}, [input.shape[:-1]], [getattr(input, "dtype", None)])


def eig(input: Tensor) -> object:
    """Compute the eigenvalues and eigenvectors of a square matrix.

    Args:
        input (Tensor): The input parameter.

    Returns:
        tuple: Result.
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend: object = get_active_backend()
        w, v = backend.execute_op("Eig", (input.data if type(input).__name__ == "Tensor" else input))
        return (
            Tensor(w, TensorConfig(w.shape, getattr(input, "dtype", None), getattr(input, "device", None))),
            Tensor(v, TensorConfig(v.shape, getattr(input, "dtype", None), getattr(input, "device", None))),
        )
    return _emit_linalg_node("Eig", [input], {}, [input.shape[:-1], input.shape], [getattr(input, "dtype", None)] * 2)
