"""Core abstractions and logic definitions for eig.py."""

from __future__ import annotations

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.base import OpDef, register_op
from ml_switcheroo_compiler.ops.linalg.utils import _emit_linalg_node


@register_op("Eig")
class Eig(OpDef):
    """Eig Operation Definition."""

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            object: The shape.
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

        Returns:
            object: The shape.
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

        Returns:
            object: The shape.
        """
        return ()


def eigh(input: Tensor, UPLO: str = "L") -> tuple[Tensor, Tensor]:
    """Computes the eigenvalues and eigenvectors of a complex Hermitian or real symmetric.

    matrix

    Args:
        input (Tensor): The symmetric or Hermitian matrix
        UPLO (str): Specifies whether the calculation is done with the lower ('L')
        or upper ('U') triangular part of the matrix. Defaults to 'L'

    Returns:
    tuple[Tensor, Tensor]: A tuple containing:
        - eigenvalues (Tensor): The eigenvalues in ascending order
        - eigenvectors (Tensor): The column eigenvectors
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        w, v = backend.execute_op("Eigh", (input.data if type(input).__name__ == "Tensor" else input), UPLO=UPLO)
        return (
            Tensor(w, TensorConfig(w.shape, getattr(input, "dtype", None), getattr(input, "device", None))),
            Tensor(v, TensorConfig(v.shape, getattr(input, "dtype", None), getattr(input, "device", None))),
        )
    return _emit_linalg_node("Eigh", [input], {"UPLO": UPLO}, [input.shape[:-1], input.shape], [getattr(input, "dtype", None)] * 2)


def eigvalsh(input: Tensor, UPLO: str = "L") -> Tensor:
    """Computes the eigenvalues of a complex Hermitian or real symmetric matrix.

    Args:
        input (Tensor): The symmetric or Hermitian matrix
        UPLO (str): Specifies whether the calculation is done with the lower ('L')
        or upper ('U') triangular part of the matrix. Defaults to 'L'

    Returns:
    Tensor: The eigenvalues in ascending order
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op("Eigvalsh", (input.data if type(input).__name__ == "Tensor" else input), UPLO=UPLO)
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

        Returns:
            object: The shape.
        """
        return ()


def eigvals(input: Tensor) -> Tensor:
    """Computes the eigenvalues of a general matrix.

    Args:
        input (Tensor): The general square matrix

    Returns:
    Tensor: The eigenvalues
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op("Eigvals", (input.data if type(input).__name__ == "Tensor" else input))
        return Tensor(
            backend.array(data),
            TensorConfig(backend.array(data).shape, getattr(input, "dtype", None), getattr(input, "device", None)),
        )
    return _emit_linalg_node("Eigvals", [input], {}, [input.shape[:-1]], [getattr(input, "dtype", None)])


def eig(input: Tensor) -> tuple[Tensor, Tensor]:
    """Computes the eigenvalues and eigenvectors of a square matrix.

    Args:
        input (Tensor): The square matrix.

    Returns:
    tuple[Tensor, Tensor]: A tuple containing:
        - eigenvalues (Tensor): The eigenvalues.
        - eigenvectors (Tensor): The right eigenvectors.
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        w, v = backend.execute_op("Eig", (input.data if type(input).__name__ == "Tensor" else input))
        return (
            Tensor(w, TensorConfig(w.shape, getattr(input, "dtype", None), getattr(input, "device", None))),
            Tensor(v, TensorConfig(v.shape, getattr(input, "dtype", None), getattr(input, "device", None))),
        )
    return _emit_linalg_node("Eig", [input], {}, [input.shape[:-1], input.shape], [getattr(input, "dtype", None)] * 2)
