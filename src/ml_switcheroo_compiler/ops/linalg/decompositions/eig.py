"""Implementation of eig operations and utilities for the ops subsystem (`ml_switcheroo_compiler.ops.linalg.decompositions.eig`)."""

from __future__ import annotations

import typing

from ml_switcheroo_compiler.backends.registry import EagerTensorProtocol
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.shape import TensorOrScalar
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.base import OpDef, register_op
from ml_switcheroo_compiler.ops.linalg.utils import _emit_linalg_node

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.interpreter.evaluator import TensorLike


@register_op("Eig")
class Eig(OpDef):
    """Eig Operation Definition."""

    def infer_shape(self, *args: EagerTensorProtocol, **kwargs: TensorOrScalar) -> tuple[tuple[int, ...], tuple[int, ...]] | tuple[int, ...]:
        """Infer eigenvalues and eigenvectors output shapes.

        Args:
            *args (EagerTensorProtocol): Input tensor or shape arguments.
            **kwargs (EagerTensorProtocol): Optional keyword arguments.

        Returns:
            tuple: Tuple of (eigenvalues_shape (..., N), eigenvectors_shape (..., N, N)).
        """
        inp = args[0] if args else kwargs.get("input")
        if inp is None:
            return ()
        shape = tuple(int(d) for d in getattr(inp, "shape", getattr(inp, "shape_metadata", inp if isinstance(inp, (list, tuple)) else ())))
        val_shape = shape[:-1] if len(shape) >= 1 else ()
        return (val_shape, shape)


@register_op("Eigh")
class Eigh(OpDef):
    """Eigh Operation Definition."""

    def infer_shape(self, *args: EagerTensorProtocol, **kwargs: TensorOrScalar) -> tuple[tuple[int, ...], tuple[int, ...]] | tuple[int, ...]:
        """Infer Hermitian/symmetric eigenvalues and eigenvectors output shapes.

        Args:
            *args (EagerTensorProtocol): Input tensor or shape arguments.
            **kwargs (EagerTensorProtocol): Optional keyword arguments.

        Returns:
            tuple: Tuple of (eigenvalues_shape (..., N), eigenvectors_shape (..., N, N)).
        """
        inp = args[0] if args else kwargs.get("input")
        if inp is None:
            return ()
        shape = tuple(int(d) for d in getattr(inp, "shape", getattr(inp, "shape_metadata", inp if isinstance(inp, (list, tuple)) else ())))
        val_shape = shape[:-1] if len(shape) >= 1 else ()
        return (val_shape, shape)


@register_op("Eigvalsh")
class Eigvalsh(OpDef):
    """Eigvalsh Operation Definition."""

    def infer_shape(self, *args: EagerTensorProtocol, **kwargs: TensorOrScalar) -> tuple[int, ...]:
        """Infer Hermitian/symmetric eigenvalues output shape (..., N).

        Args:
            *args (EagerTensorProtocol): Input tensor or shape arguments.
            **kwargs (EagerTensorProtocol): Optional keyword arguments.

        Returns:
            tuple[int, ...]: Eigenvalues output shape (..., N).
        """
        inp = args[0] if args else kwargs.get("input")
        if inp is None:
            return ()
        shape = tuple(int(d) for d in getattr(inp, "shape", getattr(inp, "shape_metadata", inp if isinstance(inp, (list, tuple)) else ())))
        return shape[:-1] if len(shape) >= 1 else ()


def eigh(input: Tensor, UPLO: str = "L") -> TensorOrScalar:
    """Compute the eigenvalues and eigenvectors of a complex Hermitian or real symmetric.

    Args:
        input (Tensor): The input parameter.
        UPLO (str): The UPLO parameter.

    Returns:
        tuple: Result.
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        w, v = typing.cast("tuple[TensorLike, TensorLike]", backend.execute_op("Eigh", (input.data if type(input).__name__ == "Tensor" else input), UPLO=UPLO))
        return (
            Tensor(w, TensorConfig(w.shape, getattr(input, "dtype", None), getattr(input, "device", None))),
            Tensor(v, TensorConfig(v.shape, getattr(input, "dtype", None), getattr(input, "device", None))),
        )
    return _emit_linalg_node("Eigh", [input], {"UPLO": UPLO}, [input.shape[:-1], input.shape], [getattr(input, "dtype", None)] * 2)


def eigvalsh(input: Tensor, UPLO: str = "L") -> TensorOrScalar:
    """Compute the eigenvalues of a complex Hermitian or real symmetric matrix.

    Args:
        input (Tensor): The input parameter.
        UPLO (str): The UPLO parameter.

    Returns:
        Tensor: Result.
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op("Eigvalsh", (input.data if type(input).__name__ == "Tensor" else input), UPLO=UPLO)
        arr = typing.cast("TensorLike", backend.array(data))
        return Tensor(
            arr,
            TensorConfig(arr.shape, getattr(input, "dtype", None), getattr(input, "device", None)),
        )
    return _emit_linalg_node("Eigvalsh", [input], {"UPLO": UPLO}, [input.shape[:-1]], [getattr(input, "dtype", None)])


@register_op("Eigvals")
class Eigvals(OpDef):
    """Eigvals Operation Definition."""

    def infer_shape(self, *args: EagerTensorProtocol, **kwargs: TensorOrScalar) -> tuple[int, ...]:
        """Infer eigenvalues output shape (..., N).

        Args:
            *args (EagerTensorProtocol): Input tensor or shape arguments.
            **kwargs (EagerTensorProtocol): Optional keyword arguments.

        Returns:
            tuple[int, ...]: Eigenvalues output shape (..., N).
        """
        inp = args[0] if args else kwargs.get("input")
        if inp is None:
            return ()
        shape = tuple(int(d) for d in getattr(inp, "shape", getattr(inp, "shape_metadata", inp if isinstance(inp, (list, tuple)) else ())))
        return shape[:-1] if len(shape) >= 1 else ()


def eigvals(input: Tensor) -> TensorOrScalar:
    """Compute the eigenvalues of a general matrix.

    Args:
        input (Tensor): The input parameter.

    Returns:
        Tensor: Result.
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op("Eigvals", (input.data if type(input).__name__ == "Tensor" else input))
        arr = typing.cast("TensorLike", backend.array(data))
        return Tensor(
            arr,
            TensorConfig(arr.shape, getattr(input, "dtype", None), getattr(input, "device", None)),
        )
    return _emit_linalg_node("Eigvals", [input], {}, [input.shape[:-1]], [getattr(input, "dtype", None)])


def eig(input: Tensor) -> TensorOrScalar:
    """Compute the eigenvalues and eigenvectors of a square matrix.

    Args:
        input (Tensor): The input parameter.

    Returns:
        tuple: Result.
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        w, v = typing.cast("tuple[TensorLike, TensorLike]", backend.execute_op("Eig", (input.data if type(input).__name__ == "Tensor" else input)))
        return (
            Tensor(w, TensorConfig(w.shape, getattr(input, "dtype", None), getattr(input, "device", None))),
            Tensor(v, TensorConfig(v.shape, getattr(input, "dtype", None), getattr(input, "device", None))),
        )
    return _emit_linalg_node("Eig", [input], {}, [input.shape[:-1], input.shape], [getattr(input, "dtype", None)] * 2)
