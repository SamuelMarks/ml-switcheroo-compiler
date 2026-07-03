"""Constants & Creation Operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ml_switcheroo_compiler.backends.registry import get_active_backend
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

from .frontend_utils import _emit_creation_node

if TYPE_CHECKING:
    pass


def blackman(M: int) -> Tensor:
    """Return the blackman window.

    Args:
        M (int): Number of points in the output window.

    Returns:
        Tensor: The window.
    """
    if config.eager_mode:
        data = get_active_backend().execute_op("Blackman", M)

        return Tensor(data, TensorConfig((M,), DType.Float32, None))
    return _emit_creation_node("Blackman", (M,), DType.Float32, {})


def bartlett(M: int) -> Tensor:
    """Return the bartlett window.

    Args:
        M (int): Number of points in the output window.

    Returns:
        Tensor: The window.
    """
    if config.eager_mode:
        data = get_active_backend().execute_op("Bartlett", M)

        return Tensor(data, TensorConfig((M,), DType.Float32, None))
    return _emit_creation_node("Bartlett", (M,), DType.Float32, {})


def hamming(M: int) -> Tensor:
    """Return the hamming window.

    Args:
        M (int): Number of points in the output window.

    Returns:
        Tensor: The window.
    """
    if config.eager_mode:
        data = get_active_backend().execute_op("Hamming", M)

        return Tensor(data, TensorConfig((M,), DType.Float32, None))
    return _emit_creation_node("Hamming", (M,), DType.Float32, {})


def hanning(M: int) -> Tensor:
    """Return the hanning window.

    Args:
        M (int): Number of points in the output window.

    Returns:
        Tensor: The window.
    """
    if config.eager_mode:
        data = get_active_backend().execute_op("Hanning", M)

        return Tensor(data, TensorConfig((M,), DType.Float32, None))
    return _emit_creation_node("Hanning", (M,), DType.Float32, {})


def kaiser(M: int, beta: float) -> Tensor:
    """Return the Kaiser window.

    Args:
        M (int): Number of points in the output window.
        beta (float): Shape parameter.

    Returns:
        Tensor: The window.
    """
    if config.eager_mode:
        data = get_active_backend().execute_op("Kaiser", M, beta)

        return Tensor(data, TensorConfig((M,), DType.Float32, None))
    return _emit_creation_node("Kaiser", (M,), DType.Float32, {"beta": beta})
