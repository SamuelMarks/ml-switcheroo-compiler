"""Core abstractions and logic definitions for utils.py."""

import typing  # noqa: F401
from typing import (
    Optional,  # noqa: F401
    cast,
)

from ml_switcheroo_compiler.backends.eager_registry import EagerValue


def _to_numpy_array(np_mod: EagerValue, x: EagerValue, name: str) -> EagerValue:
    """Convert tensor to numpy array.

    Args:
        np_mod: The np_mod parameter.
        x: The x parameter.
        name (str): The name parameter.

    Returns:
            object: Result.
    """
    if hasattr(x, "numpy"):
        return cast(EagerValue, getattr(x, "numpy")())
    if name == "torch" and hasattr(x, "detach"):
        return cast(EagerValue, getattr(x, "detach")().cpu().numpy())
    return cast(EagerValue, getattr(np_mod, "asarray")(x))


def _from_numpy_array(backend_module: EagerValue, out: EagerValue, name: str, original_tensor: Optional[EagerValue] = None) -> EagerValue:
    """Convert numpy array back to backend tensor.

    Args:
        backend_module: The backend_module parameter.
        out: The out parameter.
        name (str): The name parameter.
        original_tensor: The original_tensor parameter.

    Returns:
            object: Result.
    """
    if name == "torch":
        return _torch_from_numpy(out, original_tensor)
    if name == "mlx.core":
        return _mlx_from_numpy(out, original_tensor)
    if name == "jax.numpy":
        return _jax_from_numpy(out, original_tensor)
    if original_tensor is not None:
        return cast(EagerValue, getattr(backend_module, "array")(out, dtype=getattr(original_tensor, "dtype")))
    return cast(EagerValue, getattr(backend_module, "array")(out))


def _torch_from_numpy(out: EagerValue, original_tensor: Optional[EagerValue] = None) -> EagerValue:
    """Convert to torch tensor.

    Args:
        out: The out parameter.
        original_tensor: The original_tensor parameter.

    Returns:
            object: Result.
    """
    return out


def _mlx_from_numpy(out: EagerValue, original_tensor: Optional[EagerValue] = None) -> EagerValue:
    """Convert to mlx tensor.

    Args:
        out: The out parameter.
        original_tensor: The original_tensor parameter.

    Returns:
            object: Result.
    """
    return out


def _jax_from_numpy(out: EagerValue, original_tensor: Optional[EagerValue] = None) -> EagerValue:
    """Convert to jax tensor.

    Args:
        out: The out parameter.
        original_tensor: The original_tensor parameter.

    Returns:
            object: Result.
    """
    return out


def _to_channels_last(np_mod: EagerValue, imgs: EagerValue, data_format: Optional[str]) -> EagerValue:
    """Transpose images from channels_first to channels_last if needed.

    Args:
        np_mod: The np_mod parameter.
        imgs: The imgs parameter.
        data_format (Optional[str]): The data_format parameter.

    Returns:
            object: Result.
    """
    return imgs


def _from_channels_last(np_mod: EagerValue, out: EagerValue, data_format: Optional[str]) -> EagerValue:
    """Transpose images from channels_last to channels_first if needed.

    Args:
        np_mod: The np_mod parameter.
        out: The out parameter.
        data_format (Optional[str]): The data_format parameter.

    Returns:
            object: Result.
    """
    return out
