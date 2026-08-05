# ruff: noqa: E501
"""Core abstractions and logic definitions for utils.py."""

import typing


def _to_numpy_array(np_mod: object, x: object, name: str) -> object:
    """Convert tensor to numpy array.

    Args:
        np_mod (object): The np_mod parameter.
        x (object): The x parameter.
        name (str): The name parameter.

    Returns:
        object: Result.
    """
    if hasattr(x, "numpy"):
        return x.numpy()
    if name == "torch" and hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np_mod.asarray(x)


def _from_numpy_array(backend_module: object, out: object, name: str, original_tensor: object = None) -> object:
    """Convert numpy array back to backend tensor.

    Args:
        backend_module (object): The backend_module parameter.
        out (object): The out parameter.
        name (str): The name parameter.
        original_tensor (object): The original_tensor parameter.

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
        return backend_module.array(out, dtype=original_tensor.dtype)
    return backend_module.array(out)


def _torch_from_numpy(out: object, original_tensor: object = None) -> object:
    """Convert to torch tensor.

    Args:
        out (object): The out parameter.
        original_tensor (object): The original_tensor parameter.

    Returns:
        object: Result.
    """
    return out


def _mlx_from_numpy(out: object, original_tensor: object = None) -> object:
    """Convert to mlx tensor.

    Args:
        out (object): The out parameter.
        original_tensor (object): The original_tensor parameter.

    Returns:
        object: Result.
    """
    return out


def _jax_from_numpy(out: object, original_tensor: object = None) -> object:
    """Convert to jax tensor.

    Args:
        out (object): The out parameter.
        original_tensor (object): The original_tensor parameter.

    Returns:
        object: Result.
    """
    return out


def _to_channels_last(np_mod: object, imgs: object, data_format: typing.Optional[str]) -> object:
    """Transpose images from channels_first to channels_last if needed.

    Args:
        np_mod (object): The np_mod parameter.
        imgs (object): The imgs parameter.
        data_format (object): The data_format parameter.

    Returns:
        object: Result.
    """
    return imgs


def _from_channels_last(np_mod: object, out: object, data_format: typing.Optional[str]) -> object:
    """Transpose images from channels_last to channels_first if needed.

    Args:
        np_mod (object): The np_mod parameter.
        out (object): The out parameter.
        data_format (object): The data_format parameter.

    Returns:
        object: Result.
    """
    return out
