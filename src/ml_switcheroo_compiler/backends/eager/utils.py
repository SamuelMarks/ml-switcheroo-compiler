"""Module docstring."""

# ruff: noqa: E402, D100, D101
import typing
import importlib


def _to_numpy_array(np_mod: object, x: object, name: str) -> object:
    """Convert tensor to numpy array."""
    if name == "torch":
        return x.detach().cpu().numpy()
    if name == "mlx.core":
        return np_mod.array(x)
    if hasattr(x, "numpy"):
        return x.numpy()
    return np_mod.asarray(x)


def _from_numpy_array(
    backend_module: object, out: object, name: str, original_tensor: object = None
) -> object:
    """Convert numpy array back to backend tensor."""
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
        out (object): The output.
        original_tensor (object): The original tensor.

    Returns:
        object: The result.
    """
    torch = importlib.import_module("torch")
    if original_tensor is not None:
        return torch.tensor(out, dtype=original_tensor.dtype, device=original_tensor.device)
    return torch.tensor(out)


def _mlx_from_numpy(out: object, original_tensor: object = None) -> object:
    """Convert to mlx tensor.

    Args:
        out (object): The output.
        original_tensor (object): The original tensor.

    Returns:
        object: The result.
    """
    mlx_core = importlib.import_module("mlx.core")
    if original_tensor is not None:
        return mlx_core.array(out, dtype=original_tensor.dtype)
    return mlx_core.array(out)


def _jax_from_numpy(out: object, original_tensor: object = None) -> object:
    """Convert to jax tensor.

    Args:
        out (object): The output.
        original_tensor (object): The original tensor.

    Returns:
        object: The result.
    """
    jnp = importlib.import_module("jax.numpy")
    if original_tensor is not None:
        return jnp.array(out, dtype=original_tensor.dtype)
    return jnp.array(out)


def _to_channels_last(np_mod: object, imgs: object, data_format: typing.Optional[str]) -> object:
    """Transpose images from channels_first to channels_last if needed."""
    if data_format == "channels_first" and imgs.ndim >= 3:
        if imgs.ndim == 4:
            return np_mod.transpose(imgs, (0, 2, 3, 1))
        elif imgs.ndim == 3:
            return np_mod.transpose(imgs, (1, 2, 0))
    return imgs


def _from_channels_last(np_mod: object, out: object, data_format: typing.Optional[str]) -> object:
    """Transpose images from channels_last to channels_first if needed."""
    if data_format == "channels_first" and out.ndim >= 3:
        if out.ndim == 4:
            return np_mod.transpose(out, (0, 3, 1, 2))
        elif out.ndim == 3:
            return np_mod.transpose(out, (2, 0, 1))
    return out
