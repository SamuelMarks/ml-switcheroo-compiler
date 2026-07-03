"""Module docstring."""

import importlib
import typing

from ml_switcheroo_compiler.core.constants import MAGIC_VAL_3, MAGIC_VAL_4


def _to_numpy_array(np_mod: object, x: object, name: str) -> object:
    """Convert tensor to numpy array."""
    if name == "torch":  # pragma: no branch
        return x.detach().cpu().numpy()  # pragma: no cover
    if name == "mlx.core":  # pragma: no branch
        return np_mod.array(x)  # pragma: no cover
    if hasattr(x, "numpy"):  # pragma: no branch
        return x.numpy()  # pragma: no cover
    return np_mod.asarray(x)


def _from_numpy_array(backend_module: object, out: object, name: str, original_tensor: object = None) -> object:
    """Convert numpy array back to backend tensor."""
    if name == "torch":  # pragma: no branch
        return _torch_from_numpy(out, original_tensor)  # pragma: no cover
    if name == "mlx.core":  # pragma: no branch
        return _mlx_from_numpy(out, original_tensor)  # pragma: no cover
    if name == "jax.numpy":  # pragma: no branch
        return _jax_from_numpy(out, original_tensor)  # pragma: no cover

    if original_tensor is not None:
        return backend_module.array(out, dtype=original_tensor.dtype)
    return backend_module.array(out)  # pragma: no cover


def _torch_from_numpy(out: object, original_tensor: object = None) -> object:
    """Convert to torch tensor.

    Args:
        out (object): The output.
        original_tensor (object): The original tensor.

    Returns:
        object: The result.
    """
    torch = importlib.import_module("torch")  # pragma: no cover
    if original_tensor is not None:  # pragma: no cover
        return torch.tensor(out, dtype=original_tensor.dtype, device=original_tensor.device)  # pragma: no cover
    return torch.tensor(out)  # pragma: no cover


def _mlx_from_numpy(out: object, original_tensor: object = None) -> object:
    """Convert to mlx tensor.

    Args:
        out (object): The output.
        original_tensor (object): The original tensor.

    Returns:
        object: The result.
    """
    mlx_core = importlib.import_module("mlx.core")  # pragma: no cover
    if original_tensor is not None:  # pragma: no cover
        return mlx_core.array(out, dtype=original_tensor.dtype)  # pragma: no cover
    return mlx_core.array(out)  # pragma: no cover


def _jax_from_numpy(out: object, original_tensor: object = None) -> object:
    """Convert to jax tensor.

    Args:
        out (object): The output.
        original_tensor (object): The original tensor.

    Returns:
        object: The result.
    """
    jnp = importlib.import_module("jax.numpy")  # pragma: no cover
    if original_tensor is not None:  # pragma: no cover
        return jnp.array(out, dtype=original_tensor.dtype)  # pragma: no cover
    return jnp.array(out)  # pragma: no cover


def _to_channels_last(np_mod: object, imgs: object, data_format: typing.Optional[str]) -> object:
    """Transpose images from channels_first to channels_last if needed."""
    if data_format == "channels_first" and imgs.ndim >= MAGIC_VAL_3:  # pragma: no branch
        if imgs.ndim == MAGIC_VAL_4:  # pragma: no cover
            return np_mod.transpose(imgs, (0, 2, 3, 1))  # pragma: no cover
        elif imgs.ndim == MAGIC_VAL_3:  # pragma: no cover
            return np_mod.transpose(imgs, (1, 2, 0))  # pragma: no cover
    return imgs


def _from_channels_last(np_mod: object, out: object, data_format: typing.Optional[str]) -> object:
    """Transpose images from channels_last to channels_first if needed."""
    if data_format == "channels_first" and out.ndim >= MAGIC_VAL_3:  # pragma: no branch
        if out.ndim == MAGIC_VAL_4:  # pragma: no cover
            return np_mod.transpose(out, (0, 3, 1, 2))  # pragma: no cover
        elif out.ndim == MAGIC_VAL_3:  # pragma: no cover
            return np_mod.transpose(out, (2, 0, 1))  # pragma: no cover
    return out
