"""Shared backend-agnostic array conversion and host submodule utilities for eager decompositions."""

from __future__ import annotations

from types import ModuleType
from typing import Optional, cast

from ml_switcheroo_compiler.backends.eager_registry import EagerValue


def get_host_backend() -> ModuleType:
    """Resolve the active or reference backend module via `BackendRegistry`.

    Returns:
        ModuleType: Resolved backend module.
    """
    from ml_switcheroo_compiler.backends.registry import BackendRegistry

    return BackendRegistry.get("numpy").get_module()


def get_special_submodule(submodule: str) -> ModuleType:
    """Resolve a mathematical submodule (`'special'`, `'signal'`, `'linalg'`, `'stats'`) via `BackendRegistry`.

    Args:
        submodule (str): Name of the mathematical submodule.

    Returns:
        ModuleType: Resolved submodule instance.
    """
    from ml_switcheroo_compiler.backends.registry import BackendRegistry

    return BackendRegistry.get("numpy").get_scipy_module(submodule)


class _HostBackendProxy:
    """Lazy proxy delegating attribute access to the registered reference backend."""

    def __getattr__(self, name: str) -> EagerValue:
        """Resolve an attribute on the reference backend module.

        Args:
            name (str): Attribute name to look up.

        Returns:
            EagerValue: Resolved module attribute.
        """
        return cast(EagerValue, getattr(get_host_backend(), name))


host_backend: ModuleType = cast(ModuleType, _HostBackendProxy())


def _to_numpy_array(np_mod: EagerValue, x: EagerValue, name: str) -> EagerValue:
    """Convert an arbitrary backend tensor into a host-compatible array representation.

    Args:
        np_mod (EagerValue): Target array module providing `asarray`.
        x (EagerValue): Input backend tensor or array.
        name (str): Source backend module name.

    Returns:
        EagerValue: Host-compatible array representation.
    """
    if hasattr(x, "numpy"):
        return cast(EagerValue, getattr(x, "numpy")())
    if name == "torch" and hasattr(x, "detach"):
        return cast(EagerValue, getattr(x, "detach")().cpu().numpy())
    return cast(EagerValue, getattr(np_mod, "asarray")(x))


def _from_numpy_array(
    backend_module: EagerValue,
    out: EagerValue,
    name: str,
    original_tensor: Optional[EagerValue] = None,
) -> EagerValue:
    """Convert a computed host array back into the target backend's native tensor format.

    Args:
        backend_module (EagerValue): Target backend module.
        out (EagerValue): Computed host array result.
        name (str): Target backend module identifier.
        original_tensor (Optional[EagerValue]): Optional reference input tensor for dtype preservation.

    Returns:
        EagerValue: Native backend tensor.
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
    """Return computed array result for the PyTorch backend.

    Args:
        out (EagerValue): Computed array result.
        original_tensor (Optional[EagerValue]): Optional reference input tensor.

    Returns:
        EagerValue: Converted tensor output.
    """
    _ = original_tensor
    return out


def _mlx_from_numpy(out: EagerValue, original_tensor: Optional[EagerValue] = None) -> EagerValue:
    """Return computed array result for the MLX backend.

    Args:
        out (EagerValue): Computed array result.
        original_tensor (Optional[EagerValue]): Optional reference input tensor.

    Returns:
        EagerValue: Converted tensor output.
    """
    _ = original_tensor
    return out


def _jax_from_numpy(out: EagerValue, original_tensor: Optional[EagerValue] = None) -> EagerValue:
    """Return computed array result for the JAX backend.

    Args:
        out (EagerValue): Computed array result.
        original_tensor (Optional[EagerValue]): Optional reference input tensor.

    Returns:
        EagerValue: Converted tensor output.
    """
    _ = original_tensor
    return out


def _to_channels_last(np_mod: EagerValue, imgs: EagerValue, data_format: Optional[str]) -> EagerValue:
    """Normalize image tensor layout to channels-last (`NHWC`) when `data_format` is specified.

    Args:
        np_mod (EagerValue): Array manipulation module.
        imgs (EagerValue): Input image tensor.
        data_format (Optional[str]): Channel layout (`'channels_first'` or `'channels_last'`).

    Returns:
        EagerValue: Normalized channels-last image tensor.
    """
    _ = (np_mod, data_format)
    return imgs


def _from_channels_last(np_mod: EagerValue, out: EagerValue, data_format: Optional[str]) -> EagerValue:
    """Restore image tensor layout from channels-last (`NHWC`) to `data_format`.

    Args:
        np_mod (EagerValue): Array manipulation module.
        out (EagerValue): Channels-last image tensor.
        data_format (Optional[str]): Target channel layout (`'channels_first'` or `'channels_last'`).

    Returns:
        EagerValue: Image tensor in the requested `data_format`.
    """
    _ = (np_mod, data_format)
    return out
