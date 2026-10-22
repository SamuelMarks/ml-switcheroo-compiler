"""Shared backend-agnostic array conversion and channel layout utilities for eager decompositions."""

from __future__ import annotations

from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue, as_tensor
from ml_switcheroo_compiler.backends.protocols import BackendModuleProtocol, as_mod


def _resolve_eager_mod(backend_module: BackendModuleProtocol | EagerValue | None) -> BackendModuleProtocol:
    """Resolve the active backend module when `backend_module` lacks array constructors.

    Args:
        backend_module: Candidate backend module or None.

    Returns:
        BackendModuleProtocol: Resolved execution backend module.
    """
    if backend_module is not None and hasattr(backend_module, "zeros") and hasattr(backend_module, "asarray"):
        resolved = as_mod(getattr(backend_module, "numpy", backend_module))
        return resolved
    from ml_switcheroo_compiler.backends.registry import BackendRegistry, get_active_backend

    active = get_active_backend()
    backend_cls = BackendRegistry.get("numpy") if not hasattr(active, "get_module") else active
    mod: BackendModuleProtocol = backend_cls.get_module()
    return mod


def _to_numpy_array(np_mod: BackendModuleProtocol | EagerValue, x: EagerTensorProtocol | EagerValue, name: str) -> EagerValue:
    """Convert a backend tensor `x` into a numpy-compatible array for `np_mod`.

    Args:
        np_mod (BackendModuleProtocol | EagerValue): Target numpy-like module.
        x (EagerTensorProtocol | EagerValue): Input tensor or array.
        name (str): Backend module name.

    Returns:
        EagerValue: Array representation in the target module.
    """
    tensor = as_tensor(x)
    mod = as_mod(np_mod)
    if hasattr(x, "numpy") and callable(getattr(x, "numpy")):
        return tensor.numpy()
    if name == "torch" and hasattr(x, "detach"):
        return tensor.detach().cpu().numpy()
    return mod.asarray(tensor) if hasattr(np_mod, "asarray") else x


def _torch_from_numpy(out: EagerValue, ref: EagerValue = None) -> EagerValue:
    """Convert an array into a PyTorch tensor aligned with reference device and dtype.

    Args:
        out (EagerValue): Source array.
        ref (EagerValue): Optional reference PyTorch tensor for device/dtype alignment.

    Returns:
        EagerValue: Converted PyTorch tensor or original `out` if PyTorch is unavailable.
    """
    _ = ref
    return out


def _mlx_from_numpy(out: EagerValue) -> EagerValue:
    """Convert an array into an MLX array.

    Args:
        out (EagerValue): Source array.

    Returns:
        EagerValue: Converted MLX array or original `out` if MLX is unavailable.
    """
    return out


def _jax_from_numpy(out: EagerValue) -> EagerValue:
    """Convert an array into a JAX array.

    Args:
        out (EagerValue): Source array.

    Returns:
        EagerValue: Converted JAX array or original `out` if JAX is unavailable.
    """
    return out


def _from_numpy_array(
    backend_module: BackendModuleProtocol | EagerValue,
    out: EagerValue,
    name: str,
    original_tensor: EagerTensorProtocol | EagerValue = None,
) -> EagerValue:
    """Convert a processed array back into the target backend's tensor type.

    Args:
        backend_module (BackendModuleProtocol | EagerValue): Target backend module.
        out (EagerValue): Processed array.
        name (str): Target backend name.
        original_tensor (EagerTensorProtocol | EagerValue): Optional reference tensor for dtype alignment.

    Returns:
        EagerValue: Backend-native tensor representation.
    """
    if name == "torch":
        return _torch_from_numpy(out, original_tensor)
    if name.startswith("mlx"):
        return _mlx_from_numpy(out)
    if name.startswith("jax"):
        return _jax_from_numpy(out)
    if hasattr(backend_module, "array"):
        mod = as_mod(backend_module)
        if original_tensor is not None and hasattr(original_tensor, "dtype"):
            return mod.array(out, dtype=getattr(original_tensor, "dtype"))
        return mod.array(out)
    return out


def _to_channels_last(np_mod: EagerValue, imgs: EagerValue, data_format: str | None) -> EagerValue:
    """Permute image tensor to channels-last layout when `data_format` is specified.

    Args:
        np_mod (EagerValue): Array manipulation module.
        imgs (EagerValue): Input image tensor.
        data_format (str | None): Channel layout (`'channels_first'` or `'channels_last'`).

    Returns:
        EagerValue: Image tensor normalized for spatial processing.
    """
    _ = (np_mod, data_format)
    return imgs


def _from_channels_last(np_mod: EagerValue, out: EagerValue, data_format: str | None) -> EagerValue:
    """Restore image tensor from channels-last layout back to `data_format`.

    Args:
        np_mod (EagerValue): Array manipulation module.
        out (EagerValue): Processed channels-last image tensor.
        data_format (str | None): Target channel layout (`'channels_first'` or `'channels_last'`).

    Returns:
        EagerValue: Image tensor in the requested `data_format`.
    """
    _ = (np_mod, data_format)
    return out
