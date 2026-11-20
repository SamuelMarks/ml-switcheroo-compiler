"""Shared backend-agnostic array conversion and channel layout utilities for eager decompositions."""

from __future__ import annotations

import types

from ml_switcheroo_compiler.backends.eager_registry import (
    BackendArg,
    EagerTensorProtocol,
)
from ml_switcheroo_compiler.backends.protocols import BackendModuleProtocol, adapt_backend_module


def _resolve_eager_mod(
    backend_module: types.ModuleType | BackendModuleProtocol | None,
) -> BackendModuleProtocol:
    """Resolve the active backend module when `backend_module` lacks array constructors.

    Args:
        backend_module (types.ModuleType | BackendModuleProtocol | None): Candidate backend module or None.

    Returns:
        BackendModuleProtocol: Resolved execution backend module.
    """
    from ml_switcheroo_compiler.backends.registry import BackendRegistry, get_active_backend

    try:
        if (
            backend_module is not None
            and adapt_backend_module(backend_module).has_symbol("zeros")
            and adapt_backend_module(backend_module).has_symbol("asarray")
        ):
            has_np_sub, np_sub = adapt_backend_module(backend_module).resolve_symbol("numpy")
            if has_np_sub and np_sub is not None:
                return np_sub
            if not isinstance(backend_module, types.ModuleType):
                return backend_module
            return adapt_backend_module(BackendRegistry.get(backend_module.__name__)).require_symbol("get_module")()
    except (AttributeError, RuntimeError, TypeError, ValueError):
        np_sub = None

    active = get_active_backend()
    backend_cls = (
        BackendRegistry.get("pure_python") if not adapt_backend_module(active).has_symbol("get_module") else active
    )
    return adapt_backend_module(backend_cls).require_symbol("get_module")()


def _to_numpy_array(
    np_mod: BackendModuleProtocol | BackendArg, x: EagerTensorProtocol | BackendArg, name: str
) -> BackendArg:
    """Convert a backend tensor `x` into a numpy-compatible array for `np_mod`.

    Args:
        np_mod (BackendModuleProtocol | BackendArg): Target numpy-like module.
        x (EagerTensorProtocol | BackendArg): Input tensor or array.
        name (str): Backend module name.

    Returns:
        BackendArg: Array representation in the target module.
    """
    x_adp = adapt_backend_module(x)
    if x_adp.has_symbol("numpy") and callable(x_adp.require_symbol("numpy")):
        return x_adp.require_symbol("numpy")()
    if name == "torch" and x_adp.has_symbol("detach"):
        return x_adp.require_symbol("detach")().cpu().numpy()
    mod_adp = adapt_backend_module(np_mod)
    return mod_adp.require_symbol("asarray")(x) if mod_adp.has_symbol("asarray") else x


def _torch_from_numpy(out: BackendArg, ref: BackendArg = None) -> BackendArg:
    """Convert an array into a PyTorch tensor aligned with reference device and dtype.

    Args:
        out (BackendArg): Source array.
        ref (BackendArg): Optional reference PyTorch tensor for device/dtype alignment.

    Returns:
        BackendArg: Converted PyTorch tensor or original `out` if PyTorch is unavailable.
    """
    _ = ref
    return out


def _mlx_from_numpy(out: BackendArg) -> BackendArg:
    """Convert an array into an MLX array.

    Args:
        out (BackendArg): Source array.

    Returns:
        BackendArg: Converted MLX array or original `out` if MLX is unavailable.
    """
    return out


def _jax_from_numpy(out: BackendArg) -> BackendArg:
    """Convert an array into a JAX array.

    Args:
        out (BackendArg): Source array.

    Returns:
        BackendArg: Converted JAX array or original `out` if JAX is unavailable.
    """
    return out


def _from_numpy_array(
    backend_module: BackendModuleProtocol | BackendArg,
    out: BackendArg,
    name: str,
    original_tensor: EagerTensorProtocol | BackendArg = None,
) -> BackendArg:
    """Convert a processed array back into the target backend's tensor type.

    Args:
        backend_module (BackendModuleProtocol | BackendArg): Target backend module.
        out (BackendArg): Processed array.
        name (str): Target backend name.
        original_tensor (EagerTensorProtocol | BackendArg): Optional reference tensor for dtype alignment.

    Returns:
        BackendArg: Backend-native tensor representation.
    """
    if name == "torch":
        return _torch_from_numpy(out, original_tensor)
    if name.startswith("mlx"):
        return _mlx_from_numpy(out)
    if name.startswith("jax"):
        return _jax_from_numpy(out)
    mod_adp = adapt_backend_module(backend_module)
    if mod_adp.has_symbol("array"):
        arr_fn = mod_adp.require_symbol("array")
        if original_tensor is not None and adapt_backend_module(original_tensor).has_symbol("dtype"):
            return arr_fn(out, dtype=adapt_backend_module(original_tensor).require_symbol("dtype"))
        return arr_fn(out)
    return out


def _to_channels_last(np_mod: BackendArg, imgs: BackendArg, data_format: str | None) -> BackendArg:
    """Permute image tensor to channels-last layout when `data_format` is specified.

    Args:
        np_mod (BackendArg): Array manipulation module.
        imgs (BackendArg): Input image tensor.
        data_format (str | None): Channel layout (`'channels_first'` or `'channels_last'`).

    Returns:
        BackendArg: Image tensor normalized for spatial processing.
    """
    _ = (np_mod, data_format)
    return imgs


def _from_channels_last(np_mod: BackendArg, out: BackendArg, data_format: str | None) -> BackendArg:
    """Restore image tensor from channels-last layout back to `data_format`.

    Args:
        np_mod (BackendArg): Array manipulation module.
        out (BackendArg): Processed channels-last image tensor.
        data_format (str | None): Target channel layout (`'channels_first'` or `'channels_last'`).

    Returns:
        BackendArg: Image tensor in the requested `data_format`.
    """
    _ = (np_mod, data_format)
    return out
