"""Vision eager common operations."""

from __future__ import annotations

import numpy as np

from ml_switcheroo_compiler.backends.eager.audio import mel_filterbank_eager, mfcc_eager
from ml_switcheroo_compiler.backends.eager_registry import EagerValue, ensure_callable, ensure_int, numpy_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager import ensure_ndarray
from ml_switcheroo_compiler.backends.protocols import BackendModuleProtocol, get_backend_member


@numpy_eager_registry.register("MelFilterbank")
def _np_mel_filterbank(backend_module: BackendModuleProtocol, _: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np mel filterbank operation on the active backend array.

    Args:
        backend_module (ModuleType): Input `backend_module` argument (backend module).
        _ (EagerValue): Input `_` argument ( ).
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    return ensure_callable(mel_filterbank_eager)(backend_module, None, kwargs.get("config", kwargs))


@numpy_eager_registry.register("Mfcc")
def _np_mfcc(backend_module: BackendModuleProtocol, spectrogram: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np mfcc operation on the active backend array.

    Args:
        backend_module (ModuleType): Input `backend_module` argument (backend module).
        spectrogram (EagerValue): Input `spectrogram` argument (spectrogram).
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    return ensure_callable(mfcc_eager)(backend_module, spectrogram, kwargs.get("config", kwargs))


@numpy_eager_registry.register("PowerIteration")
def _np_power_iteration(backend_module: BackendModuleProtocol, w: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np power iteration operation on the active backend array.

    Args:
        backend_module (ModuleType): Input `backend_module` argument (backend module).
        w (EagerValue): Input `w` argument (w).
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    num_iters = kwargs.get("num_iters", 1)
    u_raw = kwargs.get("u", None)
    if u_raw is None:
        u = ensure_ndarray(np.ones(get_backend_member(w, "shape")[:-2] + (get_backend_member(w, "shape")[-2], 1), dtype=get_backend_member(w, "dtype")))
    elif get_backend_member(u_raw, "ndim", 1) < get_backend_member(w, "ndim", 2):
        u = ensure_ndarray(np.asarray(get_backend_member(np, "expand_dims")(u_raw, axis=-1)))
    else:
        u = ensure_ndarray(np.asarray(u_raw))
    v = ensure_ndarray(np.zeros_like(u))
    for _ in range(ensure_int(num_iters)):
        w_t = ensure_ndarray(np.asarray(get_backend_member(np, "swapaxes")(w, -1, -2)))
        v = ensure_ndarray(np.matmul(w_t, u))
        v = ensure_ndarray(v / (np.linalg.norm(v, axis=-2, keepdims=True) + 1e-12))
        u = ensure_ndarray(np.asarray(get_backend_member(np, "matmul")(w, v)))
        u = ensure_ndarray(u / (np.linalg.norm(u, axis=-2, keepdims=True) + 1e-12))
    sigma = np.asarray(get_backend_member(np, "matmul")(np.swapaxes(u, -1, -2), get_backend_member(np, "matmul")(w, v)))
    return (np.squeeze(v, axis=-1), np.squeeze(u, axis=-1), np.squeeze(np.squeeze(sigma, axis=-1), axis=-1))


__all__ = [
    "__cached__",
    "__doc__",
    "__file__",
    "__loader__",
    "__name__",
    "__package__",
    "__spec__",
    "_np_mel_filterbank",
    "_np_mfcc",
    "_np_power_iteration",
    "ensure_ndarray",
    "np",
    "numpy_eager_registry",
]
