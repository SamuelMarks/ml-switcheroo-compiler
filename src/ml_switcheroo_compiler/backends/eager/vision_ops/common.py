"""Generic backend-parametric eager reference operations for `vision_ops.common`."""

from __future__ import annotations

import types

from ml_switcheroo_compiler.backends.eager.audio import mel_filterbank_eager, mfcc_eager
from ml_switcheroo_compiler.backends.eager.utils import _resolve_eager_mod
from ml_switcheroo_compiler.backends.eager_registry import (
    BackendArg,
    EagerTensorProtocol,
    coerce_int,
    global_eager_registry,
)
from ml_switcheroo_compiler.backends.protocols import BackendModuleProtocol, adapt_backend_module


def _ensure_arr(val: BackendArg) -> EagerTensorProtocol:
    """Coerce `val` to an eager array via the active backend module.

    Args:
        val (BackendArg): Input array or scalar value.

    Returns:
        EagerTensorProtocol: Coerced backend array.
    """
    return _resolve_eager_mod(None).asarray(val)


@global_eager_registry.register("MelFilterbank")
def _np_mel_filterbank(
    backend_module: BackendModuleProtocol | types.ModuleType | None, _: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np mel filterbank operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        _ (BackendArg): Input `_` argument ( ).
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    from ml_switcheroo_compiler.backends.eager.audio import MelFilterbankConfig
    from ml_switcheroo_compiler.backends.eager_registry import coerce_float

    raw_cfg = kwargs.get("config", kwargs)
    if isinstance(raw_cfg, MelFilterbankConfig):
        cfg: MelFilterbankConfig | dict[str, int | float] = raw_cfg
    elif isinstance(raw_cfg, dict):
        cfg = {str(k): (v if isinstance(v, (int, float)) else coerce_float(v)) for k, v in raw_cfg.items()}
    else:
        cfg = MelFilterbankConfig()
    return mel_filterbank_eager(backend_module, None, cfg)


@global_eager_registry.register("Mfcc")
def _np_mfcc(
    backend_module: BackendModuleProtocol | types.ModuleType | None, spectrogram: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np mfcc operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        spectrogram (BackendArg): Input `spectrogram` argument (spectrogram).
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    from ml_switcheroo_compiler.backends.eager.audio import MFCCConfig
    from ml_switcheroo_compiler.backends.eager_registry import coerce_float

    raw_cfg = kwargs.get("config", kwargs)
    if isinstance(raw_cfg, MFCCConfig):
        cfg: MFCCConfig | dict[str, int | float] = raw_cfg
    elif isinstance(raw_cfg, dict):
        cfg = {str(k): (v if isinstance(v, (int, float)) else coerce_float(v)) for k, v in raw_cfg.items()}
    else:
        cfg = MFCCConfig()
    return mfcc_eager(backend_module, spectrogram, cfg)


@global_eager_registry.register("PowerIteration")
def _np_power_iteration(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    w: BackendArg,
    *args: BackendArg,
    **kwargs: BackendArg,
) -> BackendArg:
    """Execute the eager np power iteration operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        w (BackendArg): Input `w` argument (w).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    num_iters = kwargs.get("num_iters", 1)
    u_raw = kwargs.get("u", None)
    if u_raw is None:
        u = _ensure_arr(
            arr_mod.ones(
                adapt_backend_module(w).require_symbol("shape")[:-2]
                + (adapt_backend_module(w).require_symbol("shape")[-2], 1),
                dtype=adapt_backend_module(w).require_symbol("dtype"),
            )
        )
    elif adapt_backend_module(u_raw).get_symbol("ndim", 1) < adapt_backend_module(w).get_symbol("ndim", 2):
        u = _ensure_arr(arr_mod.asarray(adapt_backend_module(arr_mod).require_symbol("expand_dims")(u_raw, axis=-1)))
    else:
        u = _ensure_arr(arr_mod.asarray(u_raw))
    v = _ensure_arr(arr_mod.zeros_like(u))
    for _ in range(coerce_int(num_iters)):
        w_t = _ensure_arr(arr_mod.asarray(adapt_backend_module(arr_mod).require_symbol("swapaxes")(w, -1, -2)))
        v = _ensure_arr(arr_mod.matmul(w_t, u))
        v = _ensure_arr(v / (arr_mod.linalg.norm(v, axis=-2, keepdims=True) + 1e-12))
        u = _ensure_arr(arr_mod.asarray(adapt_backend_module(arr_mod).require_symbol("matmul")(w, v)))
        u = _ensure_arr(u / (arr_mod.linalg.norm(u, axis=-2, keepdims=True) + 1e-12))
    sigma = arr_mod.asarray(
        adapt_backend_module(arr_mod).require_symbol("matmul")(
            arr_mod.swapaxes(u, -1, -2), adapt_backend_module(arr_mod).require_symbol("matmul")(w, v)
        )
    )
    return (
        arr_mod.squeeze(v, axis=-1),
        arr_mod.squeeze(u, axis=-1),
        arr_mod.squeeze(arr_mod.squeeze(sigma, axis=-1), axis=-1),
    )


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
    "global_eager_registry",
]
