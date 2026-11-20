"""Generic backend-parametric eager reference operations for `vision_ops.filtering`."""

from __future__ import annotations

import types
import typing

from ml_switcheroo_compiler.backends.eager import median_filter_eager
from ml_switcheroo_compiler.backends.eager.signal import gaussian_blur_eager
from ml_switcheroo_compiler.backends.eager.utils import _resolve_eager_mod
from ml_switcheroo_compiler.backends.eager_registry import (
    BackendArg,
    EagerTensorProtocol,
    coerce_opt_str,
    global_eager_registry,
)
from ml_switcheroo_compiler.backends.protocols import (
    BackendModuleProtocol,
)

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.ops.configs import BlurConfig


def _ensure_arr(val: BackendArg) -> EagerTensorProtocol:
    """Coerce `val` to an eager array via the active backend module.

    Args:
        val (BackendArg): Input array or scalar value.

    Returns:
        EagerTensorProtocol: Coerced backend array.
    """
    return _resolve_eager_mod(None).asarray(val)


@global_eager_registry.register("Degeneration")
def _np_degeneration(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    images: BackendArg,
    **kwargs: float | int | str | bool | None,
) -> EagerTensorProtocol:
    """Execute the eager np degeneration operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): The backend module.
        images (BackendArg): Input `images` argument (images).
        **kwargs (Union[float, int, str, bool, None]): Keyword args.

    Returns:
        EagerTensorProtocol: Computed `EagerTensorProtocol` output.
    """
    return _ensure_arr(images)


@global_eager_registry.register("GaussianBlur")
def _np_gaussian_blur(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    images: BackendArg,
    **kwargs: BackendArg | BlurConfig,
) -> EagerTensorProtocol:
    """Execute the eager np gaussian blur operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Active NumPy backend module.
        images (BackendArg): Input image array.
        **kwargs (BackendArg | BlurConfig): Blur configuration or keyword parameters.

    Returns:
        EagerTensorProtocol: Computed `EagerTensorProtocol` output.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    from ml_switcheroo_compiler.backends.eager_registry import coerce_float, coerce_shape
    from ml_switcheroo_compiler.ops.configs import BlurConfig

    raw_cfg: BackendArg | BlurConfig = kwargs.get("config", kwargs)
    if isinstance(raw_cfg, BlurConfig):
        blur_cfg = raw_cfg
    elif isinstance(raw_cfg, dict):
        raw_ks = raw_cfg.get("kernel_size", (3, 3))
        ks_val: int | tuple[int, int] = (
            raw_ks if isinstance(raw_ks, int) else (coerce_shape(raw_ks)[0], coerce_shape(raw_ks)[1])
        )
        raw_sig = raw_cfg.get("sigma", (1.0, 1.0))
        if isinstance(raw_sig, (int, float)):
            sig_val: float | tuple[float, float] = float(raw_sig)
        elif isinstance(raw_sig, (list, tuple)) and len(raw_sig) >= 2:
            sig_val = (coerce_float(raw_sig[0]), coerce_float(raw_sig[1]))
        else:
            sig_val = 1.0
        blur_cfg = BlurConfig(
            kernel_size=ks_val,
            sigma=sig_val,
            data_format=coerce_opt_str(raw_cfg.get("data_format", None)),
        )
    else:
        blur_cfg = BlurConfig(kernel_size=3, sigma=1.0)
    return _ensure_arr(arr_mod.asarray(gaussian_blur_eager(backend_module, images, blur_cfg)))


@global_eager_registry.register("MedianFilter")
def _np_median_filter(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    images: BackendArg,
    **kwargs: float | int | str | bool | None,
) -> EagerTensorProtocol:
    """Execute the eager np median filter operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): The backend module.
        images (BackendArg): Input `images` argument (images).
        **kwargs (Union[float, int, str, bool, None]): Keyword args.

    Returns:
        EagerTensorProtocol: Computed `EagerTensorProtocol` output.
    """
    from ml_switcheroo_compiler.backends.eager_registry import coerce_int

    arr_mod = _resolve_eager_mod(backend_module)
    raw_k = kwargs.get("kernel_size", (3, 3))
    if isinstance(raw_k, int):
        k_pair: tuple[int, int] = (raw_k, raw_k)
    elif isinstance(raw_k, (list, tuple)) and len(raw_k) >= 2:
        k_pair = (int(coerce_int(raw_k[0])), int(coerce_int(raw_k[1])))
    else:
        kv = int(coerce_int(raw_k))
        k_pair = (kv, kv)
    df = coerce_opt_str(kwargs.get("data_format", None))
    pad_mode = coerce_opt_str(kwargs.get("padding", None)) or "same"
    return _ensure_arr(
        arr_mod.asarray(
            median_filter_eager(backend_module, images, kernel_size=k_pair, padding=pad_mode, data_format=df)
        )
    )


@global_eager_registry.register("Sharpen")
def _np_sharpen(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    images: BackendArg,
    **kwargs: float | int | str | bool | None,
) -> EagerTensorProtocol:
    """Execute the eager np sharpen operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): The backend module.
        images (BackendArg): Input `images` argument (images).
        **kwargs (Union[float, int, str, bool, None]): Keyword args.

    Returns:
        EagerTensorProtocol: Computed `EagerTensorProtocol` output.
    """
    return _ensure_arr(images)


@global_eager_registry.register("RandomGaussianBlur")
def _np_random_gaussian_blur(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    images: BackendArg,
    kernel_size: tuple[int, int] | int,
    sigma: tuple[float, float] | float,
    **kwargs: float | int | str | None,
) -> EagerTensorProtocol:
    """Execute the eager np random gaussian blur operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Active backend module.
        images (BackendArg): Input images array.
        kernel_size (tuple[int, int] | int): Blur kernel size.
        sigma (tuple[float, float] | float): Blur standard deviation.
        **kwargs (Union[float, int, str, None]): Keyword args.

    Returns:
        EagerTensorProtocol: Blurred images array.
    """
    return _np_gaussian_blur(backend_module, images, kernel_size=kernel_size, sigma=sigma, **kwargs)


@global_eager_registry.register("RandomSharpness")
def _np_random_sharpness(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    images: BackendArg,
    factor: float,
    **kwargs: float | int | str | None,
) -> EagerTensorProtocol:
    """Execute the eager np random sharpness operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Active backend module.
        images (BackendArg): Input images array.
        factor (float): Sharpness adjustment factor.
        **kwargs (Union[float, int, str, None]): Keyword args.

    Returns:
        EagerTensorProtocol: Sharpened images array.
    """
    return _np_sharpen(backend_module, images, factor=factor, **kwargs)


__all__ = [
    "_np_degeneration",
    "_np_gaussian_blur",
    "_np_median_filter",
    "_np_random_gaussian_blur",
    "_np_random_sharpness",
    "_np_sharpen",
    "global_eager_registry",
]
