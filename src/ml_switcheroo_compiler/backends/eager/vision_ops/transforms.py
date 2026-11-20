"""Generic backend-parametric eager reference operations for `vision_ops.transforms`."""

from __future__ import annotations

import types

from ml_switcheroo_compiler.backends.eager.utils import _resolve_eager_mod
from ml_switcheroo_compiler.backends.eager.vision_augmentation import (
    random_elastic_transform_eager,
    random_perspective_eager,
    random_shear_eager,
)
from ml_switcheroo_compiler.backends.eager_registry import (
    BackendArg,
    EagerTensorProtocol,
    global_eager_registry,
)
from ml_switcheroo_compiler.backends.protocols import (
    BackendModuleProtocol,
)


def _ensure_arr(val: BackendArg) -> EagerTensorProtocol:
    """Coerce `val` to an eager array via the active backend module.

    Args:
        val (BackendArg): Input array or scalar value.

    Returns:
        EagerTensorProtocol: Coerced backend array.
    """
    return _resolve_eager_mod(None).asarray(val)


@global_eager_registry.register("RandomShear")
def _np_random_shear(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    images: BackendArg,
    y_factor: tuple[float, float] | float,
    x_factor: tuple[float, float] | float | None = None,
    **kwargs: float | int | str | None,
) -> EagerTensorProtocol:
    """Execute the eager np random shear operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Active backend module.
        images (BackendArg): Input images array.
        y_factor (tuple[float, float] | float): Vertical shear factor.
        x_factor (tuple[float, float] | float | None): Horizontal shear factor.
        **kwargs (Union[float, int, str, None]): Additional keyword arguments.

    Returns:
        EagerTensorProtocol: Sheared images array.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    return _ensure_arr(arr_mod.asarray(random_shear_eager(backend_module, images, y_factor, x_factor, **kwargs)))


@global_eager_registry.register("RandomPerspective")
def _np_random_perspective(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    images: BackendArg,
    factor: float,
    **kwargs: float | int | str | None,
) -> EagerTensorProtocol:
    """Execute the eager np random perspective operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Active backend module.
        images (BackendArg): Input images array.
        factor (float): Perspective distortion factor.
        **kwargs (Union[float, int, str, None]): Additional keyword arguments.

    Returns:
        EagerTensorProtocol: Transformed images array.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    return _ensure_arr(arr_mod.asarray(random_perspective_eager(backend_module, images, factor, **kwargs)))


@global_eager_registry.register("RandomElasticTransform")
def _np_random_elastic_transform(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    images: BackendArg,
    alpha: float,
    sigma: float,
    **kwargs: float | int | str | None,
) -> EagerTensorProtocol:
    """Execute the eager np random elastic transform operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Active backend module.
        images (BackendArg): Input images array.
        alpha (float): Scaling factor for displacement.
        sigma (float): Elastic filter smoothing factor.
        **kwargs (Union[float, int, str, None]): Additional keyword arguments.

    Returns:
        EagerTensorProtocol: Elastically transformed images array.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    return _ensure_arr(arr_mod.asarray(random_elastic_transform_eager(backend_module, images, alpha, sigma, **kwargs)))
