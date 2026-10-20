"""Math Ops Gamma."""

from __future__ import annotations

import typing

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import EagerValue, numpy_eager_registry
from ml_switcheroo_compiler.backends.protocols import BackendModuleProtocol


@numpy_eager_registry.register("Mvlgamma")
def _np_mvlgamma(
    backend_module: BackendModuleProtocol,
    *args: EagerValue,
    **kwargs: int | float,
) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
    """Execute the eager np mvlgamma operation on the active backend array.

    Args:
        backend_module (ModuleType): Active backend module.
        *args (Union[np.ndarray, int, float]): Input arguments.
        **kwargs (Union[int, float]): Additional keyword arguments.

    Returns:
        np.ndarray: Log multivariate gamma evaluation.
    """
    return typing.cast(np.ndarray[tuple[int, ...], np.dtype[np.generic]], backend_module.mvlgamma(*args, **kwargs))
