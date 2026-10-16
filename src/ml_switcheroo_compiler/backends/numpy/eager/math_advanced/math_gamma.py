"""Math Ops Gamma."""

from __future__ import annotations

from types import ModuleType
from typing import Union, cast

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry


@numpy_eager_registry.register("Mvlgamma")
def _np_mvlgamma(
    backend_module: ModuleType,
    *args: Union[np.ndarray[tuple[int, ...], np.dtype[np.generic]], int, float],
    **kwargs: Union[int, float],
) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
    """Evaluate _np_mvlgamma operation.

    Args:
        backend_module (ModuleType): Active backend module.
        *args (Union[np.ndarray, int, float]): Input arguments.
        **kwargs (Union[int, float]): Additional keyword arguments.

    Returns:
        np.ndarray: Log multivariate gamma evaluation.
    """
    return cast(np.ndarray[tuple[int, ...], np.dtype[np.generic]], backend_module.mvlgamma(*args, **kwargs))
