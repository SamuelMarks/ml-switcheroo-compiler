# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Math Ops."""

from collections.abc import Sequence
from typing import Optional

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager.math_nan import _xlogy

from .math_general import _get_np_arg, _get_sc


@numpy_eager_registry.register("Mvlgamma")
def _np_mvlgamma(backend_module, *args, **kwargs):
    """Evaluate _np_mvlgamma operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return backend_module.mvlgamma(*args, **kwargs)
