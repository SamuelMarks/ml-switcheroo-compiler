# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Math Ops."""

from collections.abc import Sequence
from typing import Any, Optional

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager.math_nan import _xlogy

from .math_misc_ext import _get_np_arg, _get_sc


@numpy_eager_registry.register("BesselI0e")
def _np_bessel_i0e(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_bessel_i0e operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    import scipy.special as sc

    return backend_module.array(sc.i0e(args[0]))


@numpy_eager_registry.register("BesselI1e")
def _np_bessel_i1e(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_bessel_i1e operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    import scipy.special as sc

    return backend_module.array(sc.i1e(args[0]))


@numpy_eager_registry.register("modified_bessel_i0")
def _np_modified_bessel_i0(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_modified_bessel_i0 operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    import numpy as np

    x = _get_np_arg(args, 0)
    if x is None:
        return None
    return np.i0(x)


@numpy_eager_registry.register("modified_bessel_i1")
def _np_modified_bessel_i1(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_modified_bessel_i1 operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    sc = _get_sc()
    x = _get_np_arg(args, 0)
    if x is None:
        return None
    if sc is None:
        import numpy as np

        t = np.linspace(0, np.pi, 100)
        t = np.reshape(t, (1,) * np.ndim(x) + (-1,)) if np.ndim(x) > 0 else t
        x_ex = np.expand_dims(x, -1) if np.ndim(x) > 0 else x
        integrand = np.exp(x_ex * np.cos(t)) * np.cos(t)
        return (1.0 / np.pi) * np.trapz(integrand, x=t, axis=-1)
    return sc.i1(x)


@numpy_eager_registry.register("modified_bessel_k0")
def _np_modified_bessel_k0(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_modified_bessel_k0 operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    sc = _get_sc()
    x = _get_np_arg(args, 0)
    if x is None:
        return None
    if sc is None:
        import numpy as np

        t = np.linspace(0, 10, 100)
        t = np.reshape(t, (1,) * np.ndim(x) + (-1,)) if np.ndim(x) > 0 else t
        x_ex = np.expand_dims(x, -1) if np.ndim(x) > 0 else x
        integrand = np.exp(-x_ex * np.cosh(t))
        return np.trapz(integrand, x=t, axis=-1)
    return sc.k0(x)


@numpy_eager_registry.register("modified_bessel_k1")
def _np_modified_bessel_k1(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_modified_bessel_k1 operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    sc = _get_sc()
    x = _get_np_arg(args, 0)
    if x is None:
        return None
    if sc is None:
        import numpy as np

        t = np.linspace(0, 10, 100)
        t = np.reshape(t, (1,) * np.ndim(x) + (-1,)) if np.ndim(x) > 0 else t
        x_ex = np.expand_dims(x, -1) if np.ndim(x) > 0 else x
        integrand = np.exp(-x_ex * np.cosh(t)) * np.cosh(t)
        return np.trapz(integrand, x=t, axis=-1)
    return sc.k1(x)
