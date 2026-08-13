# ruff: noqa: E402, D100, D103, D104, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, D101, D102, D107, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Math Ops."""

from collections.abc import Sequence
from typing import Any, Optional

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager.math_nan import _xlogy

from .math_misc_ext import _get_np_arg, _get_sc


def _poly_recurrence(n: Any, x: Any, p0: float, p1_func: Any, p_next_func: Any) -> Any:  # noqa: D417
    """Evaluate _poly_recurrence logic eagerly backed by NumPy.

    Args:
        n (object): The n parameter.
        x (object): The x parameter.
        p0 (float): The p0 parameter.
        p1_func (object): The p1_func parameter.
        p_next_func (object): The p_next_func parameter.

    Returns: Any: Result.
    """
    import numpy as np

    n = np.asarray(n, dtype=int)
    x = np.asarray(x)
    n_b, x_b = np.broadcast_arrays(n, x)
    max_n: Any = np.max(n_b)

    if max_n < 0:
        return np.zeros_like(x_b)

    T = [np.ones_like(x_b) * p0]
    if max_n >= 1:
        T.append(p1_func(x_b))

    for i in range(2, max_n + 1):
        T.append(p_next_func(i - 1, x_b, T[-1], T[-2]))

    T = np.stack(T)
    indices = np.indices(n_b.shape)
    return T[tuple([n_b] + list(indices))]


@numpy_eager_registry.register("chebyshev_polynomial_t")
def _np_chebyshev_polynomial_t(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_chebyshev_polynomial_t operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.

    Raises:
        ValueError: An exception.
    """
    x, n = _get_np_arg(args, 0), _get_np_arg(args, 1)
    if n is None or x is None:
        raise ValueError("Expected 2 arguments x and n.")
    return _poly_recurrence(n, x, 1, lambda x: x, lambda n, x, t1, t2: 2 * x * t1 - t2)


@numpy_eager_registry.register("chebyshev_polynomial_u")
def _np_chebyshev_polynomial_u(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_chebyshev_polynomial_u operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.

    Raises:
        ValueError: An exception.
    """
    x, n = _get_np_arg(args, 0), _get_np_arg(args, 1)
    if n is None or x is None:
        raise ValueError("Expected 2 arguments x and n.")
    return _poly_recurrence(n, x, 1, lambda x: 2 * x, lambda n, x, t1, t2: 2 * x * t1 - t2)


@numpy_eager_registry.register("hermite_polynomial_h")
def _np_hermite_polynomial_h(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_hermite_polynomial_h operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.

    Raises:
        ValueError: An exception.
    """
    x, n = _get_np_arg(args, 0), _get_np_arg(args, 1)
    if n is None or x is None:
        raise ValueError("Expected 2 arguments x and n.")
    return _poly_recurrence(n, x, 1, lambda x: 2 * x, lambda n, x, t1, t2: 2 * x * t1 - 2 * n * t2)


@numpy_eager_registry.register("hermite_polynomial_he")
def _np_hermite_polynomial_he(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_hermite_polynomial_he operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.

    Raises:
        ValueError: An exception.
    """
    x, n = _get_np_arg(args, 0), _get_np_arg(args, 1)
    if n is None or x is None:
        raise ValueError("Expected 2 arguments x and n.")
    return _poly_recurrence(n, x, 1, lambda x: x, lambda n, x, t1, t2: x * t1 - n * t2)


@numpy_eager_registry.register("laguerre_polynomial_l")
def _np_laguerre_polynomial_l(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_laguerre_polynomial_l operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.

    Raises:
        ValueError: An exception.
    """
    x, n = _get_np_arg(args, 0), _get_np_arg(args, 1)
    if n is None or x is None:
        raise ValueError("Expected 2 arguments x and n.")
    return _poly_recurrence(n, x, 1, lambda x: 1 - x, lambda n, x, t1, t2: ((2 * n + 1 - x) * t1 - n * t2) / (n + 1))


@numpy_eager_registry.register("legendre_polynomial_p")
def _np_legendre_polynomial_p(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_legendre_polynomial_p operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.

    Raises:
        ValueError: An exception.
    """
    x, n = _get_np_arg(args, 0), _get_np_arg(args, 1)
    if n is None or x is None:
        raise ValueError("Expected 2 arguments x and n.")
    return _poly_recurrence(n, x, 1, lambda x: x, lambda n, x, t1, t2: ((2 * n + 1) * x * t1 - n * t2) / (n + 1))


@numpy_eager_registry.register("shifted_chebyshev_polynomial_t")
def _np_shifted_chebyshev_polynomial_t(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement shifted_chebyshev_polynomial_t.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    sc = _get_sc()
    n, x = _get_np_arg(args, 0), _get_np_arg(args, 1)
    if n is None or x is None or sc is None:
        return None
    return sc.eval_sh_chebyt(np.asarray(n, dtype=int), x)


@numpy_eager_registry.register("shifted_chebyshev_polynomial_u")
def _np_shifted_chebyshev_polynomial_u(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement shifted_chebyshev_polynomial_u.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    sc = _get_sc()
    n, x = _get_np_arg(args, 0), _get_np_arg(args, 1)
    if n is None or x is None or sc is None:
        return None
    return sc.eval_sh_chebyu(np.asarray(n, dtype=int), x)


@numpy_eager_registry.register("shifted_chebyshev_polynomial_v")
def _np_shifted_chebyshev_polynomial_v(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement shifted_chebyshev_polynomial_v.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    sc = _get_sc()
    # V_n(x) = U_n(x) - U_{n-1}(x) / 2
    n, x = _get_np_arg(args, 0), _get_np_arg(args, 1)
    if n is None or x is None or sc is None:
        return None
    n_int = np.asarray(n, dtype=int)
    u_n = sc.eval_sh_chebyu(n_int, x)
    u_nm1 = sc.eval_sh_chebyu(n_int - 1, x)
    return u_n - u_nm1 / 2.0


@numpy_eager_registry.register("shifted_chebyshev_polynomial_w")
def _np_shifted_chebyshev_polynomial_w(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement shifted_chebyshev_polynomial_w.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    sc = _get_sc()
    # W_n(x) = U_n(x) + U_{n-1}(x) / 2
    n, x = _get_np_arg(args, 0), _get_np_arg(args, 1)
    if n is None or x is None or sc is None:
        return None
    n_int = np.asarray(n, dtype=int)
    u_n = sc.eval_sh_chebyu(n_int, x)
    u_nm1 = sc.eval_sh_chebyu(n_int - 1, x)
    return u_n + u_nm1 / 2.0
