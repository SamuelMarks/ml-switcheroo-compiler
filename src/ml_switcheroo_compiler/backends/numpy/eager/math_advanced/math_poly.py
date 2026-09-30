"""Polynomial mathematical functions for numpy eager backend."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from types import ModuleType
from typing import Union

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager.math_advanced.math_general import (
    _get_np_arg,
    _get_sc,
)


def _poly_recurrence(
    n: Union[np.ndarray, int, Sequence[int]],
    x: Union[np.ndarray, float, int, Sequence[float]],
    p0: float,
    p1_func: Callable[[np.ndarray], np.ndarray],
    p_next_func: Callable[[int, np.ndarray, np.ndarray, np.ndarray], np.ndarray],
) -> np.ndarray:
    """Evaluate polynomial recurrence relations iteratively.

    Args:
        n (Union[np.ndarray, int, Sequence[int]]): Degree of polynomial.
        x (Union[np.ndarray, float, int, Sequence[float]]): Input values to evaluate.
        p0 (float): Initial value for degree 0.
        p1_func (Callable[[np.ndarray], np.ndarray]): Function computing degree 1 from x.
        p_next_func (Callable[[int, np.ndarray, np.ndarray, np.ndarray], np.ndarray]):
            Function computing next degree from previous degrees.

    Returns:
        np.ndarray: Evaluated orthogonal polynomial array.
    """
    n_arr = np.asarray(n, dtype=int)
    x_arr = np.asarray(x)
    n_b, x_b = np.broadcast_arrays(n_arr, x_arr)
    max_n = int(np.max(n_b)) if n_b.size > 0 else -1

    if max_n < 0:
        return np.zeros_like(x_b)

    t_list: list[np.ndarray] = [np.ones_like(x_b) * p0]
    if max_n >= 1:
        t_list.append(p1_func(x_b))

    for i in range(2, max_n + 1):
        t_list.append(p_next_func(i - 1, x_b, t_list[-1], t_list[-2]))

    t_stacked = np.stack(t_list)
    indices = np.indices(n_b.shape)
    return t_stacked[tuple([n_b] + list(indices))]


@numpy_eager_registry.register("chebyshev_polynomial_t")
def _np_chebyshev_polynomial_t(
    backend_module: ModuleType,
    *args: Union[np.ndarray, float, int, Sequence[float]],
    **kwargs: Union[float, int, str, bool, None],
) -> np.ndarray:
    """Evaluate Chebyshev polynomial of first kind T_n(x).

    Args:
        backend_module (ModuleType): Active backend module.
        *args (Union[np.ndarray, float, int, Sequence[float]]): Positional arguments (x, n).
        **kwargs (Union[float, int, str, bool, None]): Keyword arguments.

    Returns:
        np.ndarray: Evaluated Chebyshev polynomial values.

    Raises:
        ValueError: If x or n arguments are not provided.
    """
    x = _get_np_arg(args, 0)
    n = _get_np_arg(args, 1)
    if n is None or x is None:
        raise ValueError("Expected 2 arguments x and n.")
    return _poly_recurrence(n, x, 1.0, lambda val: val, lambda idx, val, t1, t2: 2.0 * val * t1 - t2)


@numpy_eager_registry.register("chebyshev_polynomial_u")
def _np_chebyshev_polynomial_u(
    backend_module: ModuleType,
    *args: Union[np.ndarray, float, int, Sequence[float]],
    **kwargs: Union[float, int, str, bool, None],
) -> np.ndarray:
    """Evaluate Chebyshev polynomial of second kind U_n(x).

    Args:
        backend_module (ModuleType): Active backend module.
        *args (Union[np.ndarray, float, int, Sequence[float]]): Positional arguments (x, n).
        **kwargs (Union[float, int, str, bool, None]): Keyword arguments.

    Returns:
        np.ndarray: Evaluated Chebyshev polynomial values.

    Raises:
        ValueError: If x or n arguments are not provided.
    """
    x = _get_np_arg(args, 0)
    n = _get_np_arg(args, 1)
    if n is None or x is None:
        raise ValueError("Expected 2 arguments x and n.")
    return _poly_recurrence(n, x, 1.0, lambda val: 2.0 * val, lambda idx, val, t1, t2: 2.0 * val * t1 - t2)


@numpy_eager_registry.register("hermite_polynomial_h")
def _np_hermite_polynomial_h(
    backend_module: ModuleType,
    *args: Union[np.ndarray, float, int, Sequence[float]],
    **kwargs: Union[float, int, str, bool, None],
) -> np.ndarray:
    """Evaluate physicist's Hermite polynomial H_n(x).

    Args:
        backend_module (ModuleType): Active backend module.
        *args (Union[np.ndarray, float, int, Sequence[float]]): Positional arguments (x, n).
        **kwargs (Union[float, int, str, bool, None]): Keyword arguments.

    Returns:
        np.ndarray: Evaluated Hermite polynomial values.

    Raises:
        ValueError: If x or n arguments are not provided.
    """
    x = _get_np_arg(args, 0)
    n = _get_np_arg(args, 1)
    if n is None or x is None:
        raise ValueError("Expected 2 arguments x and n.")
    return _poly_recurrence(n, x, 1.0, lambda val: 2.0 * val, lambda idx, val, t1, t2: 2.0 * val * t1 - 2.0 * idx * t2)


@numpy_eager_registry.register("hermite_polynomial_he")
def _np_hermite_polynomial_he(
    backend_module: ModuleType,
    *args: Union[np.ndarray, float, int, Sequence[float]],
    **kwargs: Union[float, int, str, bool, None],
) -> np.ndarray:
    """Evaluate probabilist's Hermite polynomial He_n(x).

    Args:
        backend_module (ModuleType): Active backend module.
        *args (Union[np.ndarray, float, int, Sequence[float]]): Positional arguments (x, n).
        **kwargs (Union[float, int, str, bool, None]): Keyword arguments.

    Returns:
        np.ndarray: Evaluated probabilist Hermite polynomial values.

    Raises:
        ValueError: If x or n arguments are not provided.
    """
    x = _get_np_arg(args, 0)
    n = _get_np_arg(args, 1)
    if n is None or x is None:
        raise ValueError("Expected 2 arguments x and n.")
    return _poly_recurrence(n, x, 1.0, lambda val: val, lambda idx, val, t1, t2: val * t1 - idx * t2)


@numpy_eager_registry.register("laguerre_polynomial_l")
def _np_laguerre_polynomial_l(
    backend_module: ModuleType,
    *args: Union[np.ndarray, float, int, Sequence[float]],
    **kwargs: Union[float, int, str, bool, None],
) -> np.ndarray:
    """Evaluate Laguerre polynomial L_n(x).

    Args:
        backend_module (ModuleType): Active backend module.
        *args (Union[np.ndarray, float, int, Sequence[float]]): Positional arguments (x, n).
        **kwargs (Union[float, int, str, bool, None]): Keyword arguments.

    Returns:
        np.ndarray: Evaluated Laguerre polynomial values.

    Raises:
        ValueError: If x or n arguments are not provided.
    """
    x = _get_np_arg(args, 0)
    n = _get_np_arg(args, 1)
    if n is None or x is None:
        raise ValueError("Expected 2 arguments x and n.")
    return _poly_recurrence(
        n,
        x,
        1.0,
        lambda val: 1.0 - val,
        lambda idx, val, t1, t2: ((2.0 * idx + 1.0 - val) * t1 - idx * t2) / (idx + 1.0),
    )


@numpy_eager_registry.register("legendre_polynomial_p")
def _np_legendre_polynomial_p(
    backend_module: ModuleType,
    *args: Union[np.ndarray, float, int, Sequence[float]],
    **kwargs: Union[float, int, str, bool, None],
) -> np.ndarray:
    """Evaluate Legendre polynomial P_n(x).

    Args:
        backend_module (ModuleType): Active backend module.
        *args (Union[np.ndarray, float, int, Sequence[float]]): Positional arguments (x, n).
        **kwargs (Union[float, int, str, bool, None]): Keyword arguments.

    Returns:
        np.ndarray: Evaluated Legendre polynomial values.

    Raises:
        ValueError: If x or n arguments are not provided.
    """
    x = _get_np_arg(args, 0)
    n = _get_np_arg(args, 1)
    if n is None or x is None:
        raise ValueError("Expected 2 arguments x and n.")
    return _poly_recurrence(
        n,
        x,
        1.0,
        lambda val: val,
        lambda idx, val, t1, t2: ((2.0 * idx + 1.0) * val * t1 - idx * t2) / (idx + 1.0),
    )


@numpy_eager_registry.register("shifted_chebyshev_polynomial_t")
def _np_shifted_chebyshev_polynomial_t(
    backend_module: ModuleType,
    *args: Union[np.ndarray, float, int, Sequence[float]],
    **kwargs: Union[float, int, str, bool, None],
) -> Union[np.ndarray, float, None]:
    """Implement shifted Chebyshev polynomial of first kind.

    Args:
        backend_module (ModuleType): Active backend module.
        *args (Union[np.ndarray, float, int, Sequence[float]]): Positional arguments (n, x).
        **kwargs (Union[float, int, str, bool, None]): Keyword arguments.

    Returns:
        Union[np.ndarray, float, None]: Evaluated shifted polynomial or None if inputs/scipy missing.
    """
    sc_module = _get_sc()
    n = _get_np_arg(args, 0)
    x = _get_np_arg(args, 1)
    if n is None or x is None or sc_module is None:
        return None
    return sc_module.eval_sh_chebyt(np.asarray(n, dtype=int), x)


@numpy_eager_registry.register("shifted_chebyshev_polynomial_u")
def _np_shifted_chebyshev_polynomial_u(
    backend_module: ModuleType,
    *args: Union[np.ndarray, float, int, Sequence[float]],
    **kwargs: Union[float, int, str, bool, None],
) -> Union[np.ndarray, float, None]:
    """Implement shifted Chebyshev polynomial of second kind.

    Args:
        backend_module (ModuleType): Active backend module.
        *args (Union[np.ndarray, float, int, Sequence[float]]): Positional arguments (n, x).
        **kwargs (Union[float, int, str, bool, None]): Keyword arguments.

    Returns:
        Union[np.ndarray, float, None]: Evaluated shifted polynomial or None if inputs/scipy missing.
    """
    sc_module = _get_sc()
    n = _get_np_arg(args, 0)
    x = _get_np_arg(args, 1)
    if n is None or x is None or sc_module is None:
        return None
    return sc_module.eval_sh_chebyu(np.asarray(n, dtype=int), x)


@numpy_eager_registry.register("shifted_chebyshev_polynomial_v")
def _np_shifted_chebyshev_polynomial_v(
    backend_module: ModuleType,
    *args: Union[np.ndarray, float, int, Sequence[float]],
    **kwargs: Union[float, int, str, bool, None],
) -> Union[np.ndarray, float, None]:
    """Implement shifted Chebyshev polynomial of third kind V_n(x).

    Args:
        backend_module (ModuleType): Active backend module.
        *args (Union[np.ndarray, float, int, Sequence[float]]): Positional arguments (n, x).
        **kwargs (Union[float, int, str, bool, None]): Keyword arguments.

    Returns:
        Union[np.ndarray, float, None]: Evaluated polynomial or None if inputs/scipy missing.
    """
    sc_module = _get_sc()
    n = _get_np_arg(args, 0)
    x = _get_np_arg(args, 1)
    if n is None or x is None or sc_module is None:
        return None
    n_int = np.asarray(n, dtype=int)
    u_n = sc_module.eval_sh_chebyu(n_int, x)
    u_nm1 = sc_module.eval_sh_chebyu(n_int - 1, x)
    return u_n - u_nm1 / 2.0


@numpy_eager_registry.register("shifted_chebyshev_polynomial_w")
def _np_shifted_chebyshev_polynomial_w(
    backend_module: ModuleType,
    *args: Union[np.ndarray, float, int, Sequence[float]],
    **kwargs: Union[float, int, str, bool, None],
) -> Union[np.ndarray, float, None]:
    """Implement shifted Chebyshev polynomial of fourth kind W_n(x).

    Args:
        backend_module (ModuleType): Active backend module.
        *args (Union[np.ndarray, float, int, Sequence[float]]): Positional arguments (n, x).
        **kwargs (Union[float, int, str, bool, None]): Keyword arguments.

    Returns:
        Union[np.ndarray, float, None]: Evaluated polynomial or None if inputs/scipy missing.
    """
    sc_module = _get_sc()
    n = _get_np_arg(args, 0)
    x = _get_np_arg(args, 1)
    if n is None or x is None or sc_module is None:
        return None
    n_int = np.asarray(n, dtype=int)
    u_n = sc_module.eval_sh_chebyu(n_int, x)
    u_nm1 = sc_module.eval_sh_chebyu(n_int - 1, x)
    return u_n + u_nm1 / 2.0
