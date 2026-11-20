"""Thin NumPy/SciPy Bessel and orthogonal polynomial eager kernel bindings."""

from __future__ import annotations

import importlib
import types
from collections.abc import Callable, Sequence

import numpy as np

from ml_switcheroo_compiler.backends.eager.reference_math.math_general import _get_np_arg
from ml_switcheroo_compiler.backends.eager_registry import (
    BackendArg,
    EagerTensorProtocol,
    coerce_float,
    numpy_eager_registry,
    require_callable,
)
from ml_switcheroo_compiler.backends.numpy.eager import ensure_ndarray
from ml_switcheroo_compiler.backends.numpy.eager.scipy_special_kernels import _get_sc
from ml_switcheroo_compiler.backends.protocols import BackendModuleProtocol, adapt_backend_module

_Float64Arr = np.ndarray[tuple[int, ...], np.dtype[np.float64]]


@numpy_eager_registry.register("BesselI0e")
def _np_bessel_i0e(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    *args: BackendArg,
    **kwargs: float | int | str | bool | None,
) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
    """Evaluate exponentially scaled modified Bessel function of order 0.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Active backend module.
        *args (BackendArg): Positional arguments.
        **kwargs (Union[float, int, str, bool, None]): Keyword arguments.

    Returns:
        np.ndarray: Computed exponentially scaled Bessel i0e result.
    """
    return ensure_ndarray(
        adapt_backend_module(backend_module).require_symbol("array")(
            importlib.import_module("scipy.special").i0e(args[0])
        )
    )


@numpy_eager_registry.register("BesselI1e")
def _np_bessel_i1e(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    *args: BackendArg,
    **kwargs: float | int | str | bool | None,
) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
    """Evaluate exponentially scaled modified Bessel function of order 1.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Active backend module.
        *args (BackendArg): Positional arguments.
        **kwargs (Union[float, int, str, bool, None]): Keyword arguments.

    Returns:
        np.ndarray: Computed exponentially scaled Bessel i1e result.
    """
    return ensure_ndarray(
        adapt_backend_module(backend_module).require_symbol("array")(
            importlib.import_module("scipy.special").i1e(args[0])
        )
    )


def _poly_recurrence(
    n: _Float64Arr | EagerTensorProtocol | int | Sequence[int],
    x: _Float64Arr | EagerTensorProtocol | float | int | Sequence[float],
    p0: float,
    p1_func: Callable[[_Float64Arr], _Float64Arr],
    p_next_func: Callable[[int, _Float64Arr, _Float64Arr, _Float64Arr], _Float64Arr],
) -> _Float64Arr:
    """Evaluate polynomial recurrence relations iteratively.

    Args:
        n (_Float64Arr | EagerTensorProtocol | int | Sequence[int]): Degree of polynomial.
        x (_Float64Arr | EagerTensorProtocol | float | int | Sequence[float]): Input values to evaluate.
        p0 (float): Initial value for degree 0.
        p1_func (Callable[[_Float64Arr], _Float64Arr]): Function computing degree 1 from x.
        p_next_func (Callable[[int, _Float64Arr, _Float64Arr, _Float64Arr], _Float64Arr]):
            Function computing next degree from previous degrees.

    Returns:
        _Float64Arr: Evaluated orthogonal polynomial array.
    """
    n_arr = np.asarray(n, dtype=int)
    x_arr = np.asarray(x)
    n_b, x_b = np.broadcast_arrays(n_arr, x_arr)
    max_n = int(np.max(n_b)) if n_b.size > 0 else -1

    if max_n < 0:
        return np.zeros_like(x_b)

    t_list: list[np.ndarray[tuple[int, ...], np.dtype[np.float64]]] = [np.ones_like(x_b) * p0]
    if max_n >= 1:
        t_list.append(p1_func(x_b))

    for i in range(2, max_n + 1):
        adapt_backend_module(t_list).require_symbol("append")(p_next_func(i - 1, x_b, t_list[-1], t_list[-2]))

    t_stacked = np.stack(t_list)
    indices = np.indices(n_b.shape)
    return ensure_ndarray(t_stacked[tuple([n_b] + list(indices))])


@numpy_eager_registry.register("chebyshev_polynomial_t")
def _np_chebyshev_polynomial_t(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    *args: BackendArg,
    **kwargs: float | int | str | bool | None,
) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
    """Evaluate Chebyshev polynomial of first kind T_n(x).

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Active backend module.
        *args (BackendArg): Positional arguments (x, n).
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
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    *args: BackendArg,
    **kwargs: float | int | str | bool | None,
) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
    """Evaluate Chebyshev polynomial of second kind U_n(x).

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Active backend module.
        *args (BackendArg): Positional arguments (x, n).
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
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    *args: BackendArg,
    **kwargs: float | int | str | bool | None,
) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
    """Evaluate physicist's Hermite polynomial H_n(x).

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Active backend module.
        *args (BackendArg): Positional arguments (x, n).
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
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    *args: BackendArg,
    **kwargs: float | int | str | bool | None,
) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
    """Evaluate probabilist's Hermite polynomial He_n(x).

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Active backend module.
        *args (BackendArg): Positional arguments (x, n).
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
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    *args: BackendArg,
    **kwargs: float | int | str | bool | None,
) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
    """Evaluate Laguerre polynomial L_n(x).

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Active backend module.
        *args (BackendArg): Positional arguments (x, n).
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
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    *args: BackendArg,
    **kwargs: float | int | str | bool | None,
) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
    """Evaluate Legendre polynomial P_n(x).

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Active backend module.
        *args (BackendArg): Positional arguments (x, n).
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
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    *args: BackendArg,
    **kwargs: float | int | str | bool | None,
) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]] | float | None:
    """Implement shifted Chebyshev polynomial of first kind.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Active backend module.
        *args (BackendArg): Positional arguments (n, x).
        **kwargs (Union[float, int, str, bool, None]): Keyword arguments.

    Returns:
        Union[np.ndarray, float, None]: Evaluated shifted polynomial or None if inputs/scipy missing.
    """
    sc_module = _get_sc()
    n = _get_np_arg(args, 0)
    x = _get_np_arg(args, 1)
    if n is None or x is None or sc_module is None:
        return None
    return ensure_ndarray(adapt_backend_module(sc_module).require_symbol("eval_sh_chebyt")(np.asarray(n, dtype=int), x))


@numpy_eager_registry.register("shifted_chebyshev_polynomial_u")
def _np_shifted_chebyshev_polynomial_u(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    *args: BackendArg,
    **kwargs: float | int | str | bool | None,
) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]] | float | None:
    """Implement shifted Chebyshev polynomial of second kind.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Active backend module.
        *args (BackendArg): Positional arguments (n, x).
        **kwargs (Union[float, int, str, bool, None]): Keyword arguments.

    Returns:
        Union[np.ndarray, float, None]: Evaluated shifted polynomial or None if inputs/scipy missing.
    """
    sc_module = _get_sc()
    n = _get_np_arg(args, 0)
    x = _get_np_arg(args, 1)
    if n is None or x is None or sc_module is None:
        return None
    return ensure_ndarray(adapt_backend_module(sc_module).require_symbol("eval_sh_chebyu")(np.asarray(n, dtype=int), x))


@numpy_eager_registry.register("shifted_chebyshev_polynomial_v")
def _np_shifted_chebyshev_polynomial_v(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    *args: BackendArg,
    **kwargs: float | int | str | bool | None,
) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]] | float | None:
    """Implement shifted Chebyshev polynomial of third kind V_n(x).

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Active backend module.
        *args (BackendArg): Positional arguments (n, x).
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
    u_n = adapt_backend_module(sc_module).require_symbol("eval_sh_chebyu")(n_int, x)
    u_nm1 = adapt_backend_module(sc_module).require_symbol("eval_sh_chebyu")(n_int - 1, x)
    return ensure_ndarray(u_n - u_nm1 / 2.0)


@numpy_eager_registry.register("shifted_chebyshev_polynomial_w")
def _np_shifted_chebyshev_polynomial_w(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    *args: BackendArg,
    **kwargs: float | int | str | bool | None,
) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]] | float | None:
    """Implement shifted Chebyshev polynomial of fourth kind W_n(x).

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Active backend module.
        *args (BackendArg): Positional arguments (n, x).
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
    u_n = adapt_backend_module(sc_module).require_symbol("eval_sh_chebyu")(n_int, x)
    u_nm1 = adapt_backend_module(sc_module).require_symbol("eval_sh_chebyu")(n_int - 1, x)
    return ensure_ndarray(u_n + u_nm1 / 2.0)


@numpy_eager_registry.register("modified_bessel_i0")
def _np_modified_bessel_i0(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    *args: BackendArg,
    **kwargs: float | int | str | bool | None,
) -> _Float64Arr | float | None:
    """Evaluate modified Bessel function of order 0.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Active backend module.
        *args (BackendArg): Positional arguments.
        **kwargs (Union[float, int, str, bool, None]): Keyword arguments.

    Returns:
        Union[EagerTensorProtocol, float, None]: Result of modified Bessel i0 or None if input missing.
    """
    arr_mod = np
    x = _get_np_arg(args, 0)
    if x is None:
        return None
    return ensure_ndarray(adapt_backend_module(arr_mod).require_symbol("i0")(x))


@numpy_eager_registry.register("modified_bessel_i1")
def _np_modified_bessel_i1(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    *args: BackendArg,
    **kwargs: float | int | str | bool | None,
) -> _Float64Arr | float | None:
    """Evaluate modified Bessel function of order 1.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Active backend module.
        *args (BackendArg): Positional arguments.
        **kwargs (Union[float, int, str, bool, None]): Keyword arguments.

    Returns:
        Union[EagerTensorProtocol, float, None]: Result of modified Bessel i1 or None if input missing.
    """
    arr_mod = np
    sc_module = _get_sc()
    x = _get_np_arg(args, 0)
    if x is None:
        return None
    if sc_module is None:
        t = np.linspace(0.0, np.pi, 100)
        dt = float(coerce_float(t[1] - t[0]))
        t_grid = np.reshape(t, (1,) * np.ndim(x) + (-1,)) if np.ndim(x) > 0 else t
        x_ex = np.expand_dims(x, -1) if np.ndim(x) > 0 else x
        integrand = np.exp(x_ex * np.cos(t_grid)) * np.cos(t_grid)
        trapz_fn = adapt_backend_module(arr_mod).get_symbol(
            "trapezoid", adapt_backend_module(arr_mod).get_symbol("trapz", None)
        )
        return ensure_ndarray(require_callable(trapz_fn)(integrand, dx=dt, axis=-1) * (1.0 / np.pi))
    return ensure_ndarray(adapt_backend_module(sc_module).require_symbol("i1")(x))


@numpy_eager_registry.register("modified_bessel_k0")
def _np_modified_bessel_k0(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    *args: BackendArg,
    **kwargs: float | int | str | bool | None,
) -> _Float64Arr | float | None:
    """Evaluate modified Bessel function of second kind of order 0.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Active backend module.
        *args (BackendArg): Positional arguments.
        **kwargs (Union[float, int, str, bool, None]): Keyword arguments.

    Returns:
        Union[EagerTensorProtocol, float, None]: Result of modified Bessel k0 or None if input missing.
    """
    arr_mod = np
    sc_module = _get_sc()
    x = _get_np_arg(args, 0)
    if x is None:
        return None
    if sc_module is None:
        t = np.linspace(0.0, 10.0, 100)
        dt = float(coerce_float(t[1] - t[0]))
        t_grid = np.reshape(t, (1,) * np.ndim(x) + (-1,)) if np.ndim(x) > 0 else t
        x_ex = np.expand_dims(x, -1) if np.ndim(x) > 0 else x
        integrand = np.exp(-x_ex * np.cosh(t_grid))
        trapz_fn = adapt_backend_module(arr_mod).get_symbol(
            "trapezoid", adapt_backend_module(arr_mod).get_symbol("trapz", None)
        )
        return ensure_ndarray(require_callable(trapz_fn)(integrand, dx=dt, axis=-1))
    return ensure_ndarray(adapt_backend_module(sc_module).require_symbol("k0")(x))


@numpy_eager_registry.register("modified_bessel_k1")
def _np_modified_bessel_k1(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    *args: BackendArg,
    **kwargs: float | int | str | bool | None,
) -> _Float64Arr | float | None:
    """Evaluate modified Bessel function of second kind of order 1.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Active backend module.
        *args (BackendArg): Positional arguments.
        **kwargs (Union[float, int, str, bool, None]): Keyword arguments.

    Returns:
        Union[EagerTensorProtocol, float, None]: Result of modified Bessel k1 or None if input missing.
    """
    arr_mod = np
    sc_module = _get_sc()
    x = _get_np_arg(args, 0)
    if x is None:
        return None
    if sc_module is None:
        t = np.linspace(0.0, 10.0, 100)
        dt = float(coerce_float(t[1] - t[0]))
        t_grid = np.reshape(t, (1,) * np.ndim(x) + (-1,)) if np.ndim(x) > 0 else t
        x_ex = np.expand_dims(x, -1) if np.ndim(x) > 0 else x
        integrand = np.exp(-x_ex * np.cosh(t_grid)) * np.cosh(t_grid)
        trapz_fn = adapt_backend_module(arr_mod).get_symbol(
            "trapezoid", adapt_backend_module(arr_mod).get_symbol("trapz", None)
        )
        return ensure_ndarray(require_callable(trapz_fn)(integrand, dx=dt, axis=-1))
    return ensure_ndarray(adapt_backend_module(sc_module).require_symbol("k1")(x))
