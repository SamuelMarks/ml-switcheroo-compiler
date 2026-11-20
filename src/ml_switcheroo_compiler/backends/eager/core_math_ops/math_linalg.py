"""Operations and utilities for `ml_switcheroo_compiler.backends.eager.core_math_ops.math_linalg`."""

from __future__ import annotations

import builtins  # noqa: F401
import types

from ml_switcheroo_compiler.backends.eager.utils import _resolve_eager_mod
from ml_switcheroo_compiler.backends.eager_registry import (
    BackendArg,
    global_eager_registry,
)
from ml_switcheroo_compiler.backends.protocols import BackendModuleProtocol, adapt_backend_module

from .math_special import _dispatch_special


@global_eager_registry.register("Det")
def _det(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager det operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    func = adapt_backend_module(backend_module).get_symbol("linalg", None)
    if func and adapt_backend_module(func).has_symbol("det"):
        return func.det(*args, **kwargs)
    if adapt_backend_module(backend_module).has_symbol("det"):
        return adapt_backend_module(backend_module).require_symbol("det")(*args, **kwargs)
    x = args[0]
    return (
        adapt_backend_module(backend_module)
        .require_symbol("linalg")
        .det(adapt_backend_module(backend_module).require_symbol("asarray")(x))
    )


@global_eager_registry.register("Eig")
def _eig(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager eig operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    func = adapt_backend_module(backend_module).get_symbol("linalg", None)
    if func and adapt_backend_module(func).has_symbol("eig"):
        return func.eig(*args, **kwargs)
    if adapt_backend_module(backend_module).has_symbol("eig"):
        return adapt_backend_module(backend_module).require_symbol("eig")(*args, **kwargs)
    x = args[0]
    return (
        adapt_backend_module(backend_module)
        .require_symbol("linalg")
        .eig(adapt_backend_module(backend_module).require_symbol("asarray")(x))
    )


@global_eager_registry.register("Eigh")
def _eigh(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager eigh operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    func = adapt_backend_module(backend_module).get_symbol("linalg", None)
    if func and adapt_backend_module(func).has_symbol("eigh"):
        return func.eigh(*args, **kwargs)
    if adapt_backend_module(backend_module).has_symbol("eigh"):
        return adapt_backend_module(backend_module).require_symbol("eigh")(*args, **kwargs)
    x = args[0]
    return (
        adapt_backend_module(backend_module)
        .require_symbol("linalg")
        .eigh(adapt_backend_module(backend_module).require_symbol("asarray")(x))
    )


@global_eager_registry.register("Eigvals")
def _eigvals(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager eigvals operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    func = adapt_backend_module(backend_module).get_symbol("linalg", None)
    if func and adapt_backend_module(func).has_symbol("eigvals"):
        return func.eigvals(*args, **kwargs)
    if adapt_backend_module(backend_module).has_symbol("eigvals"):
        return adapt_backend_module(backend_module).require_symbol("eigvals")(*args, **kwargs)
    x = args[0]
    return (
        adapt_backend_module(backend_module)
        .require_symbol("linalg")
        .eigvals(adapt_backend_module(backend_module).require_symbol("asarray")(x))
    )


@global_eager_registry.register("Eigvalsh")
def _eigvalsh(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager eigvalsh operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    func = adapt_backend_module(backend_module).get_symbol("linalg", None)
    if func and adapt_backend_module(func).has_symbol("eigvalsh"):
        return func.eigvalsh(*args, **kwargs)
    if adapt_backend_module(backend_module).has_symbol("eigvalsh"):
        return adapt_backend_module(backend_module).require_symbol("eigvalsh")(*args, **kwargs)
    x = args[0]
    return (
        adapt_backend_module(backend_module)
        .require_symbol("linalg")
        .eigvalsh(adapt_backend_module(backend_module).require_symbol("asarray")(x))
    )


@global_eager_registry.register("Cholesky")
def _cholesky(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager cholesky operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    func = adapt_backend_module(backend_module).get_symbol("linalg", None)
    if func and adapt_backend_module(func).has_symbol("cholesky"):
        return func.cholesky(*args, **kwargs)
    if adapt_backend_module(backend_module).has_symbol("cholesky"):
        return adapt_backend_module(backend_module).require_symbol("cholesky")(*args, **kwargs)
    x = args[0]
    return (
        adapt_backend_module(backend_module)
        .require_symbol("linalg")
        .cholesky(adapt_backend_module(backend_module).require_symbol("asarray")(x))
    )


@global_eager_registry.register("CholeskyEx")
def _cholesky_ex(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager cholesky ex operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    chol = _cholesky(backend_module, *args, **kwargs)
    info: BackendArg
    if adapt_backend_module(backend_module).has_symbol("zeros"):
        info = adapt_backend_module(backend_module).require_symbol("zeros")(
            (), dtype=adapt_backend_module(backend_module).get_symbol("int32", int)
        )
    else:
        info = 0
    return (chol, info)


@global_eager_registry.register("Norm")
def _norm(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager norm operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    func = adapt_backend_module(backend_module).get_symbol("linalg", None)
    if func and adapt_backend_module(func).has_symbol("norm"):
        return func.norm(*args, **kwargs)
    if adapt_backend_module(backend_module).has_symbol("norm"):
        return adapt_backend_module(backend_module).require_symbol("norm")(*args, **kwargs)
    x = args[0]
    return (
        adapt_backend_module(backend_module)
        .require_symbol("linalg")
        .norm(adapt_backend_module(backend_module).require_symbol("asarray")(x), **kwargs)
    )


@global_eager_registry.register("Pinv")
def _pinv(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager pinv operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    func = adapt_backend_module(backend_module).get_symbol("linalg", None)
    if func and adapt_backend_module(func).has_symbol("pinv"):
        return func.pinv(*args, **kwargs)
    if adapt_backend_module(backend_module).has_symbol("pinv"):
        return adapt_backend_module(backend_module).require_symbol("pinv")(*args, **kwargs)
    x = args[0]
    return (
        adapt_backend_module(backend_module)
        .require_symbol("linalg")
        .pinv(adapt_backend_module(backend_module).require_symbol("asarray")(x), **kwargs)
    )


@global_eager_registry.register("Qr")
def _qr(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager qr operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    func = adapt_backend_module(backend_module).get_symbol("linalg", None)
    if func and adapt_backend_module(func).has_symbol("qr"):
        return func.qr(*args, **kwargs)
    if adapt_backend_module(backend_module).has_symbol("qr"):
        return adapt_backend_module(backend_module).require_symbol("qr")(*args, **kwargs)
    x = args[0]
    return (
        adapt_backend_module(backend_module)
        .require_symbol("linalg")
        .qr(adapt_backend_module(backend_module).require_symbol("asarray")(x), **kwargs)
    )


@global_eager_registry.register("Svd")
def _svd(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager svd operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    func = adapt_backend_module(backend_module).get_symbol("linalg", None)
    if func and adapt_backend_module(func).has_symbol("svd"):
        return func.svd(*args, **kwargs)
    if adapt_backend_module(backend_module).has_symbol("svd"):
        return adapt_backend_module(backend_module).require_symbol("svd")(*args, **kwargs)
    x = args[0]
    return (
        adapt_backend_module(backend_module)
        .require_symbol("linalg")
        .svd(adapt_backend_module(backend_module).require_symbol("asarray")(x), **kwargs)
    )


@global_eager_registry.register("Tensorinv")
def _tensorinv(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager tensorinv operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    func = adapt_backend_module(backend_module).get_symbol("linalg", None)
    if func and adapt_backend_module(func).has_symbol("tensorinv"):
        return func.tensorinv(*args, **kwargs)
    if adapt_backend_module(backend_module).has_symbol("tensorinv"):
        return adapt_backend_module(backend_module).require_symbol("tensorinv")(*args, **kwargs)
    x = args[0]
    return (
        adapt_backend_module(backend_module)
        .require_symbol("linalg")
        .tensorinv(adapt_backend_module(backend_module).require_symbol("asarray")(x), **kwargs)
    )


@global_eager_registry.register("Inv")
def _inv(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager inv operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    func = adapt_backend_module(backend_module).get_symbol("linalg", None)
    if func and adapt_backend_module(func).has_symbol("inv"):
        return func.inv(*args, **kwargs)
    if adapt_backend_module(backend_module).has_symbol("inv"):
        return adapt_backend_module(backend_module).require_symbol("inv")(*args, **kwargs)
    x = args[0]
    return (
        adapt_backend_module(backend_module)
        .require_symbol("linalg")
        .inv(adapt_backend_module(backend_module).require_symbol("asarray")(x))
    )


@global_eager_registry.register("Vander")
def _vander(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager vander operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    return adapt_backend_module(backend_module).require_symbol("vander")(*args, **kwargs)


@global_eager_registry.register("Schur")
def _np_schur(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np schur operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    return _dispatch_special("Schur", "schur", backend_module, *args, **kwargs)


@global_eager_registry.register("Svd")
def _np_svd(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np svd operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    func = adapt_backend_module(backend_module).get_symbol(
        "svd", adapt_backend_module(backend_module).get_symbol("svd", None)
    )
    if func is not None:
        return func(*args, **kwargs)
    return _resolve_eager_mod(backend_module).linalg.svd(*args, **kwargs)


@global_eager_registry.register("Svdvals")
def _np_svdvals(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np svdvals operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    func = adapt_backend_module(backend_module).get_symbol(
        "svdvals", adapt_backend_module(backend_module).get_symbol("svdvals", None)
    )
    if func is not None:
        return func(*args, **kwargs)
    return _resolve_eager_mod(backend_module).linalg.svd(*args, **kwargs)[1]


@global_eager_registry.register("Tensorinv")
def _np_tensorinv(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np tensorinv operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    func = adapt_backend_module(backend_module).get_symbol(
        "tensorinv", adapt_backend_module(backend_module).get_symbol("tensorinv", None)
    )
    if func is not None:
        return func(*args, **kwargs)
    return _resolve_eager_mod(backend_module).linalg.tensorinv(*args, **kwargs)


@global_eager_registry.register("TriInv")
def _np_triinv(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np triinv operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    func = adapt_backend_module(backend_module).get_symbol(
        "triinv", adapt_backend_module(backend_module).get_symbol("triinv", None)
    )
    if func is not None:
        return func(*args, **kwargs)
    return _resolve_eager_mod(backend_module).linalg.inv(args[0])


@global_eager_registry.register("UniqueInverse")
def _np_uniqueinverse(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np uniqueinverse operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    func = adapt_backend_module(backend_module).get_symbol(
        "uniqueinverse", adapt_backend_module(backend_module).get_symbol("uniqueinverse", None)
    )
    if func is not None:
        return func(*args, **kwargs)
    return _resolve_eager_mod(backend_module).unique(args[0], return_inverse=True)


@global_eager_registry.register("Vander")
def _np_vander(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np vander operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    func = adapt_backend_module(backend_module).get_symbol(
        "vander", adapt_backend_module(backend_module).get_symbol("vander", None)
    )
    if func is not None:
        return func(*args, **kwargs)
    return _resolve_eager_mod(backend_module).vander(*args, **kwargs)


@global_eager_registry.register("VectorNorm")
def _np_vectornorm(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np vectornorm operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    func = adapt_backend_module(backend_module).get_symbol(
        "vectornorm", adapt_backend_module(backend_module).get_symbol("vectornorm", None)
    )
    if func is not None:
        return func(*args, **kwargs)
    return _resolve_eager_mod(backend_module).linalg.norm(*args, **kwargs)
