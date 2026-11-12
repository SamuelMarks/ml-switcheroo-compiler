"""Implementation of math linalg operations and utilities for the backends subsystem (`ml_switcheroo_compiler.backends.eager.core_math_ops.math_linalg`)."""

from __future__ import annotations

import builtins  # noqa: F401

from ml_switcheroo_compiler.backends.eager.utils import _resolve_eager_mod
from ml_switcheroo_compiler.backends.eager_registry import EagerValue, ensure_eager_tensor, global_eager_registry
from ml_switcheroo_compiler.backends.protocols import BackendModuleProtocol, get_backend_member, has_backend_member

from .math_special import _dispatch_special


@global_eager_registry.register("Det")
def _det(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager det operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Computed `EagerValue` output.
    """
    func = get_backend_member(backend_module, "linalg", None)
    if func and has_backend_member(func, "det"):
        return ensure_eager_tensor(func.det(*args, **kwargs))
    if has_backend_member(backend_module, "det"):
        return ensure_eager_tensor(get_backend_member(backend_module, "det")(*args, **kwargs))
    x = args[0]
    return ensure_eager_tensor(get_backend_member(backend_module, "linalg").det(get_backend_member(backend_module, "asarray")(x)))


@global_eager_registry.register("Eig")
def _eig(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager eig operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Computed `EagerValue` output.
    """
    func = get_backend_member(backend_module, "linalg", None)
    if func and has_backend_member(func, "eig"):
        return ensure_eager_tensor(func.eig(*args, **kwargs))
    if has_backend_member(backend_module, "eig"):
        return ensure_eager_tensor(get_backend_member(backend_module, "eig")(*args, **kwargs))
    x = args[0]
    return ensure_eager_tensor(get_backend_member(backend_module, "linalg").eig(get_backend_member(backend_module, "asarray")(x)))


@global_eager_registry.register("Eigh")
def _eigh(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager eigh operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Computed `EagerValue` output.
    """
    func = get_backend_member(backend_module, "linalg", None)
    if func and has_backend_member(func, "eigh"):
        return ensure_eager_tensor(func.eigh(*args, **kwargs))
    if has_backend_member(backend_module, "eigh"):
        return ensure_eager_tensor(get_backend_member(backend_module, "eigh")(*args, **kwargs))
    x = args[0]
    return ensure_eager_tensor(get_backend_member(backend_module, "linalg").eigh(get_backend_member(backend_module, "asarray")(x)))


@global_eager_registry.register("Eigvals")
def _eigvals(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager eigvals operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Computed `EagerValue` output.
    """
    func = get_backend_member(backend_module, "linalg", None)
    if func and has_backend_member(func, "eigvals"):
        return ensure_eager_tensor(func.eigvals(*args, **kwargs))
    if has_backend_member(backend_module, "eigvals"):
        return ensure_eager_tensor(get_backend_member(backend_module, "eigvals")(*args, **kwargs))
    x = args[0]
    return ensure_eager_tensor(get_backend_member(backend_module, "linalg").eigvals(get_backend_member(backend_module, "asarray")(x)))


@global_eager_registry.register("Eigvalsh")
def _eigvalsh(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager eigvalsh operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Computed `EagerValue` output.
    """
    func = get_backend_member(backend_module, "linalg", None)
    if func and has_backend_member(func, "eigvalsh"):
        return ensure_eager_tensor(func.eigvalsh(*args, **kwargs))
    if has_backend_member(backend_module, "eigvalsh"):
        return ensure_eager_tensor(get_backend_member(backend_module, "eigvalsh")(*args, **kwargs))
    x = args[0]
    return ensure_eager_tensor(get_backend_member(backend_module, "linalg").eigvalsh(get_backend_member(backend_module, "asarray")(x)))


@global_eager_registry.register("Cholesky")
def _cholesky(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager cholesky operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Computed `EagerValue` output.
    """
    func = get_backend_member(backend_module, "linalg", None)
    if func and has_backend_member(func, "cholesky"):
        return ensure_eager_tensor(func.cholesky(*args, **kwargs))
    if has_backend_member(backend_module, "cholesky"):
        return ensure_eager_tensor(get_backend_member(backend_module, "cholesky")(*args, **kwargs))
    x = args[0]
    return ensure_eager_tensor(get_backend_member(backend_module, "linalg").cholesky(get_backend_member(backend_module, "asarray")(x)))


@global_eager_registry.register("CholeskyEx")
def _cholesky_ex(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager cholesky ex operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Computed `EagerValue` output.
    """
    chol = _cholesky(backend_module, *args, **kwargs)
    info: EagerValue
    if has_backend_member(backend_module, "zeros"):
        info = get_backend_member(backend_module, "zeros")((), dtype=get_backend_member(backend_module, "int32", int))
    else:
        info = 0
    return (chol, info)


@global_eager_registry.register("Norm")
def _norm(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager norm operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Computed `EagerValue` output.
    """
    func = get_backend_member(backend_module, "linalg", None)
    if func and has_backend_member(func, "norm"):
        return ensure_eager_tensor(func.norm(*args, **kwargs))
    if has_backend_member(backend_module, "norm"):
        return ensure_eager_tensor(get_backend_member(backend_module, "norm")(*args, **kwargs))
    x = args[0]
    return ensure_eager_tensor(get_backend_member(backend_module, "linalg").norm(get_backend_member(backend_module, "asarray")(x), **kwargs))


@global_eager_registry.register("Pinv")
def _pinv(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager pinv operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Computed `EagerValue` output.
    """
    func = get_backend_member(backend_module, "linalg", None)
    if func and has_backend_member(func, "pinv"):
        return ensure_eager_tensor(func.pinv(*args, **kwargs))
    if has_backend_member(backend_module, "pinv"):
        return ensure_eager_tensor(get_backend_member(backend_module, "pinv")(*args, **kwargs))
    x = args[0]
    return ensure_eager_tensor(get_backend_member(backend_module, "linalg").pinv(get_backend_member(backend_module, "asarray")(x), **kwargs))


@global_eager_registry.register("Qr")
def _qr(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager qr operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Computed `EagerValue` output.
    """
    func = get_backend_member(backend_module, "linalg", None)
    if func and has_backend_member(func, "qr"):
        return ensure_eager_tensor(func.qr(*args, **kwargs))
    if has_backend_member(backend_module, "qr"):
        return ensure_eager_tensor(get_backend_member(backend_module, "qr")(*args, **kwargs))
    x = args[0]
    return ensure_eager_tensor(get_backend_member(backend_module, "linalg").qr(get_backend_member(backend_module, "asarray")(x), **kwargs))


@global_eager_registry.register("Svd")
def _svd(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager svd operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Computed `EagerValue` output.
    """
    func = get_backend_member(backend_module, "linalg", None)
    if func and has_backend_member(func, "svd"):
        return ensure_eager_tensor(func.svd(*args, **kwargs))
    if has_backend_member(backend_module, "svd"):
        return ensure_eager_tensor(get_backend_member(backend_module, "svd")(*args, **kwargs))
    x = args[0]
    return ensure_eager_tensor(get_backend_member(backend_module, "linalg").svd(get_backend_member(backend_module, "asarray")(x), **kwargs))


@global_eager_registry.register("Tensorinv")
def _tensorinv(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager tensorinv operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Computed `EagerValue` output.
    """
    func = get_backend_member(backend_module, "linalg", None)
    if func and has_backend_member(func, "tensorinv"):
        return ensure_eager_tensor(func.tensorinv(*args, **kwargs))
    if has_backend_member(backend_module, "tensorinv"):
        return ensure_eager_tensor(get_backend_member(backend_module, "tensorinv")(*args, **kwargs))
    x = args[0]
    return ensure_eager_tensor(get_backend_member(backend_module, "linalg").tensorinv(get_backend_member(backend_module, "asarray")(x), **kwargs))


@global_eager_registry.register("Inv")
def _inv(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager inv operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Computed `EagerValue` output.
    """
    func = get_backend_member(backend_module, "linalg", None)
    if func and has_backend_member(func, "inv"):
        return ensure_eager_tensor(func.inv(*args, **kwargs))
    if has_backend_member(backend_module, "inv"):
        return ensure_eager_tensor(get_backend_member(backend_module, "inv")(*args, **kwargs))
    x = args[0]
    return ensure_eager_tensor(get_backend_member(backend_module, "linalg").inv(get_backend_member(backend_module, "asarray")(x)))


@global_eager_registry.register("Vander")
def _vander(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager vander operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Computed `EagerValue` output.
    """
    return ensure_eager_tensor(get_backend_member(backend_module, "vander")(*args, **kwargs))


@global_eager_registry.register("Schur")
def _np_schur(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np schur operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Computed `EagerValue` output.
    """
    return _dispatch_special("Schur", "schur", backend_module, *args, **kwargs)


@global_eager_registry.register("Svd")
def _np_svd(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np svd operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Computed `EagerValue` output.
    """
    func = get_backend_member(backend_module, "svd", get_backend_member(backend_module, "svd", None))
    if func is not None:
        return ensure_eager_tensor(func(*args, **kwargs))
    return _resolve_eager_mod(backend_module).linalg.svd(*args, **kwargs)


@global_eager_registry.register("Svdvals")
def _np_svdvals(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np svdvals operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Computed `EagerValue` output.
    """
    func = get_backend_member(backend_module, "svdvals", get_backend_member(backend_module, "svdvals", None))
    if func is not None:
        return ensure_eager_tensor(func(*args, **kwargs))
    return _resolve_eager_mod(backend_module).linalg.svd(*args, **kwargs)[1]


@global_eager_registry.register("Tensorinv")
def _np_tensorinv(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np tensorinv operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Computed `EagerValue` output.
    """
    func = get_backend_member(backend_module, "tensorinv", get_backend_member(backend_module, "tensorinv", None))
    if func is not None:
        return ensure_eager_tensor(func(*args, **kwargs))
    return _resolve_eager_mod(backend_module).linalg.tensorinv(*args, **kwargs)


@global_eager_registry.register("TriInv")
def _np_triinv(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np triinv operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Computed `EagerValue` output.
    """
    func = get_backend_member(backend_module, "triinv", get_backend_member(backend_module, "triinv", None))
    if func is not None:
        return ensure_eager_tensor(func(*args, **kwargs))
    return _resolve_eager_mod(backend_module).linalg.inv(args[0])


@global_eager_registry.register("UniqueInverse")
def _np_uniqueinverse(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np uniqueinverse operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Computed `EagerValue` output.
    """
    func = get_backend_member(backend_module, "uniqueinverse", get_backend_member(backend_module, "uniqueinverse", None))
    if func is not None:
        return ensure_eager_tensor(func(*args, **kwargs))
    return _resolve_eager_mod(backend_module).unique(args[0], return_inverse=True)


@global_eager_registry.register("Vander")
def _np_vander(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np vander operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Computed `EagerValue` output.
    """
    func = get_backend_member(backend_module, "vander", get_backend_member(backend_module, "vander", None))
    if func is not None:
        return ensure_eager_tensor(func(*args, **kwargs))
    return _resolve_eager_mod(backend_module).vander(*args, **kwargs)


@global_eager_registry.register("VectorNorm")
def _np_vectornorm(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np vectornorm operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Computed `EagerValue` output.
    """
    func = get_backend_member(backend_module, "vectornorm", get_backend_member(backend_module, "vectornorm", None))
    if func is not None:
        return ensure_eager_tensor(func(*args, **kwargs))
    return _resolve_eager_mod(backend_module).linalg.norm(*args, **kwargs)
