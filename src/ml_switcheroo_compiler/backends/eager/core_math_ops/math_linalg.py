"""Implementation of math linalg operations and utilities for the backends subsystem (`ml_switcheroo_compiler.backends.eager.core_math_ops.math_linalg`)."""

from __future__ import annotations

import builtins  # noqa: F401

from ml_switcheroo_compiler.backends.eager.utils import _resolve_eager_mod
from ml_switcheroo_compiler.backends.eager_registry import EagerValue, as_tensor, global_eager_registry
from ml_switcheroo_compiler.backends.protocols import BackendModuleProtocol
from ml_switcheroo_compiler.core.shape import _get_attr, _has_attr

from .math_special import _dispatch_special


@global_eager_registry.register("Det")
def _det(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager det operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    func = _get_attr(backend_module, "linalg", None)
    if func and _has_attr(func, "det"):
        return as_tensor(func.det(*args, **kwargs))
    if _has_attr(backend_module, "det"):
        return as_tensor(_get_attr(backend_module, "det")(*args, **kwargs))
    x = args[0]
    return as_tensor(_get_attr(backend_module, "linalg").det(_get_attr(backend_module, "asarray")(x)))


@global_eager_registry.register("Eig")
def _eig(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager eig operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    func = _get_attr(backend_module, "linalg", None)
    if func and _has_attr(func, "eig"):
        return as_tensor(func.eig(*args, **kwargs))
    if _has_attr(backend_module, "eig"):
        return as_tensor(_get_attr(backend_module, "eig")(*args, **kwargs))
    x = args[0]
    return as_tensor(_get_attr(backend_module, "linalg").eig(_get_attr(backend_module, "asarray")(x)))


@global_eager_registry.register("Eigh")
def _eigh(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager eigh operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    func = _get_attr(backend_module, "linalg", None)
    if func and _has_attr(func, "eigh"):
        return as_tensor(func.eigh(*args, **kwargs))
    if _has_attr(backend_module, "eigh"):
        return as_tensor(_get_attr(backend_module, "eigh")(*args, **kwargs))
    x = args[0]
    return as_tensor(_get_attr(backend_module, "linalg").eigh(_get_attr(backend_module, "asarray")(x)))


@global_eager_registry.register("Eigvals")
def _eigvals(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager eigvals operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    func = _get_attr(backend_module, "linalg", None)
    if func and _has_attr(func, "eigvals"):
        return as_tensor(func.eigvals(*args, **kwargs))
    if _has_attr(backend_module, "eigvals"):
        return as_tensor(_get_attr(backend_module, "eigvals")(*args, **kwargs))
    x = args[0]
    return as_tensor(_get_attr(backend_module, "linalg").eigvals(_get_attr(backend_module, "asarray")(x)))


@global_eager_registry.register("Eigvalsh")
def _eigvalsh(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager eigvalsh operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    func = _get_attr(backend_module, "linalg", None)
    if func and _has_attr(func, "eigvalsh"):
        return as_tensor(func.eigvalsh(*args, **kwargs))
    if _has_attr(backend_module, "eigvalsh"):
        return as_tensor(_get_attr(backend_module, "eigvalsh")(*args, **kwargs))
    x = args[0]
    return as_tensor(_get_attr(backend_module, "linalg").eigvalsh(_get_attr(backend_module, "asarray")(x)))


@global_eager_registry.register("Cholesky")
def _cholesky(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager cholesky operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    func = _get_attr(backend_module, "linalg", None)
    if func and _has_attr(func, "cholesky"):
        return as_tensor(func.cholesky(*args, **kwargs))
    if _has_attr(backend_module, "cholesky"):
        return as_tensor(_get_attr(backend_module, "cholesky")(*args, **kwargs))
    x = args[0]
    return as_tensor(_get_attr(backend_module, "linalg").cholesky(_get_attr(backend_module, "asarray")(x)))


@global_eager_registry.register("CholeskyEx")
def _cholesky_ex(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager cholesky ex operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    chol = _cholesky(backend_module, *args, **kwargs)
    if _has_attr(backend_module, "zeros"):
        info = _get_attr(backend_module, "zeros")((), dtype=_get_attr(backend_module, "int32", int))
    else:
        info = 0
    return (chol, info)


@global_eager_registry.register("Norm")
def _norm(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager norm operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    func = _get_attr(backend_module, "linalg", None)
    if func and _has_attr(func, "norm"):
        return as_tensor(func.norm(*args, **kwargs))
    if _has_attr(backend_module, "norm"):
        return as_tensor(_get_attr(backend_module, "norm")(*args, **kwargs))
    x = args[0]
    return as_tensor(_get_attr(backend_module, "linalg").norm(_get_attr(backend_module, "asarray")(x), **kwargs))


@global_eager_registry.register("Pinv")
def _pinv(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager pinv operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    func = _get_attr(backend_module, "linalg", None)
    if func and _has_attr(func, "pinv"):
        return as_tensor(func.pinv(*args, **kwargs))
    if _has_attr(backend_module, "pinv"):
        return as_tensor(_get_attr(backend_module, "pinv")(*args, **kwargs))
    x = args[0]
    return as_tensor(_get_attr(backend_module, "linalg").pinv(_get_attr(backend_module, "asarray")(x), **kwargs))


@global_eager_registry.register("Qr")
def _qr(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager qr operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    func = _get_attr(backend_module, "linalg", None)
    if func and _has_attr(func, "qr"):
        return as_tensor(func.qr(*args, **kwargs))
    if _has_attr(backend_module, "qr"):
        return as_tensor(_get_attr(backend_module, "qr")(*args, **kwargs))
    x = args[0]
    return as_tensor(_get_attr(backend_module, "linalg").qr(_get_attr(backend_module, "asarray")(x), **kwargs))


@global_eager_registry.register("Svd")
def _svd(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager svd operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    func = _get_attr(backend_module, "linalg", None)
    if func and _has_attr(func, "svd"):
        return as_tensor(func.svd(*args, **kwargs))
    if _has_attr(backend_module, "svd"):
        return as_tensor(_get_attr(backend_module, "svd")(*args, **kwargs))
    x = args[0]
    return as_tensor(_get_attr(backend_module, "linalg").svd(_get_attr(backend_module, "asarray")(x), **kwargs))


@global_eager_registry.register("Tensorinv")
def _tensorinv(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager tensorinv operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    func = _get_attr(backend_module, "linalg", None)
    if func and _has_attr(func, "tensorinv"):
        return as_tensor(func.tensorinv(*args, **kwargs))
    if _has_attr(backend_module, "tensorinv"):
        return as_tensor(_get_attr(backend_module, "tensorinv")(*args, **kwargs))
    x = args[0]
    return as_tensor(_get_attr(backend_module, "linalg").tensorinv(_get_attr(backend_module, "asarray")(x), **kwargs))


@global_eager_registry.register("Inv")
def _inv(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager inv operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    func = _get_attr(backend_module, "linalg", None)
    if func and _has_attr(func, "inv"):
        return as_tensor(func.inv(*args, **kwargs))
    if _has_attr(backend_module, "inv"):
        return as_tensor(_get_attr(backend_module, "inv")(*args, **kwargs))
    x = args[0]
    return as_tensor(_get_attr(backend_module, "linalg").inv(_get_attr(backend_module, "asarray")(x)))


@global_eager_registry.register("Vander")
def _vander(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager vander operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    return as_tensor(_get_attr(backend_module, "vander")(*args, **kwargs))


@global_eager_registry.register("Schur")
def _np_schur(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np schur operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    return _dispatch_special("Schur", "schur", backend_module, *args, **kwargs)


@global_eager_registry.register("Svd")
def _np_svd(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np svd operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    func = _get_attr(backend_module, "svd", _get_attr(backend_module, "svd", None))
    if func is not None:
        return as_tensor(func(*args, **kwargs))
    return _resolve_eager_mod(backend_module).linalg.svd(*args, **kwargs)


@global_eager_registry.register("Svdvals")
def _np_svdvals(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np svdvals operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    func = _get_attr(backend_module, "svdvals", _get_attr(backend_module, "svdvals", None))
    if func is not None:
        return as_tensor(func(*args, **kwargs))
    return _resolve_eager_mod(backend_module).linalg.svd(*args, **kwargs)[1]


@global_eager_registry.register("Tensorinv")
def _np_tensorinv(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np tensorinv operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    func = _get_attr(backend_module, "tensorinv", _get_attr(backend_module, "tensorinv", None))
    if func is not None:
        return as_tensor(func(*args, **kwargs))
    return _resolve_eager_mod(backend_module).linalg.tensorinv(*args, **kwargs)


@global_eager_registry.register("TriInv")
def _np_triinv(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np triinv operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    func = _get_attr(backend_module, "triinv", _get_attr(backend_module, "triinv", None))
    if func is not None:
        return as_tensor(func(*args, **kwargs))
    return _resolve_eager_mod(backend_module).linalg.inv(args[0])


@global_eager_registry.register("UniqueInverse")
def _np_uniqueinverse(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np uniqueinverse operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    func = _get_attr(backend_module, "uniqueinverse", _get_attr(backend_module, "uniqueinverse", None))
    if func is not None:
        return as_tensor(func(*args, **kwargs))
    return _resolve_eager_mod(backend_module).unique(args[0], return_inverse=True)


@global_eager_registry.register("Vander")
def _np_vander(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np vander operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    func = _get_attr(backend_module, "vander", _get_attr(backend_module, "vander", None))
    if func is not None:
        return as_tensor(func(*args, **kwargs))
    return _resolve_eager_mod(backend_module).vander(*args, **kwargs)


@global_eager_registry.register("VectorNorm")
def _np_vectornorm(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np vectornorm operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    func = _get_attr(backend_module, "vectornorm", _get_attr(backend_module, "vectornorm", None))
    if func is not None:
        return as_tensor(func(*args, **kwargs))
    return _resolve_eager_mod(backend_module).linalg.norm(*args, **kwargs)
