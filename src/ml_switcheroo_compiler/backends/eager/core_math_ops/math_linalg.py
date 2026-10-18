"""math_linalg module."""

from __future__ import annotations

import builtins  # noqa: F401
from typing import cast

from ml_switcheroo_compiler.backends.eager.utils import get_special_submodule, host_backend
from ml_switcheroo_compiler.backends.eager_registry import EagerValue, global_eager_registry


@global_eager_registry.register("Det")
def _det(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager det operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    func = getattr(backend_module, "linalg", None)
    if func and hasattr(func, "det"):
        return cast(EagerValue, func.det(*args, **kwargs))
    if hasattr(backend_module, "det"):
        return cast(EagerValue, getattr(backend_module, "det")(*args, **kwargs))
    x = args[0]
    return cast(EagerValue, getattr(backend_module, "linalg").det(getattr(backend_module, "asarray")(x)))


@global_eager_registry.register("Eig")
def _eig(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager eig operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    func = getattr(backend_module, "linalg", None)
    if func and hasattr(func, "eig"):
        return cast(EagerValue, func.eig(*args, **kwargs))
    if hasattr(backend_module, "eig"):
        return cast(EagerValue, getattr(backend_module, "eig")(*args, **kwargs))
    x = args[0]
    return cast(EagerValue, getattr(backend_module, "linalg").eig(getattr(backend_module, "asarray")(x)))


@global_eager_registry.register("Eigh")
def _eigh(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager eigh operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    func = getattr(backend_module, "linalg", None)
    if func and hasattr(func, "eigh"):
        return cast(EagerValue, func.eigh(*args, **kwargs))
    if hasattr(backend_module, "eigh"):
        return cast(EagerValue, getattr(backend_module, "eigh")(*args, **kwargs))
    x = args[0]
    return cast(EagerValue, getattr(backend_module, "linalg").eigh(getattr(backend_module, "asarray")(x)))


@global_eager_registry.register("Eigvals")
def _eigvals(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager eigvals operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    func = getattr(backend_module, "linalg", None)
    if func and hasattr(func, "eigvals"):
        return cast(EagerValue, func.eigvals(*args, **kwargs))
    if hasattr(backend_module, "eigvals"):
        return cast(EagerValue, getattr(backend_module, "eigvals")(*args, **kwargs))
    x = args[0]
    return cast(EagerValue, getattr(backend_module, "linalg").eigvals(getattr(backend_module, "asarray")(x)))


@global_eager_registry.register("Eigvalsh")
def _eigvalsh(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager eigvalsh operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    func = getattr(backend_module, "linalg", None)
    if func and hasattr(func, "eigvalsh"):
        return cast(EagerValue, func.eigvalsh(*args, **kwargs))
    if hasattr(backend_module, "eigvalsh"):
        return cast(EagerValue, getattr(backend_module, "eigvalsh")(*args, **kwargs))
    x = args[0]
    return cast(EagerValue, getattr(backend_module, "linalg").eigvalsh(getattr(backend_module, "asarray")(x)))


@global_eager_registry.register("Cholesky")
def _cholesky(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager cholesky operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    func = getattr(backend_module, "linalg", None)
    if func and hasattr(func, "cholesky"):
        return cast(EagerValue, func.cholesky(*args, **kwargs))
    if hasattr(backend_module, "cholesky"):
        return cast(EagerValue, getattr(backend_module, "cholesky")(*args, **kwargs))
    x = args[0]
    return cast(EagerValue, getattr(backend_module, "linalg").cholesky(getattr(backend_module, "asarray")(x)))


@global_eager_registry.register("CholeskyEx")
def _cholesky_ex(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager cholesky ex operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    chol = _cholesky(backend_module, *args, **kwargs)
    if hasattr(backend_module, "zeros"):
        info = getattr(backend_module, "zeros")((), dtype=getattr(backend_module, "int32", int))
    else:
        info = 0
    return (chol, info)


@global_eager_registry.register("Norm")
def _norm(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager norm operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    func = getattr(backend_module, "linalg", None)
    if func and hasattr(func, "norm"):
        return cast(EagerValue, func.norm(*args, **kwargs))
    if hasattr(backend_module, "norm"):
        return cast(EagerValue, getattr(backend_module, "norm")(*args, **kwargs))
    x = args[0]
    return cast(EagerValue, getattr(backend_module, "linalg").norm(getattr(backend_module, "asarray")(x), **kwargs))


@global_eager_registry.register("Pinv")
def _pinv(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager pinv operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    func = getattr(backend_module, "linalg", None)
    if func and hasattr(func, "pinv"):
        return cast(EagerValue, func.pinv(*args, **kwargs))
    if hasattr(backend_module, "pinv"):
        return cast(EagerValue, getattr(backend_module, "pinv")(*args, **kwargs))
    x = args[0]
    return cast(EagerValue, getattr(backend_module, "linalg").pinv(getattr(backend_module, "asarray")(x), **kwargs))


@global_eager_registry.register("Qr")
def _qr(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager qr operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    func = getattr(backend_module, "linalg", None)
    if func and hasattr(func, "qr"):
        return cast(EagerValue, func.qr(*args, **kwargs))
    if hasattr(backend_module, "qr"):
        return cast(EagerValue, getattr(backend_module, "qr")(*args, **kwargs))
    x = args[0]
    return cast(EagerValue, getattr(backend_module, "linalg").qr(getattr(backend_module, "asarray")(x), **kwargs))


@global_eager_registry.register("Svd")
def _svd(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager svd operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    func = getattr(backend_module, "linalg", None)
    if func and hasattr(func, "svd"):
        return cast(EagerValue, func.svd(*args, **kwargs))
    if hasattr(backend_module, "svd"):
        return cast(EagerValue, getattr(backend_module, "svd")(*args, **kwargs))
    x = args[0]
    return cast(EagerValue, getattr(backend_module, "linalg").svd(getattr(backend_module, "asarray")(x), **kwargs))


@global_eager_registry.register("Tensorinv")
def _tensorinv(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager tensorinv operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    func = getattr(backend_module, "linalg", None)
    if func and hasattr(func, "tensorinv"):
        return cast(EagerValue, func.tensorinv(*args, **kwargs))
    if hasattr(backend_module, "tensorinv"):
        return cast(EagerValue, getattr(backend_module, "tensorinv")(*args, **kwargs))
    x = args[0]
    return cast(EagerValue, getattr(backend_module, "linalg").tensorinv(getattr(backend_module, "asarray")(x), **kwargs))


@global_eager_registry.register("Inv")
def _inv(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager inv operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    func = getattr(backend_module, "linalg", None)
    if func and hasattr(func, "inv"):
        return cast(EagerValue, func.inv(*args, **kwargs))
    if hasattr(backend_module, "inv"):
        return cast(EagerValue, getattr(backend_module, "inv")(*args, **kwargs))
    x = args[0]
    return cast(EagerValue, getattr(backend_module, "linalg").inv(getattr(backend_module, "asarray")(x)))


@global_eager_registry.register("Vander")
def _vander(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager vander operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    return cast(EagerValue, getattr(backend_module, "vander")(*args, **kwargs))


@global_eager_registry.register("Schur")
def _np_schur(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np schur operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    func = getattr(backend_module, "schur", getattr(backend_module, "schur", None))
    if func is not None:
        return cast(EagerValue, func(*args, **kwargs))
    return cast(EagerValue, get_special_submodule("linalg").schur(*args, **kwargs))


@global_eager_registry.register("Svd")
def _np_svd(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np svd operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    func = getattr(backend_module, "svd", getattr(backend_module, "svd", None))
    if func is not None:
        return cast(EagerValue, func(*args, **kwargs))
    return cast(EagerValue, host_backend.linalg.svd(*args, **kwargs))


@global_eager_registry.register("Svdvals")
def _np_svdvals(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np svdvals operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    func = getattr(backend_module, "svdvals", getattr(backend_module, "svdvals", None))
    if func is not None:
        return cast(EagerValue, func(*args, **kwargs))
    return cast(EagerValue, host_backend.linalg.svd(*args, **kwargs)[1])


@global_eager_registry.register("Tensorinv")
def _np_tensorinv(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np tensorinv operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    func = getattr(backend_module, "tensorinv", getattr(backend_module, "tensorinv", None))
    if func is not None:
        return cast(EagerValue, func(*args, **kwargs))
    return cast(EagerValue, host_backend.linalg.tensorinv(*args, **kwargs))


@global_eager_registry.register("TriInv")
def _np_triinv(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np triinv operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    func = getattr(backend_module, "triinv", getattr(backend_module, "triinv", None))
    if func is not None:
        return cast(EagerValue, func(*args, **kwargs))
    return cast(EagerValue, host_backend.linalg.inv(args[0]))


@global_eager_registry.register("UniqueInverse")
def _np_uniqueinverse(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np uniqueinverse operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    func = getattr(backend_module, "uniqueinverse", getattr(backend_module, "uniqueinverse", None))
    if func is not None:
        return cast(EagerValue, func(*args, **kwargs))
    return cast(EagerValue, host_backend.unique(args[0], return_inverse=True))


@global_eager_registry.register("Vander")
def _np_vander(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np vander operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    func = getattr(backend_module, "vander", getattr(backend_module, "vander", None))
    if func is not None:
        return cast(EagerValue, func(*args, **kwargs))
    return cast(EagerValue, host_backend.vander(*args, **kwargs))


@global_eager_registry.register("VectorNorm")
def _np_vectornorm(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np vectornorm operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    func = getattr(backend_module, "vectornorm", getattr(backend_module, "vectornorm", None))
    if func is not None:
        return cast(EagerValue, func(*args, **kwargs))
    return cast(EagerValue, host_backend.linalg.norm(*args, **kwargs))
