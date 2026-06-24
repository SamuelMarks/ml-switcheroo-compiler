"""Linalg Ops."""

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry


@numpy_eager_registry.register("Matmul")
def _np_matmul(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    return backend_module.matmul(*args, **kwargs)


@numpy_eager_registry.register("Cross")
def _np_cross(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    return backend_module.cross(*args, **kwargs)


@numpy_eager_registry.register("Norm")
def _np_norm(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    import numpy as np

    return np.linalg.norm(*args, **kwargs)


@numpy_eager_registry.register("DotGeneral")
def _np_dot_general(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    from ml_switcheroo_compiler.backends.numpy.eager.linalg_extras import _dot_general

    return _dot_general(*args, **kwargs)


@numpy_eager_registry.register("Einsum")
def _np_einsum(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    return backend_module.einsum(*args, **kwargs)


@numpy_eager_registry.register("Cholesky")
def _np_cholesky(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    return backend_module.linalg.cholesky(*args, **kwargs)


@numpy_eager_registry.register("Eigh")
def _np_eigh(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    return backend_module.linalg.eigh(*args, **kwargs)


@numpy_eager_registry.register("Inv")
def _np_inv(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    return backend_module.linalg.inv(*args, **kwargs)


@numpy_eager_registry.register("Pinv")
def _np_pinv(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    return backend_module.linalg.pinv(*args, **kwargs)


@numpy_eager_registry.register("Qr")
def _np_qr(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    return backend_module.linalg.qr(*args, **kwargs)


@numpy_eager_registry.register("Svd")
def _np_svd(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    return backend_module.linalg.svd(*args, **kwargs)


@numpy_eager_registry.register("Det")
def _np_det(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    return backend_module.linalg.det(*args, **kwargs)


@numpy_eager_registry.register("Slogdet")
def _np_slogdet(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    return backend_module.linalg.slogdet(*args, **kwargs)


@numpy_eager_registry.register("Solve")
def _np_solve(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    return backend_module.linalg.solve(*args, **kwargs)


@numpy_eager_registry.register("Eigvalsh")
def _np_eigvalsh(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    return backend_module.linalg.eigvalsh(*args, **kwargs)


@numpy_eager_registry.register("MatrixPower")
def _np_matrix_power(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    return backend_module.linalg.matrix_power(*args, **kwargs)


@numpy_eager_registry.register("Eig")
def _np_eig(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    import numpy as np

    return np.linalg.eig(*args, **kwargs)


@numpy_eager_registry.register("Lstsq")
def _np_lstsq(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    import numpy as np

    return np.linalg.lstsq(*args, **kwargs)


@numpy_eager_registry.register("Irfft")
def _np_irfft(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    import numpy as np  # pragma: no cover

    return np.fft.irfft(*args, **kwargs)  # pragma: no cover


@numpy_eager_registry.register("Polar")
def _np_polar(backend_module: object, abs: object, angle: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        abs: Arg.
        angle: Arg.
    """
    import numpy as np

    return abs * np.exp(1j * angle)


@numpy_eager_registry.register("ViewAsComplex")
def _np_view_as_complex(backend_module: object, x: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        x: Arg.
    """
    import numpy as np

    # Assume x has shape (..., 2)
    # Return complex array
    x_np = np.asarray(x)
    return x_np[..., 0] + 1j * x_np[..., 1]


@numpy_eager_registry.register("ViewAsReal")
def _np_view_as_real(backend_module: object, x: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        x: Arg.
    """
    import numpy as np

    x_np = np.asarray(x)
    return np.stack([np.real(x_np), np.imag(x_np)], axis=-1)


@numpy_eager_registry.register("Fft2d")
def _np_fft2d(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    import numpy as np

    return np.fft.fft2(*args, **kwargs)


@numpy_eager_registry.register("Ifft2d")
def _np_ifft2d(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    import numpy as np

    return np.fft.ifft2(*args, **kwargs)
