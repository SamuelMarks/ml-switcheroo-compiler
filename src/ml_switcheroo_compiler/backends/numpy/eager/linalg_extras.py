"""Linalg extras module."""

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry
import numpy as np

"""Module docstring."""


def _get_uncontracted_dims(dims: list[int], batch: list[int], contracting: list[int]) -> list[int]:
    """Function docstring.

    Args:
        dims: Arg.
        batch: Arg.
        contracting: Arg.
    """
    skip_set = set(batch) | set(contracting)
    return [dims[i] for i in range(len(dims)) if (i not in skip_set)]


def _parse_dot_dimension_numbers(dimension_numbers: object) -> tuple:
    """Function docstring.

    Args:
        dimension_numbers: Arg.
    """
    (contracting, batch) = dimension_numbers
    (a_contracting, b_contracting) = contracting
    (a_batch, b_batch) = batch
    return (a_contracting, b_contracting, a_batch, b_batch)


def _dot_general(a: object, b: object, dimension_numbers: object) -> object:
    r"""Execute _dot_general.\n\n    Args:\n        a (Any): Argument a.\n        b (Any): Argument b.\n        dimension_numbers (Any): Argument dimension_numbers.\n\n    Returns:\n    Any: The result.\n."""
    (a_dims, b_dims, out_dims) = _build_einsum_equation(a.ndim, b.ndim, dimension_numbers)
    return np.einsum(a, a_dims, b, b_dims, out_dims)


def _build_einsum_equation(
    a_ndim: int, b_ndim: int, dimension_numbers: object
) -> tuple[(list[int], list[int], list[int])]:
    """Function docstring.

    Args:
        a_ndim: Arg.
        b_ndim: Arg.
        dimension_numbers: Arg.
    """
    (a_contracting, b_contracting, a_batch, b_batch) = _parse_dot_dimension_numbers(
        dimension_numbers
    )
    a_dims = list(range(a_ndim))
    b_dims = list(range(a_ndim, (a_ndim + b_ndim)))
    for i, a_b in enumerate(a_batch):
        b_dims[b_batch[i]] = a_dims[a_b]
    for i, a_c in enumerate(a_contracting):
        b_dims[b_contracting[i]] = a_dims[a_c]
    out_dims = [a_dims[i] for i in a_batch]
    out_dims.extend(_get_uncontracted_dims(a_dims, a_batch, a_contracting))
    out_dims.extend(_get_uncontracted_dims(b_dims, b_batch, b_contracting))
    return (a_dims, b_dims, out_dims)


@numpy_eager_registry.register("Trace")
def _np_trace(backend_module: object, *args: object, **kwargs: object) -> object:
    return np.trace(args[0], **kwargs)


@numpy_eager_registry.register("MatrixRank")
def _np_matrix_rank(backend_module: object, *args: object, **kwargs: object) -> object:
    return np.linalg.matrix_rank(args[0], **kwargs)


@numpy_eager_registry.register("MatrixTranspose")
def _np_matrix_transpose(backend_module: object, *args: object, **kwargs: object) -> object:
    return np.swapaxes(args[0], -1, -2)


@numpy_eager_registry.register("Sqrtm")
def _np_sqrtm(backend_module: object, *args: object, **kwargs: object) -> object:
    from scipy.linalg import sqrtm

    return sqrtm(args[0])


@numpy_eager_registry.register("Adjoint")
def _np_adjoint(backend_module: object, *args: object, **kwargs: object) -> object:
    import numpy as np

    return np.conj(np.swapaxes(args[0], -1, -2))


@numpy_eager_registry.register("CholeskySolve")
def _np_cholesky_solve(backend_module: object, *args: object, **kwargs: object) -> object:
    return args[0]  # dummy


@numpy_eager_registry.register("EighTridiagonal")
def _np_eigh_tridiagonal(backend_module: object, *args: object, **kwargs: object) -> object:
    return args[0]  # dummy
