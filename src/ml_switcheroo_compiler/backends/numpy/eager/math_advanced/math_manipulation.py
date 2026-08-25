# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""math_manipulation module."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager.math_nan import _xlogy


@numpy_eager_registry.register("Append")
def _np_append(backend_module: object, *args: object, **kwargs: object) -> object:
    """Append values to the end of an array.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: object: The computed result.
    """
    return backend_module.append(*args, **kwargs)


@numpy_eager_registry.register("ArgWhere")
def _np_argwhere(backend_module: object, *args: object, **kwargs: object) -> object:
    """Find the indices of array elements that are non-zero, grouped by element.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: object: The computed result.
    """
    return backend_module.argwhere(*args, **kwargs)


@numpy_eager_registry.register("Choose")
def _np_choose(backend_module: object, *args: object, **kwargs: object) -> object:
    """Construct an array from an index array and a list of arrays to choose from.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: object: The computed result.
    """
    return backend_module.choose(*args, **kwargs)


@numpy_eager_registry.register("ColumnStack")
def _np_column_stack(backend_module: object, *args: object, **kwargs: object) -> object:
    """Stack 1-D arrays as columns into a 2-D array.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: object: The computed result.
    """
    return backend_module.column_stack(*args, **kwargs)


@numpy_eager_registry.register("Compress")
def _np_compress(backend_module: object, *args: object, **kwargs: object) -> object:
    """Return selected slices of an array along given axis.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: object: The computed result.
    """
    return backend_module.compress(*args, **kwargs)


@numpy_eager_registry.register("Delete")
def _np_delete_(backend_module: object, *args: object, **kwargs: object) -> object:
    """Implement Delete via delete.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: object: The computed result.
    """
    return backend_module.delete(*args, **kwargs)


@numpy_eager_registry.register("Extract")
def _np_extract_(backend_module: object, *args: object, **kwargs: object) -> object:
    """Implement Extract via extract.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: object: The computed result.
    """
    return backend_module.extract(*args, **kwargs)


@numpy_eager_registry.register("Flatnonzero")
def _np_flatnonzero_(backend_module: object, *args: object, **kwargs: object) -> object:
    """Implement Flatnonzero via flatnonzero.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: object: The computed result.
    """
    return backend_module.flatnonzero(*args, **kwargs)


@numpy_eager_registry.register("Flip")
def _np_flip_op_(backend_module: object, *args: object, **kwargs: object) -> object:
    """Implement Flip via flip.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: object: The computed result.
    """
    return backend_module.flip(*args, **kwargs)


@numpy_eager_registry.register("Fliplr")
def _np_fliplr_(backend_module: object, *args: object, **kwargs: object) -> object:
    """Implement Fliplr via fliplr.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: object: The computed result.
    """
    return backend_module.fliplr(*args, **kwargs)


@numpy_eager_registry.register("Flipud")
def _np_flipud_(backend_module: object, *args: object, **kwargs: object) -> object:
    """Implement Flipud via flipud.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: object: The computed result.
    """
    return backend_module.flipud(*args, **kwargs)


@numpy_eager_registry.register("Reverse")
def _np_flip_reverse_(backend_module: object, *args: object, **kwargs: object) -> object:
    """Implement Reverse via flip.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: object: The computed result.
    """
    return backend_module.flip(*args, **kwargs)


@numpy_eager_registry.register("Insert")
def _np_insert_(backend_module: object, *args: object, **kwargs: object) -> object:
    """Implement Insert via insert.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: object: The computed result.
    """
    return backend_module.insert(*args, **kwargs)


@numpy_eager_registry.register("Nonzero")
def _np_nonzero_(backend_module: object, *args: object, **kwargs: object) -> object:
    """Implement Nonzero via nonzero.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: object: The computed result.
    """
    return backend_module.nonzero(*args, **kwargs)


@numpy_eager_registry.register("Resize")
def _np_resize_(backend_module: object, *args: object, **kwargs: object) -> object:
    """Implement Resize via resize.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: object: The computed result.
    """
    return backend_module.resize(*args, **kwargs)


@numpy_eager_registry.register("ExtractPatchesOptions")
def _np_extractpatchesoptions(backend_module: object, *args: object, **kwargs: object) -> object:
    """Implement ExtractPatchesOptions.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.ops.vision.bbox import ExtractPatchesOptions

    return ExtractPatchesOptions(*args, **kwargs)


@numpy_eager_registry.register("TensorArrayStack")
def _np_tensorarraystack(backend_module: object, *args: object, **kwargs: object) -> object:
    """Implement TensorArrayStack.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    handle: object = args[0]
    return backend_module.stack(handle)
