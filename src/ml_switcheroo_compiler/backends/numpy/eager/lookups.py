"""Numpy lookup operations."""

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry


@numpy_eager_registry.register("Hashing")
def _np_hashing(backend_module: object, inputs: object, num_bins: int, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        inputs: Arg.
        num_bins: Arg.
        kwargs: Arg.
    """
    return inputs


@numpy_eager_registry.register("IntegerLookup")
def _np_integer_lookup(backend_module: object, inputs: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        inputs: Arg.
        kwargs: Arg.
    """
    return inputs


@numpy_eager_registry.register("Lookup")
def _np_lookup(backend_module: object, inputs: object, vocabulary: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        inputs: Arg.
        vocabulary: Arg.
        kwargs: Arg.
    """
    return np.zeros_like(inputs, dtype=np.int32)


@numpy_eager_registry.register("StringLookup")
def _np_string_lookup(backend_module: object, inputs: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        inputs: Arg.
        kwargs: Arg.
    """
    return inputs
