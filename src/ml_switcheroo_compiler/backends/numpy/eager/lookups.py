# ruff: noqa: E501
"""Numpy lookup operations."""

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry


@numpy_eager_registry.register("Hashing")
def _np_hashing(backend_module: object, inputs: object, num_bins: int, **kwargs: object) -> object:
    """Evaluate _np_hashing operation.

    Args:
        backend_module (object): The backend_module parameter.
        inputs (object): The inputs parameter.
        num_bins (int): The num_bins parameter.
        **kwargs (object): Keyword args.

    Returns:
        object: Result.
    """
    return inputs


@numpy_eager_registry.register("IntegerLookup")
def _np_integer_lookup(backend_module: object, inputs: object, **kwargs: object) -> object:
    """Evaluate _np_integer_lookup operation.

    Args:
        backend_module (object): The backend_module parameter.
        inputs (object): The inputs parameter.
        **kwargs (object): Keyword args.

    Returns:
        object: Result.
    """
    return inputs


@numpy_eager_registry.register("Lookup")
def _np_lookup(backend_module: object, inputs: object, vocabulary: object, **kwargs: object) -> object:
    """Evaluate _np_lookup operation.

    Args:
        backend_module (object): The backend_module parameter.
        inputs (object): The inputs parameter.
        vocabulary (object): The vocabulary parameter.
        **kwargs (object): Keyword args.

    Returns:
        object: Result.
    """
    inputs = np.asarray(inputs)
    vocab = np.asarray(vocabulary)
    # basic mapping fallback
    res = np.zeros_like(inputs, dtype=np.int32)
    for i, v in enumerate(vocab):
        res[inputs == v] = i
    return res


@numpy_eager_registry.register("StringLookup")
def _np_string_lookup(backend_module: object, inputs: object, **kwargs: object) -> object:
    """Evaluate _np_string_lookup operation.

    Args:
        backend_module (object): The backend_module parameter.
        inputs (object): The inputs parameter.
        **kwargs (object): Keyword args.

    Returns:
        object: Result.
    """
    return inputs
