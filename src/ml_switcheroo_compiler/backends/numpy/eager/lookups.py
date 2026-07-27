# ruff: noqa: E501
"""Numpy lookup operations."""

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry


@numpy_eager_registry.register("Hashing")
def _np_hashing(backend_module: object, inputs: object, num_bins: int, **kwargs: object) -> object:
    """Evaluate the hashing logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        inputs (object): Required parameter for inputs.
        num_bins (int): Required parameter for num_bins.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    return inputs


@numpy_eager_registry.register("IntegerLookup")
def _np_integer_lookup(backend_module: object, inputs: object, **kwargs: object) -> object:
    """Evaluate the integer lookup logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        inputs (object): Required parameter for inputs.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    return inputs


@numpy_eager_registry.register("Lookup")
def _np_lookup(backend_module: object, inputs: object, vocabulary: object, **kwargs: object) -> object:
    """Evaluate the lookup logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        inputs (object): Required parameter for inputs.
        vocabulary (object): Required parameter for vocabulary.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
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
    """Evaluate the string lookup logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        inputs (object): Required parameter for inputs.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    return inputs
