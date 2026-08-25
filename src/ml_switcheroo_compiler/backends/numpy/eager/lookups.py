# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
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
            tuple[int, ...]: Result.
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
            tuple[int, ...]: Result.
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
            tuple[int, ...]: Result.
    """
    inputs: object = np.asarray(inputs)
    vocab: object = np.asarray(vocabulary)
    # basic mapping fallback
    res: object = np.zeros_like(inputs, dtype=np.int32)
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
            tuple[int, ...]: Result.
    """
    return inputs
