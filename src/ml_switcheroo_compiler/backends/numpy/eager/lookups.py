# ruff: noqa: E402, D100, D103, D104, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, D101, D102, D107, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Numpy lookup operations."""

from typing import Any

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry


@numpy_eager_registry.register("Hashing")
def _np_hashing(backend_module: Any, inputs: Any, num_bins: int, **kwargs: Any) -> Any:
    """Evaluate _np_hashing operation.

    Args:
        backend_module (object): The backend_module parameter.
        inputs (object): The inputs parameter.
        num_bins (int): The num_bins parameter.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return inputs


@numpy_eager_registry.register("IntegerLookup")
def _np_integer_lookup(backend_module: Any, inputs: Any, **kwargs: Any) -> Any:
    """Evaluate _np_integer_lookup operation.

    Args:
        backend_module (object): The backend_module parameter.
        inputs (object): The inputs parameter.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return inputs


@numpy_eager_registry.register("Lookup")
def _np_lookup(backend_module: Any, inputs: Any, vocabulary: Any, **kwargs: Any) -> Any:
    """Evaluate _np_lookup operation.

    Args:
        backend_module (object): The backend_module parameter.
        inputs (object): The inputs parameter.
        vocabulary (object): The vocabulary parameter.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    inputs = np.asarray(inputs)
    vocab = np.asarray(vocabulary)
    # basic mapping fallback
    res = np.zeros_like(inputs, dtype=np.int32)
    for i, v in enumerate(vocab):
        res[inputs == v] = i
    return res


@numpy_eager_registry.register("StringLookup")
def _np_string_lookup(backend_module: Any, inputs: Any, **kwargs: Any) -> Any:
    """Evaluate _np_string_lookup operation.

    Args:
        backend_module (object): The backend_module parameter.
        inputs (object): The inputs parameter.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return inputs
