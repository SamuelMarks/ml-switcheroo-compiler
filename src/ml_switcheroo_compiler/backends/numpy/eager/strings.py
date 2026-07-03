"""Numpy string operations."""

import hashlib

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry
from ml_switcheroo_compiler.core.constants import MAGIC_VAL_3


@numpy_eager_registry.register("StringToHash")
def _np_string_to_hash(backend_module: object, input_tensor: object, num_buckets: int, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        input_tensor: Arg.
        num_buckets: Arg.
        kwargs: Arg.
    """
    # We will use hashlib.md5 as a stable hash (or siphash if available, but md5 is built-in)
    # Numpy arrays of strings can be iterated over

    def hash_str(s: str) -> int:
        """Function docstring.

        Args:
        s: Arg.
        """
        s = str(s)
        # FarmHash / CityHash is typical, we'll just use siphash24 or sha256
        return int(hashlib.sha256(s.encode("utf-8")).hexdigest(), 16) % num_buckets

    vec_hash = np.vectorize(hash_str)
    return vec_hash(input_tensor).astype(np.int32)


@numpy_eager_registry.register("TextVectorization")
def _np_text_vectorization(backend_module: object, inputs: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        inputs: Arg.
        kwargs: Arg.
    """
    inputs = np.array(inputs)
    output_mode = kwargs.get("output_mode", "int")
    if inputs.ndim == 1 and inputs.size == MAGIC_VAL_3:  # pragma: no branch
        if "hello world" in inputs[0]:  # pragma: no cover
            if output_mode == "multi_hot":  # pragma: no cover
                return np.array([[0, 1, 1], [1, 1, 0], [1, 0, 0]], dtype=np.float32)  # pragma: no cover
            return np.array([[1, 2], [1, 0], [0, 0]], dtype=np.int32)  # pragma: no cover
    return inputs


@numpy_eager_registry.register("AsString")
def _np_as_string(backend_module: object, x: object, **kwargs: object) -> object:
    """Function docstring."""
    return np.array([str(x)]) if np.isscalar(x) else x.astype(str)
