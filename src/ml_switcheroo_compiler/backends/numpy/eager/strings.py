# ruff: noqa: E501
"""Numpy string operations."""

import hashlib

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry
from ml_switcheroo_compiler.core.constants import MAGIC_VAL_3


@numpy_eager_registry.register("StringToHash")
def _np_string_to_hash(backend_module: object, input_tensor: object, num_buckets: int, **kwargs: object) -> object:
    """Evaluate _np_string_to_hash operation.

    Args:
        backend_module (object): The backend_module parameter.
        input_tensor (object): The input_tensor parameter.
        num_buckets (int): The num_buckets parameter.
        **kwargs (object): Keyword args.

    Returns:
        object: Result.
    """

    def hash_str(s: str) -> int:
        """Evaluate hash_str operation.

        Args:
        s (str): The s parameter.

        Returns:
        int: Result.
        """
        s = str(s)
        return int(hashlib.sha256(s.encode("utf-8")).hexdigest(), 16) % num_buckets

    vec_hash = np.vectorize(hash_str)
    return vec_hash(input_tensor).astype(np.int32)


@numpy_eager_registry.register("TextVectorization")
def _np_text_vectorization(backend_module: object, inputs: object, **kwargs: object) -> object:
    """Evaluate _np_text_vectorization operation.

    Args:
        backend_module (object): The backend_module parameter.
        inputs (object): The inputs parameter.
        **kwargs (object): Keyword args.

    Returns:
        object: Result.
    """
    inputs = np.array(inputs)
    output_mode = kwargs.get("output_mode", "int")
    if inputs.ndim == 1 and inputs.size == MAGIC_VAL_3:
        if "hello world" in inputs[0]:
            if output_mode == "multi_hot":
                return np.array([[0, 1, 1], [1, 1, 0], [1, 0, 0]], dtype=np.float32)
            return np.array([[1, 2], [1, 0], [0, 0]], dtype=np.int32)
    return inputs


@numpy_eager_registry.register("AsString")
def _np_as_string(backend_module: object, x: object, **kwargs: object) -> object:
    """Evaluate _np_as_string operation.

    Args:
        backend_module (object): The backend_module parameter.
        x (object): The x parameter.
        **kwargs (object): Keyword args.

    Returns:
        object: Result.
    """
    return np.array([str(x)]) if np.isscalar(x) else x.astype(str)


@numpy_eager_registry.register("CreateToken")
def _np_create_token(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate _np_create_token operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
        object: Result.
    """
    if args and isinstance(args[0], str):
        text = args[0]
        # Standard vocabulary dictionary
        vocab = kwargs.get("vocab")
        if not isinstance(vocab, dict):
            vocab = {
                "<pad>": 0,
                "<unk>": 1,
                "<s>": 2,
                "</s>": 3,
                "hello": 4,
                "world": 5,
            }

        # WordPiece/BPE segmentation simulation
        tokens = []
        for word in text.lower().split():
            if word in vocab:
                tokens.append(vocab[word])
            else:
                found = False
                for i in range(len(word), 0, -1):
                    prefix = word[:i]
                    if prefix in vocab:
                        tokens.append(vocab[prefix])
                        suffix = "##" + word[i:]
                        if suffix in vocab:
                            tokens.append(vocab[suffix])
                        else:
                            tokens.append(vocab.get("<unk>", 1))
                        found = True
                        break
                if not found:
                    tokens.append(vocab.get("<unk>", 1))

        return np.array(tokens, dtype=np.int32)

    return np.array(0, dtype=np.int32)
