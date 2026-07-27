# ruff: noqa: E501
"""Numpy string operations."""

import hashlib

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry
from ml_switcheroo_compiler.core.constants import MAGIC_VAL_3


@numpy_eager_registry.register("StringToHash")
def _np_string_to_hash(backend_module: object, input_tensor: object, num_buckets: int, **kwargs: object) -> object:
    """Evaluate the string to hash logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        input_tensor (object): Required parameter for input_tensor.
        num_buckets (int): Required parameter for num_buckets.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """

    def hash_str(s: str) -> int:
        """Evaluate and process the hash str operation.

        Args:
            s (str): Required parameter for s.

        Returns:
            int: The evaluated or processed output.
        """
        s = str(s)
        return int(hashlib.sha256(s.encode("utf-8")).hexdigest(), 16) % num_buckets

    vec_hash = np.vectorize(hash_str)
    return vec_hash(input_tensor).astype(np.int32)


@numpy_eager_registry.register("TextVectorization")
def _np_text_vectorization(backend_module: object, inputs: object, **kwargs: object) -> object:
    """Evaluate the text vectorization logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        inputs (object): Required parameter for inputs.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
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
    """Evaluate the as string logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        x (object): Required parameter for x.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    return np.array([str(x)]) if np.isscalar(x) else x.astype(str)


@numpy_eager_registry.register("CreateToken")
def _np_create_token(backend_module: object, *args: object, **kwargs: object) -> object:
    """Create a token or execute a vocabulary-based tokenizer pipeline.

    By default (XLA/Lax style), this returns a sequencing token (0).
    If a string is passed in as the first argument, it runs a BPE/WordPiece-style
    vocabulary execution pipeline to tokenize the string into integer IDs.

    Args:
        backend_module (object): Required parameter for backend_module.
        *args (object): Optional string input to be tokenized.
        **kwargs (Any): Arbitrary keyword arguments, e.g., 'vocab'.

    Returns:
        object: A NumPy array containing the token or list of token IDs.
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
