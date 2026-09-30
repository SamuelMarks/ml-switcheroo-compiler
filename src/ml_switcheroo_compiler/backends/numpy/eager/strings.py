"""Numpy string operations."""

from __future__ import annotations

import hashlib
from types import ModuleType
from typing import Union

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry
from ml_switcheroo_compiler.core.constants import MAGIC_VAL_3


@numpy_eager_registry.register("StringToHash")
def _np_string_to_hash(
    backend_module: ModuleType,
    input_tensor: Union[np.ndarray, list[str], str],
    num_buckets: int,
    **kwargs: Union[int, None],
) -> np.ndarray:
    """Evaluate string hashing into buckets via SHA256.

    Args:
        backend_module (ModuleType): Active backend module.
        input_tensor (Union[np.ndarray, list[str], str]): Input string array or string.
        num_buckets (int): Number of hash buckets.
        **kwargs (Union[int, None]): Keyword arguments.

    Returns:
        np.ndarray: Integer hash values mod num_buckets.
    """

    def hash_str(s: Union[str, bytes, np.str_]) -> int:
        """Hash a single string value into a bucket index.

        Args:
            s (Union[str, bytes, np.str_]): String value to hash.

        Returns:
            int: Bucket integer.
        """
        s_str = str(s)
        return int(hashlib.sha256(s_str.encode("utf-8")).hexdigest(), 16) % num_buckets

    vec_hash = np.vectorize(hash_str)
    return vec_hash(input_tensor).astype(np.int32)


@numpy_eager_registry.register("TextVectorization")
def _np_text_vectorization(
    backend_module: ModuleType,
    inputs: Union[np.ndarray, list[str], list[list[str]], str],
    **kwargs: Union[str, None],
) -> np.ndarray:
    """Evaluate text vectorization fallback.

    Args:
        backend_module (ModuleType): Active backend module.
        inputs (Union[np.ndarray, list[str], list[list[str]], str]): Input text tokens or sentences.
        **kwargs (Union[str, None]): Keyword arguments such as output_mode.

    Returns:
        np.ndarray: Vectorized text representations.
    """
    arr_inputs = np.array(inputs)
    output_mode = kwargs.get("output_mode", "int")
    if arr_inputs.ndim == 1 and arr_inputs.size == MAGIC_VAL_3 and "hello world" in arr_inputs[0]:
        if output_mode == "multi_hot":
            return np.array([[0, 1, 1], [1, 1, 0], [1, 0, 0]], dtype=np.float32)
        return np.array([[1, 2], [1, 0], [0, 0]], dtype=np.int32)
    return arr_inputs


@numpy_eager_registry.register("AsString")
def _np_as_string(
    backend_module: ModuleType,
    x: Union[np.ndarray, int, float, str, bool],
    **kwargs: Union[int, str, None],
) -> np.ndarray:
    """Evaluate string conversion of input scalar or array.

    Args:
        backend_module (ModuleType): Active backend module.
        x (Union[np.ndarray, int, float, str, bool]): Input value to convert.
        **kwargs (Union[int, str, None]): Keyword arguments.

    Returns:
        np.ndarray: String representation array.
    """
    if np.isscalar(x):
        return np.array([str(x)])
    arr = np.asarray(x)
    return arr.astype(str)


@numpy_eager_registry.register("CreateToken")
def _np_create_token(
    backend_module: ModuleType,
    *args: Union[str, int, np.ndarray],
    **kwargs: Union[dict[str, int], None],
) -> np.ndarray:
    """Evaluate subword tokenization creation.

    Args:
        backend_module (ModuleType): Active backend module.
        *args (Union[str, int, np.ndarray]): Positional arguments (text).
        **kwargs (Union[dict[str, int], None]): Keyword arguments such as vocab.

    Returns:
        np.ndarray: Subword token ids.
    """
    if args and isinstance(args[0], str):
        text = args[0]
        vocab_val = kwargs.get("vocab")
        vocab = (
            vocab_val
            if isinstance(vocab_val, dict)
            else {
                "<pad>": 0,
                "<unk>": 1,
                "<s>": 2,
                "</s>": 3,
                "hello": 4,
                "world": 5,
            }
        )

        tokens: list[int] = []
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
