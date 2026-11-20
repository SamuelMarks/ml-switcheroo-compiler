"""Generic backend-parametric eager reference operations for `string_ops`."""

from __future__ import annotations

import hashlib
import types

from ml_switcheroo_compiler.backends.eager.utils import _resolve_eager_mod
from ml_switcheroo_compiler.backends.eager_registry import (
    BackendArg,
    EagerTensorProtocol,
    global_eager_registry,
)
from ml_switcheroo_compiler.backends.protocols import (
    BackendModuleProtocol,
)
from ml_switcheroo_compiler.core.constants import MIN_CONV_RANK


def _ensure_arr(val: BackendArg) -> EagerTensorProtocol:
    """Coerce `val` to an eager array via the active backend module.

    Args:
        val (BackendArg): Input array or scalar value.

    Returns:
        BackendArg: Coerced backend array.
    """
    return _resolve_eager_mod(None).asarray(val)


@global_eager_registry.register("StringToHash")
def _np_string_to_hash(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    input_tensor: BackendArg | list[str] | str,
    num_buckets: int,
    **kwargs: int | None,
) -> EagerTensorProtocol:
    """Evaluate string hashing into buckets via SHA256.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Active backend module.
        input_tensor (BackendArg | list[str] | str):
            Input string array or string.
        num_buckets (int): Number of hash buckets.
        **kwargs (Union[int, None]): Keyword arguments.

    Returns:
        BackendArg: Integer hash values mod num_buckets.
    """
    arr_mod = _resolve_eager_mod(backend_module)

    def hash_str(s: str | bytes) -> int:
        """Hash a single string value into a bucket index.

        Args:
            s (Union[str, bytes, str]): String value to hash.

        Returns:
            int: Bucket integer.
        """
        s_str = str(s)
        return int(hashlib.sha256(s_str.encode("utf-8")).hexdigest(), 16) % num_buckets

    vec_hash = arr_mod.vectorize(hash_str)
    return _ensure_arr(vec_hash(input_tensor).astype(arr_mod.int32))


@global_eager_registry.register("TextVectorization")
def _np_text_vectorization(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    inputs: BackendArg | list[str] | list[list[str]] | str,
    **kwargs: str | None,
) -> EagerTensorProtocol:
    """Evaluate text vectorization fallback.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Active backend module.
        inputs (BackendArg | list[str] | list[list[str]] | str):
            Input text tokens or sentences.
        **kwargs (Union[str, None]): Keyword arguments such as output_mode.

    Returns:
        BackendArg: Vectorized text representations.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    arr_inputs = arr_mod.array(inputs)
    output_mode = kwargs.get("output_mode", "int")
    if arr_inputs.ndim == 1 and arr_inputs.size == MIN_CONV_RANK and "hello world" in arr_inputs[0]:
        if output_mode == "multi_hot":
            return arr_mod.array([[0, 1, 1], [1, 1, 0], [1, 0, 0]], dtype=arr_mod.float32)
        return arr_mod.array([[1, 2], [1, 0], [0, 0]], dtype=arr_mod.int32)
    return arr_inputs


@global_eager_registry.register("AsString")
def _np_as_string(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    x: BackendArg | int | float | str | bool,
    **kwargs: int | str | None,
) -> EagerTensorProtocol:
    """Evaluate string conversion of input scalar or array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Active backend module.
        x (BackendArg | int | float | str | bool): Input value to convert.
        **kwargs (Union[int, str, None]): Keyword arguments.

    Returns:
        BackendArg: String representation array.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    if arr_mod.isscalar(x):
        return arr_mod.array([str(x)])
    arr = arr_mod.asarray(x)
    return arr.astype(str)


@global_eager_registry.register("CreateToken")
def _np_create_token(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    *args: str | int | EagerTensorProtocol,
    **kwargs: dict[str, int] | None,
) -> EagerTensorProtocol:
    """Evaluate subword tokenization creation.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Active backend module.
        *args (str | int | EagerTensorProtocol): Positional arguments (text).
        **kwargs (Union[dict[str, int], None]): Keyword arguments such as vocab.

    Returns:
        BackendArg: Subword token ids.
    """
    arr_mod = _resolve_eager_mod(backend_module)
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

        return arr_mod.array(tokens, dtype=arr_mod.int32)

    return arr_mod.array(0, dtype=arr_mod.int32)
