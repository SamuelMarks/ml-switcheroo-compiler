"""Module lookup.py."""

from typing import Any

# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Lookup and hash table ops."""

from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor


class MutableHashTable:
    """Mutable hash table backing ops."""

    def __init__(self, key_dtype: DType, value_dtype: DType, default_value: Tensor) -> None:  # type: ignore
        """Initialize MutableHashTable.

        Args:
            key_dtype (DType): The data type of the keys.
            value_dtype (DType): The data type of the values.
            default_value (Tensor): The default value for missing keys.
        """
        self.key_dtype = key_dtype
        self.value_dtype = value_dtype
        self.default_value = default_value

    def lookup(self, keys: Tensor) -> Any:  # type: ignore
        """Lookup keys.

        Args:
            keys (Tensor): The keys parameter.

        Returns:
            Tensor: Result.
        """
        from ml_switcheroo_compiler.core.tensor import TensorConfig

        return Tensor(None, TensorConfig(getattr(keys, "shape", ()), self.value_dtype, getattr(keys, "device", "cpu")))

    def insert(self, keys: Tensor, values: Tensor) -> None:  # type: ignore
        """Insert keys and values.

        Args:
            keys: Keys tensor.
            values: Values tensor.
        """
        self._keys = keys
        self._values = values


class DenseHashTable:
    """Dense static hash table ops."""

    def __init__(self, key_dtype: DType, value_dtype: DType, default_value: Tensor, empty_key: Tensor, deleted_key: Tensor) -> None:  # type: ignore
        """Initialize DenseHashTable.

        Args:
            key_dtype (DType): The data type of the keys.
            value_dtype (DType): The data type of the values.
            default_value (Tensor): The default value for missing keys.
            empty_key (Tensor): The key representing an empty slot.
            deleted_key (Tensor): The key representing a deleted slot.
        """
        self.key_dtype = key_dtype
        self.value_dtype = value_dtype
        self.default_value = default_value
        self.empty_key = empty_key
        self.deleted_key = deleted_key

    def lookup(self, keys: Tensor) -> Any:  # type: ignore
        """Lookup keys.

        Args:
            keys (Tensor): The keys parameter.

        Returns:
            Tensor: Result.
        """
        from ml_switcheroo_compiler.core.tensor import TensorConfig

        return Tensor(None, TensorConfig(getattr(keys, "shape", ()), self.value_dtype, getattr(keys, "device", "cpu")))

    def insert(self, keys: Tensor, values: Tensor) -> None:  # type: ignore
        """Insert keys and values.

        Args:
            keys: Keys tensor.
            values: Values tensor.
        """
        self._keys = keys
        self._values = values
