"""Lookup and hash table operations."""

from __future__ import annotations

from ml_switcheroo_compiler.backends.registry import get_active_backend
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.shape import TensorOrScalar
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node


class MutableHashTable:
    """Mutable hash table backing lookup and insert operations."""

    def __init__(self, key_dtype: DType, value_dtype: DType, default_value: Tensor) -> None:
        """Initialize MutableHashTable.

        Args:
            key_dtype (DType): Data type of the keys.
            value_dtype (DType): Data type of the values.
            default_value (Tensor): Default value returned for missing keys.
        """
        self.key_dtype = key_dtype
        self.value_dtype = value_dtype
        self.default_value = default_value
        self._table: dict[TensorOrScalar, TensorOrScalar] = {}
        self._keys: Tensor | None = None
        self._values: Tensor | None = None

    def lookup(self, keys: Tensor) -> TensorOrScalar:
        """Look up keys in the hash table, falling back to `default_value` when absent.

        Args:
            keys (Tensor): Keys tensor to look up.

        Returns:
            TensorOrScalar: Looked-up values tensor with shape matching `keys`.
        """
        key_shape = getattr(keys, "shape", ())
        key_device = getattr(keys, "device", "cpu")
        if not config.eager_mode:
            return _emit_shape_node(
                "LookupTableFind",
                [keys, self.default_value],
                {"key_dtype": str(self.key_dtype), "value_dtype": str(self.value_dtype)},
                key_shape,
                self.value_dtype,
            )
        raw_keys = getattr(keys, "data", keys)
        raw_default = getattr(self.default_value, "data", self.default_value)
        raw_default = raw_default.item() if hasattr(raw_default, "item") else raw_default
        if raw_keys is not None and hasattr(raw_keys, "tolist"):
            flat_k = raw_keys.tolist()
            looked_up: TensorOrScalar = [self._table.get(k, raw_default) for k in flat_k] if isinstance(flat_k, list) else self._table.get(flat_k, raw_default)
            arr = get_active_backend().array(looked_up, dtype=str(self.value_dtype.value if hasattr(self.value_dtype, "value") else self.value_dtype))
            return Tensor(arr, TensorConfig(key_shape, self.value_dtype, key_device))
        return Tensor(raw_default, TensorConfig(key_shape, self.value_dtype, key_device))

    def insert(self, keys: Tensor, values: Tensor) -> None:
        """Insert key-value pairs into the mutable hash table.

        Args:
            keys (Tensor): Keys tensor.
            values (Tensor): Values tensor.
        """
        self._keys = keys
        self._values = values
        raw_k = getattr(keys, "data", keys)
        raw_v = getattr(values, "data", values)
        if raw_k is not None and raw_v is not None and hasattr(raw_k, "tolist") and hasattr(raw_v, "tolist"):
            k_list = raw_k.tolist()
            v_list = raw_v.tolist()
            if isinstance(k_list, list) and isinstance(v_list, list):
                for k, v in zip(k_list, v_list):
                    self._table[k] = v
            else:
                self._table[k_list] = v_list


class DenseHashTable(MutableHashTable):
    """Dense static hash table with explicit empty and deleted key sentinels."""

    def __init__(
        self,
        key_dtype: DType,
        value_dtype: DType,
        default_value: Tensor,
        empty_key: Tensor,
        deleted_key: Tensor,
    ) -> None:
        """Initialize DenseHashTable.

        Args:
            key_dtype (DType): Data type of the keys.
            value_dtype (DType): Data type of the values.
            default_value (Tensor): Default value returned for missing keys.
            empty_key (Tensor): Sentinel key representing an empty slot.
            deleted_key (Tensor): Sentinel key representing a deleted slot.
        """
        super().__init__(key_dtype, value_dtype, default_value)
        self.empty_key = empty_key
        self.deleted_key = deleted_key
