from typing import Any

# ruff: noqa: E402, D100, D103, D104, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, D101, D102, D107, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Tensor Array."""
import uuid

from ml_switcheroo_ir import LogicalNode

from ml_switcheroo_compiler.core.device import Device, DeviceType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.tracing.state import global_tracing_state
from ml_switcheroo_compiler.tracing.tracer import ProxyTensor


class TensorArray:
    """Represents a TensorArray."""

    def __init__(self, size: int, element_shape: tuple[int, ...], dtype: str) -> None:
        """Init.

        Args:
            size (int): The size parameter.
            element_shape (tuple): The element_shape parameter.
            dtype (str): The dtype parameter.
        """
        self.size = size
        self.element_shape = element_shape
        self.dtype = dtype
        self.id = str(uuid.uuid4())
        self._data = [None] * size

    def read(self, index: Tensor) -> Tensor:
        """Read from the TensorArray.

        Args:
            index (Tensor): The index parameter.

        Returns:
            Tensor: Result.
        """
        if not global_tracing_state.is_tracing:
            from ml_switcheroo_compiler.backends.registry import get_active_backend

            idx = int(get_active_backend().asarray(index.data))
            val = self._data[idx]
            if val is None:
                val = get_active_backend().execute_op("Zeros", self.element_shape)
            return Tensor(val, TensorConfig(self.element_shape, self.dtype, Device(DeviceType.CPU)))

        out_id = str(uuid.uuid4())
        node = LogicalNode(
            id=out_id,
            op_type="TensorArrayRead",
            inputs=[self.id, index.data.id],  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism
            attributes={},
            shape_metadata=self.element_shape,
        )
        global_tracing_state.add_node(node)
        proxy = ProxyTensor(id=out_id, shape=self.element_shape, dtype=self.dtype)
        return Tensor(proxy, TensorConfig(self.element_shape, self.dtype, Device(DeviceType.CPU)))

    def write(self, index: Tensor, value: Tensor) -> "TensorArray":
        """Write to the TensorArray.

        Args:
            index (Tensor): The index parameter.
            value (Tensor): The value parameter.

        Returns: Any: Result.
        """
        if not global_tracing_state.is_tracing:
            from ml_switcheroo_compiler.backends.registry import get_active_backend

            idx = int(get_active_backend().asarray(index.data))
            self._data[idx] = value.data  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism
            return self

        out_id = str(uuid.uuid4())
        node = LogicalNode(
            id=out_id,
            op_type="TensorArrayWrite",
            inputs=[self.id, index.data.id, value.data.id],  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism
            attributes={},
            shape_metadata=(),
        )
        global_tracing_state.add_node(node)
        return self

    def stack(self) -> Tensor:
        """Stack the TensorArray.

        Returns:
        Tensor: Result.
        """
        out_shape = (self.size,) + self.element_shape
        if not global_tracing_state.is_tracing:
            from ml_switcheroo_compiler.backends.registry import get_active_backend

            arrs = [d if d is not None else get_active_backend().execute_op("Zeros", self.element_shape) for d in self._data]
            res = get_active_backend().execute_op("Stack", arrs)
            return Tensor(res, TensorConfig(out_shape, self.dtype, Device(DeviceType.CPU)))

        out_id = str(uuid.uuid4())
        node = LogicalNode(
            id=out_id,
            op_type="TensorArrayStack",
            inputs=[self.id],
            attributes={},
            shape_metadata=out_shape,
        )
        global_tracing_state.add_node(node)
        proxy = ProxyTensor(id=out_id, shape=out_shape, dtype=self.dtype)
        return Tensor(proxy, TensorConfig(out_shape, self.dtype, Device(DeviceType.CPU)))
