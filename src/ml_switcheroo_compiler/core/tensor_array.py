"""Dynamic indexable tensor array container for control-flow loops and scan operations."""

from __future__ import annotations

import typing
import uuid

from ml_switcheroo_ir import LogicalNode

from ml_switcheroo_compiler.core.device import Device, DeviceType
from ml_switcheroo_compiler.core.shape import TensorOrScalar
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.tracing.state import global_tracing_state
from ml_switcheroo_compiler.tracing.tracer import ProxyTensor


class TensorArray:
    """Fixed-capacity array of fixed-shape tensors supporting indexed read, write, and stack."""

    def __init__(self, size: int, element_shape: tuple[int, ...], dtype: str) -> None:
        """Initialize `TensorArray`.

        Args:
            size (int): Number of tensor slots in the array.
            element_shape (tuple[int, ...]): Shape of each element tensor.
            dtype (str): Data type string of each element tensor.
        """
        self.size = size
        self.element_shape = element_shape
        self.dtype = dtype
        self.id = str(uuid.uuid4())
        self._data: list[TensorOrScalar] = [None] * size

    def read(self, index: Tensor) -> Tensor:
        """Read the tensor at slot `index` from the `TensorArray`.

        Args:
            index (Tensor): Scalar integer index tensor.

        Returns:
            Tensor: Element `Tensor` at `index`.
        """
        if not global_tracing_state.is_tracing:
            from ml_switcheroo_compiler.backends.registry import get_active_backend

            idx = int(typing.cast(int, get_active_backend().asarray(index.data)))
            val = self._data[idx]
            if val is None:
                val = get_active_backend().execute_op("Zeros", self.element_shape)
            return Tensor(val, TensorConfig(self.element_shape, self.dtype, Device(DeviceType.CPU)))

        out_id = str(uuid.uuid4())
        node = LogicalNode(
            id=out_id,
            op_type="TensorArrayRead",
            inputs=[getattr(self, "id"), getattr(index.data, "id")],
            attributes={},
            shape_metadata=self.element_shape,
        )
        global_tracing_state.add_node(node)
        proxy = ProxyTensor(id=out_id, shape=self.element_shape, dtype=self.dtype)
        return Tensor(proxy, TensorConfig(self.element_shape, self.dtype, Device(DeviceType.CPU)))

    def write(self, index: Tensor, value: Tensor) -> TensorArray:
        """Write `value` into slot `index` of the `TensorArray`.

        Args:
            index (Tensor): Scalar integer index tensor.
            value (Tensor): Element tensor to store at `index`.

        Returns:
            TensorArray: Updated `TensorArray` (`self`).
        """
        if not global_tracing_state.is_tracing:
            from ml_switcheroo_compiler.backends.registry import get_active_backend

            idx = int(typing.cast(int, get_active_backend().asarray(index.data)))
            self._data[idx] = value.data
            return self

        out_id = str(uuid.uuid4())
        node = LogicalNode(
            id=out_id,
            op_type="TensorArrayWrite",
            inputs=[getattr(self, "id"), getattr(index.data, "id"), getattr(value.data, "id")],
            attributes={},
            shape_metadata=(),
        )
        global_tracing_state.add_node(node)
        return self

    def stack(self) -> Tensor:
        """Stack all elements of the `TensorArray` along a new leading axis 0.

        Returns:
            Tensor: Stacked `Tensor` of shape `(self.size,) + self.element_shape`.
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
