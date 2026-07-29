"""Tensor Array."""

import uuid

from ml_switcheroo_ir import LogicalNode

from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.tracing.state import global_tracing_state
from ml_switcheroo_compiler.tracing.tracer import ProxyTensor


class TensorArray:
    """Represents a TensorArray."""

    def __init__(self, size: int, element_shape: tuple[int, ...], dtype: str) -> None:
        """Init."""
        self.size = size
        self.element_shape = element_shape
        self.dtype = dtype
        self.id = str(uuid.uuid4())
        self._data = [None] * size

    def read(self, index: Tensor) -> Tensor:
        """Reads from the TensorArray."""
        if not global_tracing_state.is_tracing:
            from ml_switcheroo_compiler.backends.registry import get_active_backend

            idx = int(get_active_backend().asarray(index.data))
            val = self._data[idx]
            if val is None:
                val = get_active_backend().execute_op("Zeros", self.element_shape)
            return Tensor(val, TensorConfig(self.element_shape, self.dtype, None))

        out_id = str(uuid.uuid4())
        node = LogicalNode(
            id=out_id,
            op_type="TensorArrayRead",
            inputs=[self.id, index.data.id],
            attributes={},
            shape_metadata=self.element_shape,
        )
        global_tracing_state.add_node(node)
        proxy = ProxyTensor(id=out_id, shape=self.element_shape, dtype=self.dtype)
        return Tensor(proxy, TensorConfig(self.element_shape, self.dtype, None))

    def write(self, index: Tensor, value: Tensor) -> "TensorArray":
        """Writes to the TensorArray."""
        if not global_tracing_state.is_tracing:
            from ml_switcheroo_compiler.backends.registry import get_active_backend

            idx = int(get_active_backend().asarray(index.data))
            self._data[idx] = value.data
            return self

        out_id = str(uuid.uuid4())
        node = LogicalNode(
            id=out_id,
            op_type="TensorArrayWrite",
            inputs=[self.id, index.data.id, value.data.id],
            attributes={},
            shape_metadata=(),
        )
        global_tracing_state.add_node(node)
        return self

    def stack(self) -> Tensor:
        """Stacks the TensorArray."""
        out_shape = (self.size,) + self.element_shape
        if not global_tracing_state.is_tracing:
            from ml_switcheroo_compiler.backends.registry import get_active_backend

            arrs = [d if d is not None else get_active_backend().execute_op("Zeros", self.element_shape) for d in self._data]
            res = get_active_backend().execute_op("Stack", arrs)
            return Tensor(res, TensorConfig(out_shape, self.dtype, None))

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
        return Tensor(proxy, TensorConfig(out_shape, self.dtype, None))
