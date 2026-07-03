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

    def read(self, index: Tensor) -> Tensor:
        """Reads from the TensorArray."""
        out_id = str(uuid.uuid4())
        node = LogicalNode(
            id=out_id,
            op_type="TensorArrayRead",
            inputs=[self.id, index.data.id],
            attributes={},
            shape_metadata=self.element_shape,
        )
        if global_tracing_state.is_tracing:  # pragma: no branch
            global_tracing_state.add_node(node)
        proxy = ProxyTensor(id=out_id, shape=self.element_shape, dtype=self.dtype)
        return Tensor(proxy, TensorConfig(self.element_shape, self.dtype, None))

    def write(self, index: Tensor, value: Tensor) -> "TensorArray":
        """Writes to the TensorArray."""
        out_id = str(uuid.uuid4())
        node = LogicalNode(
            id=out_id,
            op_type="TensorArrayWrite",
            inputs=[self.id, index.data.id, value.data.id],
            attributes={},
            shape_metadata=(),
        )
        if global_tracing_state.is_tracing:  # pragma: no branch
            global_tracing_state.add_node(node)
        return self

    def stack(self) -> Tensor:
        """Stacks the TensorArray."""
        out_id = str(uuid.uuid4())
        out_shape = (self.size,) + self.element_shape
        node = LogicalNode(
            id=out_id,
            op_type="TensorArrayStack",
            inputs=[self.id],
            attributes={},
            shape_metadata=out_shape,
        )
        if global_tracing_state.is_tracing:  # pragma: no branch
            global_tracing_state.add_node(node)
        proxy = ProxyTensor(id=out_id, shape=out_shape, dtype=self.dtype)
        return Tensor(proxy, TensorConfig(out_shape, self.dtype, None))
