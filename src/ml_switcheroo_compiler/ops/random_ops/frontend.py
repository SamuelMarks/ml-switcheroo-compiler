"""Generate random ops frontend."""

import uuid

from ml_switcheroo_ir import LogicalNode

from ml_switcheroo_compiler.backends.registry import get_active_backend
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.tracing import ProxyTensor, global_tracing_state


def sobol_sample(dim: int, num_results: int, skip: int = 0) -> Tensor:
    """Sobol sequence generator.

    Args:
        dim (int): The dim parameter.
        num_results (int): The num_results parameter.
        skip (int): The skip parameter.

    Returns:
        Tensor: Result.

    Raises:
        RuntimeError: An exception.
    """
    dtype = DType("float32")
    out_shape = (num_results, dim)
    device = config.default_device

    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("SobolSample", dim, num_results, skip)
        return Tensor(data, TensorConfig(out_shape, dtype, device))

    if not global_tracing_state.is_tracing:
        msg = "Cannot emit node outside tracing context."
        raise RuntimeError(msg)

    out_id = str(uuid.uuid4())
    node = LogicalNode(
        id=out_id,
        op_type="SobolSample",
        inputs=[],
        attributes={"dim": dim, "num_results": num_results, "skip": skip},
        shape_metadata=out_shape,
    )
    global_tracing_state.add_node(node)

    proxy = ProxyTensor(id=out_id, shape=out_shape, dtype=dtype.value)
    return Tensor(proxy, TensorConfig(out_shape, dtype, device))
