"""Module frontend.py."""

# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Generate random ops frontend."""

import uuid

from ml_switcheroo_ir import LogicalNode

from ml_switcheroo_compiler.backends.registry import get_active_backend
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.tracing import ProxyTensor, global_tracing_state


def sobol_sample(dim: int, num_results: int, skip: int = 0) -> object:
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
    dtype: object = DType("float32")
    out_shape: object = (num_results, dim)
    device: object = config.default_device

    if config.eager_mode:
        backend: object = get_active_backend()
        data: object = backend.execute_op("SobolSample", dim, num_results, skip)
        return Tensor(data, TensorConfig(out_shape, dtype, device))

    if not global_tracing_state.is_tracing:
        msg: object = "Cannot emit node outside tracing context."
        raise RuntimeError(msg)

    out_id: object = str(uuid.uuid4())
    node: object = LogicalNode(
        id=out_id,
        op_type="SobolSample",
        inputs=[],
        attributes={"dim": dim, "num_results": num_results, "skip": skip},
        shape_metadata=out_shape,
    )
    global_tracing_state.add_node(node)

    proxy: object = ProxyTensor(id=out_id, shape=out_shape, dtype=dtype.value)
    return Tensor(proxy, TensorConfig(out_shape, dtype, device))
