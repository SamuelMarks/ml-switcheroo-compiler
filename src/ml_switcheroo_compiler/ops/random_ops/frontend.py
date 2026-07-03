"""Random ops frontend."""

import uuid

from ml_switcheroo_ir import LogicalNode

from ml_switcheroo_compiler.backends.registry import get_active_backend  # pragma: no cover
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.dtype import DType  # pragma: no cover
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.tracing import ProxyTensor, global_tracing_state


def sobol_sample(dim: int, num_results: int, skip: int = 0) -> Tensor:
    """Sobol sequence generator.

    Args:
        dim (int): The dimensionality of the sequence.
        num_results (int): The number of results.
        skip (int): The number of sequence elements to skip.

    Returns:
        Tensor: A tensor of shape (num_results, dim).
    """
    dtype = DType("float32")  # pragma: no cover
    out_shape = (num_results, dim)  # pragma: no cover
    device = config.default_device  # pragma: no cover

    if config.eager_mode:  # pragma: no cover
        backend = get_active_backend()  # pragma: no cover
        data = backend.execute_op("SobolSample", dim, num_results, skip)  # pragma: no cover
        return Tensor(data, TensorConfig(out_shape, dtype, device))  # pragma: no cover

    if not global_tracing_state.is_tracing:  # pragma: no cover
        msg = "Cannot emit node outside tracing context."  # pragma: no cover
        raise RuntimeError(msg)  # pragma: no cover

    out_id = str(uuid.uuid4())  # pragma: no cover
    node = LogicalNode(  # pragma: no cover
        id=out_id,
        op_type="SobolSample",
        inputs=[],
        attributes={"dim": dim, "num_results": num_results, "skip": skip},
        shape_metadata=out_shape,
    )
    global_tracing_state.add_node(node)  # pragma: no cover

    proxy = ProxyTensor(id=out_id, shape=out_shape, dtype=dtype.value)  # pragma: no cover
    return Tensor(proxy, TensorConfig(out_shape, dtype, device))  # pragma: no cover
