"""Random ops frontend."""

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.tracing import ProxyTensor, _tracer
from ml_switcheroo_ir import LogicalNode
import uuid


def sobol_sample(dim: int, num_results: int, skip: int = 0) -> Tensor:
    """Sobol sequence generator.

    Args:
        dim (int): The dimensionality of the sequence.
        num_results (int): The number of results.
        skip (int): The number of sequence elements to skip.

    Returns:
        Tensor: A tensor of shape (num_results, dim).
    """
    from ml_switcheroo_compiler.core.dtype import DType

    dtype = DType("float32")
    out_shape = (num_results, dim)
    device = config.default_device

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op("SobolSample", dim, num_results, skip)
        return Tensor(data, TensorConfig(out_shape, dtype, device))

    if not _tracer.is_tracing:
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
    _tracer.add_node(node)

    proxy = ProxyTensor(id=out_id, shape=out_shape, dtype=dtype.value)
    return Tensor(proxy, TensorConfig(out_shape, dtype, device))
