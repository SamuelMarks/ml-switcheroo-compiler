# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Gradient computation and autodiff utilities."""

import contextlib
import math
import typing
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from ml_switcheroo_ir import LogicalGraph, LogicalNode

from ml_switcheroo_compiler.backends.registry import get_active_backend
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.control_flow_utils import _trace_function
from ml_switcheroo_compiler.ops.registry import register_util
from ml_switcheroo_compiler.tracing.state import global_tracing_state
from ml_switcheroo_compiler.tracing.tracer import ProxyTensor
from ml_switcheroo_compiler.transforms.autodiff_rules.common import UnconnectedGradients
from ml_switcheroo_compiler.transforms.autodiff_rules.vjp_registry import register_vjp


def checkpoint(fun):
    """Gradient checkpointing / rematerialization.

    Args:
        fun (Callable[..., object]): The function to checkpoint.

    Returns:
        Callable[..., object]: The checkpointed function.
    """

    def wrapper(*args, **kwargs):
        """Evaluate wrapper operation.

        Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        from ml_switcheroo_compiler.core.config import config
        from ml_switcheroo_compiler.tracing.state import global_tracing_state

        if config.eager_mode or not global_tracing_state.is_tracing:
            return fun(*args, **kwargs)

        import uuid

        from ml_switcheroo_ir import LogicalNode

        from ml_switcheroo_compiler.core.dtype import DType
        from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
        from ml_switcheroo_compiler.ops.control_flow_utils import _trace_function
        from ml_switcheroo_compiler.tracing.tracer import ProxyTensor

        tensor_args = [a for a in args if isinstance(a, Tensor)]
        fwd_block = _trace_function(fun, tuple(tensor_args), f"checkpoint_{uuid.uuid4().hex[:6]}")

        # Infer output metadata from the traced block
        out_node_id = fwd_block.outputs[0]
        nodes_dict = {n.id: n for n in (fwd_block.nodes if isinstance(fwd_block.nodes, list) else fwd_block.nodes.values())}
        out_node = nodes_dict[out_node_id]
        real_out_node = nodes_dict[out_node.inputs[0]]

        shape = real_out_node.shape_metadata
        # Try to infer dtype. Proxy tensors typically just use float32 as default if not specified
        dtype = "float32"
        if hasattr(real_out_node, "attributes") and "dtype" in real_out_node.attributes:
            dtype = real_out_node.attributes["dtype"]
        elif tensor_args:
            dtype = getattr(getattr(tensor_args[0], "dtype", None), "value", "float32")

        device = "cpu"
        if tensor_args:
            device = getattr(tensor_args[0], "device", "cpu")

        out_id = str(uuid.uuid4())
        node = LogicalNode(
            id=out_id,
            op_type="Checkpoint",
            inputs=[a.data.id for a in tensor_args if hasattr(a, "data") and hasattr(a.data, "id")],
            attributes={"subgraph": fwd_block},
            shape_metadata=shape,
        )
        global_tracing_state.add_node(node)

        proxy = ProxyTensor(id=out_id, shape=shape, dtype=dtype)
        return Tensor(proxy, TensorConfig(shape, DType(dtype), device))

    return wrapper


def remat(fun):
    """Gradient checkpointing / rematerialization alias.

    Args:
        fun (Callable[..., object]): The function to rematerialize.

    Returns:
        Callable[..., object]: The rematerialized function.
    """
    return checkpoint(fun)


def recompute_grad(fun):
    """Gradient checkpointing / rematerialization.

    Args:
        fun (Callable[..., object]): The function to recompute gradients for.

    Returns:
        Callable[..., object]: The recomputing function.
    """
    return checkpoint(fun)
