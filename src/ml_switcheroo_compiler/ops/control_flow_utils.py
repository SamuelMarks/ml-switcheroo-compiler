"""Provide higher-order control flow primitives for tracing and eager execution.

This module implements functional control flow operators such as conditional branching,
while loops, scanning, vectorizing maps (vmap), and parallel maps (pmap). These
operators support both eager execution (using NumPy/Python loops) and tracing into an
intermediate representation (IR) graph for compilation
"""

from __future__ import annotations

import uuid
from typing import Callable, cast

from ml_switcheroo_ir import LogicalGraph, LogicalNode
from ml_switcheroo_ir.types import TensorSpec

from ml_switcheroo_compiler.backends.registry import EagerTensorProtocol, EagerValue
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.tracing.state import global_tracing_state
from ml_switcheroo_compiler.tracing.tracer import ProxyTensor, increment_trace_count


def _wrap_proxy_inputs(
    args: tuple[Tensor | tuple[Tensor, ...] | int | float | bool | None, ...],
    subgraph: LogicalGraph,
) -> tuple[list[str], list[Tensor | tuple[Tensor, ...] | int | float | bool | None]]:
    """Evaluate _wrap_proxy_inputs operation.

    Args:
        args (Any): The args parameter.
        subgraph (Any): The subgraph parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    proxy_args: list[Tensor | tuple[Tensor, ...] | int | float | bool | None] = []
    input_ids: list[str] = []
    for _i, arg in enumerate(args):
        if isinstance(arg, tuple):
            from typing import cast

            sub_ids, sub_args = _wrap_proxy_inputs(arg, subgraph)
            input_ids.extend(sub_ids)
            proxy_args.append(cast("tuple[Tensor, ...]", tuple(sub_args)))
        elif isinstance(arg, Tensor):
            in_id = str(uuid.uuid4())
            node = LogicalNode(
                id=in_id,
                op_type="Input",
                inputs=[],
                shape_metadata=arg.shape,
            )
            subgraph.nodes[in_id] = node
            input_ids.append(in_id)
            if hasattr(subgraph, "input_specs"):
                subgraph.input_specs[in_id] = TensorSpec(shape=arg.shape, dtype=arg.dtype)
            proxy = ProxyTensor(id=in_id, shape=arg.shape, dtype=arg.dtype.value)
            proxy.concrete_value = arg.data
            proxy_tensor = Tensor(proxy, TensorConfig(arg.shape, arg.dtype, arg.device))
            proxy_args.append(proxy_tensor)
        else:
            proxy_args.append(arg)
    return input_ids, proxy_args


def _get_tensor_ids(obj: EagerValue) -> list[str]:
    """Evaluate _get_tensor_ids operation.

    Args:
        obj (Any): The obj parameter.

    Returns:
            tuple[int, ...]: Result.

    Raises:
        TypeError: An exception.
    """
    if isinstance(obj, Tensor):
        cur = obj
        while isinstance(cur, Tensor) and isinstance(getattr(cur, "data", None), Tensor):
            cur = cast(Tensor, cur.data)
        if hasattr(cur, "data") and hasattr(cur.data, "id"):
            return [str(getattr(cur.data, "id"))]
        if hasattr(cur, "id"):
            return [str(cur.id)]
        return [str(uuid.uuid4())]
    if isinstance(obj, (tuple, list)):
        ids = []
        for o in obj:
            ids.extend(_get_tensor_ids(o))
        return ids

    msg = "Control flow functions must return a Tensor or a tuple of Tensors."
    raise TypeError(msg)


def _process_trace_outputs(out: Tensor | tuple[Tensor, ...] | list[Tensor], subgraph: LogicalGraph) -> str:
    """Evaluate _process_trace_outputs operation.

    Args:
        out: The out parameter.
        subgraph (LogicalGraph): The subgraph parameter.

    Returns:
        str: Result.
    """
    out_ids = _get_tensor_ids(out)

    out_node = LogicalNode(
        id=str(uuid.uuid4()),
        op_type="Output",
        inputs=out_ids,
        shape_metadata=(),
    )
    subgraph.nodes[out_node.id] = out_node
    subgraph.outputs = [out_node.id]
    return str(out_node.id)


def _trace_function(func: Callable[..., Tensor | tuple[Tensor, ...] | list[Tensor]], args: tuple[Tensor, ...], name: str) -> LogicalGraph:
    """Trace a Python function's execution into a LogicalGraph.

    Args:
        func (Callable): The func parameter.
        args (tuple): The args parameter.
        name (str): The name parameter.

    Returns:
        LogicalGraph: Result.
    """
    from ml_switcheroo_compiler.core.config import config as compiler_config

    prev_graph = global_tracing_state.active_graph
    is_tracing = global_tracing_state.is_tracing
    prev_eager = compiler_config.eager_mode

    compiler_config.eager_mode = False
    cast("Callable[..., EagerTensorProtocol]", increment_trace_count)(func)

    subgraph = global_tracing_state.start_tracing(name=name)
    input_ids, proxy_args = _wrap_proxy_inputs(args, subgraph)
    subgraph.inputs = input_ids

    try:
        out = func(*proxy_args)
        out_node_id = _process_trace_outputs(out, subgraph)
    finally:
        global_tracing_state.stop_tracing()
        global_tracing_state.active_graph = prev_graph
        global_tracing_state.is_tracing = is_tracing
        compiler_config.eager_mode = prev_eager

    setattr(subgraph, "id", name)
    subgraph.outputs = [out_node_id]
    return subgraph
