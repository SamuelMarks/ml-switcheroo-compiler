"""Module control_flow_utils.py."""

from __future__ import annotations

# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915

"""Provide higher-order control flow primitives for tracing and eager execution.

This module implements functional control flow operators such as conditional branching,
while loops, scanning, vectorizing maps (vmap), and parallel maps (pmap). These
operators support both eager execution (using NumPy/Python loops) and tracing into an
intermediate representation (IR) graph for compilation
"""


import uuid
from typing import Any, Callable

from ml_switcheroo_ir import LogicalNode

from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ir.core import IRBlock
from ml_switcheroo_compiler.tracing.state import global_tracing_state
from ml_switcheroo_compiler.tracing.tracer import ProxyTensor, increment_trace_count


def _wrap_proxy_inputs(args: tuple[Any, ...], subgraph: Any) -> tuple[list[str], list[Any]]:
    """Evaluate _wrap_proxy_inputs operation.

    Args:
        args (object): The args parameter.
        subgraph (object): The subgraph parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    proxy_args = []
    input_ids = []
    for _i, arg in enumerate(args):
        if isinstance(arg, tuple):
            sub_ids, sub_args = _wrap_proxy_inputs(arg, subgraph)
            input_ids.extend(sub_ids)
            proxy_args.append(tuple(sub_args))
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
            proxy = ProxyTensor(id=in_id, shape=arg.shape, dtype=arg.dtype.value)  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism
            proxy.concrete_value = arg.data  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism
            proxy_tensor = Tensor(proxy, TensorConfig(arg.shape, arg.dtype, arg.device))  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism
            proxy_args.append(proxy_tensor)  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism
        else:
            proxy_args.append(arg)
    return input_ids, proxy_args


def _get_tensor_ids(obj: Any) -> list[str]:
    """Evaluate _get_tensor_ids operation.

    Args:
        obj (object): The obj parameter.

    Returns:
            tuple[int, ...]: Result.

    Raises:
        TypeError: An exception.
    """
    if isinstance(obj, Tensor):
        return [obj.data.id]  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism
    if isinstance(obj, (tuple, list)):
        ids = []
        for o in obj:
            ids.extend(_get_tensor_ids(o))
        return ids

    msg = "Control flow functions must return a Tensor or a tuple of Tensors."
    raise TypeError(msg)


def _process_trace_outputs(out: Any, subgraph: IRBlock) -> str:
    """Evaluate _process_trace_outputs operation.

    Args:
        out (object): The out parameter.
        subgraph (IRBlock): The subgraph parameter.

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
    return out_node.id  # type: ignore


def _trace_function(func: Callable[..., Any], args: tuple[Tensor, ...], name: str) -> IRBlock:  # type: ignore
    """Trace a Python function's execution into an IRBlock.

    Args:
        func (Callable): The func parameter.
        args (tuple): The args parameter.
        name (str): The name parameter.

    Returns:
        IRBlock: Result.
    """
    from ml_switcheroo_compiler.core.config import config as compiler_config

    prev_graph = global_tracing_state.active_graph
    is_tracing = global_tracing_state.is_tracing
    prev_eager = compiler_config.eager_mode

    compiler_config.eager_mode = False
    increment_trace_count(func)

    subgraph = global_tracing_state.start_tracing(name=name)
    input_ids, proxy_args = _wrap_proxy_inputs(args, subgraph)

    try:
        out = func(*proxy_args)
        out_node_id = _process_trace_outputs(out, subgraph)
    finally:
        global_tracing_state.stop_tracing()
        global_tracing_state.active_graph = prev_graph
        global_tracing_state.is_tracing = is_tracing
        compiler_config.eager_mode = prev_eager

    return IRBlock(
        id=name,
        nodes=list(subgraph.nodes.values()),
        inputs=input_ids,
        outputs=[out_node_id],
    )
