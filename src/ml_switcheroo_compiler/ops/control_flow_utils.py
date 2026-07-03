"""Provides higher-order control flow primitives for tracing and eager execution.

This module implements functional control flow operators such as conditional branching,
while loops, scanning, vectorizing maps (vmap), and parallel maps (pmap). These
operators support both eager execution (using NumPy/Python loops) and tracing into an
intermediate representation (IR) graph for compilation
"""

from __future__ import annotations

import uuid
from typing import Callable

from ml_switcheroo_ir import LogicalNode

from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ir.core import IRBlock
from ml_switcheroo_compiler.tracing.state import global_tracing_state
from ml_switcheroo_compiler.tracing.tracer import ProxyTensor, increment_trace_count


def _wrap_proxy_inputs(args: tuple[object, ...], subgraph: object) -> tuple[list[str], list[object]]:
    """Function docstring.

    Args:
        args: Arg.
        subgraph: Arg.
    """
    proxy_args = []
    input_ids = []
    for _i, arg in enumerate(args):
        if isinstance(arg, tuple):  # pragma: no branch
            sub_ids, sub_args = _wrap_proxy_inputs(arg, subgraph)  # pragma: no cover
            input_ids.extend(sub_ids)  # pragma: no cover
            proxy_args.append(tuple(sub_args))  # pragma: no cover
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
            proxy = ProxyTensor(id=in_id, shape=arg.shape, dtype=arg.dtype.value)
            proxy_tensor = Tensor(proxy, TensorConfig(arg.shape, arg.dtype, arg.device))
            proxy_args.append(proxy_tensor)
        else:
            proxy_args.append(arg)
    return input_ids, proxy_args


def _get_tensor_ids(obj: object) -> list[str]:
    """Function docstring.

    Args:
        obj: Arg.
    """
    if isinstance(obj, Tensor):
        return [obj.data.id]
    if isinstance(obj, (tuple, list)):
        ids = []
        for o in obj:
            ids.extend(_get_tensor_ids(o))
        return ids

    msg = "Control flow functions must return a Tensor or a tuple of Tensors."
    raise TypeError(msg)


def _process_trace_outputs(out: object, subgraph: IRBlock) -> str:
    """Function docstring.

    Args:
        out: Arg.
        subgraph: Arg.
    """
    out_ids = _get_tensor_ids(out)

    out_node = LogicalNode(
        id=str(uuid.uuid4()),
        op_type="Output",
        inputs=out_ids,
        shape_metadata=(),
    )
    subgraph.nodes[out_node.id] = out_node
    return out_node.id


def _trace_function(func: Callable, args: tuple[Tensor, ...], name: str) -> IRBlock:
    """Traces a Python function's execution into an IRBlock.

    This helper function temporarily redirects the active tracer to capture the
    operations
    performed by the given function when applied to proxy tensor arguments. It
    constructs
    input and output logical nodes to form a complete subgraph

    Args:
        func (Callable): The Python function to trace
        args (tuple[Tensor, ...]): The concrete or proxy tensor arguments to pass to the
        function
        name (str): The name to assign to the traced subgraph/IRBlock

    Returns:
    IRBlock: The traced intermediate representation block containing the logical
    nodes

    Raises:
    TypeError: If the traced function does not return a Tensor or a tuple of
    Tensors
    """
    prev_graph = global_tracing_state.active_graph
    is_tracing = global_tracing_state.is_tracing

    increment_trace_count(func)

    subgraph = global_tracing_state.start_tracing(name=name)
    input_ids, proxy_args = _wrap_proxy_inputs(args, subgraph)

    out = func(*proxy_args)

    out_node_id = _process_trace_outputs(out, subgraph)

    global_tracing_state.stop_tracing()
    global_tracing_state.active_graph = prev_graph
    global_tracing_state.is_tracing = is_tracing

    return IRBlock(
        id=name,
        nodes=list(subgraph.nodes.values()),
        inputs=input_ids,
        outputs=[out_node_id],
    )
