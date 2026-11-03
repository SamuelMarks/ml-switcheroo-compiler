"""Provide higher-order control flow primitives for tracing and eager execution.

This module implements functional control flow operators such as conditional branching,
while loops, scanning, vectorizing maps (vmap), and parallel maps (pmap). These
operators support both eager execution (using NumPy/Python loops) and tracing into an
intermediate representation (IR) graph for compilation
"""

from __future__ import annotations

import typing
import uuid
from collections.abc import Sequence

from ml_switcheroo_ir import LogicalGraph, LogicalNode
from ml_switcheroo_ir.types import TensorSpec

from ml_switcheroo_compiler.core.shape import (
    StructuralOperandProtocol,
    TensorOrScalar,
    invoke_callable,
    is_traced_data,
)
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.tracing.state import global_tracing_state
from ml_switcheroo_compiler.tracing.tracer import ProxyTensor, increment_trace_count

if typing.TYPE_CHECKING:
    pass


class SubgraphWithId(LogicalGraph):
    """LogicalGraph subclass carrying an explicit subgraph id property."""

    @property
    def id(self) -> str:
        """Return subgraph identifier.

        Returns:
            str: Subgraph identifier.
        """
        return self.name

    @id.setter
    def id(self, val: str) -> None:
        """Set subgraph identifier.

        Args:
            val (str): New subgraph identifier.
        """
        self.name = val


@typing.runtime_checkable
class _HasInputSpecsProtocol(typing.Protocol):
    """Protocol for graph objects exposing an input_specs dictionary."""

    input_specs: dict[str, TensorSpec]


def _wrap_proxy_inputs(
    args: Sequence[TensorOrScalar],
    subgraph: LogicalGraph,
) -> tuple[list[str], list[TensorOrScalar]]:
    """Compute the wrap proxy inputs tensor operation.

    Args:
        args (Sequence[TensorOrScalar]): The args parameter.
        subgraph (LogicalGraph): The subgraph parameter.

    Returns:
        tuple[list[str], list[TensorOrScalar]]: Result.
    """
    proxy_args: list[TensorOrScalar] = []
    input_ids: list[str] = []
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
            if isinstance(subgraph, _HasInputSpecsProtocol):
                from ml_switcheroo_ir.types import DType as IRDType

                ir_dt: IRDType | str = IRDType(arg.dtype.value) if arg.dtype.value in IRDType._value2member_map_ else str(arg.dtype.value)
                subgraph.input_specs[in_id] = TensorSpec(shape=arg.shape, dtype=ir_dt)
            proxy = ProxyTensor(id=in_id, shape=arg.shape, dtype=arg.dtype.value)
            proxy.concrete_value = arg.data
            proxy_tensor = Tensor(proxy, TensorConfig(arg.shape, arg.dtype, arg.device))
            proxy_args.append(proxy_tensor)
        else:
            proxy_args.append(arg)
    return input_ids, proxy_args


@typing.runtime_checkable
class _HasIdProtocol(typing.Protocol):
    """Protocol for objects exposing an id property."""

    @property
    def id(self) -> str:
        """Return identifier string.

        Returns:
            str: Identifier string.
        """
        ...


def _get_tensor_ids(obj: TensorOrScalar) -> list[str]:
    """Compute the get tensor ids tensor operation.

    Args:
        obj (TensorOrScalar): The obj parameter.

    Returns:
        list[str]: Result.

    Raises:
        TypeError: An exception.
    """
    if isinstance(obj, Tensor):
        cur: Tensor = obj
        while isinstance(cur.data, Tensor):
            cur = cur.data
        if is_traced_data(cur.data):
            return [str(cur.data.id)]
        raw_cur: StructuralOperandProtocol | TensorOrScalar = cur
        if isinstance(raw_cur, _HasIdProtocol):
            return [str(raw_cur.id)]
        return [str(uuid.uuid4())]
    if isinstance(obj, (tuple, list)):
        ids = []
        for o in obj:
            ids.extend(_get_tensor_ids(o))
        return ids

    msg = "Control flow functions must return a Tensor or a tuple of Tensors."
    raise TypeError(msg)


def _process_trace_outputs(out: TensorOrScalar, subgraph: LogicalGraph) -> str:
    """Compute the process trace outputs tensor operation.

    Args:
        out (TensorOrScalar): The out parameter.
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


def _trace_function(func: TensorOrScalar, args: Sequence[TensorOrScalar], name: str) -> LogicalGraph:
    """Trace a Python function's execution into a LogicalGraph.

    Args:
        func (TensorOrScalar): The func parameter.
        args (Sequence[TensorOrScalar]): The args parameter.
        name (str): The name parameter.

    Returns:
        LogicalGraph: Result.
    """
    from ml_switcheroo_compiler.core.config import config as compiler_config

    prev_graph = global_tracing_state.active_graph
    is_tracing = global_tracing_state.is_tracing
    prev_eager = compiler_config.eager_mode

    compiler_config.eager_mode = False
    increment_trace_count(func)

    raw_subgraph = global_tracing_state.start_tracing(name=name)
    subgraph = SubgraphWithId(
        name=raw_subgraph.name,
        nodes=raw_subgraph.nodes,
        inputs=raw_subgraph.inputs,
        input_specs=raw_subgraph.input_specs,
        outputs=raw_subgraph.outputs,
        initializers=raw_subgraph.initializers,
    )
    global_tracing_state.active_graph = subgraph
    input_ids, proxy_args = _wrap_proxy_inputs(args, subgraph)
    subgraph.inputs = input_ids

    try:
        out = invoke_callable(func, *proxy_args)
        out_node_id = _process_trace_outputs(out, subgraph)
    finally:
        global_tracing_state.stop_tracing()
        global_tracing_state.active_graph = prev_graph
        global_tracing_state.is_tracing = is_tracing
        compiler_config.eager_mode = prev_eager

    subgraph.id = name
    subgraph.outputs = [out_node_id]
    return subgraph
