"""Custom reverse-mode VJP rule wrapper (`custom_vjp` and `CustomVJPFunction`)."""

from __future__ import annotations

import typing
import uuid

from ml_switcheroo_ir import AttributeValue, LogicalGraph, LogicalNode

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.shape import OpValue, is_variadic_callable
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ir.callback_registry import register_custom_rule_handle
from ml_switcheroo_compiler.ops.control_flow_utils import _trace_function
from ml_switcheroo_compiler.tracing.state import global_tracing_state
from ml_switcheroo_compiler.tracing.tracer import ProxyTensor

from .custom_jvp_ops import _to_constant_attr
from .utils import (
    DifferentiableFn,
    PrimalTree,
    _extract_shape,
    _IdDataProtocol,
    _unwrap_data,
    as_primal_tree,
)

CustomVJPRule = typing.Union[DifferentiableFn, LogicalGraph, None]  # noqa: UP007


@typing.runtime_checkable
class _GraphHolderProtocol(typing.Protocol):
    """Protocol for objects exposing a `.graph` attribute."""

    graph: LogicalGraph


def _has_vjp_graph_holder(val: OpValue) -> typing.TypeGuard[_GraphHolderProtocol]:
    """Check whether `val` exposes `graph`.

    Args:
        val (OpValue): Candidate backward rule callable.

    Returns:
        bool: True if `val` has `graph`.
    """
    return isinstance(val, _GraphHolderProtocol)


def _has_vjp_id_data(val: OpValue) -> typing.TypeGuard[_IdDataProtocol]:
    """Check whether `val` exposes `id`.

    Args:
        val (OpValue): Candidate proxy data.

    Returns:
        bool: True if `val` has `id`.
    """
    return isinstance(val, _IdDataProtocol)


class CustomVJPFunction:
    """Wrapper for functions with user-defined reverse-mode VJP rules."""

    def __init__(self, fun: DifferentiableFn) -> None:
        """Initialize the custom VJP wrapper.

        Args:
            fun (DifferentiableFn): The base function being wrapped for custom VJPs.
        """
        self.fun: DifferentiableFn = fun
        self.fwd: DifferentiableFn | None = None
        self.bwd: CustomVJPRule = None
        self.bwd_subgraph: LogicalGraph | None = None
        self._tracing_fwd = False

    def defvjp(
        self,
        fwd: DifferentiableFn,
        bwd: CustomVJPRule,
        bwd_subgraph: LogicalGraph | None = None,
    ) -> None:
        """Define the forward and backward passes.

        Args:
            fwd (DifferentiableFn): The forward pass callable returning `(primal, residuals)`.
            bwd (CustomVJPRule): The backward pass callable or declarative `LogicalGraph`.
            bwd_subgraph (LogicalGraph | None): Declarative custom backward IR subgraph.
        """
        self.fwd = fwd
        self.bwd = bwd
        raw_bwd: OpValue = bwd
        if bwd_subgraph is not None:
            self.bwd_subgraph = bwd_subgraph
        elif isinstance(bwd, LogicalGraph):
            self.bwd_subgraph = bwd
        elif _has_vjp_graph_holder(raw_bwd) and isinstance(raw_bwd.graph, LogicalGraph):
            self.bwd_subgraph = raw_bwd.graph
        else:
            self.bwd_subgraph = None

    def defvjp_subgraph(self, bwd_subgraph: LogicalGraph) -> None:
        """Attach a declarative backward IR subgraph directly.

        Args:
            bwd_subgraph (LogicalGraph): Custom backward IR subgraph.
        """
        self.bwd_subgraph = bwd_subgraph

    def _extract_tensor_args(self, args: tuple[PrimalTree | OpValue, ...]) -> list[Tensor]:
        """Extract all tensor instances from the provided arguments.

        Args:
            args (tuple[PrimalTree | OpValue, ...]): The positional arguments.

        Returns:
            list[Tensor]: A list containing only the tensor arguments.
        """
        return [a for a in args if isinstance(a, Tensor)]

    def _trace_fwd_graph(self, tensor_args: list[Tensor]) -> LogicalGraph | None:
        """Trace the forward pass of the custom VJP function to construct its logical graph.

        Args:
            tensor_args (list[Tensor]): The list of tensor arguments.

        Returns:
            LogicalGraph | None: The traced logical graph for the forward pass.
        """
        if self.fwd is None or self.bwd is None:
            return None

        self._tracing_fwd = True
        try:
            return _trace_function(self.fwd, tuple(tensor_args), "fwd_pass")
        finally:
            self._tracing_fwd = False

    def _resolve_output_metadata(self, tensor_args: list[Tensor]) -> tuple[tuple[int, ...], str, str]:
        """Determine the shape, dtype, and device for the output based on the input tensors.

        Args:
            tensor_args (list[Tensor]): The list of tensor arguments.

        Returns:
            tuple[tuple[int, ...], str, str]: A tuple containing the output shape, dtype, and device string.
        """
        shape: tuple[int, ...] = ()
        dtype = "float32"
        device = "cpu"
        if tensor_args:
            first_arg = tensor_args[0]
            shape = _extract_shape(first_arg)
            dtype = first_arg.dtype.value
            device = str(first_arg.device)
        return (shape, dtype, device)

    def _emit_vjp_node(
        self,
        tensor_args: list[Tensor],
        fwd_graph: LogicalGraph | None,
        primal_graph: LogicalGraph | None,
    ) -> Tensor:
        """Emit a CustomVJP node into the active computation graph.

        Args:
            tensor_args (list[Tensor]): The tensor arguments.
            fwd_graph (LogicalGraph | None): The traced forward graph.
            primal_graph (LogicalGraph | None): The primal logical graph.

        Returns:
            Tensor: The proxy tensor representing the custom VJP output.
        """
        out_id = str(uuid.uuid4())
        meta = self._resolve_output_metadata(tensor_args)
        in_ids: list[str] = []
        for a in tensor_args:
            raw_data: OpValue = _unwrap_data(a)
            if _has_vjp_id_data(raw_data):
                in_ids.append(str(raw_data.id))
            else:
                cid = str(uuid.uuid4())
                arr = _unwrap_data(a)
                c_node = LogicalNode(
                    id=cid,
                    op_type="Constant",
                    inputs=[],
                    attributes={"value": _to_constant_attr(arr)},
                    shape_metadata=_extract_shape(arr),
                )
                global_tracing_state.add_node(c_node)
                in_ids.append(cid)

        subgraphs: dict[str, LogicalGraph] = {}
        if self.bwd_subgraph is not None:
            subgraphs["bwd"] = self.bwd_subgraph
        if primal_graph is not None:
            subgraphs["primal"] = primal_graph
        if fwd_graph is not None:
            subgraphs["fwd"] = fwd_graph

        bwd_attr: AttributeValue = (
            self.bwd
            if isinstance(self.bwd, LogicalGraph) or self.bwd is None
            else register_custom_rule_handle(self.bwd)
        )
        node_attrs: dict[str, AttributeValue] = {
            "primal_graph": primal_graph,
            "fwd_graph": fwd_graph,
            "bwd_fn": bwd_attr,
        }
        if self.bwd_subgraph is not None:
            node_attrs["bwd_graph"] = self.bwd_subgraph
            node_attrs["custom_backward_subgraph"] = self.bwd_subgraph
        node = LogicalNode(
            id=out_id,
            op_type="CustomVJP",
            inputs=in_ids,
            attributes=node_attrs,
            subgraphs=subgraphs,
            shape_metadata=meta[0],
        )
        global_tracing_state.add_node(node)
        proxy = ProxyTensor(id=out_id, shape=meta[0], dtype=meta[1])
        return Tensor(proxy, TensorConfig(meta[0], DType(meta[1]), meta[2]))

    def __call__(self, *args: PrimalTree | OpValue, **kwargs: PrimalTree | OpValue) -> PrimalTree:
        """Execute the function, tracing the forward and primal graphs if tracing is active.

        Args:
            *args (PrimalTree | OpValue): Positional arguments.
            **kwargs (PrimalTree | OpValue): Keyword arguments.

        Returns:
            PrimalTree: The computed primal output.
        """
        if config.eager_mode or not global_tracing_state.is_tracing or self._tracing_fwd:
            _callable_fn_1: OpValue = self.fun
            assert is_variadic_callable(_callable_fn_1)
            return as_primal_tree(_callable_fn_1(*args, **kwargs))

        tensor_args = self._extract_tensor_args(args)
        fwd_graph = self._trace_fwd_graph(tensor_args)
        primal_graph = _trace_function(self.fun, tuple(tensor_args), "primal_pass")
        return self._emit_vjp_node(tensor_args, fwd_graph, primal_graph)


def custom_vjp(fun: DifferentiableFn) -> CustomVJPFunction:
    """Wrap a function to allow defining custom vector-Jacobian product (VJP) rules.

    Args:
        fun (DifferentiableFn): The original function.

    Returns:
        CustomVJPFunction: The wrapped function that supports custom VJPs.
    """
    return CustomVJPFunction(fun)
