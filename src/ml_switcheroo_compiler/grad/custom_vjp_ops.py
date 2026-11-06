"""Gradient computation and autodiff utilities."""

from __future__ import annotations

import typing
import uuid

from ml_switcheroo_ir import AttributeValue

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.shape import TensorOrScalar, invoke_callable
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ir.core import LogicalGraph, LogicalNode
from ml_switcheroo_compiler.ops.control_flow_utils import _trace_function
from ml_switcheroo_compiler.tracing.state import global_tracing_state
from ml_switcheroo_compiler.tracing.tracer import ProxyTensor

from .utils import GradValue, _extract_shape, _IdDataProtocol, _unwrap_data


def _is_vjp_attr_dict(
    val: dict[str, AttributeValue],
) -> typing.TypeGuard[dict[str, LogicalGraph | GradValue | AttributeValue]]:
    """Type guard narrowing custom VJP node attributes dict.

    Args:
        val (dict[str, AttributeValue]): Attributes dict.

    Returns:
        bool: True if val is a dict.
    """
    return isinstance(val, dict)


@typing.runtime_checkable
class _GraphHolderProtocol(typing.Protocol):
    """Protocol for objects exposing a `.graph` attribute."""

    graph: LogicalGraph


class CustomVJPFunction:
    """Wrap for custom_vjp functions."""

    def __init__(self, fun: GradValue) -> None:
        """Initialize the custom VJP wrapper.

        Args:
            fun (GradValue): The base function being wrapped for custom VJPs.
        """
        self.fun = fun
        self.fwd: GradValue = None
        self.bwd: GradValue = None
        self.bwd_subgraph: LogicalGraph | None = None
        self._tracing_fwd = False

    def defvjp(self, fwd: GradValue, bwd: GradValue, bwd_subgraph: LogicalGraph | None = None) -> None:
        """Define the forward and backward passes.

        Args:
            fwd (GradValue): The forward pass.
            bwd (GradValue): The backward pass.
            bwd_subgraph (LogicalGraph | None): Declarative custom backward IR subgraph.
        """
        self.fwd = fwd
        self.bwd = bwd
        raw_bwd: TensorOrScalar = bwd
        if bwd_subgraph is not None:
            self.bwd_subgraph = bwd_subgraph
        elif isinstance(bwd, LogicalGraph):
            self.bwd_subgraph = bwd
        elif isinstance(raw_bwd, _GraphHolderProtocol) and isinstance(raw_bwd.graph, LogicalGraph):
            self.bwd_subgraph = raw_bwd.graph
        else:
            self.bwd_subgraph = None

    def defvjp_subgraph(self, bwd_subgraph: LogicalGraph) -> None:
        """Attach a declarative backward IR subgraph directly.

        Args:
            bwd_subgraph (LogicalGraph): Custom backward IR subgraph.
        """
        self.bwd_subgraph = bwd_subgraph

    def _extract_tensor_args(self, args: tuple[GradValue, ...]) -> list[Tensor]:
        """Extract all tensor instances from the provided arguments.

        Args:
            args (tuple[GradValue, ...]): The positional arguments.

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
            raw_data: TensorOrScalar = _unwrap_data(a)
            if isinstance(raw_data, _IdDataProtocol):
                in_ids.append(str(raw_data.id))
            else:
                cid = str(uuid.uuid4())
                arr = _unwrap_data(a)
                c_attrs: dict[str, AttributeValue] = {}
                c_node = LogicalNode(
                    id=cid,
                    op_type="Constant",
                    inputs=[],
                    attributes=c_attrs,
                    shape_metadata=_extract_shape(arr),
                )
                assert _is_vjp_attr_dict(c_node.attributes)
                c_node.attributes["value"] = arr
                global_tracing_state.add_node(c_node)
                in_ids.append(cid)

        vjp_attrs: dict[str, GradValue] = {
            "primal_graph": primal_graph,
            "fwd_graph": fwd_graph,
            "bwd_fn": self.bwd,
        }
        subgraphs: dict[str, LogicalGraph] = {}
        if self.bwd_subgraph is not None:
            vjp_attrs["bwd_graph"] = self.bwd_subgraph
            vjp_attrs["custom_backward_subgraph"] = self.bwd_subgraph
            subgraphs["bwd"] = self.bwd_subgraph
        if primal_graph is not None:
            subgraphs["primal"] = primal_graph
        if fwd_graph is not None:
            subgraphs["fwd"] = fwd_graph

        node_attrs: dict[str, AttributeValue] = {}
        node = LogicalNode(
            id=out_id,
            op_type="CustomVJP",
            inputs=in_ids,
            attributes=node_attrs,
            subgraphs=subgraphs,
            shape_metadata=meta[0],
        )
        assert _is_vjp_attr_dict(node.attributes)
        for k, v in vjp_attrs.items():
            node.attributes[k] = v
        global_tracing_state.add_node(node)
        proxy = ProxyTensor(id=out_id, shape=meta[0], dtype=meta[1])
        return Tensor(proxy, TensorConfig(meta[0], DType(meta[1]), meta[2]))

    def __call__(self, *args: GradValue, **kwargs: GradValue) -> GradValue:
        """Execute the function, tracing the forward and primal graphs if tracing is active.

        Args:
            *args (GradValue): Positional arguments.
            **kwargs (GradValue): Keyword arguments.

        Returns:
            GradValue: The computed primal output.
        """
        if config.eager_mode or not global_tracing_state.is_tracing or self._tracing_fwd:
            return invoke_callable(self.fun, *args, **kwargs)

        tensor_args = self._extract_tensor_args(args)
        fwd_graph = self._trace_fwd_graph(tensor_args)
        primal_graph = _trace_function(self.fun, tuple(tensor_args), "primal_pass")
        return self._emit_vjp_node(tensor_args, fwd_graph, primal_graph)


def custom_vjp(fun: GradValue) -> CustomVJPFunction:
    """Wrap a function to allow defining custom vector-Jacobian product (VJP) rules.

    Args:
        fun (GradValue): The original function.

    Returns:
        CustomVJPFunction: The wrapped function that supports custom VJPs.
    """
    return CustomVJPFunction(fun)
