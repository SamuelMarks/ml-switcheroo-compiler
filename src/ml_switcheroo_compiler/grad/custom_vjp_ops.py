"""Gradient computation and autodiff utilities."""

from __future__ import annotations

import uuid
from collections.abc import Callable, Mapping
from typing import Optional, cast

from ml_switcheroo_ir import AttributeValue

from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ir.core import LogicalGraph, LogicalNode
from ml_switcheroo_compiler.ops.control_flow_utils import _trace_function
from ml_switcheroo_compiler.tracing.state import global_tracing_state
from ml_switcheroo_compiler.tracing.tracer import ProxyTensor

from .utils import GradValue


class CustomVJPFunction:
    """Wrap for custom_vjp functions."""

    def __init__(self, fun: Callable[..., GradValue]) -> None:
        """Initialize the custom VJP wrapper.

        Args:
            fun (Callable[..., GradValue]): The base function being wrapped for custom VJPs.
        """
        self.fun = fun
        self.fwd: GradValue = None
        self.bwd: GradValue = None
        self.bwd_subgraph: Optional[LogicalGraph] = None
        self._tracing_fwd = False

    def defvjp(self, fwd: GradValue, bwd: GradValue, bwd_subgraph: Optional[LogicalGraph] = None) -> None:
        """Define the forward and backward passes.

        Args:
            fwd (GradValue): The forward pass.
            bwd (GradValue): The backward pass.
            bwd_subgraph (Optional[LogicalGraph]): Declarative custom backward IR subgraph.
        """
        self.fwd = fwd
        self.bwd = bwd
        if bwd_subgraph is not None:
            self.bwd_subgraph = bwd_subgraph
        elif isinstance(bwd, LogicalGraph):
            self.bwd_subgraph = bwd
        elif hasattr(bwd, "graph") and isinstance(getattr(bwd, "graph"), LogicalGraph):
            self.bwd_subgraph = getattr(bwd, "graph")
        else:
            self.bwd_subgraph = None

    def defvjp_subgraph(self, bwd_subgraph: LogicalGraph) -> None:
        """Attach a declarative backward IR subgraph directly.

        Args:
            bwd_subgraph (LogicalGraph): Custom backward IR subgraph.
        """
        self.bwd_subgraph = bwd_subgraph

    def _extract_tensor_args(self, args: tuple[Tensor | int | float | str | bool | None, ...]) -> list[Tensor]:
        """Extract all tensor instances from the provided arguments.

        Args:
            args (tuple[Tensor | int | float | str | bool | None, ...]): The positional arguments.

        Returns:
            list[Tensor]: A list containing only the tensor arguments.
        """
        return [a for a in args if isinstance(a, Tensor)]

    def _trace_fwd_graph(self, tensor_args: list[Tensor]) -> Optional[LogicalGraph]:
        """Trace the forward pass of the custom VJP function to construct its logical graph.

        Args:
            tensor_args (list[Tensor]): The list of tensor arguments.

        Returns:
            Optional[LogicalGraph]: The traced logical graph for the forward pass.
        """
        if self.fwd is None or self.bwd is None:
            return None

        self._tracing_fwd = True
        try:
            return cast(Optional[LogicalGraph], cast("Callable[..., EagerTensorProtocol]", _trace_function)(self.fwd, tuple(tensor_args), "fwd_pass"))
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
            shape = getattr(first_arg, "shape", ())
            dtype = getattr(getattr(first_arg, "dtype", None), "value", "float32")
            device = str(getattr(first_arg, "device", "cpu"))
        return (shape, dtype, device)

    def _emit_vjp_node(
        self,
        tensor_args: list[Tensor],
        fwd_graph: Optional[LogicalGraph],
        primal_graph: Optional[LogicalGraph],
    ) -> Tensor:
        """Emit a CustomVJP node into the active computation graph.

        Args:
            tensor_args (list[Tensor]): The tensor arguments.
            fwd_graph (Optional[LogicalGraph]): The traced forward graph.
            primal_graph (Optional[LogicalGraph]): The primal logical graph.

        Returns:
            Tensor: The proxy tensor representing the custom VJP output.
        """
        out_id = str(uuid.uuid4())
        meta = self._resolve_output_metadata(tensor_args)
        in_ids: list[str] = []
        for a in tensor_args:
            if hasattr(a, "data") and hasattr(a.data, "id"):
                in_ids.append(getattr(a.data, "id"))
            else:
                cid = str(uuid.uuid4())
                arr = getattr(a, "data", a)
                c_node = LogicalNode(
                    id=cid,
                    op_type="Constant",
                    inputs=[],
                    attributes=cast("Mapping[str, AttributeValue]", {"value": arr}),
                    shape_metadata=getattr(arr, "shape", ()),
                )
                global_tracing_state.add_node(c_node)
                in_ids.append(cid)

        vjp_attrs = {"primal_graph": primal_graph, "fwd_graph": fwd_graph, "bwd_fn": self.bwd}
        subgraphs: dict[str, LogicalGraph] = {}
        if self.bwd_subgraph is not None:
            vjp_attrs["bwd_graph"] = self.bwd_subgraph
            vjp_attrs["custom_backward_subgraph"] = self.bwd_subgraph
            subgraphs["bwd"] = self.bwd_subgraph
        if primal_graph is not None:
            subgraphs["primal"] = primal_graph
        if fwd_graph is not None:
            subgraphs["fwd"] = fwd_graph

        node = LogicalNode(
            id=out_id,
            op_type="CustomVJP",
            inputs=in_ids,
            attributes=cast("Mapping[str, AttributeValue]", vjp_attrs),
            subgraphs=subgraphs,
            shape_metadata=meta[0],
        )
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
            return self.fun(*args, **kwargs)

        tensor_args = getattr(self, "_extract_tensor_args")(args)
        fwd_graph = self._trace_fwd_graph(tensor_args)
        primal_graph = cast("Callable[..., LogicalGraph]", _trace_function)(self.fun, tuple(tensor_args), "primal_pass")
        return self._emit_vjp_node(tensor_args, fwd_graph, primal_graph)


def custom_vjp(fun: Callable[..., GradValue]) -> GradValue:
    """Wrap a function to allow defining custom vector-Jacobian product (VJP) rules.

    Args:
        fun (Callable[..., GradValue]): The original function.

    Returns:
        GradValue: The wrapped function that supports custom VJPs.
    """
    return cast(GradValue, CustomVJPFunction(fun))
