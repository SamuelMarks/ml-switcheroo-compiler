# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Gradient computation and autodiff utilities."""

import contextlib
import math
import typing
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Optional

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


class CustomVJPFunction:
    """Wrap for custom_vjp functions."""

    def __init__(self, fun) -> None:
        """Initialize the custom VJP wrapper.

        Args:
            fun (Callable[..., object]): The base function being wrapped for custom VJPs.
        """
        self.fun = fun
        self.fwd = None
        self.bwd = None
        self.bwd_subgraph: Optional[LogicalGraph] = None
        self._tracing_fwd = False

    def defvjp(self, fwd, bwd, bwd_subgraph: Optional[LogicalGraph] = None) -> None:
        """Define the forward and backward passes.

        Args:
            fwd (Callable[..., object]): The forward pass.
            bwd (Callable[..., object]): The backward pass.
            bwd_subgraph (Optional[LogicalGraph]): Declarative custom backward IR subgraph.
        """
        self.fwd = fwd
        self.bwd = bwd
        if bwd_subgraph is not None:
            self.bwd_subgraph = bwd_subgraph
        elif isinstance(bwd, LogicalGraph):
            self.bwd_subgraph = bwd
        elif hasattr(bwd, "graph") and isinstance(bwd.graph, LogicalGraph):
            self.bwd_subgraph = bwd.graph
        else:
            self.bwd_subgraph = None

    def defvjp_subgraph(self, bwd_subgraph: LogicalGraph) -> None:
        """Attach a declarative backward IR subgraph directly.

        Args:
            bwd_subgraph (LogicalGraph): Custom backward IR subgraph.
        """
        self.bwd_subgraph = bwd_subgraph

    def _extract_tensor_args(self, args):
        """Extract all tensor instances from the provided arguments.

        Args:
            args (tuple): The positional arguments.

        Returns:
            list: A list containing only the tensor arguments.
        """
        return [a for a in args if isinstance(a, Tensor)]

    def _trace_fwd_graph(self, tensor_args):
        """Trace the forward pass of the custom VJP function to construct its logical graph.

        Args:
            tensor_args (list): The list of tensor arguments.

        Returns: Tensor: The traced logical graph for the forward pass.
        """
        if self.fwd is None or self.bwd is None:
            return None

        self._tracing_fwd = True
        try:
            return _trace_function(self.fwd, tuple(tensor_args), "fwd_pass")
        finally:
            self._tracing_fwd = False

    def _resolve_output_metadata(self, tensor_args) -> tuple[tuple[int, ...], str, str]:
        """Determine the shape, dtype, and device for the output based on the input tensors.

        Args:
            tensor_args (list): The list of tensor arguments.

        Returns:
            tuple: A tuple containing the output shape, dtype, and device string.
        """
        shape = ()
        dtype = "float32"
        device = "cpu"
        if tensor_args:
            first_arg = tensor_args[0]
            shape = getattr(first_arg, "shape", ())
            dtype = getattr(getattr(first_arg, "dtype", None), "value", "float32")
            device = getattr(first_arg, "device", "cpu")
        return (shape, dtype, device)

    def _emit_vjp_node(self, tensor_args, fwd_graph, primal_graph):
        """Emit a CustomVJP node into the active computation graph.

        Args:
            tensor_args (list): The tensor arguments.
            fwd_graph (object): The traced forward graph.
            primal_graph (object): The primal logical graph.

        Returns: Tensor: The proxy tensor representing the custom VJP output.
        """
        out_id = str(uuid.uuid4())
        meta = self._resolve_output_metadata(tensor_args)
        in_ids: list[str] = []
        for a in tensor_args:
            if hasattr(a, "data") and hasattr(a.data, "id"):
                in_ids.append(a.data.id)
            else:
                cid = str(uuid.uuid4())
                arr = getattr(a, "data", a)
                c_node = LogicalNode(
                    id=cid,
                    op_type="Constant",
                    inputs=[],
                    attributes={"value": arr},
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
            attributes=vjp_attrs,
            subgraphs=subgraphs,
            shape_metadata=meta[0],
        )
        global_tracing_state.add_node(node)
        proxy = ProxyTensor(id=out_id, shape=meta[0], dtype=meta[1])
        return Tensor(proxy, TensorConfig(meta[0], DType(meta[1]), meta[2]))

    def __call__(self, *args, **kwargs):
        """Execute the function, tracing the forward and primal graphs if tracing is active.

        Args:
            *args (object): Positional arguments.
            **kwargs (object): Keyword arguments.

        Returns: Tensor: The computed primal output.
        """
        if config.eager_mode or not global_tracing_state.is_tracing or self._tracing_fwd:
            return self.fun(*args, **kwargs)

        tensor_args = self._extract_tensor_args(args)
        fwd_graph = self._trace_fwd_graph(tensor_args)
        primal_graph = _trace_function(self.fun, tuple(tensor_args), "primal_pass")
        return self._emit_vjp_node(tensor_args, fwd_graph, primal_graph)


def custom_vjp(fun):
    """Wrap a function to allow defining custom vector-Jacobian product (VJP) rules.

    Args:
        fun (Callable): The original function.

    Returns:
        Callable: The wrapped function that supports custom VJPs.
    """
    return CustomVJPFunction(fun)
