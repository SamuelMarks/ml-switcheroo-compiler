# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Module base.py."""

from __future__ import annotations

"""Define base definitions for the operation registry."""


import functools
import uuid
from typing import Callable, TypeVar

from ml_switcheroo_ir import LogicalNode

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

# We import dispatcher directly
from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

# Global operation registry

T = TypeVar("T", bound="OpDef")


class OpDef:
    """Abstract base class for all operations in the compiler.

    Acts strictly as a definition schema holding operation metadata.
    """

    op_type: str = "Unknown"

    def __call__(self, *args, **kwargs):
        """Universal dispatcher for the operation.

        Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        return dispatch_op(self.op_type, *args, **kwargs)

    def infer_shape(self, *args, **kwargs):
        """infer_shape function.

        Args:
            args: Positional args.
            kwargs: Keyword args.

        Args:
            message (str): The message.
            input_vars (list): The input vars.
            node (object): The node.
            **kwargs (object): Keyword arguments.
        self (object): The self parameter.

        Returns:
        object: Result.
        """
        return tuple()
        """Infer the output shape(s) and dtype(s) of the operation.

        Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        ...

    def eager_eval(self, *args, **kwargs):
        """Evaluate eager_eval operation.

        Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        return backend.execute_op(self.op_type, *args, **kwargs)


def emit_ir_node(
    graph,
    op_type: str,
    inputs: list[str],
    shape_metadata=None,
    attributes=None,
) -> str:
    """Emit a new node into the IR graph directly.

    Args:
        graph (object): The graph parameter.
        op_type (str): The op_type parameter.
        inputs (list): The inputs parameter.
        shape_metadata (object): The shape_metadata parameter.
        attributes (object): The attributes parameter.

    Returns:
        str: Result.
    """
    nid = f"{op_type.lower()}_{uuid.uuid4().hex[:6]}"
    node = LogicalNode(
        id=nid,
        op_type=op_type,
        inputs=inputs,
        shape_metadata=shape_metadata,
        attributes=attributes or {},
    )
    if graph is not None:
        graph.nodes[nid] = node
    else:
        from ml_switcheroo_compiler.tracing.state import global_tracing_state

        global_tracing_state.add_node(node)
    return nid


def dispatch_eager(op_name: str):
    """Evaluate dispatch_eager operation.

    Args:
        op_name (str): The op_name parameter.

    Returns:
        Callable: Result.
    """

    def decorator(func):
        """Evaluate decorator operation.

        Args:
            func (Callable): The func parameter.

        Returns:
            Callable: Result.
        """

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            """Evaluate wrapper operation.

            Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

            Returns:
            tuple[int, ...]: Result.
            """
            if config.eager_mode:
                from ml_switcheroo_compiler.backends.registry import get_active_backend

                backend = get_active_backend()
                raw_args = [a.data if isinstance(a, Tensor) else a for a in args]
                data = backend.execute_op(op_name, *raw_args, **kwargs)
                first_tensor = next((a for a in args if isinstance(a, Tensor)), None)
                device = first_tensor.device if first_tensor is not None else None
                dtype = first_tensor.dtype if first_tensor is not None else None
                if isinstance(data, tuple):
                    return tuple(
                        Tensor(
                            backend.array(d),
                            TensorConfig(backend.array(d).shape, getattr(d, "dtype", dtype), device),
                        )
                        for d in data
                    )
                return Tensor(backend.array(data), TensorConfig(backend.array(data).shape, dtype, device))
            return func(*args, **kwargs)

        return wrapper

    return decorator


__all__ = [
    "OpDef",
    "dispatch_eager",
    "emit_ir_node",
    "get_op",
    "register_op",
]

from ml_switcheroo_compiler.ops.registry import get_op, register_op  # noqa: E402
