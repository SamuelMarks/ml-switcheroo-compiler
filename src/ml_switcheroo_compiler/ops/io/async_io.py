"""Asynchronous and non-blocking I/O operations and future abstractions."""

from __future__ import annotations

import asyncio
from collections.abc import Generator

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ir.core import IRNode
from ml_switcheroo_compiler.ops.base import OpDef, register_op
from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node


class AsyncIOFuture:
    """Awaitable future representing an asynchronous or non-blocking I/O operation.

    Resolves to a concrete IR node or evaluated Tensor upon completion or graph execution.

    Attributes:
        node (IRNode | None): Symbolic IRNode representing the operation in tracing mode.
        tensor (Tensor | None): Evaluated Tensor in eager mode.
    """

    def __init__(
        self,
        node: IRNode | None = None,
        tensor: Tensor | None = None,
        asyncio_future: asyncio.Future[object] | None = None,
    ) -> None:
        """Initialize AsyncIOFuture.

        Args:
            node (IRNode | None): Associated IRNode.
            tensor (Tensor | None): Associated Tensor.
            asyncio_future (Optional[asyncio.Future[object]]): Underlying asyncio Future if available.
        """
        self.node = node
        self.tensor = tensor
        self._asyncio_future = asyncio_future

    def done(self) -> bool:
        """Check if the asynchronous operation has completed.

        Returns:
            bool: True if completed or immediately resolved, False otherwise.
        """
        if self._asyncio_future is not None:
            return self._asyncio_future.done()
        return True

    def result(self) -> object:
        """Retrieve the resolved result or underlying IR node.

        Returns:
            object: Evaluated tensor, loaded data, or symbolic IR node.
        """
        if self._asyncio_future is not None and self._asyncio_future.done():
            return self._asyncio_future.result()
        if self.tensor is not None:
            return self.tensor
        return self.node

    def __await__(self) -> Generator[object, None, object]:
        """Make AsyncIOFuture awaitable in asyncio event loops.

        Returns:
            Generator[object, None, object]: Awaitable generator.
        """
        if self._asyncio_future is not None:
            return self._asyncio_future.__await__()

        async def _resolve() -> object:
            """Resolve future result asynchronously.

            Returns:
                object: Evaluated result.
            """
            return self.result()

        return _resolve().__await__()


@register_op("NonblockingLoad")
class NonblockingLoad(OpDef):
    """Operation for asynchronous non-blocking tensor loading."""

    op_name = "NonblockingLoad"

    def infer_shape(self, *args: object, **kwargs: object) -> tuple[int, ...]:
        """Infer shape of loaded tensor.

        Args:
            *args (object): Positional arguments.
            **kwargs (object): Keyword arguments including optional 'shape'.

        Returns:
            tuple[int, ...]: Inferred output shape.
        """
        shape = kwargs.get("shape", args[1] if len(args) > 1 else None)
        if shape is not None and isinstance(shape, (list, tuple)):
            return tuple(int(d) for d in shape)
        return ()


@register_op("NonblockingSave")
class NonblockingSave(OpDef):
    """Operation for asynchronous non-blocking tensor saving."""

    op_name = "NonblockingSave"

    def infer_shape(self, *args: object, **kwargs: object) -> tuple[int, ...]:
        """Infer shape of nonblocking save result (scalar empty tuple).

        Args:
            *args (object): Positional arguments.
            **kwargs (object): Keyword arguments.

        Returns:
            tuple[int, ...]: Empty tuple indicating no tensor payload output.
        """
        return ()


def nonblocking_load(
    file_path: str,
    shape: tuple[int, ...] | None = None,
    dtype: str = "float32",
    **kwargs: object,
) -> AsyncIOFuture:
    """Asynchronously load a tensor from disk without blocking the main compute thread.

    In eager mode with an active asyncio loop, queues loading in executor.
    In tracing mode, emits a NonblockingLoad IR node and returns an awaitable future.

    Args:
        file_path (str): Path to target file on disk.
        shape (tuple[int, ...] | None): Optional expected shape of tensor.
        dtype (str): Expected data type string.
        **kwargs (object): Additional format or storage options.

    Returns:
        AsyncIOFuture: Awaitable future resolving to the loaded Tensor or IRNode.
    """
    import contextvars

    from ml_switcheroo_compiler.ops.io.numpy_io import _fallback_load
    from ml_switcheroo_compiler.tracing.state import global_tracing_state

    if config.eager_mode:
        try:
            loop = asyncio.get_running_loop()
            ctx = contextvars.copy_context()
            fut: asyncio.Future[object] = loop.run_in_executor(None, ctx.run, _fallback_load, file_path)
            return AsyncIOFuture(asyncio_future=fut)
        except RuntimeError:
            loaded_data = _fallback_load(file_path)
            t_shape = getattr(loaded_data, "shape", shape or ())
            t = Tensor(loaded_data, TensorConfig(t_shape, dtype, config.default_device))
            return AsyncIOFuture(tensor=t)

    node_shape = shape or ()
    t_out = _emit_shape_node("NonblockingLoad", [file_path], {"file_path": file_path, "dtype": dtype, **kwargs}, node_shape, dtype)
    node_id = getattr(t_out.data, "id", None) if hasattr(t_out, "data") else None
    ir_node: IRNode | None = None
    if node_id and global_tracing_state.active_graph and hasattr(global_tracing_state.active_graph, "nodes"):
        ir_node = global_tracing_state.active_graph.nodes.get(node_id)
    return AsyncIOFuture(node=ir_node, tensor=t_out)


def nonblocking_save(
    file_path: str,
    tensor: object,
    **kwargs: object,
) -> AsyncIOFuture:
    """Asynchronously save a tensor to disk without blocking the execution pipeline.

    In eager mode with an active asyncio loop, queues saving in executor.
    In tracing mode, emits a NonblockingSave IR node.

    Args:
        file_path (str): Target save destination path.
        tensor (object): Tensor or numerical array to serialize.
        **kwargs (object): Additional serializer options.

    Returns:
        AsyncIOFuture: Awaitable future resolving upon completion or graph execution.
    """
    import contextvars

    from ml_switcheroo_compiler.ops.io.numpy_io import save
    from ml_switcheroo_compiler.tracing.state import global_tracing_state

    if config.eager_mode:
        try:
            loop = asyncio.get_running_loop()
            ctx = contextvars.copy_context()
            fut: asyncio.Future[object] = loop.run_in_executor(None, ctx.run, save, file_path, tensor)
            return AsyncIOFuture(asyncio_future=fut)
        except RuntimeError:
            save(file_path, tensor)
            return AsyncIOFuture()

    t_out = _emit_shape_node("NonblockingSave", [tensor], {"file_path": file_path, **kwargs}, (), "float32")
    node_id = getattr(t_out.data, "id", None) if hasattr(t_out, "data") else None
    ir_node: IRNode | None = None
    if node_id and global_tracing_state.active_graph and hasattr(global_tracing_state.active_graph, "nodes"):
        ir_node = global_tracing_state.active_graph.nodes.get(node_id)
    return AsyncIOFuture(node=ir_node, tensor=t_out)
