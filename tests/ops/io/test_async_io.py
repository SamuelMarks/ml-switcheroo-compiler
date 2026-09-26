"""Unit tests for asynchronous and non-blocking I/O operations and future abstractions."""

import asyncio
import os
import tempfile

import numpy as np
import pytest

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
from ml_switcheroo_compiler.ops.io.async_io import (
    AsyncIOFuture,
    NonblockingLoad,
    NonblockingSave,
    nonblocking_load,
    nonblocking_save,
)
from ml_switcheroo_compiler.tracing.state import global_tracing_state


@pytest.mark.asyncio
async def test_async_io_future_await_and_result() -> None:
    """Verify AsyncIOFuture behavior in an asyncio event loop."""
    loop = asyncio.get_running_loop()
    async_fut: asyncio.Future[object] = loop.create_future()
    async_fut.set_result(42)

    future = AsyncIOFuture(asyncio_future=async_fut)
    assert future.done()
    res = await future
    assert res == 42
    assert future.result() == 42

    # Direct tensor / node future without asyncio future
    node = IRNode(id="test_node", op_type="NonblockingLoad", inputs=[])
    direct_future = AsyncIOFuture(node=node)
    assert direct_future.done()
    assert direct_future.result() == node
    res_node = await direct_future
    assert res_node == node


def test_nonblocking_load_and_save_infer_shape() -> None:
    """Verify shape inference for NonblockingLoad and NonblockingSave."""
    op_load = NonblockingLoad()
    op_save = NonblockingSave()

    assert op_load.infer_shape(shape=(4, 8, 16)) == (4, 8, 16)
    assert op_load.infer_shape() == ()
    assert op_save.infer_shape() == ()


@pytest.mark.asyncio
async def test_nonblocking_load_and_save_eager_async() -> None:
    """Verify nonblocking_load and nonblocking_save in eager mode within an asyncio loop."""
    orig_eager = config.eager_mode
    config.eager_mode = True
    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = os.path.join(tmp_dir, "test_array.npy")
            arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

            # Save asynchronously
            save_fut = nonblocking_save(file_path, arr)
            await save_fut
            assert os.path.exists(file_path)

            # Load asynchronously
            load_fut = nonblocking_load(file_path, shape=(2, 2))
            loaded = await load_fut
            assert isinstance(loaded, np.ndarray) or hasattr(loaded, "shape")
            np.testing.assert_allclose(np.asarray(loaded), arr)
    finally:
        config.eager_mode = orig_eager


def test_nonblocking_load_and_save_eager_sync_fallback() -> None:
    """Verify nonblocking_load and nonblocking_save when no event loop is running."""
    orig_eager = config.eager_mode
    config.eager_mode = True
    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = os.path.join(tmp_dir, "test_sync.npy")
            arr = np.array([10.0, 20.0, 30.0], dtype=np.float32)

            save_fut = nonblocking_save(file_path, arr)
            assert save_fut.done()
            assert os.path.exists(file_path)

            load_fut = nonblocking_load(file_path, shape=(3,))
            assert load_fut.done()
            res = load_fut.result()
            np.testing.assert_allclose(np.asarray(getattr(res, "data", res)), arr)
    finally:
        config.eager_mode = orig_eager


def test_nonblocking_load_and_save_tracing_mode() -> None:
    """Verify nonblocking_load and nonblocking_save in graph tracing mode."""
    orig_eager = config.eager_mode
    orig_tracing = global_tracing_state.is_tracing
    orig_graph = global_tracing_state.active_graph

    config.eager_mode = False
    g = IRGraph(name="test_async_graph")
    global_tracing_state.is_tracing = True
    global_tracing_state.active_graph = g

    try:
        # Load in tracing mode
        load_fut = nonblocking_load("fake_path.npy", shape=(4, 8))
        assert load_fut.node is not None
        assert load_fut.node.op_type == "NonblockingLoad"
        assert load_fut.node.id in g.nodes

        # Save in tracing mode
        save_fut = nonblocking_save("save_path.npy", load_fut.tensor)
        assert save_fut.node is not None
        assert save_fut.node.op_type == "NonblockingSave"
        assert save_fut.node.id in g.nodes

        # Tracing with active_graph = None (lines 166 and 207 branches)
        global_tracing_state.active_graph = None
        load_fut_no_graph = nonblocking_load("fake2.npy")
        assert load_fut_no_graph.node is None
        save_fut_no_graph = nonblocking_save("save2.npy", load_fut.tensor)
        assert save_fut_no_graph.node is None
    finally:
        config.eager_mode = orig_eager
        global_tracing_state.is_tracing = orig_tracing
        global_tracing_state.active_graph = orig_graph
