"""Tests for test_strategy_server."""

from __future__ import annotations

import io
import json
import types
import typing

import numpy as np
import pytest

from ml_switcheroo_compiler.distributed.strategy import Server
from tests.testing_utils import EagerBoundCallable, adapt_backend_module, require_callable

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import BackendArg


class _TestStrategyServerState(typing.TypedDict, total=False):
    """Typed shared state for `_test_strategy_server` split helpers."""

    DataParallelStrategy: type
    MagicMock: type
    MultiWorkerMirroredStrategy: type
    io: types.ModuleType
    json: types.ModuleType
    np: types.ModuleType
    patch: EagerBoundCallable
    server: Server


def _test_strategy_server_part_1(_state: _TestStrategyServerState) -> None:
    """Execute part 1 of test_strategy_server.

    Args:
        _state: Shared test state dictionary.
    """
    import io
    import json
    from unittest.mock import MagicMock, patch

    import numpy as np

    from ml_switcheroo_compiler.distributed.strategy import Server

    server = Server(server_def={"mock": "def"}, job_name="mock_job", task_index=1)

    with patch("ml_switcheroo_compiler.backends.registry.get_active_backend") as mock_backend:
        mock_backend.return_value = MagicMock()
        server.start()
        assert mock_backend.return_value.start_server.called

    with patch("ml_switcheroo_compiler.backends.registry.get_active_backend", side_effect=RuntimeError("mock fail")):
        with patch("socket.socket"):
            with patch("threading.Thread") as mock_thread:
                with pytest.warns(UserWarning, match="Backend failed to start custom server: mock fail"):
                    server.start()
                assert server._running
                assert mock_thread.called
    _state.update({"MagicMock": MagicMock, "io": io, "json": json})
    _state.update({"np": np, "patch": require_callable(patch)})
    _state.update({"server": server})


def _test_strategy_server_part_2(_state: _TestStrategyServerState) -> None:
    """Execute part 2 of test_strategy_server.

    Args:
        _state: Shared test state dictionary.
    """
    MagicMock, patch, server = _state["MagicMock"], _state["patch"], _state["server"]
    with patch("select.select") as mock_select:
        mock_sock = MagicMock()
        server._server = mock_sock
        pytest.MonkeyPatch().setattr(server, "_running", True, raising=False)

        # Test empty header
        mock_conn = MagicMock()
        pytest.MonkeyPatch().setattr(mock_conn.recv, "return_value", b"", raising=False)
        pytest.MonkeyPatch().setattr(mock_sock.accept, "return_value", (mock_conn, "addr"), raising=False)
        pytest.MonkeyPatch().setattr(mock_select, "return_value", ([mock_sock], [], []), raising=False)

        # We need to break the loop
        def set_not_running(*args: BackendArg, **kwargs: BackendArg) -> BackendArg:
            """Test function set_not_running.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.

            Returns:
                BackendArg: Computed output value from `set_not_running`.
            """
            pytest.MonkeyPatch().setattr(server, "_running", False, raising=False)
            return b""

        pytest.MonkeyPatch().setattr(mock_conn.recv, "side_effect", set_not_running, raising=False)
        server._run_server()
    _state.update({"MagicMock": MagicMock, "patch": require_callable(patch), "server": server})


def _test_strategy_server_part_3(_state: _TestStrategyServerState) -> None:
    """Execute part 3 of test_strategy_server.

    Args:
        _state: Shared test state dictionary.
    """
    MagicMock, patch = _state["MagicMock"], _state["patch"]
    server = _state["server"]
    with patch("select.select") as mock_select:
        mock_sock = MagicMock()
        server._server = mock_sock
        pytest.MonkeyPatch().setattr(server, "_running", True, raising=False)

        # Test valid header pull
        mock_conn = MagicMock()
        header = json.dumps({"action": "pull", "tensor_id": "test_t"}).encode("utf-8")
        header_len = len(header).to_bytes(4, "big")

        # Provide header len, header str, break loop
        def mock_recv(size: BackendArg) -> BackendArg:
            """Test function mock_recv.

            Args:
                size: Test parameter `size`.

            Returns:
                BackendArg: Computed output value from `mock_recv`.
            """
            if size == 4:
                pytest.MonkeyPatch().setattr(server, "_running", False, raising=False)
                return header_len
            return header

        pytest.MonkeyPatch().setattr(mock_conn.recv, "side_effect", mock_recv, raising=False)
        pytest.MonkeyPatch().setattr(mock_sock.accept, "return_value", (mock_conn, "addr"), raising=False)
        pytest.MonkeyPatch().setattr(mock_select, "return_value", ([mock_sock], [], []), raising=False)

        server.state_store["test_t"] = np.array([1.0, 2.0])
        server._run_server()
        assert mock_conn.sendall.called
    _state.update({"MagicMock": MagicMock, "json": json, "np": np})
    _state.update({"patch": require_callable(patch), "server": server})


def _test_strategy_server_part_4(_state: _TestStrategyServerState) -> None:
    """Execute part 4 of test_strategy_server.

    Args:
        _state: Shared test state dictionary.
    """
    MagicMock, patch, server = _state["MagicMock"], _state["patch"], _state["server"]
    with patch("select.select") as mock_select:
        mock_sock = MagicMock()
        server._server = mock_sock
        pytest.MonkeyPatch().setattr(server, "_running", True, raising=False)

        # Test invalid pull
        mock_conn = MagicMock()
        header = json.dumps({"action": "pull", "tensor_id": "test_missing"}).encode("utf-8")
        header_len = len(header).to_bytes(4, "big")

        def mock_recv(size: BackendArg) -> BackendArg:
            """Test function mock_recv.

            Args:
                size: Test parameter `size`.

            Returns:
                BackendArg: Computed output value from `mock_recv`.
            """
            if size == 4:
                pytest.MonkeyPatch().setattr(server, "_running", False, raising=False)
                return header_len
            return header

        pytest.MonkeyPatch().setattr(mock_conn.recv, "side_effect", mock_recv, raising=False)
        pytest.MonkeyPatch().setattr(mock_sock.accept, "return_value", (mock_conn, "addr"), raising=False)
        pytest.MonkeyPatch().setattr(mock_select, "return_value", ([mock_sock], [], []), raising=False)

        server._run_server()
        assert not mock_conn.sendall.called
    _state.update({"MagicMock": MagicMock, "json": json, "patch": require_callable(patch)})
    _state.update({"server": server})


def _test_strategy_server_part_5(_state: _TestStrategyServerState) -> None:
    """Execute part 5 of test_strategy_server.

    Args:
        _state: Shared test state dictionary.
    """
    MagicMock = _state["MagicMock"]
    patch, server = _state["patch"], _state["server"]
    with patch("select.select") as mock_select:
        mock_sock = MagicMock()
        server._server = mock_sock
        pytest.MonkeyPatch().setattr(server, "_running", True, raising=False)

        # Test push
        mock_conn = MagicMock()
        header = json.dumps({"action": "push", "tensor_id": "test_push"}).encode("utf-8")
        header_len = len(header).to_bytes(4, "big")

        data_arr = np.array([3.0, 4.0])
        bio = io.BytesIO()
        np.save(bio, data_arr, allow_pickle=False)
        data = bio.getvalue()
        data_len = len(data).to_bytes(8, "big")

        class RecvSeq:
            """Test class RecvSeq."""

            def __init__(self) -> None:
                """Test function __init__."""
                self.calls = 0

            def __call__(self, size: BackendArg) -> BackendArg:
                """Test function __call__.

                Args:
                    size: Test parameter `size`.

                Returns:
                    BackendArg: Computed output value from `__call__`.
                """
                self.calls += 1
                if self.calls == 1:
                    return header_len
                elif self.calls == 2:
                    return header
                elif self.calls == 3:
                    return data_len
                elif self.calls == 4:
                    pytest.MonkeyPatch().setattr(server, "_running", False, raising=False)
                    return data
                return b""

        pytest.MonkeyPatch().setattr(mock_conn.recv, "side_effect", RecvSeq(), raising=False)
        pytest.MonkeyPatch().setattr(mock_sock.accept, "return_value", (mock_conn, "addr"), raising=False)
        pytest.MonkeyPatch().setattr(mock_select, "return_value", ([mock_sock], [], []), raising=False)

        server.state_store["test_push"] = np.array([1.0, 2.0])
        server._run_server()
        from ml_switcheroo_compiler.backends.numpy.eager import ensure_ndarray

        assert np.array_equal(ensure_ndarray(server.state_store["test_push"]), np.array([4.0, 6.0]))
    _state.update({"MagicMock": MagicMock, "io": io, "json": json})
    _state.update({"np": np, "patch": require_callable(patch), "server": server})


def _test_strategy_server_part_6(_state: _TestStrategyServerState) -> None:
    """Execute part 6 of test_strategy_server.

    Args:
        _state: Shared test state dictionary.
    """
    MagicMock = _state["MagicMock"]
    patch, server = _state["patch"], _state["server"]
    with patch("select.select") as mock_select:
        mock_sock = MagicMock()
        server._server = mock_sock
        pytest.MonkeyPatch().setattr(server, "_running", True, raising=False)

        # Test push without existing state
        mock_conn = MagicMock()
        header = json.dumps({"action": "push", "tensor_id": "test_push2"}).encode("utf-8")
        header_len = len(header).to_bytes(4, "big")

        data_arr = np.array([3.0, 4.0])
        bio = io.BytesIO()
        np.save(bio, data_arr, allow_pickle=False)
        data = bio.getvalue()
        data_len = len(data).to_bytes(8, "big")

        class RecvSeq2:
            """Test class RecvSeq2."""

            def __init__(self) -> None:
                """Test function __init__."""
                self.calls = 0

            def __call__(self, size: BackendArg) -> BackendArg:
                """Test function __call__.

                Args:
                    size: Test parameter `size`.

                Returns:
                    BackendArg: Computed output value from `__call__`.
                """
                self.calls += 1
                if self.calls == 1:
                    return header_len
                elif self.calls == 2:
                    return header
                elif self.calls == 3:
                    return data_len
                elif self.calls == 4:
                    pytest.MonkeyPatch().setattr(server, "_running", False, raising=False)
                    return data
                return b""

        pytest.MonkeyPatch().setattr(mock_conn.recv, "side_effect", RecvSeq2(), raising=False)
        pytest.MonkeyPatch().setattr(mock_sock.accept, "return_value", (mock_conn, "addr"), raising=False)
        pytest.MonkeyPatch().setattr(mock_select, "return_value", ([mock_sock], [], []), raising=False)

        server._run_server()
        from ml_switcheroo_compiler.backends.numpy.eager import ensure_ndarray

        assert np.array_equal(ensure_ndarray(server.state_store["test_push2"]), np.array([3.0, 4.0]))
    _state.update({"MagicMock": MagicMock, "io": io, "json": json})
    _state.update({"np": np, "patch": require_callable(patch), "server": server})


def _test_strategy_server_part_7(_state: _TestStrategyServerState) -> None:
    """Execute part 7 of test_strategy_server.

    Args:
        _state: Shared test state dictionary.
    """
    MagicMock, patch, server = _state["MagicMock"], _state["patch"], _state["server"]
    with patch("select.select", side_effect=OSError("select error")):
        pytest.MonkeyPatch().setattr(server, "_running", True, raising=False)

        class MockServer:
            """Test class MockServer."""

            def __init__(self) -> None:
                """Test function __init__."""
                self.calls = 0

            def setblocking(self, b: BackendArg) -> None:
                """Test function setblocking.

                Args:
                    b: Test parameter `b`.
                """
                pass

            def close(self) -> None:
                """Test function close."""
                pass

        pytest.MonkeyPatch().setattr(server, "_server", MockServer(), raising=False)

        def mock_warn(msg: BackendArg, stacklevel: BackendArg) -> None:
            """Test function mock_warn.

            Args:
                msg: Test parameter `msg`.
                stacklevel: Test parameter `stacklevel`.
            """
            pytest.MonkeyPatch().setattr(server, "_running", False, raising=False)

        with patch("warnings.warn", side_effect=mock_warn):
            server._run_server()

    with patch(
        "ml_switcheroo_compiler.backends.registry.get_active_backend",
        side_effect=RuntimeError("join mock fail"),
    ):
        with pytest.warns(UserWarning, match="Backend failed to start custom server: join mock fail"):
            server.join()

    server._server = MagicMock()
    _state.update({"MagicMock": MagicMock, "patch": require_callable(patch), "server": server})


def _test_strategy_server_part_8(_state: _TestStrategyServerState) -> None:
    """Execute part 8 of test_strategy_server.

    Args:
        _state (_TestStrategyServerState): Shared test state dictionary.
    """
    server = _state["server"]
    pytest.MonkeyPatch().setattr(
        adapt_backend_module(server._server).require_symbol("close"),
        "side_effect",
        OSError("close error"),
        raising=False,
    )
    with pytest.warns(UserWarning, match="Server loop error: close error"):
        server.join()
    from unittest.mock import MagicMock, patch

    from ml_switcheroo_compiler.distributed.strategy import DataParallelStrategy, MultiWorkerMirroredStrategy, Server

    server = Server()
    server.join()

    pytest.MonkeyPatch().setattr(server, "_running", True, raising=False)
    pytest.MonkeyPatch().setattr(server, "_thread", MagicMock(), raising=False)
    server.join()

    # Run the extra coverage tests
    server = Server()
    pytest.MonkeyPatch().setattr(server, "_server", None, raising=False)
    server._run_server()

    server._server = MagicMock()
    pytest.MonkeyPatch().setattr(server, "_running", True, raising=False)
    _state.update(
        {
            "DataParallelStrategy": DataParallelStrategy,
            "MagicMock": MagicMock,
            "MultiWorkerMirroredStrategy": MultiWorkerMirroredStrategy,
        }
    )
    _state.update({"patch": require_callable(patch), "server": server})


def _test_strategy_server_part_9(_state: _TestStrategyServerState) -> None:
    """Execute part 9 of test_strategy_server.

    Args:
        _state: Shared test state dictionary.
    """
    MagicMock, patch, server = _state["MagicMock"], _state["patch"], _state["server"]
    with patch("select.select") as mock_select:
        # Test push break on chunk
        mock_conn = MagicMock()
        header = json.dumps({"action": "push", "tensor_id": "test_push"}).encode("utf-8")
        header_len = len(header).to_bytes(4, "big")
        data_len = (100).to_bytes(8, "big")

        class RecvSeqBreak:
            """Test class RecvSeqBreak."""

            def __init__(self) -> None:
                """Test function __init__."""
                self.calls = 0

            def __call__(self, size: BackendArg) -> BackendArg:
                """Test function __call__.

                Args:
                    size: Test parameter `size`.

                Returns:
                    BackendArg: Computed output value from `__call__`.
                """
                self.calls += 1
                if self.calls == 1:
                    return header_len
                elif self.calls == 2:
                    return header
                elif self.calls == 3:
                    return data_len
                elif self.calls == 4:
                    pytest.MonkeyPatch().setattr(server, "_running", False, raising=False)
                    return b""  # chunk break
                return b""

        pytest.MonkeyPatch().setattr(mock_conn.recv, "side_effect", RecvSeqBreak(), raising=False)
        pytest.MonkeyPatch().setattr(
            adapt_backend_module(server._server).require_symbol("accept"),
            "return_value",
            (mock_conn, "addr"),
            raising=False,
        )
        pytest.MonkeyPatch().setattr(mock_select, "return_value", ([server._server], [], []), raising=False)
        with pytest.warns(UserWarning, match="Server loop error"):
            server._run_server()
    _state.update({"MagicMock": MagicMock, "json": json, "patch": require_callable(patch)})
    _state.update({"server": server})


def _test_strategy_server_part_10(_state: _TestStrategyServerState) -> None:
    """Execute part 10 of test_strategy_server.

    Args:
        _state: Shared test state dictionary.
    """
    server = _state["server"]
    pytest.MonkeyPatch().setattr(server, "_running", True, raising=False)
    _state.update({"server": server})


def _test_strategy_server_part_11(_state: _TestStrategyServerState) -> None:
    """Execute part 11 of test_strategy_server.

    Args:
        _state: Shared test state dictionary.
    """
    MagicMock, patch, server = _state["MagicMock"], _state["patch"], _state["server"]
    with patch("select.select") as mock_select:
        # Test push continue on empty data len
        mock_conn = MagicMock()
        header = json.dumps({"action": "push", "tensor_id": "test_push"}).encode("utf-8")
        header_len = len(header).to_bytes(4, "big")

        class RecvSeqContinue:
            """Test class RecvSeqContinue."""

            def __init__(self) -> None:
                """Test function __init__."""
                self.calls = 0

            def __call__(self, size: BackendArg) -> BackendArg:
                """Test function __call__.

                Args:
                    size: Test parameter `size`.

                Returns:
                    BackendArg: Computed output value from `__call__`.
                """
                self.calls += 1
                if self.calls == 1:
                    return header_len
                elif self.calls == 2:
                    return header
                elif self.calls == 3:
                    pytest.MonkeyPatch().setattr(server, "_running", False, raising=False)
                    return b""  # continue
                return b""

        pytest.MonkeyPatch().setattr(mock_conn.recv, "side_effect", RecvSeqContinue(), raising=False)
        pytest.MonkeyPatch().setattr(
            adapt_backend_module(server._server).require_symbol("accept"),
            "return_value",
            (mock_conn, "addr"),
            raising=False,
        )
        pytest.MonkeyPatch().setattr(mock_select, "return_value", ([server._server], [], []), raising=False)
        server._run_server()
    _state.update({"MagicMock": MagicMock, "json": json, "patch": require_callable(patch)})
    _state.update({"server": server})


def _test_strategy_server_part_12(_state: _TestStrategyServerState) -> None:
    """Execute part 12 of test_strategy_server.

    Args:
        _state: Shared test state dictionary.
    """
    DataParallelStrategy, MagicMock, MultiWorkerMirroredStrategy, patch = (
        _state["DataParallelStrategy"],
        _state["MagicMock"],
        _state["MultiWorkerMirroredStrategy"],
        _state["patch"],
    )
    server = _state["server"]

    class MockBackendJoin:
        """Test class MockBackendJoin."""

        def join_server(self, s: BackendArg) -> None:
            """Test function join_server.

            Args:
                s: Test parameter `s`.
            """
            self.called = True

    mock_b = MockBackendJoin()
    with patch("ml_switcheroo_compiler.backends.registry.get_active_backend", return_value=mock_b):
        pytest.MonkeyPatch().setattr(server, "_thread", MagicMock(), raising=False)
        server.join()
        assert mock_b.called

    mwms = MultiWorkerMirroredStrategy(target_env="browser")
    assert mwms.get_communication_protocol() == "webrtc"

    mwms = MultiWorkerMirroredStrategy(target_env="host")
    assert mwms.get_communication_protocol() == "tcp"

    dp = DataParallelStrategy(mesh_axis="data")
    assert dp.mesh_axis == "data"
    assert dp.get_communication_protocol() == "tcp"


def test_strategy_server() -> None:
    """Test function test_strategy_server."""
    _state: _TestStrategyServerState = {}
    _test_strategy_server_part_1(_state)
    _test_strategy_server_part_2(_state)
    _test_strategy_server_part_3(_state)
    _test_strategy_server_part_4(_state)
    _test_strategy_server_part_5(_state)
    _test_strategy_server_part_6(_state)
    _test_strategy_server_part_7(_state)
    _test_strategy_server_part_8(_state)
    _test_strategy_server_part_9(_state)
    _test_strategy_server_part_10(_state)
    _test_strategy_server_part_11(_state)
    _test_strategy_server_part_12(_state)
    assert len(_state) > 0
