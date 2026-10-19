"""Tests for test_strategy_server."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol, EagerTensorProtocol, EagerValue


def _test_strategy_server_part_1(_state: dict[str, EagerTensorProtocol]) -> None:
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

    with patch("ml_switcheroo_compiler.backends.registry.get_active_backend", side_effect=Exception("mock fail")):
        with patch("socket.socket"):
            with patch("threading.Thread") as mock_thread:
                server.start()
                assert server._running
                assert mock_thread.called
    getattr(_state, "update")({"MagicMock": MagicMock, "Server": Server, "io": io, "json": json})
    getattr(_state, "update")({"mock_backend": mock_backend, "mock_thread": mock_thread, "np": np, "patch": patch})
    getattr(_state, "update")({"server": server})
    _state.update(locals())


def _test_strategy_server_part_2(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 2 of test_strategy_server.

    Args:
        _state: Shared test state dictionary.
    """
    MagicMock, patch, server = _state["MagicMock"], _state["patch"], _state["server"]
    with patch("select.select") as mock_select:
        mock_sock = MagicMock()
        setattr(server, "_server", mock_sock)
        setattr(server, "_running", True)

        # Test empty header
        mock_conn = MagicMock()
        setattr(mock_conn.recv, "return_value", b"")
        setattr(mock_sock.accept, "return_value", (mock_conn, "addr"))
        setattr(mock_select, "return_value", ([mock_sock], [], []))

        # We need to break the loop
        def set_not_running(*args: EagerValue, **kwargs: EagerValue) -> EagerValue:
            """Test function set_not_running.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.
            """
            setattr(server, "_running", False)
            return b""

        setattr(mock_conn.recv, "side_effect", set_not_running)
        server._run_server()
    getattr(_state, "update")({"mock_conn": mock_conn, "mock_select": mock_select, "mock_sock": mock_sock, "set_not_running": set_not_running})
    _state.update(locals())


def _test_strategy_server_part_3(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 3 of test_strategy_server.

    Args:
        _state: Shared test state dictionary.
    """
    MagicMock, json, np, patch = _state["MagicMock"], _state["json"], _state["np"], _state["patch"]
    server = _state["server"]
    with patch("select.select") as mock_select:
        mock_sock = MagicMock()
        setattr(server, "_server", mock_sock)
        setattr(server, "_running", True)

        # Test valid header pull
        mock_conn = MagicMock()
        header = json.dumps({"action": "pull", "tensor_id": "test_t"}).encode("utf-8")
        header_len = len(header).to_bytes(4, "big")

        # Provide header len, header str, break loop
        def mock_recv(size: EagerValue) -> EagerValue:
            """Test function mock_recv.

            Args:
                size: Test parameter `size`.
            """
            if size == 4:
                setattr(server, "_running", False)
                return header_len
            return header

        setattr(mock_conn.recv, "side_effect", mock_recv)
        setattr(mock_sock.accept, "return_value", (mock_conn, "addr"))
        setattr(mock_select, "return_value", ([mock_sock], [], []))

        server.state_store["test_t"] = np.array([1.0, 2.0])
        server._run_server()
        assert mock_conn.sendall.called
    getattr(_state, "update")({"header": header, "header_len": header_len, "mock_conn": mock_conn, "mock_recv": mock_recv})
    getattr(_state, "update")({"mock_select": mock_select, "mock_sock": mock_sock})
    _state.update(locals())


def _test_strategy_server_part_4(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 4 of test_strategy_server.

    Args:
        _state: Shared test state dictionary.
    """
    MagicMock, json, patch, server = _state["MagicMock"], _state["json"], _state["patch"], _state["server"]
    with patch("select.select") as mock_select:
        mock_sock = MagicMock()
        setattr(server, "_server", mock_sock)
        setattr(server, "_running", True)

        # Test invalid pull
        mock_conn = MagicMock()
        header = json.dumps({"action": "pull", "tensor_id": "test_missing"}).encode("utf-8")
        header_len = len(header).to_bytes(4, "big")

        def mock_recv(size: EagerValue) -> EagerValue:
            """Test function mock_recv.

            Args:
                size: Test parameter `size`.
            """
            if size == 4:
                setattr(server, "_running", False)
                return header_len
            return header

        setattr(mock_conn.recv, "side_effect", mock_recv)
        setattr(mock_sock.accept, "return_value", (mock_conn, "addr"))
        setattr(mock_select, "return_value", ([mock_sock], [], []))

        server._run_server()
        assert not mock_conn.sendall.called
    getattr(_state, "update")({"header": header, "header_len": header_len, "mock_conn": mock_conn, "mock_recv": mock_recv})
    getattr(_state, "update")({"mock_select": mock_select, "mock_sock": mock_sock})
    _state.update(locals())


def _test_strategy_server_part_5(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 5 of test_strategy_server.

    Args:
        _state: Shared test state dictionary.
    """
    MagicMock, io, json, np = _state["MagicMock"], _state["io"], _state["json"], _state["np"]
    patch, server = _state["patch"], _state["server"]
    with patch("select.select") as mock_select:
        mock_sock = MagicMock()
        setattr(server, "_server", mock_sock)
        setattr(server, "_running", True)

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

            def __call__(self, size: EagerValue) -> EagerValue:
                """Test function __call__.

                Args:
                    size: Test parameter `size`.
                """
                self.calls += 1
                if self.calls == 1:
                    return header_len
                elif self.calls == 2:
                    return header
                elif self.calls == 3:
                    return data_len
                elif self.calls == 4:
                    setattr(server, "_running", False)
                    return data
                return b""

        setattr(mock_conn.recv, "side_effect", RecvSeq())
        setattr(mock_sock.accept, "return_value", (mock_conn, "addr"))
        setattr(mock_select, "return_value", ([mock_sock], [], []))

        server.state_store["test_push"] = np.array([1.0, 2.0])
        server._run_server()
        assert np.array_equal(server.state_store["test_push"], np.array([4.0, 6.0]))
    getattr(_state, "update")({"RecvSeq": RecvSeq, "bio": bio, "data": data, "data_arr": data_arr})
    getattr(_state, "update")({"data_len": data_len, "header": header, "header_len": header_len, "mock_conn": mock_conn})
    getattr(_state, "update")({"mock_select": mock_select, "mock_sock": mock_sock})
    _state.update(locals())


def _test_strategy_server_part_6(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 6 of test_strategy_server.

    Args:
        _state: Shared test state dictionary.
    """
    MagicMock, io, json, np = _state["MagicMock"], _state["io"], _state["json"], _state["np"]
    patch, server = _state["patch"], _state["server"]
    with patch("select.select") as mock_select:
        mock_sock = MagicMock()
        setattr(server, "_server", mock_sock)
        setattr(server, "_running", True)

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

            def __call__(self, size: EagerValue) -> EagerValue:
                """Test function __call__.

                Args:
                    size: Test parameter `size`.
                """
                self.calls += 1
                if self.calls == 1:
                    return header_len
                elif self.calls == 2:
                    return header
                elif self.calls == 3:
                    return data_len
                elif self.calls == 4:
                    setattr(server, "_running", False)
                    return data
                return b""

        setattr(mock_conn.recv, "side_effect", RecvSeq2())
        setattr(mock_sock.accept, "return_value", (mock_conn, "addr"))
        setattr(mock_select, "return_value", ([mock_sock], [], []))

        server._run_server()
        assert np.array_equal(server.state_store["test_push2"], np.array([3.0, 4.0]))
    getattr(_state, "update")({"RecvSeq2": RecvSeq2, "bio": bio, "data": data, "data_arr": data_arr})
    getattr(_state, "update")({"data_len": data_len, "header": header, "header_len": header_len, "mock_conn": mock_conn})
    getattr(_state, "update")({"mock_select": mock_select, "mock_sock": mock_sock})
    _state.update(locals())


def _test_strategy_server_part_7(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 7 of test_strategy_server.

    Args:
        _state: Shared test state dictionary.
    """
    MagicMock, patch, server = _state["MagicMock"], _state["patch"], _state["server"]
    with cast("EagerCallableProtocol", patch)("select.select", side_effect=cast("EagerCallableProtocol", Exception)("select error")):
        setattr(server, "_running", True)

        class MockServer:
            """Test class MockServer."""

            def __init__(self) -> None:
                """Test function __init__."""
                self.calls = 0

            def setblocking(self, b: EagerValue) -> None:
                """Test function setblocking.

                Args:
                    b: Test parameter `b`.
                """
                pass

            def close(self) -> None:
                """Test function close."""
                pass

        setattr(server, "_server", MockServer())

        def mock_warn(msg: EagerValue, stacklevel: EagerValue) -> None:
            """Test function mock_warn.

            Args:
                msg: Test parameter `msg`.
                stacklevel: Test parameter `stacklevel`.
            """
            setattr(server, "_running", False)

        with patch("warnings.warn", side_effect=mock_warn):
            server._run_server()

    with cast("EagerCallableProtocol", patch)("ml_switcheroo_compiler.backends.registry.get_active_backend", side_effect=cast("EagerCallableProtocol", Exception)("join mock fail")):
        server.join()

    setattr(server, "_server", MagicMock())
    getattr(_state, "update")({"MockServer": MockServer, "mock_warn": mock_warn})
    _state.update(locals())


def _test_strategy_server_part_8(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 8 of test_strategy_server.

    Args:
        _state: Shared test state dictionary.
    """
    server = _state["server"]
    setattr(server._server.close, "side_effect", Exception("close error"))
    server.join()
    from unittest.mock import MagicMock, patch

    from ml_switcheroo_compiler.distributed.strategy import DataParallelStrategy, MultiWorkerMirroredStrategy, Server

    server = cast("EagerTensorProtocol", Server())
    server.join()

    setattr(server, "_running", True)
    setattr(server, "_thread", MagicMock())
    server.join()

    # Run the extra coverage tests
    server = cast("EagerTensorProtocol", Server())
    setattr(server, "_server", None)
    server._run_server()

    setattr(server, "_server", MagicMock())
    setattr(server, "_running", True)
    getattr(_state, "update")({"DataParallelStrategy": DataParallelStrategy, "MagicMock": MagicMock, "MultiWorkerMirroredStrategy": MultiWorkerMirroredStrategy, "Server": Server})
    getattr(_state, "update")({"patch": patch, "server": server})
    _state.update(locals())


def _test_strategy_server_part_9(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 9 of test_strategy_server.

    Args:
        _state: Shared test state dictionary.
    """
    MagicMock, json, patch, server = _state["MagicMock"], _state["json"], _state["patch"], _state["server"]
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

            def __call__(self, size: EagerValue) -> EagerValue:
                """Test function __call__.

                Args:
                    size: Test parameter `size`.
                """
                self.calls += 1
                if self.calls == 1:
                    return header_len
                elif self.calls == 2:
                    return header
                elif self.calls == 3:
                    return data_len
                elif self.calls == 4:
                    setattr(server, "_running", False)
                    return b""  # chunk break
                return b""

        setattr(mock_conn.recv, "side_effect", RecvSeqBreak())
        setattr(server._server.accept, "return_value", (mock_conn, "addr"))
        setattr(mock_select, "return_value", ([server._server], [], []))
        server._run_server()
    getattr(_state, "update")({"RecvSeqBreak": RecvSeqBreak, "data_len": data_len, "header": header, "header_len": header_len})
    getattr(_state, "update")({"mock_conn": mock_conn, "mock_select": mock_select})
    _state.update(locals())


def _test_strategy_server_part_10(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 10 of test_strategy_server.

    Args:
        _state: Shared test state dictionary.
    """
    server = _state["server"]
    setattr(server, "_running", True)
    _state.update(locals())


def _test_strategy_server_part_11(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 11 of test_strategy_server.

    Args:
        _state: Shared test state dictionary.
    """
    MagicMock, json, patch, server = _state["MagicMock"], _state["json"], _state["patch"], _state["server"]
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

            def __call__(self, size: EagerValue) -> EagerValue:
                """Test function __call__.

                Args:
                    size: Test parameter `size`.
                """
                self.calls += 1
                if self.calls == 1:
                    return header_len
                elif self.calls == 2:
                    return header
                elif self.calls == 3:
                    setattr(server, "_running", False)
                    return b""  # continue
                return b""

        setattr(mock_conn.recv, "side_effect", RecvSeqContinue())
        setattr(server._server.accept, "return_value", (mock_conn, "addr"))
        setattr(mock_select, "return_value", ([server._server], [], []))
        server._run_server()
    getattr(_state, "update")({"RecvSeqContinue": RecvSeqContinue, "header": header, "header_len": header_len, "mock_conn": mock_conn})
    getattr(_state, "update")({"mock_select": mock_select})
    _state.update(locals())


def _test_strategy_server_part_12(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 12 of test_strategy_server.

    Args:
        _state: Shared test state dictionary.
    """
    DataParallelStrategy, MagicMock, MultiWorkerMirroredStrategy, patch = _state["DataParallelStrategy"], _state["MagicMock"], _state["MultiWorkerMirroredStrategy"], _state["patch"]
    server = _state["server"]

    class MockBackendJoin:
        """Test class MockBackendJoin."""

        def join_server(self, s: EagerValue) -> None:
            """Test function join_server.

            Args:
                s: Test parameter `s`.
            """
            self.called = True

    mock_b = MockBackendJoin()
    with cast("EagerCallableProtocol", patch)("ml_switcheroo_compiler.backends.registry.get_active_backend", return_value=mock_b):
        setattr(server, "_thread", MagicMock())
        server.join()
        assert getattr(mock_b, "called", False)

    mwms = MultiWorkerMirroredStrategy(target_env="browser")
    assert mwms.get_communication_protocol() == "webrtc"

    mwms = MultiWorkerMirroredStrategy(target_env="host")
    assert mwms.get_communication_protocol() == "tcp"

    dp = DataParallelStrategy(mesh_axis="data")
    assert dp.mesh_axis == "data"
    assert dp.get_communication_protocol() == "tcp"
    getattr(_state, "update")({"MockBackendJoin": MockBackendJoin, "dp": dp, "mock_b": mock_b, "mwms": mwms})


def test_strategy_server() -> None:
    """Test function test_strategy_server."""
    _state: dict[str, EagerTensorProtocol] = {}
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
