"""Tests for test_strategy_server."""

from __future__ import annotations

import typing

from tests.testing_utils import ensure_callable, ensure_eager_tensor, ensure_mutable, ensure_tensor_dict

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue


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
    _state.update(ensure_tensor_dict({"MagicMock": MagicMock, "Server": Server, "io": io, "json": json}))
    _state.update(ensure_tensor_dict({"mock_backend": mock_backend, "mock_thread": mock_thread, "np": np, "patch": patch}))
    _state.update(ensure_tensor_dict({"server": server}))


def _test_strategy_server_part_2(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 2 of test_strategy_server.

    Args:
        _state: Shared test state dictionary.
    """
    MagicMock, patch, server = _state["MagicMock"], _state["patch"], _state["server"]
    with patch("select.select") as mock_select:
        mock_sock = MagicMock()
        ensure_mutable(server)._server = ensure_eager_tensor(mock_sock)
        ensure_mutable(server)._running = ensure_eager_tensor(True)

        # Test empty header
        mock_conn = MagicMock()
        ensure_mutable(mock_conn.recv).return_value = ensure_eager_tensor(b"")
        ensure_mutable(mock_sock.accept).return_value = ensure_eager_tensor((mock_conn, "addr"))
        ensure_mutable(mock_select).return_value = ensure_eager_tensor(([mock_sock], [], []))

        # We need to break the loop
        def set_not_running(*args: EagerValue, **kwargs: EagerValue) -> EagerValue:
            """Test function set_not_running.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.

            Returns:
                EagerValue: Computed output value from `set_not_running`.
            """
            ensure_mutable(server)._running = ensure_eager_tensor(False)
            return b""

        ensure_mutable(mock_conn.recv).side_effect = ensure_eager_tensor(set_not_running)
        server._run_server()
    _state.update(ensure_tensor_dict({"mock_conn": mock_conn, "mock_select": mock_select, "mock_sock": mock_sock, "set_not_running": set_not_running}))
    _state.update(ensure_tensor_dict({"MagicMock": MagicMock, "patch": patch, "server": server}))


def _test_strategy_server_part_3(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 3 of test_strategy_server.

    Args:
        _state: Shared test state dictionary.
    """
    MagicMock, json, np, patch = _state["MagicMock"], _state["json"], _state["np"], _state["patch"]
    server = _state["server"]
    with patch("select.select") as mock_select:
        mock_sock = MagicMock()
        ensure_mutable(server)._server = ensure_eager_tensor(mock_sock)
        ensure_mutable(server)._running = ensure_eager_tensor(True)

        # Test valid header pull
        mock_conn = MagicMock()
        header = json.dumps({"action": "pull", "tensor_id": "test_t"}).encode("utf-8")
        header_len = len(header).to_bytes(4, "big")

        # Provide header len, header str, break loop
        def mock_recv(size: EagerValue) -> EagerValue:
            """Test function mock_recv.

            Args:
                size: Test parameter `size`.

            Returns:
                EagerValue: Computed output value from `mock_recv`.
            """
            if size == 4:
                ensure_mutable(server)._running = ensure_eager_tensor(False)
                return header_len
            return header

        ensure_mutable(mock_conn.recv).side_effect = ensure_eager_tensor(mock_recv)
        ensure_mutable(mock_sock.accept).return_value = ensure_eager_tensor((mock_conn, "addr"))
        ensure_mutable(mock_select).return_value = ensure_eager_tensor(([mock_sock], [], []))

        server.state_store["test_t"] = np.array([1.0, 2.0])
        server._run_server()
        assert mock_conn.sendall.called
    _state.update(ensure_tensor_dict({"header": header, "header_len": header_len, "mock_conn": mock_conn, "mock_recv": mock_recv}))
    _state.update(ensure_tensor_dict({"mock_select": mock_select, "mock_sock": mock_sock}))
    _state.update(ensure_tensor_dict({"MagicMock": MagicMock, "json": json, "np": np}))
    _state.update(ensure_tensor_dict({"patch": patch, "server": server}))


def _test_strategy_server_part_4(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 4 of test_strategy_server.

    Args:
        _state: Shared test state dictionary.
    """
    MagicMock, json, patch, server = _state["MagicMock"], _state["json"], _state["patch"], _state["server"]
    with patch("select.select") as mock_select:
        mock_sock = MagicMock()
        ensure_mutable(server)._server = ensure_eager_tensor(mock_sock)
        ensure_mutable(server)._running = ensure_eager_tensor(True)

        # Test invalid pull
        mock_conn = MagicMock()
        header = json.dumps({"action": "pull", "tensor_id": "test_missing"}).encode("utf-8")
        header_len = len(header).to_bytes(4, "big")

        def mock_recv(size: EagerValue) -> EagerValue:
            """Test function mock_recv.

            Args:
                size: Test parameter `size`.

            Returns:
                EagerValue: Computed output value from `mock_recv`.
            """
            if size == 4:
                ensure_mutable(server)._running = ensure_eager_tensor(False)
                return header_len
            return header

        ensure_mutable(mock_conn.recv).side_effect = ensure_eager_tensor(mock_recv)
        ensure_mutable(mock_sock.accept).return_value = ensure_eager_tensor((mock_conn, "addr"))
        ensure_mutable(mock_select).return_value = ensure_eager_tensor(([mock_sock], [], []))

        server._run_server()
        assert not mock_conn.sendall.called
    _state.update(ensure_tensor_dict({"header": header, "header_len": header_len, "mock_conn": mock_conn, "mock_recv": mock_recv}))
    _state.update(ensure_tensor_dict({"mock_select": mock_select, "mock_sock": mock_sock}))
    _state.update(ensure_tensor_dict({"MagicMock": MagicMock, "json": json, "patch": patch}))
    _state.update(ensure_tensor_dict({"server": server}))


def _test_strategy_server_part_5(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 5 of test_strategy_server.

    Args:
        _state: Shared test state dictionary.
    """
    MagicMock, io, json, np = _state["MagicMock"], _state["io"], _state["json"], _state["np"]
    patch, server = _state["patch"], _state["server"]
    with patch("select.select") as mock_select:
        mock_sock = MagicMock()
        ensure_mutable(server)._server = ensure_eager_tensor(mock_sock)
        ensure_mutable(server)._running = ensure_eager_tensor(True)

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

                Returns:
                    EagerValue: Computed output value from `__call__`.
                """
                self.calls += 1
                if self.calls == 1:
                    return header_len
                elif self.calls == 2:
                    return header
                elif self.calls == 3:
                    return data_len
                elif self.calls == 4:
                    ensure_mutable(server)._running = ensure_eager_tensor(False)
                    return data
                return b""

        ensure_mutable(mock_conn.recv).side_effect = ensure_eager_tensor(RecvSeq())
        ensure_mutable(mock_sock.accept).return_value = ensure_eager_tensor((mock_conn, "addr"))
        ensure_mutable(mock_select).return_value = ensure_eager_tensor(([mock_sock], [], []))

        server.state_store["test_push"] = np.array([1.0, 2.0])
        server._run_server()
        assert np.array_equal(server.state_store["test_push"], np.array([4.0, 6.0]))
    _state.update(ensure_tensor_dict({"RecvSeq": RecvSeq, "bio": bio, "data": data, "data_arr": data_arr}))
    _state.update(ensure_tensor_dict({"data_len": data_len, "header": header, "header_len": header_len, "mock_conn": mock_conn}))
    _state.update(ensure_tensor_dict({"mock_select": mock_select, "mock_sock": mock_sock}))
    _state.update(ensure_tensor_dict({"MagicMock": MagicMock, "io": io, "json": json}))
    _state.update(ensure_tensor_dict({"np": np, "patch": patch, "server": server}))


def _test_strategy_server_part_6(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 6 of test_strategy_server.

    Args:
        _state: Shared test state dictionary.
    """
    MagicMock, io, json, np = _state["MagicMock"], _state["io"], _state["json"], _state["np"]
    patch, server = _state["patch"], _state["server"]
    with patch("select.select") as mock_select:
        mock_sock = MagicMock()
        ensure_mutable(server)._server = ensure_eager_tensor(mock_sock)
        ensure_mutable(server)._running = ensure_eager_tensor(True)

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

                Returns:
                    EagerValue: Computed output value from `__call__`.
                """
                self.calls += 1
                if self.calls == 1:
                    return header_len
                elif self.calls == 2:
                    return header
                elif self.calls == 3:
                    return data_len
                elif self.calls == 4:
                    ensure_mutable(server)._running = ensure_eager_tensor(False)
                    return data
                return b""

        ensure_mutable(mock_conn.recv).side_effect = ensure_eager_tensor(RecvSeq2())
        ensure_mutable(mock_sock.accept).return_value = ensure_eager_tensor((mock_conn, "addr"))
        ensure_mutable(mock_select).return_value = ensure_eager_tensor(([mock_sock], [], []))

        server._run_server()
        assert np.array_equal(server.state_store["test_push2"], np.array([3.0, 4.0]))
    _state.update(ensure_tensor_dict({"RecvSeq2": RecvSeq2, "bio": bio, "data": data, "data_arr": data_arr}))
    _state.update(ensure_tensor_dict({"data_len": data_len, "header": header, "header_len": header_len, "mock_conn": mock_conn}))
    _state.update(ensure_tensor_dict({"mock_select": mock_select, "mock_sock": mock_sock}))
    _state.update(ensure_tensor_dict({"MagicMock": MagicMock, "io": io, "json": json}))
    _state.update(ensure_tensor_dict({"np": np, "patch": patch, "server": server}))


def _test_strategy_server_part_7(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 7 of test_strategy_server.

    Args:
        _state: Shared test state dictionary.
    """
    MagicMock, patch, server = _state["MagicMock"], _state["patch"], _state["server"]
    with ensure_callable(patch)("select.select", side_effect=ensure_callable(Exception)("select error")):
        ensure_mutable(server)._running = ensure_eager_tensor(True)

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

        ensure_mutable(server)._server = ensure_eager_tensor(MockServer())

        def mock_warn(msg: EagerValue, stacklevel: EagerValue) -> None:
            """Test function mock_warn.

            Args:
                msg: Test parameter `msg`.
                stacklevel: Test parameter `stacklevel`.
            """
            ensure_mutable(server)._running = ensure_eager_tensor(False)

        with patch("warnings.warn", side_effect=mock_warn):
            server._run_server()

    with ensure_callable(patch)("ml_switcheroo_compiler.backends.registry.get_active_backend", side_effect=ensure_callable(Exception)("join mock fail")):
        server.join()

    ensure_mutable(server)._server = ensure_eager_tensor(MagicMock())
    _state.update(ensure_tensor_dict({"MockServer": MockServer, "mock_warn": mock_warn}))
    _state.update(ensure_tensor_dict({"MagicMock": MagicMock, "patch": patch, "server": server}))


def _test_strategy_server_part_8(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 8 of test_strategy_server.

    Args:
        _state: Shared test state dictionary.
    """
    server = _state["server"]
    ensure_mutable(server._server.close).side_effect = ensure_eager_tensor(Exception("close error"))
    server.join()
    from unittest.mock import MagicMock, patch

    from ml_switcheroo_compiler.distributed.strategy import DataParallelStrategy, MultiWorkerMirroredStrategy, Server

    server = ensure_eager_tensor(Server())
    server.join()

    ensure_mutable(server)._running = ensure_eager_tensor(True)
    ensure_mutable(server)._thread = ensure_eager_tensor(MagicMock())
    server.join()

    # Run the extra coverage tests
    server = ensure_eager_tensor(Server())
    ensure_mutable(server)._server = ensure_eager_tensor(None)
    server._run_server()

    ensure_mutable(server)._server = ensure_eager_tensor(MagicMock())
    ensure_mutable(server)._running = ensure_eager_tensor(True)
    _state.update(ensure_tensor_dict({"DataParallelStrategy": DataParallelStrategy, "MagicMock": MagicMock, "MultiWorkerMirroredStrategy": MultiWorkerMirroredStrategy, "Server": Server}))
    _state.update(ensure_tensor_dict({"patch": patch, "server": server}))


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

                Returns:
                    EagerValue: Computed output value from `__call__`.
                """
                self.calls += 1
                if self.calls == 1:
                    return header_len
                elif self.calls == 2:
                    return header
                elif self.calls == 3:
                    return data_len
                elif self.calls == 4:
                    ensure_mutable(server)._running = ensure_eager_tensor(False)
                    return b""  # chunk break
                return b""

        ensure_mutable(mock_conn.recv).side_effect = ensure_eager_tensor(RecvSeqBreak())
        ensure_mutable(server._server.accept).return_value = ensure_eager_tensor((mock_conn, "addr"))
        ensure_mutable(mock_select).return_value = ensure_eager_tensor(([server._server], [], []))
        server._run_server()
    _state.update(ensure_tensor_dict({"RecvSeqBreak": RecvSeqBreak, "data_len": data_len, "header": header, "header_len": header_len}))
    _state.update(ensure_tensor_dict({"mock_conn": mock_conn, "mock_select": mock_select}))
    _state.update(ensure_tensor_dict({"MagicMock": MagicMock, "json": json, "patch": patch}))
    _state.update(ensure_tensor_dict({"server": server}))


def _test_strategy_server_part_10(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 10 of test_strategy_server.

    Args:
        _state: Shared test state dictionary.
    """
    server = _state["server"]
    ensure_mutable(server)._running = ensure_eager_tensor(True)
    _state.update(ensure_tensor_dict({"server": server}))


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

                Returns:
                    EagerValue: Computed output value from `__call__`.
                """
                self.calls += 1
                if self.calls == 1:
                    return header_len
                elif self.calls == 2:
                    return header
                elif self.calls == 3:
                    ensure_mutable(server)._running = ensure_eager_tensor(False)
                    return b""  # continue
                return b""

        ensure_mutable(mock_conn.recv).side_effect = ensure_eager_tensor(RecvSeqContinue())
        ensure_mutable(server._server.accept).return_value = ensure_eager_tensor((mock_conn, "addr"))
        ensure_mutable(mock_select).return_value = ensure_eager_tensor(([server._server], [], []))
        server._run_server()
    _state.update(ensure_tensor_dict({"RecvSeqContinue": RecvSeqContinue, "header": header, "header_len": header_len, "mock_conn": mock_conn}))
    _state.update(ensure_tensor_dict({"mock_select": mock_select}))
    _state.update(ensure_tensor_dict({"MagicMock": MagicMock, "json": json, "patch": patch}))
    _state.update(ensure_tensor_dict({"server": server}))


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
    with ensure_callable(patch)("ml_switcheroo_compiler.backends.registry.get_active_backend", return_value=mock_b):
        ensure_mutable(server)._thread = ensure_eager_tensor(MagicMock())
        server.join()
        assert ensure_eager_tensor(mock_b).called

    mwms = MultiWorkerMirroredStrategy(target_env="browser")
    assert mwms.get_communication_protocol() == "webrtc"

    mwms = MultiWorkerMirroredStrategy(target_env="host")
    assert mwms.get_communication_protocol() == "tcp"

    dp = DataParallelStrategy(mesh_axis="data")
    assert dp.mesh_axis == "data"
    assert dp.get_communication_protocol() == "tcp"
    _state.update(ensure_tensor_dict({"MockBackendJoin": MockBackendJoin, "dp": dp, "mock_b": mock_b, "mwms": mwms}))


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
