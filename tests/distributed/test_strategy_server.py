import pytest


def test_strategy_server():
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
        with patch("socket.socket") as mock_socket:
            with patch("threading.Thread") as mock_thread:
                server.start()
                assert server._running
                assert mock_thread.called

    # Test _run_server logic
    with patch("select.select") as mock_select:
        mock_sock = MagicMock()
        server._server = mock_sock
        server._running = True

        # Test empty header
        mock_conn = MagicMock()
        mock_conn.recv.return_value = b""
        mock_sock.accept.return_value = (mock_conn, "addr")
        mock_select.return_value = ([mock_sock], [], [])

        # We need to break the loop
        def set_not_running(*args, **kwargs):
            server._running = False
            return b""

        mock_conn.recv.side_effect = set_not_running
        server._run_server()

    with patch("select.select") as mock_select:
        mock_sock = MagicMock()
        server._server = mock_sock
        server._running = True

        # Test valid header pull
        mock_conn = MagicMock()
        header = json.dumps({"action": "pull", "tensor_id": "test_t"}).encode("utf-8")
        header_len = len(header).to_bytes(4, "big")

        # Provide header len, header str, break loop
        def mock_recv(size):
            if size == 4:
                server._running = False
                return header_len
            return header

        mock_conn.recv.side_effect = mock_recv
        mock_sock.accept.return_value = (mock_conn, "addr")
        mock_select.return_value = ([mock_sock], [], [])

        server.state_store["test_t"] = np.array([1.0, 2.0])
        server._run_server()
        assert mock_conn.sendall.called

    with patch("select.select") as mock_select:
        mock_sock = MagicMock()
        server._server = mock_sock
        server._running = True

        # Test invalid pull
        mock_conn = MagicMock()
        header = json.dumps({"action": "pull", "tensor_id": "test_missing"}).encode("utf-8")
        header_len = len(header).to_bytes(4, "big")

        def mock_recv(size):
            if size == 4:
                server._running = False
                return header_len
            return header

        mock_conn.recv.side_effect = mock_recv
        mock_sock.accept.return_value = (mock_conn, "addr")
        mock_select.return_value = ([mock_sock], [], [])

        server._run_server()
        assert not mock_conn.sendall.called

    with patch("select.select") as mock_select:
        mock_sock = MagicMock()
        server._server = mock_sock
        server._running = True

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
            def __init__(self):
                self.calls = 0

            def __call__(self, size):
                self.calls += 1
                if self.calls == 1:
                    return header_len
                elif self.calls == 2:
                    return header
                elif self.calls == 3:
                    return data_len
                elif self.calls == 4:
                    server._running = False
                    return data
                return b""

        mock_conn.recv.side_effect = RecvSeq()
        mock_sock.accept.return_value = (mock_conn, "addr")
        mock_select.return_value = ([mock_sock], [], [])

        server.state_store["test_push"] = np.array([1.0, 2.0])
        server._run_server()
        assert np.array_equal(server.state_store["test_push"], np.array([4.0, 6.0]))

    with patch("select.select") as mock_select:
        mock_sock = MagicMock()
        server._server = mock_sock
        server._running = True

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
            def __init__(self):
                self.calls = 0

            def __call__(self, size):
                self.calls += 1
                if self.calls == 1:
                    return header_len
                elif self.calls == 2:
                    return header
                elif self.calls == 3:
                    return data_len
                elif self.calls == 4:
                    server._running = False
                    return data
                return b""

        mock_conn.recv.side_effect = RecvSeq2()
        mock_sock.accept.return_value = (mock_conn, "addr")
        mock_select.return_value = ([mock_sock], [], [])

        server._run_server()
        assert np.array_equal(server.state_store["test_push2"], np.array([3.0, 4.0]))

    with patch("select.select", side_effect=Exception("select error")):
        server._running = True

        class MockServer:
            def __init__(self):
                self.calls = 0

            def setblocking(self, b):
                pass

            def close(self):
                pass

        server._server = MockServer()

        def mock_warn(msg, stacklevel):
            server._running = False

        with patch("warnings.warn", side_effect=mock_warn):
            server._run_server()

    with patch("ml_switcheroo_compiler.backends.registry.get_active_backend", side_effect=Exception("join mock fail")):
        server.join()

    server._server = MagicMock()
    server._server.close.side_effect = Exception("close error")
    server.join()
    from unittest.mock import MagicMock, patch

    from ml_switcheroo_compiler.distributed.strategy import MultiWorkerMirroredStrategy, Server, TPUStrategy

    server = Server()
    server.join()

    server._running = True
    server._thread = MagicMock()
    server.join()

    # Run the extra coverage tests
    server = Server()
    server._server = None
    server._run_server()

    server._server = MagicMock()
    server._running = True

    with patch("select.select") as mock_select:
        # Test push break on chunk
        mock_conn = MagicMock()
        header = json.dumps({"action": "push", "tensor_id": "test_push"}).encode("utf-8")
        header_len = len(header).to_bytes(4, "big")
        data_len = (100).to_bytes(8, "big")

        class RecvSeqBreak:
            def __init__(self):
                self.calls = 0

            def __call__(self, size):
                self.calls += 1
                if self.calls == 1:
                    return header_len
                elif self.calls == 2:
                    return header
                elif self.calls == 3:
                    return data_len
                elif self.calls == 4:
                    server._running = False
                    return b""  # chunk break
                return b""

        mock_conn.recv.side_effect = RecvSeqBreak()
        server._server.accept.return_value = (mock_conn, "addr")
        mock_select.return_value = ([server._server], [], [])
        server._run_server()

    server._running = True
    with patch("select.select") as mock_select:
        # Test push continue on empty data len
        mock_conn = MagicMock()
        header = json.dumps({"action": "push", "tensor_id": "test_push"}).encode("utf-8")
        header_len = len(header).to_bytes(4, "big")

        class RecvSeqContinue:
            def __init__(self):
                self.calls = 0

            def __call__(self, size):
                self.calls += 1
                if self.calls == 1:
                    return header_len
                elif self.calls == 2:
                    return header
                elif self.calls == 3:
                    server._running = False
                    return b""  # continue
                return b""

        mock_conn.recv.side_effect = RecvSeqContinue()
        server._server.accept.return_value = (mock_conn, "addr")
        mock_select.return_value = ([server._server], [], [])
        server._run_server()

    # Join with mock backend
    class MockBackendJoin:
        def join_server(self, s):
            self.called = True

    mock_b = MockBackendJoin()
    with patch("ml_switcheroo_compiler.backends.registry.get_active_backend", return_value=mock_b):
        server._thread = MagicMock()
        server.join()
        assert getattr(mock_b, "called", False)

    mwms = MultiWorkerMirroredStrategy(target_env="browser")
    assert mwms.get_communication_protocol() == "webrtc"

    mwms = MultiWorkerMirroredStrategy(target_env="host")
    assert mwms.get_communication_protocol() == "tcp"

    tpu = TPUStrategy()
    with pytest.raises(RuntimeError):
        tpu.sync()

    class MockBackendTPU:
        def mock_sync(self, resolver, *args, **kwargs):
            return "synced"

    tpu.config = {"registry_hooks": {"sync": "mock_sync"}}
    with patch("ml_switcheroo_compiler.backends.registry.get_active_backend", return_value=MockBackendTPU()):
        assert tpu.sync() == "synced"
