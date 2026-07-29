"""Test Numpy distributed ipc coverage."""

from unittest.mock import MagicMock

import numpy as np

from ml_switcheroo_compiler.backends.numpy.distributed.ipc import _exchange_ipc_data, _exchange_ipc_data_coordinator, _exchange_ipc_data_worker, _ipc_all_gather, _ipc_all_reduce, _ipc_reduce_scatter


def test_ipc_coordinator(monkeypatch):
    """Test IPC coordinator."""

    class MockConnection:
        def recv(self):
            return (1, np.array([2]))  # rank 1, data 2

        def send(self, data):
            pass

        def close(self):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    class MockListener:
        def __init__(self, *args, **kwargs):
            pass

        def accept(self):
            return MockConnection()

        def close(self):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    def mock_client(*args, **kwargs):
        return MockConnection()

    monkeypatch.setattr("ml_switcheroo_compiler.backends.numpy.distributed.ipc.Listener", MockListener)
    monkeypatch.setattr("ml_switcheroo_compiler.backends.numpy.distributed.ipc.Client", mock_client)

    res = _exchange_ipc_data_coordinator(2, np.array([1]), 0.1, 0.01)
    assert len(res) == 2
    assert np.array_equal(res[0], np.array([1]))
    assert np.array_equal(res[1], np.array([2]))

    # Exception branch
    class ErrorListener:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            raise Exception("test error")

        def __exit__(self, *args):
            pass

    monkeypatch.setattr("ml_switcheroo_compiler.backends.numpy.distributed.ipc.Listener", ErrorListener)
    res_err = _exchange_ipc_data_coordinator(2, np.array([1]), 0.1, 0.01)
    assert len(res_err) == 2
    assert np.array_equal(res_err[0], np.array([1]))
    assert np.array_equal(res_err[1], np.array([1]))


def test_ipc_worker(monkeypatch):
    """Test IPC worker."""

    class MockConnection:
        def recv(self):
            return [np.array([1]), np.array([2])]

        def send(self, data):
            pass

        def close(self):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    class MockListener:
        def __init__(self, *args, **kwargs):
            pass

        def accept(self):
            return MockConnection()

        def close(self):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    def mock_client(*args, **kwargs):
        return MockConnection()

    monkeypatch.setattr("ml_switcheroo_compiler.backends.numpy.distributed.ipc.Listener", MockListener)
    monkeypatch.setattr("ml_switcheroo_compiler.backends.numpy.distributed.ipc.Client", mock_client)

    res = _exchange_ipc_data_worker(1, 2, np.array([2]), 0.1, 0.01)
    assert len(res) == 2
    assert np.array_equal(res[0], np.array([1]))

    # Timeout exception inside retry loop
    def mock_client_err(*args, **kwargs):
        raise Exception("test connection error")

    import time

    orig_time = time.time

    # fast forward time to hit timeout
    def mock_time():
        mock_time.count += 1
        return orig_time() + mock_time.count * 10

    mock_time.count = 0
    monkeypatch.setattr(time, "time", mock_time)
    monkeypatch.setattr("ml_switcheroo_compiler.backends.numpy.distributed.ipc.Client", mock_client_err)

    res_err = _exchange_ipc_data_worker(1, 2, np.array([2]), 0.1, 0.01)
    assert len(res_err) == 2
    assert np.array_equal(res_err[0], np.array([2]))


def test_ipc_exchange_ipc_data(monkeypatch):
    """Test standard exchange entrypoint."""
    monkeypatch.setattr("ml_switcheroo_compiler.backends.numpy.distributed.ipc._exchange_ipc_data_coordinator", lambda *args: ["mock_coord"])
    monkeypatch.setattr("ml_switcheroo_compiler.backends.numpy.distributed.ipc._exchange_ipc_data_worker", lambda *args: ["mock_worker"])

    assert _exchange_ipc_data(0, 2, np.array([1])) == ["mock_coord"]
    assert _exchange_ipc_data(1, 2, np.array([2])) == ["mock_worker"]


def test_ipc_collectives(monkeypatch):
    """Test ipc collectives fallback logic."""
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "2")

    mock_mesh = MagicMock()
    mock_mesh.size = 2

    # We mock IPC to return two 1D arrays
    monkeypatch.setattr("ml_switcheroo_compiler.backends.numpy.distributed.ipc._exchange_ipc_data", lambda *args: [np.array([1, 2]), np.array([3, 4])])

    # sum op
    res_reduce_scatter_sum = _ipc_reduce_scatter(np.array([1, 2]), "sum", axis=0, mesh=mock_mesh)
    assert np.array_equal(res_reduce_scatter_sum, np.array([4]))

    # max op
    res_reduce_scatter_max = _ipc_reduce_scatter(np.array([1, 2]), "max", axis=0, mesh=mock_mesh)
    assert np.array_equal(res_reduce_scatter_max, np.array([3]))

    # min op
    res_reduce_scatter_min = _ipc_reduce_scatter(np.array([1, 2]), "min", axis=0, mesh=mock_mesh)
    assert np.array_equal(res_reduce_scatter_min, np.array([1]))

    # fallback op
    res_reduce_scatter_fallback = _ipc_reduce_scatter(np.array([1, 2]), "unknown", axis=0, mesh=mock_mesh)
    assert np.array_equal(res_reduce_scatter_fallback, np.array([4]))

    # sum op
    res_all_reduce_sum = _ipc_all_reduce(np.array([1, 2]), "sum", mesh=mock_mesh)
    assert np.array_equal(res_all_reduce_sum, np.array([4, 6]))

    # max op
    res_all_reduce_max = _ipc_all_reduce(np.array([1, 2]), "max", mesh=mock_mesh)
    assert np.array_equal(res_all_reduce_max, np.array([3, 4]))

    # min op
    res_all_reduce_min = _ipc_all_reduce(np.array([1, 2]), "min", mesh=mock_mesh)
    assert np.array_equal(res_all_reduce_min, np.array([1, 2]))

    # fallback op
    res_all_reduce_fallback = _ipc_all_reduce(np.array([1, 2]), "unknown", mesh=mock_mesh)
    assert np.array_equal(res_all_reduce_fallback, np.array([4, 6]))

    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)


def test_ipc_collectives_no_mesh():
    """Test ipc collectives when mesh is None (line 127)."""
    import numpy as np

    from ml_switcheroo_compiler.backends.numpy.distributed.ipc import _ipc_all_reduce, _ipc_reduce_scatter

    t = np.array([1, 2])

    res_gather = _ipc_all_gather(t, axis=0, mesh=None)
    assert np.array_equal(res_gather, np.expand_dims(t, axis=0))

    res_gather_no_axis = _ipc_all_gather(t, axis=None, mesh=None)
    assert np.array_equal(res_gather_no_axis, t)

    res_reduce_scatter = _ipc_reduce_scatter(t, "sum", axis=0, mesh=None)
    assert np.array_equal(res_reduce_scatter, t)

    res_all_reduce = _ipc_all_reduce(t, "sum", mesh=None)
    assert np.array_equal(res_all_reduce, t)


def test_ipc_exchange_ipc_data_coordinator_timeout_fallback(monkeypatch):
    """Test coordinator timeout fallback loop."""
    import time

    from ml_switcheroo_compiler.backends.numpy.distributed.ipc import _exchange_ipc_data_coordinator

    class MockConnection:
        def recv(self):
            return (1, np.array([2]))  # rank 1, data 2

        def send(self, data):
            pass

        def close(self):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    class MockListener:
        def __init__(self, *args, **kwargs):
            pass

        def accept(self):
            return MockConnection()

        def close(self):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    # Ensure Client fails so we hit the timeout
    def mock_client_err(*args, **kwargs):
        raise Exception("test connection error")

    monkeypatch.setattr("ml_switcheroo_compiler.backends.numpy.distributed.ipc.Listener", MockListener)
    monkeypatch.setattr("ml_switcheroo_compiler.backends.numpy.distributed.ipc.Client", mock_client_err)

    orig_time = time.time

    # fast forward time to hit timeout quickly
    def mock_time():
        mock_time.count += 1
        return orig_time() + mock_time.count * 10

    mock_time.count = 0
    monkeypatch.setattr(time, "time", mock_time)

    res = _exchange_ipc_data_coordinator(2, np.array([1]), 0.1, 0.01)
    # The timeout breaks the loop, it should still return gathered_sorted
    assert len(res) == 2
    assert np.array_equal(res[0], np.array([1]))
    assert np.array_equal(res[1], np.array([2]))


def test_ipc_coordinator_sleep(monkeypatch):
    """Test coordinator sleep branch before timeout."""
    import time

    from ml_switcheroo_compiler.backends.numpy.distributed.ipc import _exchange_ipc_data_coordinator

    class MockConnection:
        def recv(self):
            return (1, np.array([2]))  # rank 1, data 2

        def send(self, data):
            pass

        def close(self):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    class MockListener:
        def __init__(self, *args, **kwargs):
            pass

        def accept(self):
            return MockConnection()

        def close(self):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    # Client fails first few times, then hits timeout to break loop
    def mock_client_err(*args, **kwargs):
        raise Exception("test connection error")

    monkeypatch.setattr("ml_switcheroo_compiler.backends.numpy.distributed.ipc.Listener", MockListener)
    monkeypatch.setattr("ml_switcheroo_compiler.backends.numpy.distributed.ipc.Client", mock_client_err)

    orig_time = time.time

    # Time advances slowly so we hit time.sleep
    def mock_time():
        mock_time.count += 1
        return orig_time() + mock_time.count * 0.001

    mock_time.count = 0
    monkeypatch.setattr(time, "time", mock_time)

    mock_sleep = MagicMock()
    monkeypatch.setattr(time, "sleep", mock_sleep)

    # timeout=0.01, retry_interval=0.001
    res = _exchange_ipc_data_coordinator(2, np.array([1]), 0.01, 0.001)
    assert mock_sleep.call_count > 0


def test_ipc_all_gather_no_axis(monkeypatch):
    """Test all gather without axis."""

    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "2")

    mock_mesh = MagicMock()
    mock_mesh.size = 2

    monkeypatch.setattr("ml_switcheroo_compiler.backends.numpy.distributed.ipc._exchange_ipc_data", lambda *args: [np.array([1]), np.array([2])])

    # Mesh>1 triggers IPC, axis=None -> concatenates along axis=0
    res_gather = _ipc_all_gather(np.array([1]), axis=None, mesh=mock_mesh)
    assert np.array_equal(res_gather, np.array([1, 2]))
