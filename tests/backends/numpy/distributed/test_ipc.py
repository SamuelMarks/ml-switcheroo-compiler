"""Test Numpy distributed ipc coverage."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast
from unittest.mock import MagicMock

import numpy as np

from ml_switcheroo_compiler.backends.numpy.distributed.ipc import _exchange_ipc_data, _exchange_ipc_data_coordinator, _exchange_ipc_data_worker, _ipc_all_gather, _ipc_all_reduce, _ipc_reduce_scatter

if TYPE_CHECKING:
    from collections.abc import Callable

    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue


def _test_ipc_coordinator_part_1(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 1 of test_ipc_coordinator.

    Args:
        _state: Shared test state dictionary.
    """

    class MockConnection:
        """Test class MockConnection."""

        def recv(self) -> EagerValue:
            """Test function recv."""
            return (1, np.array([2]))  # rank 1, data 2

        def send(self, data: EagerValue) -> None:
            """Test function send."""
            pass

        def close(self) -> None:
            """Test function close."""
            pass

        def __enter__(self) -> EagerValue:
            """Test function __enter__."""
            return cast("EagerTensorProtocol", self)

        def __exit__(self, *args: EagerValue) -> None:
            """Test function __exit__."""
            pass

    getattr(_state, "update")({"MockConnection": MockConnection})
    _state.update(locals())


def _test_ipc_coordinator_part_2(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 2 of test_ipc_coordinator.

    Args:
        _state: Shared test state dictionary.
    """
    MockConnection, monkeypatch = _state["MockConnection"], _state["monkeypatch"]

    class MockListener:
        """Test class MockListener."""

        def __init__(self, *args: EagerValue, **kwargs: EagerValue) -> None:
            """Test function __init__."""
            pass

        def accept(self) -> EagerValue:
            """Test function accept."""
            return MockConnection()

        def close(self) -> None:
            """Test function close."""
            pass

        def __enter__(self) -> EagerValue:
            """Test function __enter__."""
            return cast("EagerTensorProtocol", self)

        def __exit__(self, *args: EagerValue) -> None:
            """Test function __exit__."""
            pass

    def mock_client(*args: EagerValue, **kwargs: EagerValue) -> EagerValue:
        """Test function mock_client."""
        return MockConnection()

    monkeypatch.setattr("ml_switcheroo_compiler.backends.numpy.distributed.ipc.Listener", MockListener)
    monkeypatch.setattr("ml_switcheroo_compiler.backends.numpy.distributed.ipc.Client", mock_client)
    getattr(_state, "update")({"MockListener": MockListener, "mock_client": mock_client})
    _state.update(locals())


def _test_ipc_coordinator_part_3(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 3 of test_ipc_coordinator.

    Args:
        _state: Shared test state dictionary.
    """
    monkeypatch = _state["monkeypatch"]
    res = _exchange_ipc_data_coordinator(2, np.array([1]), 0.1, 0.01)
    assert len(res) == 2
    assert cast("Callable[..., EagerTensorProtocol]", np.array_equal)(res[0], cast("Callable[..., EagerTensorProtocol]", np.array)([1]))
    assert cast("Callable[..., EagerTensorProtocol]", np.array_equal)(res[1], cast("Callable[..., EagerTensorProtocol]", np.array)([2]))

    # Exception branch
    class ErrorListener:
        """Test class ErrorListener."""

        def __init__(self, *args: EagerValue, **kwargs: EagerValue) -> None:
            """Test function __init__."""
            pass

        def __enter__(self) -> None:
            """Test function __enter__."""
            raise Exception("test error")

        def __exit__(self, *args: EagerValue) -> None:
            """Test function __exit__."""
            pass

    monkeypatch.setattr("ml_switcheroo_compiler.backends.numpy.distributed.ipc.Listener", ErrorListener)
    res_err = _exchange_ipc_data_coordinator(2, np.array([1]), 0.1, 0.01)
    assert len(res_err) == 2
    assert cast("Callable[..., EagerTensorProtocol]", np.array_equal)(res_err[0], cast("Callable[..., EagerTensorProtocol]", np.array)([1]))
    assert cast("Callable[..., EagerTensorProtocol]", np.array_equal)(res_err[1], cast("Callable[..., EagerTensorProtocol]", np.array)([1]))
    getattr(_state, "update")({"ErrorListener": ErrorListener, "res": res, "res_err": res_err})


def test_ipc_coordinator(monkeypatch: EagerTensorProtocol) -> None:
    """Test IPC coordinator."""
    _state: dict[str, EagerTensorProtocol] = {"monkeypatch": monkeypatch}
    _test_ipc_coordinator_part_1(_state)
    _test_ipc_coordinator_part_2(_state)
    _test_ipc_coordinator_part_3(_state)
    assert isinstance(_state, dict)


def _test_ipc_worker_part_1(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 1 of test_ipc_worker.

    Args:
        _state: Shared test state dictionary.
    """

    class MockConnection:
        """Test class MockConnection."""

        def recv(self) -> EagerValue:
            """Test function recv."""
            return [np.array([1]), np.array([2])]

        def send(self, data: EagerValue) -> None:
            """Test function send."""
            pass

        def close(self) -> None:
            """Test function close."""
            pass

        def __enter__(self) -> EagerValue:
            """Test function __enter__."""
            return cast("EagerTensorProtocol", self)

        def __exit__(self, *args: EagerValue) -> None:
            """Test function __exit__."""
            pass

    getattr(_state, "update")({"MockConnection": MockConnection})
    _state.update(locals())


def _test_ipc_worker_part_2(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 2 of test_ipc_worker.

    Args:
        _state: Shared test state dictionary.
    """
    MockConnection, monkeypatch = _state["MockConnection"], _state["monkeypatch"]

    class MockListener:
        """Test class MockListener."""

        def __init__(self, *args: EagerValue, **kwargs: EagerValue) -> None:
            """Test function __init__."""
            pass

        def accept(self) -> EagerValue:
            """Test function accept."""
            return MockConnection()

        def close(self) -> None:
            """Test function close."""
            pass

        def __enter__(self) -> EagerValue:
            """Test function __enter__."""
            return cast("EagerTensorProtocol", self)

        def __exit__(self, *args: EagerValue) -> None:
            """Test function __exit__."""
            pass

    def mock_client(*args: EagerValue, **kwargs: EagerValue) -> EagerValue:
        """Test function mock_client."""
        return MockConnection()

    monkeypatch.setattr("ml_switcheroo_compiler.backends.numpy.distributed.ipc.Listener", MockListener)
    monkeypatch.setattr("ml_switcheroo_compiler.backends.numpy.distributed.ipc.Client", mock_client)
    getattr(_state, "update")({"MockListener": MockListener, "mock_client": mock_client})
    _state.update(locals())


def _test_ipc_worker_part_3(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 3 of test_ipc_worker.

    Args:
        _state: Shared test state dictionary.
    """
    monkeypatch = _state["monkeypatch"]
    res = _exchange_ipc_data_worker(1, 2, np.array([2]), 0.1, 0.01)
    assert len(res) == 2
    assert cast("Callable[..., EagerTensorProtocol]", np.array_equal)(res[0], cast("Callable[..., EagerTensorProtocol]", np.array)([1]))

    def mock_client_err(*args: EagerValue, **kwargs: EagerValue) -> None:
        """Test function mock_client_err."""
        raise Exception("test connection error")

    import time

    orig_time = time.time

    # fast forward time to hit timeout
    def mock_time() -> EagerValue:
        """Test function mock_time."""
        setattr(mock_time, "count", getattr(mock_time, "count") + 1)
        return cast("EagerValue", orig_time() + getattr(mock_time, "count") * 10)

    setattr(mock_time, "count", 0)
    monkeypatch.setattr(time, "time", mock_time)
    monkeypatch.setattr("ml_switcheroo_compiler.backends.numpy.distributed.ipc.Client", mock_client_err)

    res_err = _exchange_ipc_data_worker(1, 2, np.array([2]), 0.1, 0.01)
    assert len(res_err) == 2
    assert cast("Callable[..., EagerTensorProtocol]", np.array_equal)(res_err[0], cast("Callable[..., EagerTensorProtocol]", np.array)([2]))
    getattr(_state, "update")({"mock_client_err": mock_client_err, "mock_time": mock_time, "orig_time": orig_time, "res": res})
    getattr(_state, "update")({"res_err": res_err, "time": time})


def test_ipc_worker(monkeypatch: EagerTensorProtocol) -> None:
    """Test IPC worker."""
    _state: dict[str, EagerTensorProtocol] = {"monkeypatch": monkeypatch}
    _test_ipc_worker_part_1(_state)
    _test_ipc_worker_part_2(_state)
    _test_ipc_worker_part_3(_state)
    assert isinstance(_state, dict)


def test_ipc_exchange_ipc_data(monkeypatch: EagerTensorProtocol) -> None:
    """Test standard exchange entrypoint."""
    monkeypatch.setattr("ml_switcheroo_compiler.backends.numpy.distributed.ipc._exchange_ipc_data_coordinator", lambda *args: ["mock_coord"])
    monkeypatch.setattr("ml_switcheroo_compiler.backends.numpy.distributed.ipc._exchange_ipc_data_worker", lambda *args: ["mock_worker"])

    assert _exchange_ipc_data(0, 2, np.array([1])) == ["mock_coord"]
    assert _exchange_ipc_data(1, 2, np.array([2])) == ["mock_worker"]


def test_ipc_collectives(monkeypatch: EagerTensorProtocol) -> None:
    """Test ipc collectives fallback logic."""
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "2")

    mock_mesh = MagicMock()
    mock_mesh.size = 2

    # We mock IPC to return two 1D arrays
    monkeypatch.setattr("ml_switcheroo_compiler.backends.numpy.distributed.ipc._exchange_ipc_data", lambda *args: [np.array([1, 2]), np.array([3, 4])])

    # sum op
    res_reduce_scatter_sum = _ipc_reduce_scatter(np.array([1, 2]), "sum", axis=0, mesh=mock_mesh)
    assert cast("Callable[..., EagerTensorProtocol]", np.array_equal)(res_reduce_scatter_sum, cast("Callable[..., EagerTensorProtocol]", np.array)([4]))

    # max op
    res_reduce_scatter_max = _ipc_reduce_scatter(np.array([1, 2]), "max", axis=0, mesh=mock_mesh)
    assert cast("Callable[..., EagerTensorProtocol]", np.array_equal)(res_reduce_scatter_max, cast("Callable[..., EagerTensorProtocol]", np.array)([3]))

    # min op
    res_reduce_scatter_min = _ipc_reduce_scatter(np.array([1, 2]), "min", axis=0, mesh=mock_mesh)
    assert cast("Callable[..., EagerTensorProtocol]", np.array_equal)(res_reduce_scatter_min, cast("Callable[..., EagerTensorProtocol]", np.array)([1]))

    # fallback op
    res_reduce_scatter_fallback = _ipc_reduce_scatter(np.array([1, 2]), "unknown", axis=0, mesh=mock_mesh)
    assert cast("Callable[..., EagerTensorProtocol]", np.array_equal)(res_reduce_scatter_fallback, cast("Callable[..., EagerTensorProtocol]", np.array)([4]))

    # sum op
    res_all_reduce_sum = _ipc_all_reduce(np.array([1, 2]), "sum", mesh=mock_mesh)
    assert cast("Callable[..., EagerTensorProtocol]", np.array_equal)(res_all_reduce_sum, cast("Callable[..., EagerTensorProtocol]", np.array)([4, 6]))

    # max op
    res_all_reduce_max = _ipc_all_reduce(np.array([1, 2]), "max", mesh=mock_mesh)
    assert cast("Callable[..., EagerTensorProtocol]", np.array_equal)(res_all_reduce_max, cast("Callable[..., EagerTensorProtocol]", np.array)([3, 4]))

    # min op
    res_all_reduce_min = _ipc_all_reduce(np.array([1, 2]), "min", mesh=mock_mesh)
    assert cast("Callable[..., EagerTensorProtocol]", np.array_equal)(res_all_reduce_min, cast("Callable[..., EagerTensorProtocol]", np.array)([1, 2]))

    # fallback op
    res_all_reduce_fallback = _ipc_all_reduce(np.array([1, 2]), "unknown", mesh=mock_mesh)
    assert cast("Callable[..., EagerTensorProtocol]", np.array_equal)(res_all_reduce_fallback, cast("Callable[..., EagerTensorProtocol]", np.array)([4, 6]))

    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)


def test_ipc_collectives_no_mesh() -> None:
    """Test ipc collectives when mesh is None (line 127)."""
    import numpy as np

    from ml_switcheroo_compiler.backends.numpy.distributed.ipc import _ipc_all_reduce, _ipc_reduce_scatter

    t = np.array([1, 2])

    res_gather = _ipc_all_gather(t, axis=0, mesh=None)
    assert cast("Callable[..., EagerTensorProtocol]", np.array_equal)(res_gather, cast("Callable[..., EagerTensorProtocol]", np.expand_dims)(t, axis=0))

    res_gather_no_axis = _ipc_all_gather(t, axis=None, mesh=None)
    assert cast("Callable[..., EagerTensorProtocol]", np.array_equal)(res_gather_no_axis, t)

    res_reduce_scatter = _ipc_reduce_scatter(t, "sum", 0, mesh=None)
    assert cast("Callable[..., EagerTensorProtocol]", np.array_equal)(res_reduce_scatter, t)

    # string fallback
    assert _ipc_all_gather("string_tensor", 0, mesh=None) == "string_tensor"
    assert _ipc_reduce_scatter("string_tensor", "sum", 0, mesh=None) == "string_tensor"
    assert _ipc_all_reduce("string_tensor", "sum", mesh=None) == "string_tensor"

    res_all_reduce = _ipc_all_reduce(t, "sum", mesh=None)
    assert cast("Callable[..., EagerTensorProtocol]", np.array_equal)(res_all_reduce, t)


def _test_ipc_exchange_ipc_data_coordinator_timeout_fallback_part_1(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 1 of test_ipc_exchange_ipc_data_coordinator_timeout_fallback.

    Args:
        _state: Shared test state dictionary.
    """
    import time

    from ml_switcheroo_compiler.backends.numpy.distributed.ipc import _exchange_ipc_data_coordinator

    class MockConnection:
        """Test class MockConnection."""

        def recv(self) -> EagerValue:
            """Test function recv."""
            return (1, np.array([2]))  # rank 1, data 2

        def send(self, data: EagerValue) -> None:
            """Test function send."""
            pass

        def close(self) -> None:
            """Test function close."""
            pass

        def __enter__(self) -> EagerValue:
            """Test function __enter__."""
            return cast("EagerTensorProtocol", self)

        def __exit__(self, *args: EagerValue) -> None:
            """Test function __exit__."""
            pass

    getattr(_state, "update")({"MockConnection": MockConnection, "_exchange_ipc_data_coordinator": _exchange_ipc_data_coordinator, "time": time})
    _state.update(locals())


def _test_ipc_exchange_ipc_data_coordinator_timeout_fallback_part_2(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 2 of test_ipc_exchange_ipc_data_coordinator_timeout_fallback.

    Args:
        _state: Shared test state dictionary.
    """
    MockConnection, monkeypatch = _state["MockConnection"], _state["monkeypatch"]

    class MockListener:
        """Test class MockListener."""

        def __init__(self, *args: EagerValue, **kwargs: EagerValue) -> None:
            """Test function __init__."""
            pass

        def accept(self) -> EagerValue:
            """Test function accept."""
            return MockConnection()

        def close(self) -> None:
            """Test function close."""
            pass

        def __enter__(self) -> EagerValue:
            """Test function __enter__."""
            return cast("EagerTensorProtocol", self)

        def __exit__(self, *args: EagerValue) -> None:
            """Test function __exit__."""
            pass

    # Ensure Client fails so we hit the timeout
    def mock_client_err(*args: EagerValue, **kwargs: EagerValue) -> None:
        """Test function mock_client_err."""
        raise Exception("test connection error")

    monkeypatch.setattr("ml_switcheroo_compiler.backends.numpy.distributed.ipc.Listener", MockListener)
    monkeypatch.setattr("ml_switcheroo_compiler.backends.numpy.distributed.ipc.Client", mock_client_err)
    getattr(_state, "update")({"MockListener": MockListener, "mock_client_err": mock_client_err})
    _state.update(locals())


def _test_ipc_exchange_ipc_data_coordinator_timeout_fallback_part_3(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 3 of test_ipc_exchange_ipc_data_coordinator_timeout_fallback.

    Args:
        _state: Shared test state dictionary.
    """
    _exchange_ipc_data_coordinator, monkeypatch, time = _state["_exchange_ipc_data_coordinator"], _state["monkeypatch"], _state["time"]
    orig_time = time.time

    def mock_time() -> EagerValue:
        """Test function mock_time."""
        setattr(mock_time, "count", getattr(mock_time, "count") + 1)
        return cast("EagerValue", orig_time() + getattr(mock_time, "count") * 10)

    setattr(mock_time, "count", 0)
    monkeypatch.setattr(time, "time", mock_time)

    res = _exchange_ipc_data_coordinator(2, np.array([1]), 0.1, 0.01)
    # The timeout breaks the loop, it should still return gathered_sorted
    assert len(res) == 2
    assert cast("Callable[..., EagerTensorProtocol]", np.array_equal)(res[0], cast("Callable[..., EagerTensorProtocol]", np.array)([1]))
    assert cast("Callable[..., EagerTensorProtocol]", np.array_equal)(res[1], cast("Callable[..., EagerTensorProtocol]", np.array)([2]))
    getattr(_state, "update")({"mock_time": mock_time, "orig_time": orig_time, "res": res})


def test_ipc_exchange_ipc_data_coordinator_timeout_fallback(monkeypatch: EagerTensorProtocol) -> None:
    """Test coordinator timeout fallback loop."""
    _state: dict[str, EagerTensorProtocol] = {"monkeypatch": monkeypatch}
    _test_ipc_exchange_ipc_data_coordinator_timeout_fallback_part_1(_state)
    _test_ipc_exchange_ipc_data_coordinator_timeout_fallback_part_2(_state)
    _test_ipc_exchange_ipc_data_coordinator_timeout_fallback_part_3(_state)
    assert isinstance(_state, dict)


def _test_ipc_coordinator_sleep_part_1(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 1 of test_ipc_coordinator_sleep.

    Args:
        _state: Shared test state dictionary.
    """
    import time

    from ml_switcheroo_compiler.backends.numpy.distributed.ipc import _exchange_ipc_data_coordinator

    class MockConnection:
        """Test class MockConnection."""

        def recv(self) -> EagerValue:
            """Test function recv."""
            return (1, np.array([2]))  # rank 1, data 2

        def send(self, data: EagerValue) -> None:
            """Test function send."""
            pass

        def close(self) -> None:
            """Test function close."""
            pass

        def __enter__(self) -> EagerValue:
            """Test function __enter__."""
            return cast("EagerTensorProtocol", self)

        def __exit__(self, *args: EagerValue) -> None:
            """Test function __exit__."""
            pass

    getattr(_state, "update")({"MockConnection": MockConnection, "_exchange_ipc_data_coordinator": _exchange_ipc_data_coordinator, "time": time})
    _state.update(locals())


def _test_ipc_coordinator_sleep_part_2(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 2 of test_ipc_coordinator_sleep.

    Args:
        _state: Shared test state dictionary.
    """
    MockConnection, monkeypatch = _state["MockConnection"], _state["monkeypatch"]

    class MockListener:
        """Test class MockListener."""

        def __init__(self, *args: EagerValue, **kwargs: EagerValue) -> None:
            """Test function __init__."""
            pass

        def accept(self) -> EagerValue:
            """Test function accept."""
            return MockConnection()

        def close(self) -> None:
            """Test function close."""
            pass

        def __enter__(self) -> EagerValue:
            """Test function __enter__."""
            return cast("EagerTensorProtocol", self)

        def __exit__(self, *args: EagerValue) -> None:
            """Test function __exit__."""
            pass

    # Client fails first few times, then hits timeout to break loop
    def mock_client_err(*args: EagerValue, **kwargs: EagerValue) -> None:
        """Test function mock_client_err."""
        raise Exception("test connection error")

    monkeypatch.setattr("ml_switcheroo_compiler.backends.numpy.distributed.ipc.Listener", MockListener)
    monkeypatch.setattr("ml_switcheroo_compiler.backends.numpy.distributed.ipc.Client", mock_client_err)
    getattr(_state, "update")({"MockListener": MockListener, "mock_client_err": mock_client_err})
    _state.update(locals())


def _test_ipc_coordinator_sleep_part_3(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 3 of test_ipc_coordinator_sleep.

    Args:
        _state: Shared test state dictionary.
    """
    _exchange_ipc_data_coordinator, monkeypatch, time = _state["_exchange_ipc_data_coordinator"], _state["monkeypatch"], _state["time"]
    orig_time = time.time

    def mock_time() -> EagerValue:
        """Test function mock_time."""
        setattr(mock_time, "count", getattr(mock_time, "count") + 1)
        return cast("EagerValue", orig_time() + getattr(mock_time, "count") * 0.001)

    setattr(mock_time, "count", 0)
    monkeypatch.setattr(time, "time", mock_time)

    mock_sleep = MagicMock()
    monkeypatch.setattr(time, "sleep", mock_sleep)

    # timeout=0.01, retry_interval=0.001
    _exchange_ipc_data_coordinator(2, np.array([1]), 0.01, 0.001)
    assert mock_sleep.call_count > 0
    getattr(_state, "update")({"mock_sleep": mock_sleep, "mock_time": mock_time, "orig_time": orig_time})


def test_ipc_coordinator_sleep(monkeypatch: EagerTensorProtocol) -> None:
    """Test coordinator sleep branch before timeout."""
    _state: dict[str, EagerTensorProtocol] = {"monkeypatch": monkeypatch}
    _test_ipc_coordinator_sleep_part_1(_state)
    _test_ipc_coordinator_sleep_part_2(_state)
    _test_ipc_coordinator_sleep_part_3(_state)
    assert isinstance(_state, dict)


def test_ipc_all_gather_no_axis(monkeypatch: EagerTensorProtocol) -> None:
    """Test all gather without axis."""
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "2")

    mock_mesh = MagicMock()
    mock_mesh.size = 2

    monkeypatch.setattr("ml_switcheroo_compiler.backends.numpy.distributed.ipc._exchange_ipc_data", lambda *args: [np.array([1]), np.array([2])])

    # Mesh>1 triggers IPC, axis=None -> concatenates along axis=0
    res_gather = _ipc_all_gather(np.array([1]), axis=None, mesh=mock_mesh)
    assert cast("Callable[..., EagerTensorProtocol]", np.array_equal)(res_gather, cast("Callable[..., EagerTensorProtocol]", np.array)([1, 2]))
