"""Test Numpy distributed ipc coverage."""

from __future__ import annotations

import types
import typing
from unittest.mock import MagicMock

import numpy as np
import pytest

from ml_switcheroo_compiler.backends.numpy.distributed.ipc import (
    _exchange_ipc_data,
    _exchange_ipc_data_coordinator,
    _exchange_ipc_data_worker,
    _ipc_all_gather,
    _ipc_all_reduce,
    _ipc_reduce_scatter,
)
from ml_switcheroo_compiler.backends.numpy.eager import ensure_ndarray
from tests.testing_utils import EagerBoundCallable, MockTestStub, require_callable

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import BackendArg


class _TestIpcCoordinatorState(typing.TypedDict, total=False):
    """Typed shared state for `_test_ipc_coordinator` split helpers."""

    MockConnection: type[MockTestStub]
    monkeypatch: pytest.MonkeyPatch


def _test_ipc_coordinator_part_1(_state: _TestIpcCoordinatorState) -> None:
    """Execute part 1 of test_ipc_coordinator.

    Args:
        _state: Shared test state dictionary.
    """

    class MockConnection(MockTestStub):
        """Test class MockConnection."""

        def recv(self) -> BackendArg:
            """Test function recv.

            Returns:
                BackendArg: Computed output value from `recv`.
            """
            return (1, np.array([2]))  # rank 1, data 2

        def send(self, data: BackendArg) -> None:
            """Test function send.

            Args:
                data: Test parameter `data`.
            """
            pass

        def close(self) -> None:
            """Test function close."""
            pass

        def __enter__(self) -> MockConnection:
            """Test function __enter__.

            Returns:
                MockConnection: Computed output value from `__enter__`.
            """
            return self

        def __exit__(self, *args: BackendArg) -> None:
            """Test function __exit__.

            Args:
                args: Test parameter `args`.
            """
            pass

    _state.update({"MockConnection": MockConnection})


def _test_ipc_coordinator_part_2(_state: _TestIpcCoordinatorState) -> None:
    """Execute part 2 of test_ipc_coordinator.

    Args:
        _state: Shared test state dictionary.
    """
    MockConnection, monkeypatch = _state["MockConnection"], _state["monkeypatch"]

    class MockListener(MockTestStub):
        """Test class MockListener."""

        def __init__(self, *args: BackendArg, **kwargs: BackendArg) -> None:
            """Test function __init__.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.
            """
            del args, kwargs
            super().__init__()

        def accept(self) -> MockTestStub:
            """Test function accept.

            Returns:
                MockTestStub: Computed output value from `accept`.
            """
            return MockConnection()

        def close(self) -> None:
            """Test function close."""
            pass

        def __enter__(self) -> MockListener:
            """Test function __enter__.

            Returns:
                MockListener: Computed output value from `__enter__`.
            """
            return self

        def __exit__(self, *args: BackendArg) -> None:
            """Test function __exit__.

            Args:
                args: Test parameter `args`.
            """
            pass

    def mock_client(*args: BackendArg, **kwargs: BackendArg) -> MockTestStub:
        """Test function mock_client.

        Args:
            args: Test parameter `args`.
            kwargs: Test parameter `kwargs`.

        Returns:
            MockTestStub: Computed output value from `mock_client`.
        """
        del args, kwargs
        return MockConnection()

    monkeypatch.setattr("ml_switcheroo_compiler.backends.numpy.distributed.ipc.Listener", MockListener)
    monkeypatch.setattr("ml_switcheroo_compiler.backends.numpy.distributed.ipc.Client", mock_client)
    _state.update({"MockConnection": MockConnection, "monkeypatch": monkeypatch})


def _test_ipc_coordinator_part_3(_state: _TestIpcCoordinatorState) -> None:
    """Execute part 3 of test_ipc_coordinator.

    Args:
        _state: Shared test state dictionary.
    """
    monkeypatch = _state["monkeypatch"]
    res = _exchange_ipc_data_coordinator(2, np.array([1]), 0.1, 0.01)
    assert len(res) == 2
    assert np.array_equal(ensure_ndarray(res[0]), np.array([1]))
    assert np.array_equal(ensure_ndarray(res[1]), np.array([2]))

    # Exception branch
    class ErrorListener(MockTestStub):
        """Test class ErrorListener."""

        def __init__(self, *args: BackendArg, **kwargs: BackendArg) -> None:
            """Test function __init__.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.
            """
            del args, kwargs
            super().__init__()

        def __enter__(self) -> None:
            """Test function __enter__.

            Raises:
                Exception: When raised during `__enter__` (test error).
            """
            raise RuntimeError("test error")

        def __exit__(self, *args: BackendArg) -> None:
            """Test function __exit__.

            Args:
                args: Test parameter `args`.
            """
            pass

    monkeypatch.setattr("ml_switcheroo_compiler.backends.numpy.distributed.ipc.Listener", ErrorListener)
    res_err = _exchange_ipc_data_coordinator(2, np.array([1]), 0.1, 0.01)
    assert len(res_err) == 2
    assert np.array_equal(ensure_ndarray(res_err[0]), np.array([1]))
    assert np.array_equal(ensure_ndarray(res_err[1]), np.array([1]))


def test_ipc_coordinator(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test IPC coordinator.

    Args:
        monkeypatch: Test parameter `monkeypatch`.
    """
    _state: _TestIpcCoordinatorState = {"monkeypatch": monkeypatch}
    _test_ipc_coordinator_part_1(_state)
    _test_ipc_coordinator_part_2(_state)
    _test_ipc_coordinator_part_3(_state)
    assert len(_state) > 0


class _TestIpcWorkerState(typing.TypedDict, total=False):
    """Typed shared state for `_test_ipc_worker` split helpers."""

    MockConnection: type[MockTestStub]
    monkeypatch: pytest.MonkeyPatch
    time: types.ModuleType


def _test_ipc_worker_part_1(_state: _TestIpcWorkerState) -> None:
    """Execute part 1 of test_ipc_worker.

    Args:
        _state: Shared test state dictionary.
    """

    class MockConnection(MockTestStub):
        """Test class MockConnection."""

        def recv(self) -> BackendArg:
            """Test function recv.

            Returns:
                BackendArg: Computed output value from `recv`.
            """
            return [np.array([1]), np.array([2])]

        def send(self, data: BackendArg) -> None:
            """Test function send.

            Args:
                data: Test parameter `data`.
            """
            pass

        def close(self) -> None:
            """Test function close."""
            pass

        def __enter__(self) -> MockConnection:
            """Test function __enter__.

            Returns:
                MockConnection: Computed output value from `__enter__`.
            """
            return self

        def __exit__(self, *args: BackendArg) -> None:
            """Test function __exit__.

            Args:
                args: Test parameter `args`.
            """
            pass

    _state.update({"MockConnection": MockConnection})


def _test_ipc_worker_part_2(_state: _TestIpcWorkerState) -> None:
    """Execute part 2 of test_ipc_worker.

    Args:
        _state: Shared test state dictionary.
    """
    MockConnection, monkeypatch = _state["MockConnection"], _state["monkeypatch"]

    class MockListener(MockTestStub):
        """Test class MockListener."""

        def __init__(self, *args: BackendArg, **kwargs: BackendArg) -> None:
            """Test function __init__.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.
            """
            del args, kwargs
            super().__init__()

        def accept(self) -> MockTestStub:
            """Test function accept.

            Returns:
                MockTestStub: Computed output value from `accept`.
            """
            return MockConnection()

        def close(self) -> None:
            """Test function close."""
            pass

        def __enter__(self) -> MockListener:
            """Test function __enter__.

            Returns:
                MockListener: Computed output value from `__enter__`.
            """
            return self

        def __exit__(self, *args: BackendArg) -> None:
            """Test function __exit__.

            Args:
                args: Test parameter `args`.
            """
            pass

    def mock_client(*args: BackendArg, **kwargs: BackendArg) -> MockTestStub:
        """Test function mock_client.

        Args:
            args: Test parameter `args`.
            kwargs: Test parameter `kwargs`.

        Returns:
            MockTestStub: Computed output value from `mock_client`.
        """
        del args, kwargs
        return MockConnection()

    monkeypatch.setattr("ml_switcheroo_compiler.backends.numpy.distributed.ipc.Listener", MockListener)
    monkeypatch.setattr("ml_switcheroo_compiler.backends.numpy.distributed.ipc.Client", mock_client)
    _state.update({"MockConnection": MockConnection, "monkeypatch": monkeypatch})


def _test_ipc_worker_part_3(_state: _TestIpcWorkerState) -> None:
    """Execute part 3 of test_ipc_worker.

    Args:
        _state: Shared test state dictionary.
    """
    monkeypatch = _state["monkeypatch"]
    res = _exchange_ipc_data_worker(1, 2, np.array([2]), 0.1, 0.01)
    assert len(res) == 2
    assert np.array_equal(ensure_ndarray(res[0]), np.array([1]))

    def mock_client_err(*args: BackendArg, **kwargs: BackendArg) -> None:
        """Test function mock_client_err.

        Args:
            args: Test parameter `args`.
            kwargs: Test parameter `kwargs`.

        Raises:
            Exception: When raised during `mock_client_err` (test connection error).
        """
        raise RuntimeError("test connection error")

    import time

    orig_time = time.time
    count = 0

    # fast forward time after one retry to hit both retry sleep and timeout
    def mock_time() -> float:
        """Test function mock_time.

        Returns:
            float: Computed output value from `mock_time`.
        """
        nonlocal count
        count += 1
        offset = 0.0 if count <= 2 else 10.0
        return orig_time() + offset

    monkeypatch.setattr(time, "time", mock_time)
    monkeypatch.setattr("ml_switcheroo_compiler.backends.numpy.distributed.ipc.Client", mock_client_err)

    res_err = _exchange_ipc_data_worker(1, 2, np.array([2]), 0.1, 0.01)
    assert len(res_err) == 2
    assert np.array_equal(ensure_ndarray(res_err[0]), np.array([2]))
    _state.update({"time": time})


def test_ipc_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test IPC worker.

    Args:
        monkeypatch: Test parameter `monkeypatch`.
    """
    _state: _TestIpcWorkerState = {"monkeypatch": monkeypatch}
    _test_ipc_worker_part_1(_state)
    _test_ipc_worker_part_2(_state)
    _test_ipc_worker_part_3(_state)
    assert len(_state) > 0


def test_ipc_exchange_ipc_data(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test standard exchange entrypoint.

    Args:
        monkeypatch: Test parameter `monkeypatch`.
    """
    monkeypatch.setattr(
        "ml_switcheroo_compiler.backends.numpy.distributed.ipc._exchange_ipc_data_coordinator",
        lambda *args: ["mock_coord"],
    )
    monkeypatch.setattr(
        "ml_switcheroo_compiler.backends.numpy.distributed.ipc._exchange_ipc_data_worker", lambda *args: ["mock_worker"]
    )

    assert _exchange_ipc_data(0, 2, np.array([1])) == ["mock_coord"]
    assert _exchange_ipc_data(1, 2, np.array([2])) == ["mock_worker"]


def test_ipc_collectives(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test ipc collectives fallback logic.

    Args:
        monkeypatch: Test parameter `monkeypatch`.
    """
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "2")

    mock_mesh = MagicMock()
    mock_mesh.size = 2

    # We mock IPC to return two 1D arrays
    monkeypatch.setattr(
        "ml_switcheroo_compiler.backends.numpy.distributed.ipc._exchange_ipc_data",
        lambda *args: [np.array([1, 2]), np.array([3, 4])],
    )

    # sum op
    res_reduce_scatter_sum = _ipc_reduce_scatter(np.array([1, 2]), "sum", axis=0, mesh=mock_mesh)
    assert np.array_equal(ensure_ndarray(res_reduce_scatter_sum), np.array([4]))

    # max op
    res_reduce_scatter_max = _ipc_reduce_scatter(np.array([1, 2]), "max", axis=0, mesh=mock_mesh)
    assert np.array_equal(ensure_ndarray(res_reduce_scatter_max), np.array([3]))

    # min op
    res_reduce_scatter_min = _ipc_reduce_scatter(np.array([1, 2]), "min", axis=0, mesh=mock_mesh)
    assert np.array_equal(ensure_ndarray(res_reduce_scatter_min), np.array([1]))

    # fallback op
    res_reduce_scatter_fallback = _ipc_reduce_scatter(np.array([1, 2]), "unknown", axis=0, mesh=mock_mesh)
    assert np.array_equal(ensure_ndarray(res_reduce_scatter_fallback), np.array([4]))

    # sum op
    res_all_reduce_sum = _ipc_all_reduce(np.array([1, 2]), "sum", mesh=mock_mesh)
    assert np.array_equal(ensure_ndarray(res_all_reduce_sum), np.array([4, 6]))

    # max op
    res_all_reduce_max = _ipc_all_reduce(np.array([1, 2]), "max", mesh=mock_mesh)
    assert np.array_equal(ensure_ndarray(res_all_reduce_max), np.array([3, 4]))

    # min op
    res_all_reduce_min = _ipc_all_reduce(np.array([1, 2]), "min", mesh=mock_mesh)
    assert np.array_equal(ensure_ndarray(res_all_reduce_min), np.array([1, 2]))

    # fallback op
    res_all_reduce_fallback = _ipc_all_reduce(np.array([1, 2]), "unknown", mesh=mock_mesh)
    assert np.array_equal(ensure_ndarray(res_all_reduce_fallback), np.array([4, 6]))

    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)


def test_ipc_collectives_no_mesh() -> None:
    """Test ipc collectives when mesh is None (line 127)."""
    import numpy as np

    from ml_switcheroo_compiler.backends.numpy.distributed.ipc import _ipc_all_reduce, _ipc_reduce_scatter

    t = np.array([1, 2])

    res_gather = _ipc_all_gather(t, axis=0, mesh=None)
    assert np.array_equal(ensure_ndarray(res_gather), np.expand_dims(t, axis=0))

    res_gather_no_axis = _ipc_all_gather(t, axis=None, mesh=None)
    assert np.array_equal(ensure_ndarray(res_gather_no_axis), t)

    res_reduce_scatter = _ipc_reduce_scatter(t, "sum", 0, mesh=None)
    assert np.array_equal(ensure_ndarray(res_reduce_scatter), t)

    # string fallback
    assert _ipc_all_gather("string_tensor", 0, mesh=None) == "string_tensor"
    assert _ipc_reduce_scatter("string_tensor", "sum", 0, mesh=None) == "string_tensor"
    assert _ipc_all_reduce("string_tensor", "sum", mesh=None) == "string_tensor"

    res_all_reduce = _ipc_all_reduce(t, "sum", mesh=None)
    assert np.array_equal(ensure_ndarray(res_all_reduce), t)


class _TestIpcExchangeIpcDataCoordinatorTimeoutFallbackState(typing.TypedDict, total=False):
    """Typed shared state for `_test_ipc_exchange_ipc_data_coordinator_timeout_fallback` split helpers."""

    MockConnection: type[MockTestStub]
    _exchange_ipc_data_coordinator: EagerBoundCallable
    monkeypatch: pytest.MonkeyPatch
    time: types.ModuleType


def _test_ipc_exchange_ipc_data_coordinator_timeout_fallback_part_1(
    _state: _TestIpcExchangeIpcDataCoordinatorTimeoutFallbackState,
) -> None:
    """Execute part 1 of test_ipc_exchange_ipc_data_coordinator_timeout_fallback.

    Args:
        _state: Shared test state dictionary.
    """
    import time

    from ml_switcheroo_compiler.backends.numpy.distributed.ipc import _exchange_ipc_data_coordinator

    class MockConnection(MockTestStub):
        """Test class MockConnection."""

        def recv(self) -> BackendArg:
            """Test function recv.

            Returns:
                BackendArg: Computed output value from `recv`.
            """
            return (1, np.array([2]))  # rank 1, data 2

        def send(self, data: BackendArg) -> None:
            """Test function send.

            Args:
                data: Test parameter `data`.
            """
            pass

        def close(self) -> None:
            """Test function close."""
            pass

        def __enter__(self) -> MockConnection:
            """Test function __enter__.

            Returns:
                MockConnection: Computed output value from `__enter__`.
            """
            return self

        def __exit__(self, *args: BackendArg) -> None:
            """Test function __exit__.

            Args:
                args: Test parameter `args`.
            """
            pass

    _state.update(
        {
            "MockConnection": MockConnection,
            "_exchange_ipc_data_coordinator": require_callable(_exchange_ipc_data_coordinator),
            "time": time,
        }
    )


def _test_ipc_exchange_ipc_data_coordinator_timeout_fallback_part_2(
    _state: _TestIpcExchangeIpcDataCoordinatorTimeoutFallbackState,
) -> None:
    """Execute part 2 of test_ipc_exchange_ipc_data_coordinator_timeout_fallback.

    Args:
        _state: Shared test state dictionary.
    """
    MockConnection, monkeypatch = _state["MockConnection"], _state["monkeypatch"]

    class MockListener(MockTestStub):
        """Test class MockListener."""

        def __init__(self, *args: BackendArg, **kwargs: BackendArg) -> None:
            """Test function __init__.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.
            """
            del args, kwargs
            super().__init__()

        def accept(self) -> MockTestStub:
            """Test function accept.

            Returns:
                MockTestStub: Computed output value from `accept`.
            """
            return MockConnection()

        def close(self) -> None:
            """Test function close."""
            pass

        def __enter__(self) -> MockListener:
            """Test function __enter__.

            Returns:
                MockListener: Computed output value from `__enter__`.
            """
            return self

        def __exit__(self, *args: BackendArg) -> None:
            """Test function __exit__.

            Args:
                args: Test parameter `args`.
            """
            pass

    # Ensure Client fails so we hit the timeout
    def mock_client_err(*args: BackendArg, **kwargs: BackendArg) -> None:
        """Test function mock_client_err.

        Args:
            args: Test parameter `args`.
            kwargs: Test parameter `kwargs`.

        Raises:
            Exception: When raised during `mock_client_err` (test connection error).
        """
        raise RuntimeError("test connection error")

    monkeypatch.setattr("ml_switcheroo_compiler.backends.numpy.distributed.ipc.Listener", MockListener)
    monkeypatch.setattr("ml_switcheroo_compiler.backends.numpy.distributed.ipc.Client", mock_client_err)
    _state.update({"MockConnection": MockConnection, "monkeypatch": monkeypatch})


def _test_ipc_exchange_ipc_data_coordinator_timeout_fallback_part_3(
    _state: _TestIpcExchangeIpcDataCoordinatorTimeoutFallbackState,
) -> None:
    """Execute part 3 of test_ipc_exchange_ipc_data_coordinator_timeout_fallback.

    Args:
        _state: Shared test state dictionary.
    """
    import time as std_time

    _exchange_ipc_data_coordinator, monkeypatch, time = (
        _state["_exchange_ipc_data_coordinator"],
        _state["monkeypatch"],
        _state["time"],
    )
    orig_time = std_time.time
    count = 0

    def mock_time() -> float:
        """Test function mock_time.

        Returns:
            float: Computed output value from `mock_time`.
        """
        nonlocal count
        count += 1
        return orig_time() + count * 10.0

    monkeypatch.setattr(time, "time", mock_time)

    res = _exchange_ipc_data_coordinator(2, np.array([1]), 0.1, 0.01)
    # The timeout breaks the loop, it should still return gathered_sorted
    assert len(res) == 2
    assert np.array_equal(res[0], np.array([1]))
    assert np.array_equal(res[1], np.array([2]))


def test_ipc_exchange_ipc_data_coordinator_timeout_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test coordinator timeout fallback loop.

    Args:
        monkeypatch: Test parameter `monkeypatch`.
    """
    _state: _TestIpcExchangeIpcDataCoordinatorTimeoutFallbackState = {"monkeypatch": monkeypatch}
    _test_ipc_exchange_ipc_data_coordinator_timeout_fallback_part_1(_state)
    _test_ipc_exchange_ipc_data_coordinator_timeout_fallback_part_2(_state)
    _test_ipc_exchange_ipc_data_coordinator_timeout_fallback_part_3(_state)
    assert len(_state) > 0


class _TestIpcCoordinatorSleepState(typing.TypedDict, total=False):
    """Typed shared state for `_test_ipc_coordinator_sleep` split helpers."""

    MockConnection: type[MockTestStub]
    _exchange_ipc_data_coordinator: EagerBoundCallable
    monkeypatch: pytest.MonkeyPatch
    time: types.ModuleType


def _test_ipc_coordinator_sleep_part_1(_state: _TestIpcCoordinatorSleepState) -> None:
    """Execute part 1 of test_ipc_coordinator_sleep.

    Args:
        _state: Shared test state dictionary.
    """
    import time

    from ml_switcheroo_compiler.backends.numpy.distributed.ipc import _exchange_ipc_data_coordinator

    class MockConnection(MockTestStub):
        """Test class MockConnection."""

        def recv(self) -> BackendArg:
            """Test function recv.

            Returns:
                BackendArg: Computed output value from `recv`.
            """
            return (1, np.array([2]))  # rank 1, data 2

        def send(self, data: BackendArg) -> None:
            """Test function send.

            Args:
                data: Test parameter `data`.
            """
            pass

        def close(self) -> None:
            """Test function close."""
            pass

        def __enter__(self) -> MockConnection:
            """Test function __enter__.

            Returns:
                MockConnection: Computed output value from `__enter__`.
            """
            return self

        def __exit__(self, *args: BackendArg) -> None:
            """Test function __exit__.

            Args:
                args: Test parameter `args`.
            """
            pass

    _state.update(
        {
            "MockConnection": MockConnection,
            "_exchange_ipc_data_coordinator": require_callable(_exchange_ipc_data_coordinator),
            "time": time,
        }
    )


def _test_ipc_coordinator_sleep_part_2(_state: _TestIpcCoordinatorSleepState) -> None:
    """Execute part 2 of test_ipc_coordinator_sleep.

    Args:
        _state: Shared test state dictionary.
    """
    MockConnection, monkeypatch = _state["MockConnection"], _state["monkeypatch"]

    class MockListener(MockTestStub):
        """Test class MockListener."""

        def __init__(self, *args: BackendArg, **kwargs: BackendArg) -> None:
            """Test function __init__.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.
            """
            del args, kwargs
            super().__init__()

        def accept(self) -> MockTestStub:
            """Test function accept.

            Returns:
                MockTestStub: Computed output value from `accept`.
            """
            return MockConnection()

        def close(self) -> None:
            """Test function close."""
            pass

        def __enter__(self) -> MockListener:
            """Test function __enter__.

            Returns:
                MockListener: Computed output value from `__enter__`.
            """
            return self

        def __exit__(self, *args: BackendArg) -> None:
            """Test function __exit__.

            Args:
                args: Test parameter `args`.
            """
            pass

    # Client fails first few times, then hits timeout to break loop
    def mock_client_err(*args: BackendArg, **kwargs: BackendArg) -> None:
        """Test function mock_client_err.

        Args:
            args: Test parameter `args`.
            kwargs: Test parameter `kwargs`.

        Raises:
            Exception: When raised during `mock_client_err` (test connection error).
        """
        raise RuntimeError("test connection error")

    monkeypatch.setattr("ml_switcheroo_compiler.backends.numpy.distributed.ipc.Listener", MockListener)
    monkeypatch.setattr("ml_switcheroo_compiler.backends.numpy.distributed.ipc.Client", mock_client_err)
    _state.update({"MockConnection": MockConnection, "monkeypatch": monkeypatch})


def _test_ipc_coordinator_sleep_part_3(_state: _TestIpcCoordinatorSleepState) -> None:
    """Execute part 3 of test_ipc_coordinator_sleep.

    Args:
        _state: Shared test state dictionary.
    """
    import time as std_time

    _exchange_ipc_data_coordinator, monkeypatch, time = (
        _state["_exchange_ipc_data_coordinator"],
        _state["monkeypatch"],
        _state["time"],
    )
    orig_time = std_time.time
    count = 0

    def mock_time() -> float:
        """Test function mock_time.

        Returns:
            float: Computed output value from `mock_time`.
        """
        nonlocal count
        count += 1
        return orig_time() + count * 0.001

    monkeypatch.setattr(time, "time", mock_time)

    mock_sleep = MagicMock()
    monkeypatch.setattr(time, "sleep", mock_sleep)

    # timeout=0.01, retry_interval=0.001
    _exchange_ipc_data_coordinator(2, np.array([1]), 0.01, 0.001)
    assert mock_sleep.call_count > 0


def test_ipc_coordinator_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test coordinator sleep branch before timeout.

    Args:
        monkeypatch: Test parameter `monkeypatch`.
    """
    _state: _TestIpcCoordinatorSleepState = {"monkeypatch": monkeypatch}
    _test_ipc_coordinator_sleep_part_1(_state)
    _test_ipc_coordinator_sleep_part_2(_state)
    _test_ipc_coordinator_sleep_part_3(_state)
    assert len(_state) > 0


def test_ipc_all_gather_no_axis(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test all gather without axis.

    Args:
        monkeypatch: Test parameter `monkeypatch`.
    """
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "2")

    mock_mesh = MagicMock()
    mock_mesh.size = 2

    monkeypatch.setattr(
        "ml_switcheroo_compiler.backends.numpy.distributed.ipc._exchange_ipc_data",
        lambda *args: [np.array([1]), np.array([2])],
    )

    # Mesh>1 triggers IPC, axis=None -> concatenates along axis=0
    res_gather = _ipc_all_gather(np.array([1]), axis=None, mesh=mock_mesh)
    assert np.array_equal(ensure_ndarray(res_gather), np.array([1, 2]))
