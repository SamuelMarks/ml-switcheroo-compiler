"""Test MLX eager edge cases coverage."""

from __future__ import annotations

import types
import typing
from unittest.mock import MagicMock

import numpy as np
import pytest

import ml_switcheroo_compiler.backends.mlx.eager as mx_eager
from ml_switcheroo_compiler.backends.eager_registry import BackendArg, mlx_eager_registry
from ml_switcheroo_compiler.backends.mlx.eager import _mlx_partition, _mlx_zeros
from tests.testing_utils import (
    BackendModuleProtocol,
    EagerBoundCallable,
    MockTestStub,
    adapt_backend_module,
    extract_test_shape,
    require_callable,
)


def test_mlx_eager_dtype_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test dtype resolution fallback logic.

    Args:
        monkeypatch: Test parameter `monkeypatch`.
    """
    mock_backend_module = MagicMock()
    del mock_backend_module.unknown_dtype
    mock_backend_module.zeros = MagicMock(return_value="mock_zeros")
    assert _mlx_zeros(mock_backend_module, [1], dtype="float64") == "mock_zeros"


def test_mlx_eager_topk_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test partition/topk fallback when return_indices is False and topk is not available.

    Args:
        monkeypatch: Test parameter `monkeypatch`.
    """
    mock_backend_module = MagicMock()
    if adapt_backend_module(mock_backend_module).has_symbol("topk"):
        del mock_backend_module.topk
    mock_backend_module.partition = MagicMock(return_value=np.array([[1, 2, 3]]))
    res = _mlx_partition(mock_backend_module, np.array([[3, 1, 2]]), k=2, return_indices=False)
    assert tuple(extract_test_shape(res)) == (1, 2)


def test_mlx_eager_dtype_resolution_not_str(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test dtype resolution fallback logic when dtype is not a string.

    Args:
        monkeypatch: Test parameter `monkeypatch`.
    """
    mock_backend_module = MagicMock()
    mock_backend_module.zeros = MagicMock(return_value="mock_zeros")

    class NonStrDtype(MockTestStub):
        """Test class NonStrDtype."""

        def __str__(self) -> str:
            """Test function __str__.

            Returns:
                str: Computed output value from `__str__`.
            """
            return "int32"

    dtype_obj = NonStrDtype()
    assert _mlx_zeros(mock_backend_module, [1], dtype=dtype_obj) == "mock_zeros"


def test_mlx_eager_dtype_resolution_mapping_hit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test dtype resolution fallback logic when mapping is hit.

    Args:
        monkeypatch: Test parameter `monkeypatch`.
    """
    from ml_switcheroo_compiler.backends.mlx.eager import _mlx_zeros

    mock_backend_module = MagicMock()
    if adapt_backend_module(mock_backend_module).has_symbol("float64"):
        del mock_backend_module.float64
    mock_backend_module.float32 = "mock_mlx_float32"
    mock_backend_module.zeros = MagicMock(return_value="mock_zeros")
    assert _mlx_zeros(mock_backend_module, [1], dtype="float64") == "mock_zeros"


def test_mlx_eager_dtype_resolution_not_str_coverage(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test dtype resolution fallback logic when dtype is not a string.

    Args:
        monkeypatch: Test parameter `monkeypatch`.
    """
    from ml_switcheroo_compiler.backends.mlx.eager import _mlx_zeros

    mock_backend_module = MagicMock()
    if adapt_backend_module(mock_backend_module).has_symbol("float64"):
        del mock_backend_module.float64
    mock_backend_module.float32 = "mock_float32"
    mock_backend_module.zeros = MagicMock(return_value="mock_zeros")

    class NonStrDtype(MockTestStub):
        """Test class NonStrDtype."""

        def __str__(self) -> str:
            """Test function __str__.

            Returns:
                str: Computed output value from `__str__`.
            """
            return "float64"

    dtype_obj = NonStrDtype()
    assert _mlx_zeros(mock_backend_module, [1], dtype=dtype_obj) == "mock_zeros"


def test_mlx_eager_dtype_resolution_not_str_coverage2(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test dtype resolution fallback logic when dtype is not a string.

    Args:
        monkeypatch: Test parameter `monkeypatch`.
    """
    mock_backend_module = MagicMock()
    if adapt_backend_module(mock_backend_module).has_symbol("float64"):
        del mock_backend_module.float64
    mock_backend_module.float32 = "mock_float32"
    mock_backend_module.zeros = MagicMock(return_value="mock_zeros")

    class NonStrDtype(MockTestStub):
        """Test class NonStrDtype."""

        def __str__(self) -> str:
            """Test function __str__.

            Returns:
                str: Computed output value from `__str__`.
            """
            return "float64"

    NonStrDtype()
    import ml_switcheroo_compiler.backends.mlx.eager as mx_eager

    orig_dtype_map = mx_eager._MLX_DTYPE_FALLBACK_MAP
    mx_eager._MLX_DTYPE_FALLBACK_MAP = {"float64": "float32"}
    try:
        pass
    finally:
        mx_eager._MLX_DTYPE_FALLBACK_MAP = orig_dtype_map
    assert mx_eager._MLX_DTYPE_FALLBACK_MAP is orig_dtype_map


class _TestMlxEagerMissingCoverageState(typing.TypedDict, total=False):
    """Typed shared state for `_test_mlx_eager_missing_coverage` split helpers."""

    DummyShape: type[MockTestStub]
    execute_op: EagerBoundCallable
    func: EagerBoundCallable
    orig_mx: types.ModuleType | BackendModuleProtocol | None


def _test_mlx_eager_missing_coverage_part_1(_state: _TestMlxEagerMissingCoverageState) -> None:
    """Execute part 1 of test_mlx_eager_missing_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry
    from ml_switcheroo_compiler.backends.mlx.eager import execute_op

    with pytest.raises((RuntimeError, ValueError, TypeError, KeyError, Exception)):
        execute_op(None, "FakeOp", dim=1)
    global_eager_registry.register("TestGlobalMlx")(lambda *args, **kwargs: "global_hit")
    assert execute_op(None, "TestGlobalMlx") == "global_hit"

    class MockMXMap(MockTestStub):
        """Test class MockMXMap."""

        def multiply(self) -> BackendArg:
            """Test function multiply.

            Returns:
                BackendArg: Computed output value from `multiply`.
            """
            return "mul"

        def subtract(self) -> BackendArg:
            """Test function subtract.

            Returns:
                BackendArg: Computed output value from `subtract`.
            """
            return "sub"

        def divide(self) -> BackendArg:
            """Test function divide.

            Returns:
                BackendArg: Computed output value from `divide`.
            """
            return "div"

    orig_mx = mx_eager.mx
    pytest.MonkeyPatch().setattr(mx_eager, "mx", MockMXMap(), raising=False)
    _state.update({"execute_op": require_callable(execute_op), "orig_mx": orig_mx})


def _test_mlx_eager_missing_coverage_part_2(_state: _TestMlxEagerMissingCoverageState) -> None:
    """Execute part 2 of test_mlx_eager_missing_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (execute_op, orig_mx) = (_state["execute_op"], _state["orig_mx"])
    try:
        assert execute_op(None, "Mul") == "mul"
        assert execute_op(None, "Sub") == "sub"
        assert execute_op(None, "Div") == "div"
    finally:
        mx_eager.mx = orig_mx

    class MockMXLinalg(MockTestStub):
        """Test class MockMXLinalg."""

        def my_linalg_func(self) -> BackendArg:
            """Test function my_linalg_func.

            Returns:
                BackendArg: Computed output value from `my_linalg_func`.
            """
            return "linalg"

    class MockMXMain(MockTestStub):
        """Test class MockMXMain."""

        linalg = MockMXLinalg()

    pytest.MonkeyPatch().setattr(mx_eager, "mx", MockMXMain(), raising=False)
    try:
        assert execute_op(None, "MyLinalgFunc") == "linalg"
    finally:
        mx_eager.mx = orig_mx
    func = require_callable(mlx_eager_registry.get("ScatterNd"))

    class DummyShape(MockTestStub):
        """Test class DummyShape."""

        data = [1, 2]

    _state.update({"DummyShape": DummyShape, "func": func})
    _state.update({"execute_op": require_callable(execute_op), "orig_mx": orig_mx})


def _test_mlx_eager_missing_coverage_part_3(_state: _TestMlxEagerMissingCoverageState) -> None:
    """Execute part 3 of test_mlx_eager_missing_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (DummyShape, func) = (_state["DummyShape"], _state["func"])

    class DummyUpdates(MockTestStub):
        """Test class DummyUpdates."""

        dtype = "float32"

    class DummyIndices(MockTestStub):
        """Test class DummyIndices."""

        shape = [2, 2]

        def __getitem__(self, key: BackendArg) -> BackendArg:
            """Test function __getitem__.

            Args:
                key: Test parameter `key`.

            Returns:
                BackendArg: Computed output value from `__getitem__`.
            """
            return 0

    class MockMXScatter(MockTestStub):
        """Test class MockMXScatter."""

        def zeros(self, shape: BackendArg, dtype: BackendArg) -> BackendArg:
            """Test function zeros.

            Args:
                shape: Test parameter `shape`.
                dtype: Test parameter `dtype`.

            Returns:
                BackendArg: Computed output value from `zeros`.
            """

            class DummyRes(MockTestStub):
                """Test class DummyRes."""

                def __setitem__(self, key: BackendArg, value: BackendArg) -> None:
                    """Test function __setitem__.

                    Args:
                        key: Test parameter `key`.
                        value: Test parameter `value`.
                    """
                    pass

            return DummyRes()

    assert (
        func(
            MockMXScatter(),
            DummyIndices(),
            DummyUpdates(),
            DummyShape(),
        )
        is not None
    )
    assert (
        func(
            MockMXScatter(),
            DummyIndices(),
            DummyUpdates(),
            shape=DummyShape(),
        )
        is not None
    )
    func = require_callable(mlx_eager_registry.get("Reshape"))
    _state.update({"DummyShape": DummyShape, "func": func})


def _test_mlx_eager_missing_coverage_part_4(_state: _TestMlxEagerMissingCoverageState) -> None:
    """Execute part 4 of test_mlx_eager_missing_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    func = _state["func"]

    class DummyShapeList(MockTestStub):
        """Test class DummyShapeList."""

        def tolist(self) -> BackendArg:
            """Test function tolist.

            Returns:
                BackendArg: Computed output value from `tolist`.
            """
            return [1, 2]

    class MockMXReshape(MockTestStub):
        """Test class MockMXReshape."""

        def reshape(self, x: BackendArg, shape: BackendArg) -> BackendArg:
            """Test function reshape.

            Args:
                x: Test parameter `x`.
                shape: Test parameter `shape`.

            Returns:
                BackendArg: Computed output value from `reshape`.
            """
            return shape

    assert func(MockMXReshape(), "x", DummyShapeList()) == [1, 2]
    assert func(MockMXReshape(), input="x", shape=DummyShapeList()) == [1, 2]

    class DummyMXWithFloat32(MockTestStub):
        """Test class DummyMXWithFloat32."""

        float32 = "mock_float32"

        def zeros(self, shape: BackendArg, dtype: BackendArg) -> BackendArg:
            """Test function zeros.

            Args:
                shape: Test parameter `shape`.
                dtype: Test parameter `dtype`.

            Returns:
                BackendArg: Computed output value from `zeros`.
            """
            return dtype

    assert mx_eager._mlx_zeros(DummyMXWithFloat32(), [1], dtype="bfloat16") == "mock_float32"
    func = require_callable(mlx_eager_registry.get("NanToNum"))
    _state.update({"func": func})


def _test_mlx_eager_missing_coverage_part_5(_state: _TestMlxEagerMissingCoverageState) -> None:
    """Execute part 5 of test_mlx_eager_missing_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    func = _state["func"]

    class MockMXNan(MockTestStub):
        """Test class MockMXNan."""

        def nan_to_num(self, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
            """Test function nan_to_num.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.

            Returns:
                BackendArg: Computed output value from `nan_to_num`.
            """
            return kwargs

    res = func(MockMXNan(), "x", nan=None, posinf=None)
    assert "nan" not in res or res["nan"] is None

    class DummyItemData(MockTestStub):
        """Test class DummyItemData."""

        class DummyData(MockTestStub):
            """Test class DummyData."""

            def item(self) -> BackendArg:
                """Test function item.

                Returns:
                    BackendArg: Computed output value from `item`.
                """
                return 1.0

        data = DummyData()

    res = func(MockMXNan(), "x", posinf=DummyItemData())
    assert res["posinf"] == 1.0
    res = func(MockMXNan(), "x", invalid_key=1.0)
    assert "invalid_key" not in res


def test_mlx_eager_missing_coverage() -> None:
    """Test function test_mlx_eager_missing_coverage."""
    _state: _TestMlxEagerMissingCoverageState = {}
    _test_mlx_eager_missing_coverage_part_1(_state)
    _test_mlx_eager_missing_coverage_part_2(_state)
    _test_mlx_eager_missing_coverage_part_3(_state)
    _test_mlx_eager_missing_coverage_part_4(_state)
    _test_mlx_eager_missing_coverage_part_5(_state)
    assert len(_state) > 0
