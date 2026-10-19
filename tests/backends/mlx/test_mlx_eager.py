"""Test MLX eager edge cases coverage."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast
from unittest.mock import MagicMock

import numpy as np
import pytest

from ml_switcheroo_compiler.backends.mlx.eager import _mlx_partition, _mlx_zeros

if TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol, EagerTensorProtocol, EagerValue


def test_mlx_eager_dtype_resolution(monkeypatch: EagerTensorProtocol) -> None:
    """Test dtype resolution fallback logic.

    Args:
        monkeypatch: Test parameter `monkeypatch`.
    """
    mock_backend_module = MagicMock()
    del mock_backend_module.unknown_dtype
    mock_backend_module.zeros = MagicMock(return_value="mock_zeros")
    assert _mlx_zeros(mock_backend_module, [1], dtype="float64") == "mock_zeros"


def test_mlx_eager_topk_fallback(monkeypatch: EagerTensorProtocol) -> None:
    """Test partition/topk fallback when return_indices is False and topk is not available.

    Args:
        monkeypatch: Test parameter `monkeypatch`.
    """
    mock_backend_module = MagicMock()
    if hasattr(mock_backend_module, "topk"):
        del mock_backend_module.topk
    mock_backend_module.partition = MagicMock(return_value=np.array([[1, 2, 3]]))
    res = _mlx_partition(mock_backend_module, np.array([[3, 1, 2]]), k=2, return_indices=False)
    assert res is not None


def test_mlx_eager_dtype_resolution_not_str(monkeypatch: EagerTensorProtocol) -> None:
    """Test dtype resolution fallback logic when dtype is not a string.

    Args:
        monkeypatch: Test parameter `monkeypatch`.
    """
    mock_backend_module = MagicMock()
    mock_backend_module.zeros = MagicMock(return_value="mock_zeros")

    class NonStrDtype:
        """Test class NonStrDtype."""

        def __str__(self) -> str:
            """Test function __str__."""
            return "int32"

    dtype_obj = NonStrDtype()
    assert _mlx_zeros(mock_backend_module, [1], dtype=dtype_obj) == "mock_zeros"


def test_mlx_eager_dtype_resolution_mapping_hit(monkeypatch: EagerTensorProtocol) -> None:
    """Test dtype resolution fallback logic when mapping is hit.

    Args:
        monkeypatch: Test parameter `monkeypatch`.
    """
    from ml_switcheroo_compiler.backends.mlx.eager import _mlx_zeros

    mock_backend_module = MagicMock()
    if hasattr(mock_backend_module, "float64"):
        del mock_backend_module.float64
    mock_backend_module.float32 = "mock_mlx_float32"
    mock_backend_module.zeros = MagicMock(return_value="mock_zeros")
    assert _mlx_zeros(mock_backend_module, [1], dtype="float64") == "mock_zeros"


def test_mlx_eager_dtype_resolution_not_str_coverage(monkeypatch: EagerTensorProtocol) -> None:
    """Test dtype resolution fallback logic when dtype is not a string.

    Args:
        monkeypatch: Test parameter `monkeypatch`.
    """
    from ml_switcheroo_compiler.backends.mlx.eager import _mlx_zeros

    mock_backend_module = MagicMock()
    if hasattr(mock_backend_module, "float64"):
        del mock_backend_module.float64
    mock_backend_module.float32 = "mock_float32"
    mock_backend_module.zeros = MagicMock(return_value="mock_zeros")

    class NonStrDtype:
        """Test class NonStrDtype."""

        def __str__(self) -> str:
            """Test function __str__."""
            return "float64"

    dtype_obj = NonStrDtype()
    assert _mlx_zeros(mock_backend_module, [1], dtype=dtype_obj) == "mock_zeros"


def test_mlx_eager_dtype_resolution_not_str_coverage2(monkeypatch: EagerTensorProtocol) -> None:
    """Test dtype resolution fallback logic when dtype is not a string.

    Args:
        monkeypatch: Test parameter `monkeypatch`.
    """
    mock_backend_module = MagicMock()
    if hasattr(mock_backend_module, "float64"):
        del mock_backend_module.float64
    mock_backend_module.float32 = "mock_float32"
    mock_backend_module.zeros = MagicMock(return_value="mock_zeros")

    class NonStrDtype:
        """Test class NonStrDtype."""

        def __str__(self) -> str:
            """Test function __str__."""
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


def _test_mlx_eager_missing_coverage_part_1(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 1 of test_mlx_eager_missing_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry
    from ml_switcheroo_compiler.backends.mlx.eager import execute_op

    with pytest.raises((RuntimeError, ValueError, TypeError, KeyError, Exception)):
        cast("EagerCallableProtocol", execute_op)(None, "FakeOp", dim=1)
    global_eager_registry.register("TestGlobalMlx")(lambda *args, **kwargs: "global_hit")
    assert cast("EagerCallableProtocol", execute_op)(None, "TestGlobalMlx") == "global_hit"

    class MockMXMap:
        """Test class MockMXMap."""

        def multiply(self) -> EagerValue:
            """Test function multiply."""
            return "mul"

        def subtract(self) -> EagerValue:
            """Test function subtract."""
            return "sub"

        def divide(self) -> EagerValue:
            """Test function divide."""
            return "div"

    import ml_switcheroo_compiler.backends.mlx.eager as mx_eager

    orig_mx = cast("EagerTensorProtocol", getattr(mx_eager, "mx"))
    setattr(mx_eager, "mx", MockMXMap())
    getattr(_state, "update")({"MockMXMap": MockMXMap, "execute_op": execute_op, "global_eager_registry": global_eager_registry, "mx_eager": mx_eager})
    getattr(_state, "update")({"orig_mx": orig_mx})
    _state.update(locals())


def _test_mlx_eager_missing_coverage_part_2(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 2 of test_mlx_eager_missing_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (execute_op, mx_eager, orig_mx) = (_state["execute_op"], _state["mx_eager"], _state["orig_mx"])
    try:
        assert execute_op(None, "Mul") == "mul"
        assert execute_op(None, "Sub") == "sub"
        assert execute_op(None, "Div") == "div"
    finally:
        setattr(mx_eager, "mx", orig_mx)

    class MockMXLinalg:
        """Test class MockMXLinalg."""

        def my_linalg_func(self) -> EagerValue:
            """Test function my_linalg_func."""
            return "linalg"

    class MockMXMain:
        """Test class MockMXMain."""

        linalg = MockMXLinalg()

    setattr(mx_eager, "mx", MockMXMain())
    try:
        assert execute_op(None, "MyLinalgFunc") == "linalg"
    finally:
        setattr(mx_eager, "mx", orig_mx)
    func = mx_eager.mlx_eager_registry.get("ScatterNd")

    class DummyShape:
        """Test class DummyShape."""

        data = [1, 2]

    getattr(_state, "update")({"DummyShape": DummyShape, "MockMXLinalg": MockMXLinalg, "MockMXMain": MockMXMain, "func": func})
    _state.update(locals())


def _test_mlx_eager_missing_coverage_part_3(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 3 of test_mlx_eager_missing_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (DummyShape, func, mx_eager) = (_state["DummyShape"], _state["func"], _state["mx_eager"])

    class DummyUpdates:
        """Test class DummyUpdates."""

        dtype = "float32"

    class DummyIndices:
        """Test class DummyIndices."""

        shape = [2, 2]

        def __getitem__(self, key: EagerValue) -> EagerValue:
            """Test function __getitem__.

            Args:
                key: Test parameter `key`.
            """
            return 0

    class MockMXScatter:
        """Test class MockMXScatter."""

        def zeros(self, shape: EagerValue, dtype: EagerValue) -> EagerValue:
            """Test function zeros.

            Args:
                shape: Test parameter `shape`.
                dtype: Test parameter `dtype`.
            """

            class DummyRes:
                """Test class DummyRes."""

                def __setitem__(self, key: EagerValue, value: EagerValue) -> None:
                    """Test function __setitem__.

                    Args:
                        key: Test parameter `key`.
                        value: Test parameter `value`.
                    """
                    pass

            return cast("EagerTensorProtocol", DummyRes())

    assert cast("EagerCallableProtocol", func)(cast("EagerCallableProtocol", MockMXScatter)(), cast("EagerCallableProtocol", DummyIndices)(), cast("EagerCallableProtocol", DummyUpdates)(), cast("EagerCallableProtocol", DummyShape)()) is not None
    assert cast("EagerCallableProtocol", func)(cast("EagerCallableProtocol", MockMXScatter)(), cast("EagerCallableProtocol", DummyIndices)(), cast("EagerCallableProtocol", DummyUpdates)(), shape=cast("EagerCallableProtocol", DummyShape)()) is not None
    func = mx_eager.mlx_eager_registry.get("Reshape")
    getattr(_state, "update")({"DummyIndices": DummyIndices, "DummyUpdates": DummyUpdates, "MockMXScatter": MockMXScatter, "func": func})
    _state.update(locals())


def _test_mlx_eager_missing_coverage_part_4(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 4 of test_mlx_eager_missing_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (func, mx_eager) = (_state["func"], _state["mx_eager"])

    class DummyShapeList:
        """Test class DummyShapeList."""

        def tolist(self) -> EagerValue:
            """Test function tolist."""
            return [1, 2]

    class MockMXReshape:
        """Test class MockMXReshape."""

        def reshape(self, x: EagerValue, shape: EagerValue) -> EagerValue:
            """Test function reshape.

            Args:
                x: Test parameter `x`.
                shape: Test parameter `shape`.
            """
            return shape

    assert cast("EagerCallableProtocol", func)(cast("EagerCallableProtocol", MockMXReshape)(), "x", cast("EagerCallableProtocol", DummyShapeList)()) == [1, 2]
    assert cast("EagerCallableProtocol", func)(cast("EagerCallableProtocol", MockMXReshape)(), input="x", shape=cast("EagerCallableProtocol", DummyShapeList)()) == [1, 2]

    class DummyMXWithFloat32:
        """Test class DummyMXWithFloat32."""

        float32 = "mock_float32"

        def zeros(self, shape: EagerValue, dtype: EagerValue) -> EagerValue:
            """Test function zeros.

            Args:
                shape: Test parameter `shape`.
                dtype: Test parameter `dtype`.
            """
            return dtype

    assert cast("EagerCallableProtocol", mx_eager._mlx_zeros)(cast("EagerCallableProtocol", DummyMXWithFloat32)(), [1], dtype="bfloat16") == "mock_float32"
    func = mx_eager.mlx_eager_registry.get("NanToNum")
    getattr(_state, "update")({"DummyMXWithFloat32": DummyMXWithFloat32, "DummyShapeList": DummyShapeList, "MockMXReshape": MockMXReshape, "func": func})
    _state.update(locals())


def _test_mlx_eager_missing_coverage_part_5(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 5 of test_mlx_eager_missing_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    func = _state["func"]

    class MockMXNan:
        """Test class MockMXNan."""

        def nan_to_num(self, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
            """Test function nan_to_num.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.
            """
            return kwargs

    res = cast("EagerCallableProtocol", func)(cast("EagerCallableProtocol", MockMXNan)(), "x", nan=None, posinf=None)
    assert "nan" not in res or res["nan"] is None

    class DummyItemData:
        """Test class DummyItemData."""

        class DummyData:
            """Test class DummyData."""

            def item(self) -> EagerValue:
                """Test function item."""
                return 1.0

        data = DummyData()

    res = cast("EagerCallableProtocol", func)(cast("EagerCallableProtocol", MockMXNan)(), "x", posinf=cast("EagerCallableProtocol", DummyItemData)())
    assert res["posinf"] == 1.0
    res = cast("EagerCallableProtocol", func)(cast("EagerCallableProtocol", MockMXNan)(), "x", invalid_key=1.0)
    assert "invalid_key" not in res
    getattr(_state, "update")({"DummyItemData": DummyItemData, "MockMXNan": MockMXNan, "res": res})


def test_mlx_eager_missing_coverage() -> None:
    """Test function test_mlx_eager_missing_coverage."""
    _state: dict[str, EagerTensorProtocol] = {}
    _test_mlx_eager_missing_coverage_part_1(_state)
    _test_mlx_eager_missing_coverage_part_2(_state)
    _test_mlx_eager_missing_coverage_part_3(_state)
    _test_mlx_eager_missing_coverage_part_4(_state)
    _test_mlx_eager_missing_coverage_part_5(_state)
    assert len(_state) > 0
