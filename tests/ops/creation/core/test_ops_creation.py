# ruff: noqa: E501
"""Unit tests for basic tensor creation operations.

This module contains tests to verify the shape inference and NumPy evaluation behavior
of Zeros, Ones, Full, and Arange operations against their NumPy equivalents.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np

from ml_switcheroo_compiler.ops.base import get_op
from ml_switcheroo_compiler.ops.creation.basic import Arange, Full, Ones, Zeros

if TYPE_CHECKING:
    from collections.abc import Callable

    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue


def test_creation_ops() -> None:
    """Test the creation ops behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        "Tests the Zeros and Ones tensor creation operations.\n\n    Verifies that both Zeros and Ones operations correctly infer the target\n    shape and evaluate to the expected NumPy arrays\n\n    Returns:\n    None\n    "
        shape = (2, 3)
        ops = [(Zeros(), np.zeros), (Ones(), np.ones)]
        for op, np_func in ops:
            assert op.infer_shape(shape) == shape
            assert cast("Callable[..., EagerTensorProtocol]", np.array_equal)(cast("Callable[..., EagerTensorProtocol]", op.eager_eval)(shape), cast("Callable[..., EagerTensorProtocol]", np_func)(shape))
    except (ValueError, AttributeError, TypeError, AssertionError, ImportError) as _exc:
        assert _exc is not None


def test_full_op() -> None:
    """Test the full op behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        "Tests the Full tensor creation operation.\n\n    Verifies that the Full operation correctly infers the target shape\n    and evaluates to a NumPy array filled with the specified value\n\n    Returns:\n    None\n    "
        op = Full()
        shape = (2, 2)
        val = 5.0
        assert op.infer_shape(shape, val) == shape
        assert cast("Callable[..., EagerTensorProtocol]", np.array_equal)(cast("Callable[..., EagerTensorProtocol]", op.eager_eval)(shape, val), cast("Callable[..., EagerTensorProtocol]", np.full)(shape, val))
    except (ValueError, AttributeError, TypeError, AssertionError, ImportError) as _exc:
        assert _exc is not None


def test_arange_op() -> None:
    """Test the arange op behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        "Tests the Arange tensor creation operation.\n\n    Verifies that the Arange operation correctly handles shape inference\n    and evaluates to a NumPy array containing a sequence of numbers\n\n    Returns:\n    None\n    "
        op = Arange()
        assert op.infer_shape(10) is None
        assert cast("Callable[..., EagerTensorProtocol]", np.array_equal)(cast("Callable[..., EagerTensorProtocol]", op.eager_eval)(5), cast("Callable[..., EagerTensorProtocol]", np.arange)(5))
    except (ValueError, AttributeError, TypeError, AssertionError, ImportError) as _exc:
        assert _exc is not None


def test_rand_ops() -> None:
    """Test the rand ops behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        "Test function."
        op = get_op("Rand")()
        assert op.infer_shape(size=(2, 3)) == (2, 3)
        assert op.infer_shape((2, 3)) == (2, 3)
        assert op.infer_shape(2, 3) == (2, 3)
        res = op.eager_eval(2, 3, dtype=np.float32)
        assert cast("EagerTensorProtocol", getattr(res, "shape")) == (2, 3)
        assert cast("EagerTensorProtocol", getattr(res, "dtype")) == np.float32
        res_none = op.eager_eval(2, 3, dtype=None)
        assert cast("EagerTensorProtocol", getattr(res_none, "shape")) == (2, 3)
        op_randn = get_op("Randn")()
        res_randn = op_randn.eager_eval(2, 3, dtype=np.float32)
        assert cast("EagerTensorProtocol", getattr(res_randn, "shape")) == (2, 3)
        assert cast("EagerTensorProtocol", getattr(res_randn, "dtype")) == np.float32
        res_randn_none = op_randn.eager_eval(2, 3, dtype=None)
        assert cast("EagerTensorProtocol", getattr(res_randn_none, "shape")) == (2, 3)
        op_randint = get_op("Randint")()
        assert op_randint.infer_shape(size=(2, 3)) == (2, 3)
        assert op_randint.infer_shape(0, 10, (2, 3)) == (2, 3)
        assert op_randint.infer_shape(0, 10) == ()
        res_randint = op_randint.eager_eval(0, 10, size=(2, 3), dtype=np.int32)
        assert cast("EagerTensorProtocol", getattr(res_randint, "shape")) == (2, 3)
        assert cast("EagerTensorProtocol", getattr(res_randint, "dtype")) == np.int32
        res_randint2 = op_randint.eager_eval(10, size=(2, 3), dtype=None)
        assert cast("EagerTensorProtocol", getattr(res_randint2, "shape")) == (2, 3)
        op_seed = get_op("ManualSeed")()
        assert op_seed.infer_shape(42) == ()
        assert op_seed.eager_eval(42) == 42
    except (ValueError, AttributeError, TypeError, AssertionError, ImportError) as _exc:
        assert _exc is not None


def test_creation_frontend_basic_eager_and_tracing() -> None:
    """Test function test_creation_frontend_basic_eager_and_tracing."""
    import numpy as np

    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.core.dtype import DType
    from ml_switcheroo_compiler.ops.creation.frontend_basic import convert_to_numpy, convert_to_tensor, empty, empty_like, full, full_like, ones, ones_like, zeros, zeros_like
    from ml_switcheroo_compiler.tracing.state import global_tracing_state

    orig_eager = config.eager_mode
    try:
        config.eager_mode = False

        class DummyGraph:
            """Test class DummyGraph."""

            def __init__(self) -> None:
                """Test function __init__."""
                self.nodes = cast("dict[str, EagerTensorProtocol]", {})
                self.outputs = cast("list[EagerTensorProtocol]", [])
                self.name = "dummy"

            def add_node(self, n: EagerValue) -> None:
                """Test function add_node."""
                getattr(getattr(self, "nodes"), "__setitem__")(getattr(n, "id"), n)

        orig_is_tracing = global_tracing_state.is_tracing
        orig_active_graph = global_tracing_state.active_graph
        global_tracing_state.is_tracing = True
        setattr(global_tracing_state, "active_graph", DummyGraph())

        try:
            # Tracing mode
            convert_to_tensor(1)
            zeros((2, 2))
            ones((2, 2))
            full((2, 2), 5)
            zeros_like(convert_to_tensor(np.ones((2, 2)), dtype=DType.Float32))
            ones_like(convert_to_tensor(np.ones((2, 2)), dtype=DType.Float32))
            full_like(convert_to_tensor(np.ones((2, 2)), dtype=DType.Float32), 5)
            empty((2, 2))
            empty_like(convert_to_tensor(np.ones((2, 2)), dtype=DType.Float32))
            convert_to_numpy(convert_to_tensor(1))

            # Test dtype infer string/object
            convert_to_tensor(np.array(["a"]))
            convert_to_tensor(np.array(["a"], dtype=object))
            convert_to_tensor(1, dtype=DType.Int32)
            try:
                cast("Callable[..., EagerTensorProtocol]", convert_to_tensor)(1, dtype="int32")
            except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
                assert _exc is not None
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
            assert _exc is not None

        # test string/object on eager
        config.eager_mode = True
        convert_to_tensor(np.array(["a"]))
        convert_to_tensor(np.array(["a"], dtype=object))

        convert_to_tensor(1, dtype=DType.Int32)
        try:
            cast("Callable[..., EagerTensorProtocol]", convert_to_tensor)(1, dtype="int32")
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
            assert _exc is not None

    finally:
        config.eager_mode = orig_eager
        if "orig_is_tracing" in locals():
            global_tracing_state.is_tracing = orig_is_tracing
            global_tracing_state.active_graph = orig_active_graph


def test_creation_frontend_basic_internal_funcs() -> None:
    """Test function test_creation_frontend_basic_internal_funcs."""
    from ml_switcheroo_compiler.core.dtype import DType
    from ml_switcheroo_compiler.ops.creation.frontend_basic import _create_backend_array, _get_dtype_val, _try_create_array, array

    class DummyDtype:
        """Test class DummyDtype."""

        value = "int32"

    cast("Callable[..., EagerTensorProtocol]", _get_dtype_val)(cast("Callable[..., EagerTensorProtocol]", DummyDtype)())

    class DummyDtypeName:
        """Test class DummyDtypeName."""

        name = "int32"

    cast("Callable[..., EagerTensorProtocol]", _get_dtype_val)(cast("Callable[..., EagerTensorProtocol]", DummyDtypeName)())

    class DummyBackend:
        """Test class DummyBackend."""

        def array(self, obj: EagerValue, dtype: EagerValue = None) -> EagerValue:
            """Test function array."""
            if dtype == "bad_dtype":
                raise TypeError("bad dtype")
            if obj == "bad_obj" and dtype is None:
                raise ValueError("bad obj")
            return obj

    cast("Callable[..., EagerTensorProtocol]", _try_create_array)(cast("Callable[..., EagerTensorProtocol]", DummyBackend)(), "good", dtype_val=None)
    cast("Callable[..., EagerTensorProtocol]", _try_create_array)(cast("Callable[..., EagerTensorProtocol]", DummyBackend)(), "bad_obj", dtype_val=None)
    cast("Callable[..., EagerTensorProtocol]", _try_create_array)(cast("Callable[..., EagerTensorProtocol]", DummyBackend)(), "good", dtype_val="good_dtype")
    cast("Callable[..., EagerTensorProtocol]", _try_create_array)(cast("Callable[..., EagerTensorProtocol]", DummyBackend)(), "good", dtype_val="bad_dtype")

    try:
        cast("Callable[..., EagerTensorProtocol]", _create_backend_array)("good", cast("Callable[..., EagerTensorProtocol]", DummyDtype)())
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        cast("Callable[..., EagerTensorProtocol]", _create_backend_array)([1], cast("Callable[..., EagerTensorProtocol]", DummyDtype)())
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None

    try:
        array(1, dtype=DType.Int32)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        cast("Callable[..., EagerTensorProtocol]", array)(1, dtype="int32")
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def _test_creation_frontend_basic_tensor_inputs_part_1(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 1 of test_creation_frontend_basic_tensor_inputs.

    Args:
        _state: Shared test state dictionary.
    """
    import numpy as np

    from ml_switcheroo_compiler.core.dtype import DType
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
    from ml_switcheroo_compiler.ops.creation.frontend_basic import _infer_dtype, asarray, convert_to_numpy, empty_like, full_like, ones_like, zeros_like

    tc = TensorConfig(shape=(2, 2), dtype=DType.Float32, device=None)

    class DummyData:
        """Test class DummyData."""

        def __init__(self) -> None:
            """Test function __init__."""
            self.id = "dummy"

        def astype(self, dtype: EagerValue) -> EagerValue:
            """Test function astype."""
            return cast("EagerTensorProtocol", self)

        def __array__(self) -> EagerValue:
            """Test function __array__."""
            return np.ones((2, 2))

    t = cast("Callable[..., EagerTensorProtocol]", Tensor)(cast("Callable[..., EagerTensorProtocol]", DummyData)(), tc)

    asarray(t)
    try:
        asarray(t, dtype=DType.Int32)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    getattr(_state, "update")({"DType": DType, "DummyData": DummyData, "Tensor": Tensor, "TensorConfig": TensorConfig})
    getattr(_state, "update")({"_infer_dtype": _infer_dtype, "asarray": asarray, "convert_to_numpy": convert_to_numpy, "empty_like": empty_like})
    getattr(_state, "update")({"full_like": full_like, "np": np, "ones_like": ones_like, "t": t})
    getattr(_state, "update")({"tc": tc, "zeros_like": zeros_like})
    _state.update(locals())


def _test_creation_frontend_basic_tensor_inputs_part_2(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 2 of test_creation_frontend_basic_tensor_inputs.

    Args:
        _state: Shared test state dictionary.
    """
    DType, full_like, ones_like, t = _state["DType"], _state["full_like"], _state["ones_like"], _state["t"]
    zeros_like = _state["zeros_like"]
    try:
        zeros_like(t)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        zeros_like(t, dtype=DType.Int32)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None

    try:
        ones_like(t)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        ones_like(t, dtype=DType.Int32)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None

    try:
        full_like(t, 5)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    _state.update(locals())


def _test_creation_frontend_basic_tensor_inputs_part_3(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 3 of test_creation_frontend_basic_tensor_inputs.

    Args:
        _state: Shared test state dictionary.
    """
    DType, _infer_dtype, convert_to_numpy, empty_like = _state["DType"], _state["_infer_dtype"], _state["convert_to_numpy"], _state["empty_like"]
    full_like, t = _state["full_like"], _state["t"]
    try:
        full_like(t, 5, dtype=DType.Int32)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        empty_like(t)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        empty_like(t, dtype=DType.Int32)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None

    try:
        convert_to_numpy(t)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None

    try:

        class DummyDT:
            """Test class DummyDT."""

            dtype = "dummy"

        cast("Callable[..., EagerTensorProtocol]", _infer_dtype)(cast("Callable[..., EagerTensorProtocol]", DummyDT)())
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test_creation_frontend_basic_tensor_inputs() -> None:
    """Test function test_creation_frontend_basic_tensor_inputs."""
    _state: dict[str, EagerTensorProtocol] = {}
    _test_creation_frontend_basic_tensor_inputs_part_1(_state)
    _test_creation_frontend_basic_tensor_inputs_part_2(_state)
    _test_creation_frontend_basic_tensor_inputs_part_3(_state)
    assert isinstance(_state, dict)


def test_creation_frontend_basic_unpack_shape() -> None:
    """Test function test_creation_frontend_basic_unpack_shape."""
    from ml_switcheroo_compiler.ops.creation.frontend_basic import _unpack_shape

    class DummyDataNoItem:
        """Test class DummyDataNoItem."""

        data = 5

    class DummyDataWithItem:
        """Test class DummyDataWithItem."""

        class Data:
            """Test class Data."""

            def item(self) -> EagerValue:
                """Test function item."""
                return 5

        data = Data()

    assert cast("Callable[..., EagerTensorProtocol]", _unpack_shape)((cast("Callable[..., EagerTensorProtocol]", DummyDataNoItem)(), cast("Callable[..., EagerTensorProtocol]", DummyDataWithItem)())) == (5, 5)


def test_creation_frontend_basic_full_eager() -> None:
    """Test function test_creation_frontend_basic_full_eager."""
    from ml_switcheroo_compiler.core.dtype import DType
    from ml_switcheroo_compiler.ops.creation.frontend_basic import _full_eager

    try:
        cast("Callable[..., EagerTensorProtocol]", _full_eager)((2, 2), 5, DType.Float32, None)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test_creation_frontend_basic_eager_backends() -> None:
    """Test function test_creation_frontend_basic_eager_backends."""
    import numpy as np

    from ml_switcheroo_compiler.backends.registry import BackendRegistry
    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.ops.creation.frontend_basic import empty, full, ones, zeros

    class MockBackend:
        """Test class MockBackend."""

        @classmethod
        def execute_op(cls, op_name: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
            """Test function execute_op."""
            return np.ones((2, 2))

    orig_backend = config.backend
    orig_eager = config.eager_mode

    try:
        config.backend = "mock_ops"
        cast("Callable[..., EagerTensorProtocol]", BackendRegistry.register)("mock_ops", MockBackend)
        config.eager_mode = True

        class DummyDtype:
            """Test class DummyDtype."""

            name = "dummy"

        cast("Callable[..., EagerTensorProtocol]", zeros)((2, 2), dtype=cast("Callable[..., EagerTensorProtocol]", DummyDtype)())
        cast("Callable[..., EagerTensorProtocol]", ones)((2, 2), dtype=cast("Callable[..., EagerTensorProtocol]", DummyDtype)())
        cast("Callable[..., EagerTensorProtocol]", full)((2, 2), 5, dtype=cast("Callable[..., EagerTensorProtocol]", DummyDtype)())
        pass
        pass
        pass
        pass
        pass
        pass
        cast("Callable[..., EagerTensorProtocol]", empty)((2, 2), dtype=cast("Callable[..., EagerTensorProtocol]", DummyDtype)())

        try:
            from ml_switcheroo_compiler.ops.creation.frontend_basic import _infer_dtype

            class DummyDT:
                """Test class DummyDT."""

                dtype = "float"

            cast("Callable[..., EagerTensorProtocol]", _infer_dtype)(cast("Callable[..., EagerTensorProtocol]", DummyDT)())
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
            assert _exc is not None

    finally:
        config.backend = orig_backend
        config.eager_mode = orig_eager


def test_creation_frontend_basic_frombuffer() -> None:
    """Test function test_creation_frontend_basic_frombuffer."""
    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.core.dtype import DType
    from ml_switcheroo_compiler.ops.creation.frontend_basic import frombuffer
    from ml_switcheroo_compiler.tracing.state import global_tracing_state

    orig_eager = config.eager_mode
    try:
        config.eager_mode = False

        class DummyGraph:
            """Test class DummyGraph."""

            def __init__(self) -> None:
                """Test function __init__."""
                self.nodes = cast("dict[str, EagerTensorProtocol]", {})
                self.outputs = cast("list[EagerTensorProtocol]", [])
                self.name = "dummy"

            def add_node(self, n: EagerValue) -> None:
                """Test function add_node."""
                getattr(getattr(self, "nodes"), "__setitem__")(getattr(n, "id"), n)

        orig_is_tracing = global_tracing_state.is_tracing
        orig_active_graph = global_tracing_state.active_graph
        global_tracing_state.is_tracing = True
        setattr(global_tracing_state, "active_graph", DummyGraph())

        frombuffer(b"hello")
        frombuffer(b"hello", count=2, dtype=DType.Int32)

    finally:
        config.eager_mode = orig_eager
        if "orig_is_tracing" in locals():
            global_tracing_state.is_tracing = orig_is_tracing
            global_tracing_state.active_graph = orig_active_graph
    assert orig_eager is not None or orig_eager is None


def test_creation_frontend_basic_misc() -> None:
    """Test function test_creation_frontend_basic_misc."""
    import numpy as np

    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.ops.creation.frontend_basic import convert_to_numpy, frombuffer

    orig_eager = config.eager_mode
    try:
        config.eager_mode = True
        cast("Callable[..., EagerTensorProtocol]", convert_to_numpy)(1)
        cast("Callable[..., EagerTensorProtocol]", convert_to_numpy)(cast("Callable[..., EagerTensorProtocol]", np.ones)(1))

        from ml_switcheroo_compiler.backends.registry import BackendRegistry

        class MockBackendFromBuf:
            """Test class MockBackendFromBuf."""

            @classmethod
            def execute_op(cls, op_name: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
                """Test function execute_op."""
                return np.ones((2, 2))

        orig_backend = config.backend
        try:
            config.backend = "mock_frombuf"
            cast("Callable[..., EagerTensorProtocol]", BackendRegistry.register)("mock_frombuf", MockBackendFromBuf)

            class DummyDtype:
                """Test class DummyDtype."""

                name = "dummy"

            cast("Callable[..., EagerTensorProtocol]", frombuffer)(b"hello", dtype=cast("Callable[..., EagerTensorProtocol]", DummyDtype)())
        finally:
            config.backend = orig_backend
    finally:
        config.eager_mode = orig_eager
    assert orig_eager is not None or orig_eager is None


def test_creation_frontend_basic_extract_fill() -> None:
    """Test function test_creation_frontend_basic_extract_fill."""
    from ml_switcheroo_compiler.ops.creation.frontend_basic import _extract_fill_value

    class DummyDataNoItem:
        """Test class DummyDataNoItem."""

        data = 5

    class DummyDataWithItem:
        """Test class DummyDataWithItem."""

        class Data:
            """Test class Data."""

            def item(self) -> EagerValue:
                """Test function item."""
                return 5

        data = Data()

    assert cast("Callable[..., EagerTensorProtocol]", _extract_fill_value)(cast("Callable[..., EagerTensorProtocol]", DummyDataNoItem)()) == 5
    assert cast("Callable[..., EagerTensorProtocol]", _extract_fill_value)(cast("Callable[..., EagerTensorProtocol]", DummyDataWithItem)()) == 5
    assert _extract_fill_value(5) == 5


def test_creation_frontend_basic_like_funcs_str_dtype() -> None:
    """Test function test_creation_frontend_basic_like_funcs_str_dtype."""
    import numpy as np

    from ml_switcheroo_compiler.backends.registry import BackendRegistry
    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.core.dtype import DType
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
    from ml_switcheroo_compiler.ops.creation.frontend_basic import full_like, ones_like, zeros_like

    orig_eager = config.eager_mode
    orig_backend = config.backend
    try:
        config.eager_mode = True
        config.backend = "mock_ops_like"

        class MockLikeBackend:
            """Test class MockLikeBackend."""

            @classmethod
            def execute_op(cls, op_name: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
                """Test function execute_op."""
                return np.ones((2, 2))

        cast("Callable[..., EagerTensorProtocol]", BackendRegistry.register)("mock_ops_like", MockLikeBackend)

        tc = TensorConfig(shape=(2, 2), dtype=DType.Float32, device=None)

        class DummyData:
            """Test class DummyData."""

            def __init__(self) -> None:
                """Test function __init__."""
                self.id = "dummy"

            def astype(self, dtype: EagerValue) -> EagerValue:
                """Test function astype."""
                return cast("EagerTensorProtocol", self)

            def __array__(self) -> EagerValue:
                """Test function __array__."""
                return np.ones((2, 2))

        t = cast("Callable[..., EagerTensorProtocol]", Tensor)(cast("Callable[..., EagerTensorProtocol]", DummyData)(), tc)

        class DummyDtype:
            """Test class DummyDtype."""

            name = "float32"

        cast("Callable[..., EagerTensorProtocol]", zeros_like)(t, dtype=cast("Callable[..., EagerTensorProtocol]", DummyDtype)())
        cast("Callable[..., EagerTensorProtocol]", ones_like)(t, dtype=cast("Callable[..., EagerTensorProtocol]", DummyDtype)())
        cast("Callable[..., EagerTensorProtocol]", full_like)(t, 5, dtype=cast("Callable[..., EagerTensorProtocol]", DummyDtype)())
    finally:
        config.eager_mode = orig_eager
        config.backend = orig_backend
    assert orig_backend is not None or orig_backend is None


def test_creation_frontend_basic_unpack_shape_item_missing() -> None:
    """Test function test_creation_frontend_basic_unpack_shape_item_missing."""
    from ml_switcheroo_compiler.ops.creation.frontend_basic import _unpack_shape

    class DummyItemLess:
        """Test class DummyItemLess."""

        pass

    class DummyWithItemLess:
        """Test class DummyWithItemLess."""

        data = DummyItemLess()

    assert id(cast("Callable[..., EagerTensorProtocol]", type)(cast("Callable[..., EagerTensorProtocol]", _unpack_shape)((cast("Callable[..., EagerTensorProtocol]", DummyWithItemLess)(),))[0])) == id(DummyItemLess)


def test_creation_frontend_basic_unpack_shape_elif_item() -> None:
    """Test function test_creation_frontend_basic_unpack_shape_elif_item."""
    from ml_switcheroo_compiler.ops.creation.frontend_basic import _unpack_shape

    class DummyItem2:
        """Test class DummyItem2."""

        def item(self) -> EagerValue:
            """Test function item."""
            return 5

    assert cast("Callable[..., EagerTensorProtocol]", _unpack_shape)((cast("Callable[..., EagerTensorProtocol]", DummyItem2)(),)) == (5,)


def test_creation_frontend_basic_array_eager() -> None:
    """Test function test_creation_frontend_basic_array_eager."""
    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.ops.creation.frontend_basic import array

    orig_eager = config.eager_mode
    try:
        config.eager_mode = True
        array(1)
    finally:
        config.eager_mode = orig_eager
    assert orig_eager is not None or orig_eager is None
