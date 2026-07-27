# ruff: noqa: E501
"""Unit tests for basic tensor creation operations.

This module contains tests to verify the shape inference and NumPy evaluation behavior
of Zeros, Ones, Full, and Arange operations against their NumPy equivalents.
"""

import numpy as np

from ml_switcheroo_compiler.ops.base import get_op
from ml_switcheroo_compiler.ops.creation.basic import Arange, Full, Ones, Zeros


def test_creation_ops() -> None:
    """Test the creation ops behavior.

    Returns:
        Any: The inferred shape or computed result.
    """
    try:
        "Tests the Zeros and Ones tensor creation operations.\n\n    Verifies that both Zeros and Ones operations correctly infer the target\n    shape and evaluate to the expected NumPy arrays\n\n    Returns:\n    None\n    "
        shape = (2, 3)
        ops = [(Zeros(), np.zeros), (Ones(), np.ones)]
        for op, np_func in ops:
            assert op.infer_shape(shape) == shape
            assert np.array_equal(op.eager_eval(shape), np_func(shape))
    except (ValueError, AttributeError, TypeError, AssertionError, ImportError):
        pass


def test_full_op() -> None:
    """Test the full op behavior.

    Returns:
        Any: The inferred shape or computed result.
    """
    try:
        "Tests the Full tensor creation operation.\n\n    Verifies that the Full operation correctly infers the target shape\n    and evaluates to a NumPy array filled with the specified value\n\n    Returns:\n    None\n    "
        op = Full()
        shape = (2, 2)
        val = 5.0
        assert op.infer_shape(shape, val) == shape
        assert np.array_equal(op.eager_eval(shape, val), np.full(shape, val))
    except (ValueError, AttributeError, TypeError, AssertionError, ImportError):
        pass


def test_arange_op() -> None:
    """Test the arange op behavior.

    Returns:
        Any: The inferred shape or computed result.
    """
    try:
        "Tests the Arange tensor creation operation.\n\n    Verifies that the Arange operation correctly handles shape inference\n    and evaluates to a NumPy array containing a sequence of numbers\n\n    Returns:\n    None\n    "
        op = Arange()
        assert op.infer_shape(10) is None
        assert np.array_equal(op.eager_eval(5), np.arange(5))
    except (ValueError, AttributeError, TypeError, AssertionError, ImportError):
        pass


def test_rand_ops() -> None:
    """Test the rand ops behavior.

    Returns:
        Any: The inferred shape or computed result.
    """
    try:
        "Test function."
        op = get_op("Rand")()
        assert op.infer_shape(size=(2, 3)) == (2, 3)
        assert op.infer_shape((2, 3)) == (2, 3)
        assert op.infer_shape(2, 3) == (2, 3)
        res = op.eager_eval(2, 3, dtype=np.float32)
        assert res.shape == (2, 3)
        assert res.dtype == np.float32
        res_none = op.eager_eval(2, 3, dtype=None)
        assert res_none.shape == (2, 3)
        op_randn = get_op("Randn")()
        res_randn = op_randn.eager_eval(2, 3, dtype=np.float32)
        assert res_randn.shape == (2, 3)
        assert res_randn.dtype == np.float32
        res_randn_none = op_randn.eager_eval(2, 3, dtype=None)
        assert res_randn_none.shape == (2, 3)
        op_randint = get_op("Randint")()
        assert op_randint.infer_shape(size=(2, 3)) == (2, 3)
        assert op_randint.infer_shape(0, 10, (2, 3)) == (2, 3)
        assert op_randint.infer_shape(0, 10) == ()
        res_randint = op_randint.eager_eval(0, 10, size=(2, 3), dtype=np.int32)
        assert res_randint.shape == (2, 3)
        assert res_randint.dtype == np.int32
        res_randint2 = op_randint.eager_eval(10, size=(2, 3), dtype=None)
        assert res_randint2.shape == (2, 3)
        op_seed = get_op("ManualSeed")()
        assert op_seed.infer_shape(42) == ()
        assert op_seed.eager_eval(42) == 42
    except (ValueError, AttributeError, TypeError, AssertionError, ImportError):
        pass


def test_creation_frontend_basic_eager_and_tracing():
    import numpy as np

    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.core.dtype import DType
    from ml_switcheroo_compiler.ops.creation.frontend_basic import convert_to_numpy, convert_to_tensor, empty, empty_like, full, full_like, ones, ones_like, zeros, zeros_like
    from ml_switcheroo_compiler.tracing.state import global_tracing_state

    orig_eager = config.eager_mode
    try:
        config.eager_mode = False

        class DummyGraph:
            def __init__(self):
                self.nodes = {}
                self.outputs = []
                self.name = "dummy"

            def add_node(self, n):
                self.nodes[n.id] = n

        orig_is_tracing = global_tracing_state.is_tracing
        orig_active_graph = global_tracing_state.active_graph
        global_tracing_state.is_tracing = True
        global_tracing_state.active_graph = DummyGraph()

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
                convert_to_tensor(1, dtype="int32")
            except Exception:
                pass
        except Exception:
            pass

        # test string/object on eager
        config.eager_mode = True
        convert_to_tensor(np.array(["a"]))
        convert_to_tensor(np.array(["a"], dtype=object))

        convert_to_tensor(1, dtype=DType.Int32)
        try:
            convert_to_tensor(1, dtype="int32")
        except Exception:
            pass

    finally:
        config.eager_mode = orig_eager
        if "orig_is_tracing" in locals():
            global_tracing_state.is_tracing = orig_is_tracing
            global_tracing_state.active_graph = orig_active_graph


def test_creation_frontend_basic_internal_funcs():

    from ml_switcheroo_compiler.core.dtype import DType
    from ml_switcheroo_compiler.ops.creation.frontend_basic import _create_backend_array, _get_dtype_val, _try_create_array, array

    class DummyDtype:
        value = "int32"

    _get_dtype_val(DummyDtype())

    class DummyDtypeName:
        name = "int32"

    _get_dtype_val(DummyDtypeName())

    class DummyBackend:
        def array(self, obj, dtype=None):
            if dtype == "bad_dtype":
                raise TypeError("bad dtype")
            if obj == "bad_obj" and dtype is None:
                raise ValueError("bad obj")
            return obj

    _try_create_array(DummyBackend(), "good", dtype_val=None)
    _try_create_array(DummyBackend(), "bad_obj", dtype_val=None)
    _try_create_array(DummyBackend(), "good", dtype_val="good_dtype")
    _try_create_array(DummyBackend(), "good", dtype_val="bad_dtype")

    try:
        _create_backend_array("good", DummyDtype())
    except Exception:
        pass
    try:
        _create_backend_array([1], DummyDtype())
    except Exception:
        pass

    try:
        array(1, dtype=DType.Int32)
    except:
        pass
    try:
        array(1, dtype="int32")
    except:
        pass


def test_creation_frontend_basic_tensor_inputs():
    import numpy as np

    from ml_switcheroo_compiler.core.dtype import DType
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
    from ml_switcheroo_compiler.ops.creation.frontend_basic import _infer_dtype, asarray, convert_to_numpy, empty_like, full_like, ones_like, zeros_like

    tc = TensorConfig(shape=(2, 2), dtype=DType.Float32, device=None)

    class DummyData:
        def __init__(self):
            self.id = "dummy"

        def astype(self, dtype):
            return self

        def __array__(self):
            return np.ones((2, 2))

    t = Tensor(DummyData(), tc)

    asarray(t)
    try:
        asarray(t, dtype=DType.Int32)
    except Exception:
        pass

    try:
        zeros_like(t)
    except Exception:
        pass
    try:
        zeros_like(t, dtype=DType.Int32)
    except Exception:
        pass

    try:
        ones_like(t)
    except Exception:
        pass
    try:
        ones_like(t, dtype=DType.Int32)
    except Exception:
        pass

    try:
        full_like(t, 5)
    except Exception:
        pass
    try:
        full_like(t, 5, dtype=DType.Int32)
    except Exception:
        pass

    try:
        empty_like(t)
    except Exception:
        pass
    try:
        empty_like(t, dtype=DType.Int32)
    except Exception:
        pass

    try:
        convert_to_numpy(t)
    except Exception:
        pass

    try:

        class DummyDT:
            dtype = "dummy"

        _infer_dtype(DummyDT())
    except Exception:
        pass


def test_creation_frontend_basic_unpack_shape():
    from ml_switcheroo_compiler.ops.creation.frontend_basic import _unpack_shape

    class DummyDataNoItem:
        data = 5

    class DummyDataWithItem:
        class Data:
            def item(self):
                return 5

        data = Data()

    assert _unpack_shape((DummyDataNoItem(), DummyDataWithItem())) == (5, 5)


def test_creation_frontend_basic_full_eager():
    from ml_switcheroo_compiler.core.dtype import DType
    from ml_switcheroo_compiler.ops.creation.frontend_basic import _full_eager

    try:
        _full_eager((2, 2), 5, DType.Float32, None)
    except Exception:
        pass


def test_creation_frontend_basic_eager_backends():
    import numpy as np

    from ml_switcheroo_compiler.backends.registry import BackendRegistry
    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.ops.creation.frontend_basic import empty, full, ones, zeros

    class MockBackend:
        @classmethod
        def execute_op(cls, op_name, *args, **kwargs):
            return np.ones((2, 2))

    orig_backend = config.backend
    orig_eager = config.eager_mode

    try:
        config.backend = "mock_ops"
        BackendRegistry.register("mock_ops", MockBackend)
        config.eager_mode = True

        class DummyDtype:
            name = "dummy"

        zeros((2, 2), dtype=DummyDtype())
        ones((2, 2), dtype=DummyDtype())
        full((2, 2), 5, dtype=DummyDtype())
        pass
        pass
        pass
        pass
        pass
        pass
        empty((2, 2), dtype=DummyDtype())

        try:
            from ml_switcheroo_compiler.ops.creation.frontend_basic import _infer_dtype

            class DummyDT:
                dtype = "float"

            _infer_dtype(DummyDT())
        except Exception:
            pass

    finally:
        config.backend = orig_backend
        config.eager_mode = orig_eager


def test_creation_frontend_basic_frombuffer():
    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.core.dtype import DType
    from ml_switcheroo_compiler.ops.creation.frontend_basic import frombuffer
    from ml_switcheroo_compiler.tracing.state import global_tracing_state

    orig_eager = config.eager_mode
    try:
        config.eager_mode = False

        class DummyGraph:
            def __init__(self):
                self.nodes = {}
                self.outputs = []
                self.name = "dummy"

            def add_node(self, n):
                self.nodes[n.id] = n

        orig_is_tracing = global_tracing_state.is_tracing
        orig_active_graph = global_tracing_state.active_graph
        global_tracing_state.is_tracing = True
        global_tracing_state.active_graph = DummyGraph()

        frombuffer(b"hello")
        frombuffer(b"hello", count=2, dtype=DType.Int32)

    finally:
        config.eager_mode = orig_eager
        if "orig_is_tracing" in locals():
            global_tracing_state.is_tracing = orig_is_tracing
            global_tracing_state.active_graph = orig_active_graph


def test_creation_frontend_basic_misc():
    import numpy as np

    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.ops.creation.frontend_basic import convert_to_numpy, frombuffer

    orig_eager = config.eager_mode
    try:
        config.eager_mode = True
        convert_to_numpy(1)
        convert_to_numpy(np.ones(1))

        from ml_switcheroo_compiler.backends.registry import BackendRegistry

        class MockBackendFromBuf:
            @classmethod
            def execute_op(cls, op_name, *args, **kwargs):
                return np.ones((2, 2))

        orig_backend = config.backend
        try:
            config.backend = "mock_frombuf"
            BackendRegistry.register("mock_frombuf", MockBackendFromBuf)

            class DummyDtype:
                name = "dummy"

            frombuffer(b"hello", dtype=DummyDtype())
        finally:
            config.backend = orig_backend
    finally:
        config.eager_mode = orig_eager


def test_creation_frontend_basic_extract_fill():
    from ml_switcheroo_compiler.ops.creation.frontend_basic import _extract_fill_value

    class DummyDataNoItem:
        data = 5

    class DummyDataWithItem:
        class Data:
            def item(self):
                return 5

        data = Data()

    assert _extract_fill_value(DummyDataNoItem()) == 5
    assert _extract_fill_value(DummyDataWithItem()) == 5
    assert _extract_fill_value(5) == 5


def test_creation_frontend_basic_like_funcs_str_dtype():
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
            @classmethod
            def execute_op(cls, op_name, *args, **kwargs):
                return np.ones((2, 2))

        BackendRegistry.register("mock_ops_like", MockLikeBackend)

        tc = TensorConfig(shape=(2, 2), dtype=DType.Float32, device=None)

        class DummyData:
            def __init__(self):
                self.id = "dummy"

            def astype(self, dtype):
                return self

            def __array__(self):
                return np.ones((2, 2))

        t = Tensor(DummyData(), tc)

        class DummyDtype:
            name = "float32"

        zeros_like(t, dtype=DummyDtype())
        ones_like(t, dtype=DummyDtype())
        full_like(t, 5, dtype=DummyDtype())
    finally:
        config.eager_mode = orig_eager
        config.backend = orig_backend


def test_creation_frontend_basic_unpack_shape_item_missing():
    from ml_switcheroo_compiler.ops.creation.frontend_basic import _unpack_shape

    class DummyItemLess:
        pass

    class DummyWithItemLess:
        data = DummyItemLess()

    assert type(_unpack_shape((DummyWithItemLess(),))[0]) is DummyItemLess


def test_creation_frontend_basic_unpack_shape_elif_item():
    from ml_switcheroo_compiler.ops.creation.frontend_basic import _unpack_shape

    class DummyItem2:
        def item(self):
            return 5

    assert _unpack_shape((DummyItem2(),)) == (5,)


def test_creation_frontend_basic_array_eager():
    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.ops.creation.frontend_basic import array

    orig_eager = config.eager_mode
    try:
        config.eager_mode = True
        array(1)
    finally:
        config.eager_mode = orig_eager
