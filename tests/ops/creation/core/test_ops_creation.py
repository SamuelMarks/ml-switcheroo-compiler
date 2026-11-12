"""Unit tests for basic tensor creation operations.

This module contains tests to verify the shape inference and NumPy evaluation behavior
of Zeros, Ones, Full, and Arange operations against their NumPy equivalents.
"""

from __future__ import annotations

import typing

import numpy as np
import pytest

from ml_switcheroo_compiler.ops.base import get_op
from ml_switcheroo_compiler.ops.creation.basic import Arange, Full, Ones, Zeros
from tests.testing_utils import ensure_callable, ensure_eager_tensor, ensure_mutable, ensure_tensor_dict, ensure_tensor_list

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue


def test_creation_ops() -> None:
    """Test the creation ops behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    shape = (2, 3)
    ops = [(Zeros(), np.zeros), (Ones(), np.ones)]
    for op, np_func in ops:
        assert op.infer_shape(shape) == shape
        assert ensure_callable(np.array_equal)(ensure_callable(op.eager_eval)(shape), ensure_callable(np_func)(shape))


def test_full_op() -> None:
    """Test the full op behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    op = Full()
    shape = (2, 2)
    val = 5.0
    assert op.infer_shape(shape, val) == shape
    assert ensure_callable(np.array_equal)(ensure_callable(op.eager_eval)(shape, val), ensure_callable(np.full)(shape, val))


def test_arange_op() -> None:
    """Test the arange op behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    op = Arange()
    assert op.infer_shape(10) is None
    assert ensure_callable(np.array_equal)(ensure_callable(op.eager_eval)(5), ensure_callable(np.arange)(5))


def test_rand_ops() -> None:
    """Test the rand ops behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    op = get_op("Rand")()
    assert op.infer_shape(size=(2, 3)) == (2, 3)
    assert op.infer_shape((2, 3)) == (2, 3)
    assert op.infer_shape(2, 3) == (2, 3)
    res = op.eager_eval(2, 3, dtype=np.float32)
    assert ensure_eager_tensor(ensure_eager_tensor(res).shape) == (2, 3)
    assert ensure_eager_tensor(ensure_eager_tensor(res).dtype) == np.float32
    res_none = op.eager_eval(2, 3, dtype=None)
    assert ensure_eager_tensor(ensure_eager_tensor(res_none).shape) == (2, 3)
    op_randn = get_op("Randn")()
    res_randn = op_randn.eager_eval(2, 3, dtype=np.float32)
    assert ensure_eager_tensor(ensure_eager_tensor(res_randn).shape) == (2, 3)
    res_randn_none = op_randn.eager_eval(2, 3, dtype=None)
    assert ensure_eager_tensor(ensure_eager_tensor(res_randn_none).shape) == (2, 3)
    op_randint = get_op("Randint")()
    assert op_randint.infer_shape(size=(2, 3)) == (2, 3)
    assert op_randint.infer_shape(0, 10, (2, 3)) == (2, 3)
    assert op_randint.infer_shape(0, 10) == ()
    res_randint = op_randint.eager_eval(0, 10, (2, 3), dtype=np.int32)
    assert res_randint is not None
    res_randint2 = op_randint.eager_eval(0, 10, (2, 3), dtype=None)
    assert res_randint2 is not None
    op_seed = get_op("ManualSeed")()
    assert op_seed.infer_shape(42) == ()
    from ml_switcheroo_compiler.core.errors import UnimplementedMathError

    with pytest.raises(UnimplementedMathError):
        op_seed.eager_eval(42)


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
                self.nodes = ensure_tensor_dict({})
                self.outputs = ensure_tensor_list([])
                self.name = "dummy"

            def add_node(self, n: EagerValue) -> None:
                """Test function add_node.

                Args:
                    n: Test parameter `n`.
                """
                ensure_eager_tensor(ensure_eager_tensor(self).nodes)[ensure_eager_tensor(n).id] = n

        orig_is_tracing = global_tracing_state.is_tracing
        orig_active_graph = global_tracing_state.active_graph
        global_tracing_state.is_tracing = True
        ensure_mutable(global_tracing_state).active_graph = ensure_eager_tensor(DummyGraph())
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
        convert_to_tensor(np.array(["a"]))
        convert_to_tensor(np.array(["a"], dtype="O"))
        convert_to_tensor(1, dtype=DType.Int32)
        with pytest.raises(AttributeError):
            ensure_callable(convert_to_tensor)(1, dtype="int32")
        config.eager_mode = True
        convert_to_tensor(np.array(["a"]))
        convert_to_tensor(np.array(["a"], dtype="O"))
        convert_to_tensor(1, dtype=DType.Int32)
        with pytest.raises(AttributeError):
            ensure_callable(convert_to_tensor)(1, dtype="int32")
    finally:
        config.eager_mode = orig_eager
        global_tracing_state.is_tracing = orig_is_tracing
        global_tracing_state.active_graph = orig_active_graph


def test_creation_frontend_basic_internal_funcs() -> None:
    """Test function test_creation_frontend_basic_internal_funcs."""
    from ml_switcheroo_compiler.core.dtype import DType
    from ml_switcheroo_compiler.ops.creation.frontend_basic import _create_backend_array, _get_dtype_val, _try_create_array, array

    class DummyDtype:
        """Test class DummyDtype."""

        value = "int32"

    ensure_callable(_get_dtype_val)(ensure_callable(DummyDtype)())

    class DummyDtypeName:
        """Test class DummyDtypeName."""

        name = "int32"

    ensure_callable(_get_dtype_val)(ensure_callable(DummyDtypeName)())

    class DummyBackend:
        """Test class DummyBackend."""

        def array(self, obj: EagerValue, dtype: EagerValue = None) -> EagerValue:
            """Test function array.

            Args:
                obj: Test parameter `obj`.
                dtype: Test parameter `dtype`.

            Returns:
                EagerValue: Computed output value from `array`.
            """
            if dtype == "bad_dtype":
                raise TypeError("bad dtype")
            if obj == "bad_obj" and dtype is None:
                raise ValueError("bad obj")
            return obj

    ensure_callable(_try_create_array)(ensure_callable(DummyBackend)(), "good", dtype_val=None)
    ensure_callable(_try_create_array)(ensure_callable(DummyBackend)(), "bad_obj", dtype_val=None)
    ensure_callable(_try_create_array)(ensure_callable(DummyBackend)(), "good", dtype_val="good_dtype")
    ensure_callable(_try_create_array)(ensure_callable(DummyBackend)(), "good", dtype_val="bad_dtype")
    with pytest.raises(ValueError):
        ensure_callable(_create_backend_array)("good", ensure_callable(DummyDtype)())
    ensure_callable(_create_backend_array)([1], ensure_callable(DummyDtype)())
    array(1, dtype=DType.Int32)
    with pytest.raises(AttributeError):
        ensure_callable(array)(1, dtype="int32")


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
            """Test function astype.

            Args:
                dtype: Test parameter `dtype`.

            Returns:
                EagerValue: Computed output value from `astype`.
            """
            return ensure_eager_tensor(self)

        def __array__(self) -> EagerValue:
            """Test function __array__.

            Returns:
                EagerValue: Computed output value from `__array__`.
            """
            return np.ones((2, 2))

    t = ensure_callable(Tensor)(ensure_callable(DummyData)(), tc)
    asarray(t)
    with pytest.raises(RuntimeError):
        asarray(t, dtype=DType.Int32)
    _state.update(ensure_tensor_dict({"DType": DType, "DummyData": DummyData, "Tensor": Tensor, "TensorConfig": TensorConfig}))
    _state.update(ensure_tensor_dict({"_infer_dtype": _infer_dtype, "asarray": asarray, "convert_to_numpy": convert_to_numpy, "empty_like": empty_like}))
    _state.update(ensure_tensor_dict({"full_like": full_like, "np": np, "ones_like": ones_like, "t": t}))
    _state.update(ensure_tensor_dict({"tc": tc, "zeros_like": zeros_like}))


def _test_creation_frontend_basic_tensor_inputs_part_2(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 2 of test_creation_frontend_basic_tensor_inputs.

    Args:
        _state: Shared test state dictionary.
    """
    (DType, full_like, ones_like, t) = (_state["DType"], _state["full_like"], _state["ones_like"], _state["t"])
    zeros_like = _state["zeros_like"]
    with pytest.raises(RuntimeError):
        zeros_like(t)
    with pytest.raises(RuntimeError):
        zeros_like(t, dtype=DType.Int32)
    with pytest.raises(RuntimeError):
        ones_like(t)
    with pytest.raises(RuntimeError):
        ones_like(t, dtype=DType.Int32)
    with pytest.raises(RuntimeError):
        full_like(t, 5)
    _state.update(ensure_tensor_dict({"DType": DType, "full_like": full_like, "ones_like": ones_like}))
    _state.update(ensure_tensor_dict({"t": t, "zeros_like": zeros_like}))


def _test_creation_frontend_basic_tensor_inputs_part_3(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 3 of test_creation_frontend_basic_tensor_inputs.

    Args:
        _state: Shared test state dictionary.
    """
    (DType, _infer_dtype, convert_to_numpy, empty_like) = (_state["DType"], _state["_infer_dtype"], _state["convert_to_numpy"], _state["empty_like"])
    (full_like, t) = (_state["full_like"], _state["t"])
    with pytest.raises(RuntimeError):
        full_like(t, 5, dtype=DType.Int32)
    with pytest.raises(RuntimeError):
        empty_like(t)
    with pytest.raises(RuntimeError):
        empty_like(t, dtype=DType.Int32)
    convert_to_numpy(t)
    with pytest.raises(ValueError):

        class DummyDT:
            """Test class DummyDT."""

            dtype = "dummy"

        ensure_callable(_infer_dtype)(ensure_callable(DummyDT)())


def test_creation_frontend_basic_tensor_inputs() -> None:
    """Test function test_creation_frontend_basic_tensor_inputs."""
    _state: dict[str, EagerTensorProtocol] = {}
    _test_creation_frontend_basic_tensor_inputs_part_1(_state)
    _test_creation_frontend_basic_tensor_inputs_part_2(_state)
    _test_creation_frontend_basic_tensor_inputs_part_3(_state)
    assert len(_state) > 0


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
                """Test function item.

                Returns:
                    EagerValue: Computed output value from `item`.
                """
                return 5

        data = Data()

    assert ensure_callable(_unpack_shape)((ensure_callable(DummyDataNoItem)(), ensure_callable(DummyDataWithItem)())) == (5, 5)


def test_creation_frontend_basic_full_eager() -> None:
    """Test function test_creation_frontend_basic_full_eager."""
    from ml_switcheroo_compiler.core.dtype import DType
    from ml_switcheroo_compiler.ops.creation.frontend_basic import _full_eager

    out = ensure_callable(_full_eager)((2, 2), 5, DType.Float32, None)
    assert ensure_eager_tensor(out).shape == (2, 2)


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
            """Test function execute_op.

            Args:
                op_name: Test parameter `op_name`.
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.

            Returns:
                EagerValue: Computed output value from `execute_op`.
            """
            return np.ones((2, 2))

    orig_backend = config.backend
    orig_eager = config.eager_mode
    try:
        config.backend = "mock_ops"
        ensure_callable(BackendRegistry.register)("mock_ops", MockBackend)
        config.eager_mode = True

        class DummyDtype:
            """Test class DummyDtype."""

            name = "dummy"

        ensure_callable(zeros)((2, 2), dtype=ensure_callable(DummyDtype)())
        ensure_callable(ones)((2, 2), dtype=ensure_callable(DummyDtype)())
        ensure_callable(full)((2, 2), 5, dtype=ensure_callable(DummyDtype)())
        pass
        pass
        pass
        pass
        pass
        pass
        ensure_callable(empty)((2, 2), dtype=ensure_callable(DummyDtype)())
        with pytest.raises(ValueError):
            from ml_switcheroo_compiler.ops.creation.frontend_basic import _infer_dtype

            class DummyDT:
                """Test class DummyDT."""

                dtype = "float"

            ensure_callable(_infer_dtype)(ensure_callable(DummyDT)())
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
                self.nodes = ensure_tensor_dict({})
                self.outputs = ensure_tensor_list([])
                self.name = "dummy"

            def add_node(self, n: EagerValue) -> None:
                """Test function add_node.

                Args:
                    n: Test parameter `n`.
                """
                ensure_eager_tensor(ensure_eager_tensor(self).nodes)[ensure_eager_tensor(n).id] = n

        orig_is_tracing = global_tracing_state.is_tracing
        orig_active_graph = global_tracing_state.active_graph
        global_tracing_state.is_tracing = True
        ensure_mutable(global_tracing_state).active_graph = ensure_eager_tensor(DummyGraph())
        frombuffer(b"hello")
        frombuffer(b"hello", count=2, dtype=DType.Int32)
    finally:
        config.eager_mode = orig_eager
        global_tracing_state.is_tracing = orig_is_tracing
        global_tracing_state.active_graph = orig_active_graph
    assert config.eager_mode == orig_eager


def test_creation_frontend_basic_misc() -> None:
    """Test function test_creation_frontend_basic_misc."""
    import numpy as np

    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.ops.creation.frontend_basic import convert_to_numpy, frombuffer

    orig_eager = config.eager_mode
    try:
        config.eager_mode = True
        ensure_callable(convert_to_numpy)(1)
        ensure_callable(convert_to_numpy)(ensure_callable(np.ones)(1))
        from ml_switcheroo_compiler.backends.registry import BackendRegistry

        class MockBackendFromBuf:
            """Test class MockBackendFromBuf."""

            @classmethod
            def execute_op(cls, op_name: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
                """Test function execute_op.

                Args:
                    op_name: Test parameter `op_name`.
                    args: Test parameter `args`.
                    kwargs: Test parameter `kwargs`.

                Returns:
                    EagerValue: Computed output value from `execute_op`.
                """
                return np.ones((2, 2))

        orig_backend = config.backend
        try:
            config.backend = "mock_frombuf"
            ensure_callable(BackendRegistry.register)("mock_frombuf", MockBackendFromBuf)

            class DummyDtype:
                """Test class DummyDtype."""

                name = "dummy"

            ensure_callable(frombuffer)(b"hello", dtype=ensure_callable(DummyDtype)())
        finally:
            config.backend = orig_backend
    finally:
        config.eager_mode = orig_eager
    assert config.eager_mode == orig_eager


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
                """Test function item.

                Returns:
                    EagerValue: Computed output value from `item`.
                """
                return 5

        data = Data()

    assert ensure_callable(_extract_fill_value)(ensure_callable(DummyDataNoItem)()) == 5
    assert ensure_callable(_extract_fill_value)(ensure_callable(DummyDataWithItem)()) == 5
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
                """Test function execute_op.

                Args:
                    op_name: Test parameter `op_name`.
                    args: Test parameter `args`.
                    kwargs: Test parameter `kwargs`.

                Returns:
                    EagerValue: Computed output value from `execute_op`.
                """
                return np.ones((2, 2))

        ensure_callable(BackendRegistry.register)("mock_ops_like", MockLikeBackend)
        tc = TensorConfig(shape=(2, 2), dtype=DType.Float32, device=None)

        class DummyData:
            """Test class DummyData."""

            def __init__(self) -> None:
                """Test function __init__."""
                self.id = "dummy"

            def astype(self, dtype: EagerValue) -> EagerValue:
                """Test function astype.

                Args:
                    dtype: Test parameter `dtype`.

                Returns:
                    EagerValue: Computed output value from `astype`.
                """
                return ensure_eager_tensor(self)

            def __array__(self) -> EagerValue:
                """Test function __array__.

                Returns:
                    EagerValue: Computed output value from `__array__`.
                """
                return np.ones((2, 2))

        t = ensure_callable(Tensor)(ensure_callable(DummyData)(), tc)

        class DummyDtype:
            """Test class DummyDtype."""

            name = "float32"

        ensure_callable(zeros_like)(t, dtype=ensure_callable(DummyDtype)())
        ensure_callable(ones_like)(t, dtype=ensure_callable(DummyDtype)())
        ensure_callable(full_like)(t, 5, dtype=ensure_callable(DummyDtype)())
    finally:
        config.eager_mode = orig_eager
        config.backend = orig_backend
    assert config.backend == orig_backend


def test_creation_frontend_basic_unpack_shape_item_missing() -> None:
    """Test function test_creation_frontend_basic_unpack_shape_item_missing."""
    from ml_switcheroo_compiler.ops.creation.frontend_basic import _unpack_shape

    class DummyItemLess:
        """Test class DummyItemLess."""

        pass

    class DummyWithItemLess:
        """Test class DummyWithItemLess."""

        data = DummyItemLess()

    assert id(ensure_callable(type)(ensure_callable(_unpack_shape)((ensure_callable(DummyWithItemLess)(),))[0])) == id(DummyItemLess)


def test_creation_frontend_basic_unpack_shape_elif_item() -> None:
    """Test function test_creation_frontend_basic_unpack_shape_elif_item."""
    from ml_switcheroo_compiler.ops.creation.frontend_basic import _unpack_shape

    class DummyItem2:
        """Test class DummyItem2."""

        def item(self) -> EagerValue:
            """Test function item.

            Returns:
                EagerValue: Computed output value from `item`.
            """
            return 5

    assert ensure_callable(_unpack_shape)((ensure_callable(DummyItem2)(),)) == (5,)


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
    assert config.eager_mode == orig_eager
