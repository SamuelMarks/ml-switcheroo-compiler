from unittest.mock import MagicMock, patch

from ml_switcheroo_compiler.backends.eager import (
    execute_generic_op,
    generic_array,
    generic_asarray,
    generic_item,
    generic_zeros,
)


def test_execute_generic_op_coverage():
    b = MagicMock()

    # BroadcastTo missing shape in kwargs
    # op_type == "BroadcastTo"
    del b.broadcastto
    b.broadcast_to.return_value = "bto"
    assert execute_generic_op(b, "BroadcastTo", "x", shape="shape_arg") == "bto"
    assert execute_generic_op(b, "BroadcastTo", "x") == "bto"  # len(args) == 1

    # We must raise AttributeError when `b.zeros(*args, **kwargs)` is called, so it hits the fallback block.
    # Zeros
    b.zeros.side_effect = [TypeError, "zeros_res"]
    assert execute_generic_op(b, "Zeros", (2,)) == "zeros_res"

    b.zeros.side_effect = [TypeError, TypeError, "zeros_res2"]
    assert execute_generic_op(b, "Zeros", (2,)) == "zeros_res2"

    # with shape.data
    shape_mock = MagicMock()
    shape_mock.data = (2,)
    b.zeros.side_effect = [TypeError, TypeError, "zeros_res3"]
    assert execute_generic_op(b, "Zeros", shape_mock) == "zeros_res3"

    # Ones
    b.ones.side_effect = [TypeError, "ones_res"]
    assert execute_generic_op(b, "Ones", (2,)) == "ones_res"

    b.ones.side_effect = [TypeError, TypeError, "ones_res2"]
    assert execute_generic_op(b, "Ones", (2,)) == "ones_res2"

    b.ones.side_effect = [TypeError, TypeError, "ones_res3"]
    assert execute_generic_op(b, "Ones", shape_mock) == "ones_res3"

    # Full
    b.full.side_effect = [TypeError, "full_res"]
    assert execute_generic_op(b, "Full", (2,), 5) == "full_res"

    b.full.side_effect = [TypeError, TypeError, "full_res2"]
    assert execute_generic_op(b, "Full", (2,), 5) == "full_res2"

    b.full.side_effect = [TypeError, TypeError, "full_res3"]
    assert execute_generic_op(b, "Full", shape_mock, 5) == "full_res3"

    # BroadcastInDim
    b.broadcastindim.side_effect = AttributeError
    b.reshape.return_value = "reshaped"
    b.broadcast_to.return_value = "broadcast_to_res"
    x_mock = MagicMock()
    x_mock.shape = (2, 3)
    assert execute_generic_op(b, "BroadcastInDim", x_mock, (2, 3, 4), [0, 1]) == "broadcast_to_res"

    # Check generic_array with mlx core array string
    class MLXArray:
        pass

    MLXArray.__name__ = "mlx.core.array"
    assert generic_array(b, MLXArray()) is not None

    # Resize
    b.resize.side_effect = AttributeError
    b.zeros.side_effect = None
    b.zeros.return_value = "resize_res"
    x_mock = MagicMock()
    x_mock.shape = (1, 2, 3)
    assert execute_generic_op(b, "Resize", x_mock, (4,)) == "resize_res"

    # DynamicUpdateSlice
    b.dynamicupdateslice.side_effect = AttributeError
    del b.dynamic_update_slice
    assert execute_generic_op(b, "DynamicUpdateSlice", "x", "update") == "x"
    b.dynamic_update_slice = MagicMock(return_value="dus_res")
    assert execute_generic_op(b, "DynamicUpdateSlice", "x", "update") == "dus_res"

    # ConvGeneralDilated, Psum, Pmean, SegmentSum
    b.convgeneraldilated.side_effect = AttributeError
    b.zeros.return_value = "conv_res"
    assert execute_generic_op(b, "ConvGeneralDilated", 1) == "conv_res"

    b.segmentsum.side_effect = AttributeError
    assert execute_generic_op(b, "SegmentSum", 1) == "conv_res"

    b.psum.side_effect = AttributeError
    assert execute_generic_op(b, "Psum", "p") == "p"

    b.pmean.side_effect = AttributeError
    assert execute_generic_op(b, "Pmean", "p") == "p"


def test_generic_utils():
    b = MagicMock()
    b.zeros.return_value = "zeros"
    assert generic_zeros(b, (1,)) == "zeros"

    b.array.return_value = "array"
    assert generic_array(b, [1]) == "array"
    assert generic_array(b, None) is None

    class MLXArray:
        pass

    MLXArray.__name__ = "mlx.core"

    class MLXArray2:
        pass

    a2 = MLXArray2()

    # Mocking type str
    class FakeType:
        def __str__(self):
            return "mlx.core.array"

    with patch("builtins.type", return_value=FakeType()):
        assert generic_array(b, a2) == a2

    del b.array
    b.convert_to_tensor.return_value = "tensor"
    assert generic_array(b, [1]) == "tensor"

    b.asarray.return_value = "asarray"
    assert generic_asarray(b, [1]) == "asarray"
    del b.asarray
    assert generic_asarray(b, [1]) == "tensor"

    b.asarray = MagicMock()
    b.asarray.return_value.item.return_value = 1.0
    assert generic_item(b, 1) == 1.0
    del b.asarray
    assert generic_item(b, 2.0) == 2.0


def test_shape_parsing_helpers():
    from unittest.mock import MagicMock

    from ml_switcheroo_compiler.backends.eager import (
        _extract_shape_value,
        _normalize_shape,
        _parse_eager_shape,
    )

    # _normalize_shape
    m1 = MagicMock()
    m1.data = (1, 2)
    assert _normalize_shape(m1) == [1, 2]

    m2 = MagicMock()
    del m2.data
    m2.tolist.return_value = [3, 4]
    assert _normalize_shape(m2) == [3, 4]

    assert _normalize_shape((5, 6)) == [5, 6]

    # _extract_shape_value
    v1 = MagicMock()
    v1.data = 7
    assert _extract_shape_value(v1) == 7

    v2 = MagicMock()
    del v2.data
    v2.item.return_value = 8
    assert _extract_shape_value(v2) == 8

    v3 = MagicMock()
    del v3.data
    del v3.item
    v3.tolist.return_value = [9]
    assert _extract_shape_value(v3) == 9

    v4 = MagicMock()
    del v4.data
    del v4.item
    v4.tolist.return_value = 10
    assert _extract_shape_value(v4) == 10

    # _parse_eager_shape
    assert _parse_eager_shape(None) is None
    assert _parse_eager_shape([]) == []
    assert _parse_eager_shape([1, 2]) == [1, 2]
