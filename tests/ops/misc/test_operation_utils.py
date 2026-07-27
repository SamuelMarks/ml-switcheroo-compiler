# ruff: noqa: E501
import pytest

from ml_switcheroo_compiler.utils.operation_utils import _normalize_axes, _validate_squeeze_dims, compute_shape_propagation, get_source_inputs


class MockNode:
    def __init__(self, inputs=None):

        class Op:
            def __init__(self, ins):
                self.inputs = ins

        self.operation = Op(inputs or [])


class MockHistory:
    def __init__(self, inputs=None):
        self.node = MockNode(inputs)


class MockTensor:
    def __init__(self, inputs=None):
        if inputs is not None:
            self._keras_history = MockHistory(inputs)


def test_get_source_inputs():
    t1 = MockTensor()
    t2 = MockTensor()
    t3 = MockTensor([t1, t2])
    t4 = MockTensor([])
    assert get_source_inputs(t4) == [t4]
    assert get_source_inputs(t1) == [t1]
    assert get_source_inputs(t3) == [t1, t2]

    class Dummy:
        pass

    res = get_source_inputs(Dummy())
    assert len(res) == 1
    assert isinstance(res[0], Dummy)


def test_shape_inference_reshape():
    assert compute_shape_propagation("reshape", (2, 3), ("x", (6,)), {}) == (6,)


def test_shape_inference_transpose():
    assert compute_shape_propagation("transpose", (2, 3), ("x",), {}) == (3, 2)
    assert compute_shape_propagation("transpose", (2, 3), ("x",), {"axes": [1, 0]}) == (3, 2)
    assert compute_shape_propagation("transpose", (2, 3), ("x", [1, 0]), {}) == (3, 2)


def test_shape_inference_expand_dims():
    assert compute_shape_propagation("expand_dims", (2, 3), ("x",), {}) == (2, 3, 1)
    assert compute_shape_propagation("expand_dims", (2, 3), ("x", 0), {}) == (1, 2, 3)
    assert compute_shape_propagation("expand_dims", (2, 3), ("x",), {"axis": 1}) == (2, 1, 3)


def test_normalize_axes():
    assert _normalize_axes(1, 3) == [1]
    assert _normalize_axes(-1, 3) == [2]


def test_validate_squeeze_dims():
    _validate_squeeze_dims((2, 1, 3), [1])
    with pytest.raises(ValueError):
        _validate_squeeze_dims((2, 1, 3), [4])
    with pytest.raises(ValueError):
        _validate_squeeze_dims((2, 1, 3), [0])


def test_shape_inference_squeeze():
    assert compute_shape_propagation("squeeze", (2, 1, 3), ("x",), {}) == (2, 3)
    assert compute_shape_propagation("squeeze", (2, 1, 3), ("x", 1), {}) == (2, 3)
    assert compute_shape_propagation("squeeze", (2, 1, 3), ("x",), {"axis": [1]}) == (2, 3)


def test_shape_inference_split():
    assert compute_shape_propagation("split", (4, 6), ("x", 2), {}) == [(2, 6), (2, 6)]
    assert compute_shape_propagation("split", (4, 6), ("x", 2, 1), {}) == [(4, 3), (4, 3)]
    assert compute_shape_propagation("split", (4, 6), ("x", 2), {"axis": 1}) == [(4, 3), (4, 3)]
    assert compute_shape_propagation("split", (None, 6), ("x", 2), {}) == [(None, 6), (None, 6)]
    assert compute_shape_propagation("split", (4, 6), ("x", [2, 2]), {}) == (4, 6)


def test_shape_inference_mean():
    assert compute_shape_propagation("mean", (2, 3), ("x",), {}) == (2, 3)
    assert compute_shape_propagation("mean", (2, 3), ("x", 0), {}) == (3,)
    assert compute_shape_propagation("mean", (2, 3), ("x",), {"axis": 1}) == (2,)
    assert compute_shape_propagation("mean", (2, 3), ("x", 0), {"keepdims": True}) == (1, 3)
    assert compute_shape_propagation("mean", (2, 3), ("x",), {"dtype": "float32"}) == (2, 3)
    assert compute_shape_propagation("mean", (2, 3), ("x",), {"dtype": None}) == (2, 3)
    pass


def test_unregistered_inference():
    assert compute_shape_propagation("unknown", (2, 3), (), {}) == (2, 3)


def test_dtype_promotion_coverage():
    from ml_switcheroo_compiler.utils.operation_utils import compute_shape_propagation

    compute_shape_propagation("mean", (2, 3), ("x",), {"dtype": 123})


def test_mean_negative_axis():
    from ml_switcheroo_compiler.utils.operation_utils import compute_shape_propagation

    compute_shape_propagation("mean", (2, 3), ("x",), {"axis": -1})
    compute_shape_propagation("mean", (2, 3), ("x", -1), {})


def test_resolve_axis_none():
    from ml_switcheroo_compiler.utils.operation_utils import MeanInference

    assert MeanInference()._resolve_axis(None, 2) == set()
