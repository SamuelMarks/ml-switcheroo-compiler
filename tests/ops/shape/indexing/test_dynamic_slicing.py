import numpy as np

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.ops.shape.dynamic_slicing import DynamicSlice, DynamicUpdateSlice, dynamic_slice, dynamic_update_slice, update_slice


class MockTensor:
    def __init__(self, shape):
        self.shape = shape


def test_dynamic_slice_eager():
    config.eager_mode = True

    class FakeTensor:
        def __init__(self, data):
            self.data = np.array(data)
            self.shape = self.data.shape
            self.dtype = "float32"
            self.device = "cpu"

    t = FakeTensor([1, 2, 3, 4, 5])

    class FakeStart:
        def __init__(self, d):
            self.data = d

    res = dynamic_slice(t, [FakeStart(1)], [2])
    assert res.config.shape == (2,)
    assert list(res.data) == [2, 3]

    res2 = dynamic_slice(t, [3], [2])
    assert res2.config.shape == (2,)
    assert list(res2.data) == [4, 5]

    config.eager_mode = False


def test_update_slice(mocker):
    mocker.patch("ml_switcheroo_compiler.ops.shape.dynamic_slicing.dynamic_update_slice", return_value="updated")
    mocker.patch("ml_switcheroo_compiler.ops.creation.frontend_basic.array", side_effect=lambda x: f"array({x})")

    t = MockTensor((5,))
    update = MockTensor((2,))

    res = update_slice(t, update, [1, MockTensor(())])
    assert res == "updated"


def test_dynamic_slice_infer_shape():
    op = DynamicSlice()

    # Args
    assert op.infer_shape("x", "starts", [2, 3]) == (2, 3)

    # Kwargs
    assert op.infer_shape(slice_sizes=[4, 5]) == (4, 5)


def test_dynamic_update_slice_infer_shape():
    op = DynamicUpdateSlice()

    t = MockTensor((2, 3))
    assert op.infer_shape(t, "update", "starts") == (2, 3)

    assert op.infer_shape("not_tensor", "update", "starts") == ()


def test_dynamic_slice_tracing(mocker):
    config.eager_mode = False
    mocker.patch("ml_switcheroo_compiler.ops.shape.dynamic_slicing._emit_shape_node", return_value="sliced")
    t = MockTensor((5,))
    t.dtype = "float32"
    res = dynamic_slice(t, [1], [2])
    assert res == "sliced"


def test_dynamic_update_slice_tracing(mocker):
    config.eager_mode = False
    mocker.patch("ml_switcheroo_compiler.ops.shape.dynamic_slicing._emit_shape_node", return_value="updated")
    t = MockTensor((5,))
    t.dtype = "float32"
    res = dynamic_update_slice(t, MockTensor((2,)), [1])
    assert res == "updated"
