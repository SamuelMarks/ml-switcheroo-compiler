# ruff: noqa: E501
import numpy as np

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops import block, delete, diag_indices, diag_indices_from, diagflat, fill_diagonal, insert
from ml_switcheroo_compiler.ops.shape.pad_and_tile import DynamicShape, Pad, Rank, Size, _compute_pad_dim, _normalize_pad_width, argsort, image_resize, meshgrid, pad, pad_circular, pad_constant, pad_reflect, pad_replicate, repeat, sort, tile, top_k, tril, triu
from ml_switcheroo_compiler.tracing.state import global_tracing_state

"Tests for shape ops."


def _t(data: object, shape: tuple) -> Tensor:
    """Helper to create test tensor."""
    return Tensor(data, TensorConfig(shape, "float32", "cpu"))


def test_block_eager() -> None:
    """Test eager."""
    config.eager_mode = True
    x = _t(np.array([1, 2]), (2,))
    y = _t(np.array([3, 4]), (2,))
    out = block([x, y])
    assert getattr(out, "shape", getattr(out, "data", out).shape) == (4,)


def test_block_tracing() -> None:
    """Test tracing."""
    config.eager_mode = False
    global_tracing_state.start_tracing()
    try:
        x = _t(np.array([1, 2]), (2,))
        y = _t(np.array([3, 4]), (2,))
        out = block([x, y])
        assert out.shape == ()
    finally:
        global_tracing_state.stop_tracing()


def test_delete_eager() -> None:
    """Test eager."""
    config.eager_mode = True
    x = _t(np.array([1, 2, 3]), (3,))
    out = delete(x, 0)
    assert getattr(out, "shape", getattr(out, "data", out).shape) == (2,)


def test_delete_tracing() -> None:
    """Test tracing."""
    config.eager_mode = False
    global_tracing_state.start_tracing()
    try:
        x = _t(np.array([1, 2, 3]), (3,))
        out = delete(x, 0)
        assert out.shape == (3,)
    finally:
        global_tracing_state.stop_tracing()


def test_diag_indices_eager() -> None:
    """Test eager."""
    config.eager_mode = True
    out = diag_indices(4)
    assert len(out) == 2
    assert out[0].shape == (4,)


def test_diag_indices_tracing() -> None:
    """Test tracing."""
    config.eager_mode = False
    global_tracing_state.start_tracing()
    try:
        out = diag_indices(4)
        assert len(out) == 2
        assert out[0].shape == (4,)
    finally:
        global_tracing_state.stop_tracing()


def test_diag_indices_from_eager() -> None:
    """Test eager."""
    config.eager_mode = True
    x = _t(np.array([[1, 2], [3, 4]]), (2, 2))
    out = diag_indices_from(x)
    assert len(out) == 2
    assert out[0].shape == (2,)


def test_diag_indices_from_tracing() -> None:
    """Test tracing."""
    config.eager_mode = False
    global_tracing_state.start_tracing()
    try:
        x = _t(np.array([[1, 2], [3, 4]]), (2, 2))
        out = diag_indices_from(x)
        assert len(out) == 2
        assert out[0].shape == (2,)
    finally:
        global_tracing_state.stop_tracing()


def test_diagflat_eager() -> None:
    """Test eager."""
    config.eager_mode = True
    x = _t(np.array([1, 2]), (2,))
    out = diagflat(x)
    assert getattr(out, "shape", getattr(out, "data", out).shape) == (2, 2)


def test_diagflat_tracing() -> None:
    """Test tracing."""
    config.eager_mode = False
    global_tracing_state.start_tracing()
    try:
        x = _t(np.array([1, 2]), (2,))
        out = diagflat(x)
        assert out.shape == (2, 2)
    finally:
        global_tracing_state.stop_tracing()


def test_fill_diagonal_eager() -> None:
    """Test eager."""
    config.eager_mode = True
    x = _t(np.array([[1, 2], [3, 4]]), (2, 2))
    v = _t(np.array(5), ())
    out = fill_diagonal(x, v)
    assert getattr(out, "shape", getattr(out, "data", out).shape) == (2, 2)


def test_fill_diagonal_tracing() -> None:
    """Test tracing."""
    config.eager_mode = False
    global_tracing_state.start_tracing()
    try:
        x = _t(np.array([[1, 2], [3, 4]]), (2, 2))
        v = _t(np.array(5), ())
        out = fill_diagonal(x, v)
        assert out.shape == (2, 2)
    finally:
        global_tracing_state.stop_tracing()


def test_insert_eager() -> None:
    """Test eager."""
    config.eager_mode = True
    x = _t(np.array([1, 3]), (2,))
    v = _t(np.array([2]), (1,))
    out = insert(x, 1, v)
    assert getattr(out, "shape", getattr(out, "data", out).shape) == (3,)


def test_insert_tracing() -> None:
    """Test tracing."""
    config.eager_mode = False
    global_tracing_state.start_tracing()
    try:
        x = _t(np.array([1, 3]), (2,))
        v = _t(np.array([2]), (1,))
        out = insert(x, 1, v)
        assert out.shape == (2,)
    finally:
        global_tracing_state.stop_tracing()


class MockTensor:
    def __init__(self, shape=()):
        self.shape = shape
        self.dtype = "float32"
        self.device = "cpu"
        self.data = np.zeros(shape)


def test_tile(mocker):
    t = Tensor(MockTensor((2, 3)).data, TensorConfig((2, 3), "float32", "cpu"))
    config.eager_mode = False
    mocker.patch("ml_switcheroo_compiler.ops.shape.pad_and_tile._emit_shape_node", return_value="tile")
    assert tile(t, [2, 2]) == "tile"
    config.eager_mode = True
    mock_backend = mocker.patch("ml_switcheroo_compiler.ops.shape.pad_and_tile.get_active_backend").return_value
    mock_backend.execute_op.return_value = MockTensor((4, 6))
    mock_backend.array.side_effect = lambda x: x
    assert tile(t, [2, 2]).config.shape == (4, 6)


def test_repeat(mocker):
    t = Tensor(MockTensor((2, 3)).data, TensorConfig((2, 3), "float32", "cpu"))
    config.eager_mode = False
    mocker.patch("ml_switcheroo_compiler.ops.shape.pad_and_tile._emit_shape_node", return_value="repeat")
    assert repeat(t, 2) == "repeat"
    config.eager_mode = True
    mock_backend = mocker.patch("ml_switcheroo_compiler.ops.shape.pad_and_tile.get_active_backend").return_value
    mock_backend.execute_op.return_value = MockTensor((4, 3))
    mock_backend.array.side_effect = lambda x: x
    assert repeat(t, 2).config.shape == (4, 3)


def test_triu(mocker):
    t = Tensor(MockTensor((2, 3)).data, TensorConfig((2, 3), "float32", "cpu"))
    config.eager_mode = False
    mocker.patch("ml_switcheroo_compiler.ops.shape.pad_and_tile._emit_shape_node", return_value="triu")
    assert triu(t) == "triu"
    config.eager_mode = True
    mock_backend = mocker.patch("ml_switcheroo_compiler.ops.shape.pad_and_tile.get_active_backend").return_value
    mock_backend.execute_op.return_value = MockTensor((2, 3))
    mock_backend.array.side_effect = lambda x: x
    assert triu(t).config.shape == (2, 3)


def test_tril(mocker):
    t = Tensor(MockTensor((2, 3)).data, TensorConfig((2, 3), "float32", "cpu"))
    config.eager_mode = False
    mocker.patch("ml_switcheroo_compiler.ops.shape.pad_and_tile._emit_shape_node", return_value="tril")
    assert tril(t) == "tril"
    config.eager_mode = True
    mock_backend = mocker.patch("ml_switcheroo_compiler.ops.shape.pad_and_tile.get_active_backend").return_value
    mock_backend.execute_op.return_value = MockTensor((2, 3))
    mock_backend.array.side_effect = lambda x: x
    assert tril(t).config.shape == (2, 3)


def test_meshgrid(mocker):
    t = Tensor(MockTensor((2,)).data, TensorConfig((2,), "float32", "cpu"))
    config.eager_mode = False
    mocker.patch("ml_switcheroo_compiler.ops.shape.pad_and_tile._emit_shape_node", return_value=["mesh1"])
    assert meshgrid(t, t)[0] == ["mesh1"]
    config.eager_mode = True
    mock_backend = mocker.patch("ml_switcheroo_compiler.ops.shape.pad_and_tile.get_active_backend").return_value
    mock_backend.execute_op.return_value = [MockTensor((2, 2)), MockTensor((2, 2))]
    mock_backend.array.side_effect = lambda x: x
    res = meshgrid(t, t)
    assert res[0].config.shape == (2, 2)
    assert res[1].config.shape == (2, 2)


def test_pad_helpers():
    assert _normalize_pad_width([(1, 2)], 1) == ((1, 2),)
    assert _normalize_pad_width((1, 2), 2) == ((1, 2), (1, 2))
    assert _normalize_pad_width(1, 2) == ((1, 1), (1, 1))
    assert _compute_pad_dim(5, (1, 2)) == 8


def test_pad_class():
    op = Pad()
    assert op.infer_shape(MockTensor((2, 3)), ((1, 1), (2, 2))) == (4, 7)
    assert op.infer_shape(None, ((1, 1), (2, 2))) == ()


def test_pad_func(mocker):
    t = Tensor(MockTensor((2, 3)).data, TensorConfig((2, 3), "float32", "cpu"))
    config.eager_mode = False
    mocker.patch("ml_switcheroo_compiler.ops.shape.pad_and_tile._emit_shape_node", return_value="pad")
    assert pad(t, 1) == "pad"
    config.eager_mode = True
    mock_backend = mocker.patch("ml_switcheroo_compiler.ops.shape.pad_and_tile.get_active_backend").return_value
    mock_backend.execute_op.return_value = MockTensor((4, 5))
    mock_backend.array.side_effect = lambda x: x
    assert pad(t, 1).config.shape == (4, 5)


def test_top_k(mocker):
    t = Tensor(MockTensor((2, 3)).data, TensorConfig((2, 3), "float32", "cpu"))
    config.eager_mode = False
    mocker.patch("ml_switcheroo_compiler.ops.shape.pad_and_tile._emit_shape_node", return_value=("val", "idx"))
    assert top_k(t, 2) == (("val", "idx"), ("val", "idx"))
    config.eager_mode = True
    mock_backend = mocker.patch("ml_switcheroo_compiler.ops.shape.pad_and_tile.get_active_backend").return_value
    mock_backend.execute_op.return_value = [MockTensor((2, 2)), MockTensor((2, 2))]
    mock_backend.array.side_effect = lambda x: x
    res = top_k(t, 2)
    assert res[0].config.shape == (2, 2)
    assert res[1].config.shape == (2, 2)


def test_argsort(mocker):
    t = Tensor(MockTensor((2, 3)).data, TensorConfig((2, 3), "float32", "cpu"))
    config.eager_mode = False
    mocker.patch("ml_switcheroo_compiler.ops.shape.pad_and_tile._emit_shape_node", return_value="argsort")
    assert argsort(t) == "argsort"
    config.eager_mode = True
    mock_backend = mocker.patch("ml_switcheroo_compiler.ops.shape.pad_and_tile.get_active_backend").return_value
    mock_backend.execute_op.return_value = MockTensor((2, 3))
    mock_backend.array.side_effect = lambda x: x
    mock_backend.array.side_effect = lambda x: x
    assert argsort(t).config.shape == (2, 3)


def test_sort(mocker):
    t = Tensor(MockTensor((2, 3)).data, TensorConfig((2, 3), "float32", "cpu"))
    config.eager_mode = False
    mocker.patch("ml_switcheroo_compiler.ops.shape.pad_and_tile._emit_shape_node", return_value="sort")
    assert sort(t) == "sort"
    config.eager_mode = True
    mock_backend = mocker.patch("ml_switcheroo_compiler.ops.shape.pad_and_tile.get_active_backend").return_value
    mock_backend.execute_op.return_value = MockTensor((2, 3))
    mock_backend.array.side_effect = lambda x: x
    mock_backend.array.side_effect = lambda x: x
    assert sort(t).config.shape == (2, 3)


def test_image_resize(mocker):
    t = Tensor(MockTensor((2, 3, 4, 3)).data, TensorConfig((2, 3, 4, 3), "float32", "cpu"))
    config.eager_mode = False
    mocker.patch("ml_switcheroo_compiler.ops.shape.pad_and_tile._emit_shape_node", return_value="resize")
    assert image_resize(t, (5, 6)) == "resize"
    config.eager_mode = True
    mock_backend = mocker.patch("ml_switcheroo_compiler.backends.registry.get_active_backend").return_value
    mock_backend.execute_op.return_value = MockTensor((2, 5, 6, 3))
    mock_array = mocker.MagicMock()
    mock_array.shape = (2, 5, 6, 3)
    mock_backend.array.return_value = mock_array
    assert image_resize(t, (5, 6)).config.shape == (2, 5, 6, 3)


def test_dynamic_shape_infer_shape():
    op = DynamicShape()
    assert op.infer_shape(MockTensor((2, 3))) == (2,)


def test_rank_infer_shape():
    op = Rank()
    assert op.infer_shape(MockTensor((2, 3))) == ()


def test_size_infer_shape():
    op = Size()
    assert op.infer_shape(MockTensor((2, 3))) == ()


def test_pad_modes(mocker):
    mocker.patch("ml_switcheroo_compiler.ops.shape.pad_and_tile.pad", return_value="padded")
    assert pad_constant("arr", "pw") == "padded"
    assert pad_reflect("arr", "pw") == "padded"
    assert pad_replicate("arr", "pw") == "padded"
    assert pad_circular("arr", "pw") == "padded"


def test_compute_meshgrid_shape():
    from ml_switcheroo_compiler.ops.shape.pad_and_tile import _compute_meshgrid_shape

    t1 = MockTensor((2,))
    t2 = MockTensor((3,))
    assert _compute_meshgrid_shape([t1], "ij") == (2,)
    assert _compute_meshgrid_shape([], "ij") == ()
    assert _compute_meshgrid_shape([t1, t2], "xy") == (3, 2)


def test_compute_pad_dim_extra():
    assert _compute_pad_dim(5, 2) == 9
    assert _compute_pad_dim(5, "invalid") == 5


def test_pad_infer_shape_short_pw():
    op = Pad()
    assert op.infer_shape(MockTensor((2, 3)), ((1, 1),)) == (4, 3)


def test_top_k_scalar(mocker):
    t = Tensor(MockTensor(()).data, TensorConfig((), "float32", "cpu"))
    config.eager_mode = False
    mocker.patch("ml_switcheroo_compiler.ops.shape.pad_and_tile._emit_shape_node", return_value=("val", "idx"))
    assert top_k(t, 2) == (("val", "idx"), ("val", "idx"))


def test_argsort_axis_dim(mocker):
    t = Tensor(MockTensor((2, 3)).data, TensorConfig((2, 3), "float32", "cpu"))
    config.eager_mode = False
    mocker.patch("ml_switcheroo_compiler.ops.shape.pad_and_tile._emit_shape_node", return_value="argsort")
    assert argsort(t, axis=0, dim=0) == "argsort"


def test_sort_axis_dim(mocker):
    t = Tensor(MockTensor((2, 3)).data, TensorConfig((2, 3), "float32", "cpu"))
    config.eager_mode = False
    mocker.patch("ml_switcheroo_compiler.ops.shape.pad_and_tile._emit_shape_node", return_value="sort")
    assert sort(t, axis=0, dim=0) == "sort"
