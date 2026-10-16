"""Tests for test_diag_and_block."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

# ruff: noqa: E501
import numpy as np

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops import block, delete, diag_indices, diag_indices_from, diagflat, fill_diagonal, insert
from ml_switcheroo_compiler.ops.shape.pad_and_tile import DynamicShape, Pad, Rank, Size, _compute_pad_dim, _normalize_pad_width, argsort, image_resize, meshgrid, pad, pad_circular, pad_constant, pad_reflect, pad_replicate, repeat, sort, tile, top_k, tril, triu
from ml_switcheroo_compiler.tracing.state import global_tracing_state

if TYPE_CHECKING:
    from collections.abc import Callable

    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue


def _t(data: EagerValue, shape: tuple[EagerValue, ...]) -> Tensor:
    """Helper to create test tensor."""
    return cast("Tensor", cast("Callable[..., EagerTensorProtocol]", Tensor)(data, cast("Callable[..., EagerTensorProtocol]", TensorConfig)(shape, "float32", "cpu")))


def test_block_eager() -> None:
    """Test eager."""
    config.eager_mode = True
    x = _t(np.array([1, 2]), (2,))
    y = _t(np.array([3, 4]), (2,))
    out = block([x, y])
    assert getattr(out, "shape", cast("EagerTensorProtocol", getattr(getattr(out, "data", out), "shape"))) == (4,)


def test_block_tracing() -> None:
    """Test tracing."""
    config.eager_mode = False
    global_tracing_state.start_tracing()
    try:
        x = _t(np.array([1, 2]), (2,))
        y = _t(np.array([3, 4]), (2,))
        out = block([x, y])
        assert cast("EagerTensorProtocol", getattr(out, "shape")) == ()
    finally:
        global_tracing_state.stop_tracing()


def test_delete_eager() -> None:
    """Test eager."""
    config.eager_mode = True
    x = _t(np.array([1, 2, 3]), (3,))
    out = delete(x, 0)
    assert getattr(out, "shape", cast("EagerTensorProtocol", getattr(getattr(out, "data", out), "shape"))) == (2,)


def test_delete_tracing() -> None:
    """Test tracing."""
    config.eager_mode = False
    global_tracing_state.start_tracing()
    try:
        x = _t(np.array([1, 2, 3]), (3,))
        out = delete(x, 0)
        assert cast("EagerTensorProtocol", getattr(out, "shape")) == (3,)
    finally:
        global_tracing_state.stop_tracing()


def test_diag_indices_eager() -> None:
    """Test eager."""
    config.eager_mode = True
    out = diag_indices(4)
    assert cast("Callable[..., EagerTensorProtocol]", len)(out) == 2
    assert cast("EagerTensorProtocol", getattr(cast("EagerTensorProtocol", getattr(out, "__getitem__")(0)), "shape")) == (4,)


def test_diag_indices_tracing() -> None:
    """Test tracing."""
    config.eager_mode = False
    global_tracing_state.start_tracing()
    try:
        out = diag_indices(4)
        assert cast("Callable[..., EagerTensorProtocol]", len)(out) == 2
        assert cast("EagerTensorProtocol", getattr(cast("EagerTensorProtocol", getattr(out, "__getitem__")(0)), "shape")) == (4,)
    finally:
        global_tracing_state.stop_tracing()


def test_diag_indices_from_eager() -> None:
    """Test eager."""
    config.eager_mode = True
    x = _t(np.array([[1, 2], [3, 4]]), (2, 2))
    out = diag_indices_from(x)
    assert cast("Callable[..., EagerTensorProtocol]", len)(out) == 2
    assert cast("EagerTensorProtocol", getattr(cast("EagerTensorProtocol", getattr(out, "__getitem__")(0)), "shape")) == (2,)


def test_diag_indices_from_tracing() -> None:
    """Test tracing."""
    config.eager_mode = False
    global_tracing_state.start_tracing()
    try:
        x = _t(np.array([[1, 2], [3, 4]]), (2, 2))
        out = diag_indices_from(x)
        assert cast("Callable[..., EagerTensorProtocol]", len)(out) == 2
        assert cast("EagerTensorProtocol", getattr(cast("EagerTensorProtocol", getattr(out, "__getitem__")(0)), "shape")) == (2,)
    finally:
        global_tracing_state.stop_tracing()


def test_diagflat_eager() -> None:
    """Test eager."""
    config.eager_mode = True
    x = _t(np.array([1, 2]), (2,))
    out = diagflat(x)
    assert getattr(out, "shape", cast("EagerTensorProtocol", getattr(getattr(out, "data", out), "shape"))) == (2, 2)


def test_diagflat_tracing() -> None:
    """Test tracing."""
    config.eager_mode = False
    global_tracing_state.start_tracing()
    try:
        x = _t(np.array([1, 2]), (2,))
        out = diagflat(x)
        assert cast("EagerTensorProtocol", getattr(out, "shape")) == (2, 2)
    finally:
        global_tracing_state.stop_tracing()


def test_fill_diagonal_eager() -> None:
    """Test eager."""
    config.eager_mode = True
    x = _t(np.array([[1, 2], [3, 4]]), (2, 2))
    v = _t(np.array(5), ())
    out = fill_diagonal(x, v)
    assert getattr(out, "shape", cast("EagerTensorProtocol", getattr(getattr(out, "data", out), "shape"))) == (2, 2)


def test_fill_diagonal_tracing() -> None:
    """Test tracing."""
    config.eager_mode = False
    global_tracing_state.start_tracing()
    try:
        x = _t(np.array([[1, 2], [3, 4]]), (2, 2))
        v = _t(np.array(5), ())
        out = fill_diagonal(x, v)
        assert cast("EagerTensorProtocol", getattr(out, "shape")) == (2, 2)
    finally:
        global_tracing_state.stop_tracing()


def test_insert_eager() -> None:
    """Test eager."""
    config.eager_mode = True
    x = _t(np.array([1, 3]), (2,))
    v = _t(np.array([2]), (1,))
    out = insert(x, 1, v)
    assert getattr(out, "shape", cast("EagerTensorProtocol", getattr(getattr(out, "data", out), "shape"))) == (3,)


def test_insert_tracing() -> None:
    """Test tracing."""
    config.eager_mode = False
    global_tracing_state.start_tracing()
    try:
        x = _t(np.array([1, 3]), (2,))
        v = _t(np.array([2]), (1,))
        out = insert(x, 1, v)
        assert cast("EagerTensorProtocol", getattr(out, "shape")) == (2,)
    finally:
        global_tracing_state.stop_tracing()


class MockTensor:
    """Test class MockTensor."""

    def __init__(self, shape: EagerValue = ()) -> None:
        """Test function __init__."""
        self.shape = shape
        self.dtype = "float32"
        self.device = "cpu"
        self.data = cast("Callable[..., EagerTensorProtocol]", np.zeros)(shape)


def test_tile(mocker: EagerTensorProtocol) -> None:
    """Test function test_tile."""
    t = Tensor(MockTensor((2, 3)).data, TensorConfig((2, 3), "float32", "cpu"))
    config.eager_mode = False
    mocker.patch("ml_switcheroo_compiler.ops.shape.pad_and_tile._emit_shape_node", return_value="tile")
    assert tile(t, [2, 2]) == "tile"
    config.eager_mode = True
    mock_backend = mocker.patch("ml_switcheroo_compiler.ops.shape.pad_and_tile.get_active_backend").return_value
    setattr(mock_backend.execute_op, "return_value", MockTensor((4, 6)))
    setattr(mock_backend.array, "side_effect", lambda x: x)
    assert cast("EagerTensorProtocol", getattr(tile(t, [2, 2]), "config")).shape == (4, 6)


def test_repeat(mocker: EagerTensorProtocol) -> None:
    """Test function test_repeat."""
    t = Tensor(MockTensor((2, 3)).data, TensorConfig((2, 3), "float32", "cpu"))
    config.eager_mode = False
    mocker.patch("ml_switcheroo_compiler.ops.shape.pad_and_tile._emit_shape_node", return_value="repeat")
    assert repeat(t, 2) == "repeat"
    config.eager_mode = True
    mock_backend = mocker.patch("ml_switcheroo_compiler.ops.shape.pad_and_tile.get_active_backend").return_value
    setattr(mock_backend.execute_op, "return_value", MockTensor((4, 3)))
    setattr(mock_backend.array, "side_effect", lambda x: x)
    assert cast("EagerTensorProtocol", getattr(repeat(t, 2), "config")).shape == (4, 3)


def test_triu(mocker: EagerTensorProtocol) -> None:
    """Test function test_triu."""
    t = Tensor(MockTensor((2, 3)).data, TensorConfig((2, 3), "float32", "cpu"))
    config.eager_mode = False
    mocker.patch("ml_switcheroo_compiler.ops.shape.pad_and_tile._emit_shape_node", return_value="triu")
    assert triu(t) == "triu"
    config.eager_mode = True
    mock_backend = mocker.patch("ml_switcheroo_compiler.ops.shape.pad_and_tile.get_active_backend").return_value
    setattr(mock_backend.execute_op, "return_value", MockTensor((2, 3)))
    setattr(mock_backend.array, "side_effect", lambda x: x)
    assert cast("EagerTensorProtocol", getattr(triu(t), "config")).shape == (2, 3)


def test_tril(mocker: EagerTensorProtocol) -> None:
    """Test function test_tril."""
    t = Tensor(MockTensor((2, 3)).data, TensorConfig((2, 3), "float32", "cpu"))
    config.eager_mode = False
    mocker.patch("ml_switcheroo_compiler.ops.shape.pad_and_tile._emit_shape_node", return_value="tril")
    assert tril(t) == "tril"
    config.eager_mode = True
    mock_backend = mocker.patch("ml_switcheroo_compiler.ops.shape.pad_and_tile.get_active_backend").return_value
    setattr(mock_backend.execute_op, "return_value", MockTensor((2, 3)))
    setattr(mock_backend.array, "side_effect", lambda x: x)
    assert cast("EagerTensorProtocol", getattr(tril(t), "config")).shape == (2, 3)


def test_meshgrid(mocker: EagerTensorProtocol) -> None:
    """Test function test_meshgrid."""
    t = Tensor(MockTensor((2,)).data, TensorConfig((2,), "float32", "cpu"))
    config.eager_mode = False
    mocker.patch("ml_switcheroo_compiler.ops.shape.pad_and_tile._emit_shape_node", return_value=["mesh1"])
    assert cast("EagerTensorProtocol", meshgrid(t, t)[0]) == cast("EagerTensorProtocol", ["mesh1"])
    config.eager_mode = True
    mock_backend = mocker.patch("ml_switcheroo_compiler.ops.shape.pad_and_tile.get_active_backend").return_value
    setattr(mock_backend.execute_op, "return_value", [MockTensor((2, 2)), MockTensor((2, 2))])
    setattr(mock_backend.array, "side_effect", lambda x: x)
    res = meshgrid(t, t)
    assert res[0].config.shape == (2, 2)
    assert res[1].config.shape == (2, 2)


def test_pad_helpers() -> None:
    """Test function test_pad_helpers."""
    assert _normalize_pad_width([(1, 2)], 1) == ((1, 2),)
    assert _normalize_pad_width((1, 2), 2) == ((1, 2), (1, 2))
    assert _normalize_pad_width(1, 2) == ((1, 1), (1, 1))
    assert _compute_pad_dim(5, (1, 2)) == 8


def test_pad_class() -> None:
    """Test function test_pad_class."""
    op = Pad()
    assert cast("Callable[..., EagerTensorProtocol]", op.infer_shape)(cast("Callable[..., EagerTensorProtocol]", MockTensor)((2, 3)), ((1, 1), (2, 2))) == (4, 7)
    assert cast("Callable[..., EagerTensorProtocol]", op.infer_shape)(None, ((1, 1), (2, 2))) == ()


def test_pad_func(mocker: EagerTensorProtocol) -> None:
    """Test function test_pad_func."""
    t = Tensor(MockTensor((2, 3)).data, TensorConfig((2, 3), "float32", "cpu"))
    config.eager_mode = False
    mocker.patch("ml_switcheroo_compiler.ops.shape.pad_and_tile._emit_shape_node", return_value="pad")
    assert pad(t, 1) == "pad"
    config.eager_mode = True
    mock_backend = mocker.patch("ml_switcheroo_compiler.ops.shape.pad_and_tile.get_active_backend").return_value
    setattr(mock_backend.execute_op, "return_value", MockTensor((4, 5)))
    setattr(mock_backend.array, "side_effect", lambda x: x)
    assert cast("EagerTensorProtocol", getattr(pad(t, 1), "config")).shape == (4, 5)


def test_top_k(mocker: EagerTensorProtocol) -> None:
    """Test function test_top_k."""
    t = Tensor(MockTensor((2, 3)).data, TensorConfig((2, 3), "float32", "cpu"))
    config.eager_mode = False
    mocker.patch("ml_switcheroo_compiler.ops.shape.pad_and_tile._emit_shape_node", return_value=("val", "idx"))
    assert top_k(t, 2) == (("val", "idx"), ("val", "idx"))
    config.eager_mode = True
    mock_backend = mocker.patch("ml_switcheroo_compiler.ops.shape.pad_and_tile.get_active_backend").return_value
    setattr(mock_backend.execute_op, "return_value", [MockTensor((2, 2)), MockTensor((2, 2))])
    setattr(mock_backend.array, "side_effect", lambda x: x)
    res = top_k(t, 2)
    assert cast("EagerTensorProtocol", getattr(cast("EagerTensorProtocol", getattr(res, "__getitem__")(0)), "config")).shape == (2, 2)
    assert cast("EagerTensorProtocol", getattr(cast("EagerTensorProtocol", getattr(res, "__getitem__")(1)), "config")).shape == (2, 2)


def test_argsort(mocker: EagerTensorProtocol) -> None:
    """Test function test_argsort."""
    t = Tensor(MockTensor((2, 3)).data, TensorConfig((2, 3), "float32", "cpu"))
    config.eager_mode = False
    mocker.patch("ml_switcheroo_compiler.ops.shape.pad_and_tile._emit_shape_node", return_value="argsort")
    assert argsort(t) == "argsort"
    config.eager_mode = True
    mock_backend = mocker.patch("ml_switcheroo_compiler.ops.shape.pad_and_tile.get_active_backend").return_value
    setattr(mock_backend.execute_op, "return_value", MockTensor((2, 3)))
    setattr(mock_backend.array, "side_effect", lambda x: x)
    setattr(mock_backend.array, "side_effect", lambda x: x)
    assert cast("EagerTensorProtocol", getattr(argsort(t), "config")).shape == (2, 3)


def test_sort(mocker: EagerTensorProtocol) -> None:
    """Test function test_sort."""
    t = Tensor(MockTensor((2, 3)).data, TensorConfig((2, 3), "float32", "cpu"))
    config.eager_mode = False
    mocker.patch("ml_switcheroo_compiler.ops.shape.pad_and_tile._emit_shape_node", return_value="sort")
    assert sort(t) == "sort"
    config.eager_mode = True
    mock_backend = mocker.patch("ml_switcheroo_compiler.ops.shape.pad_and_tile.get_active_backend").return_value
    setattr(mock_backend.execute_op, "return_value", MockTensor((2, 3)))
    setattr(mock_backend.array, "side_effect", lambda x: x)
    setattr(mock_backend.array, "side_effect", lambda x: x)
    assert cast("EagerTensorProtocol", getattr(sort(t), "config")).shape == (2, 3)


def test_image_resize(mocker: EagerTensorProtocol) -> None:
    """Test function test_image_resize."""
    t = Tensor(MockTensor((2, 3, 4, 3)).data, TensorConfig((2, 3, 4, 3), "float32", "cpu"))
    config.eager_mode = False
    mocker.patch("ml_switcheroo_compiler.ops.shape.pad_and_tile._emit_shape_node", return_value="resize")
    assert image_resize(t, (5, 6)) == "resize"
    config.eager_mode = True
    mock_backend = mocker.patch("ml_switcheroo_compiler.backends.registry.get_active_backend").return_value
    setattr(mock_backend.execute_op, "return_value", MockTensor((2, 5, 6, 3)))
    mock_array = mocker.MagicMock()
    setattr(mock_array, "shape", (2, 5, 6, 3))
    setattr(mock_backend.array, "return_value", mock_array)
    assert cast("EagerTensorProtocol", getattr(image_resize(t, (5, 6)), "config")).shape == (2, 5, 6, 3)


def test_dynamic_shape_infer_shape() -> None:
    """Test function test_dynamic_shape_infer_shape."""
    op = DynamicShape()
    assert cast("Callable[..., EagerTensorProtocol]", op.infer_shape)(cast("Callable[..., EagerTensorProtocol]", MockTensor)((2, 3))) == (2,)


def test_rank_infer_shape() -> None:
    """Test function test_rank_infer_shape."""
    op = Rank()
    assert cast("Callable[..., EagerTensorProtocol]", op.infer_shape)(cast("Callable[..., EagerTensorProtocol]", MockTensor)((2, 3))) == ()


def test_size_infer_shape() -> None:
    """Test function test_size_infer_shape."""
    op = Size()
    assert cast("Callable[..., EagerTensorProtocol]", op.infer_shape)(cast("Callable[..., EagerTensorProtocol]", MockTensor)((2, 3))) == ()


def test_pad_modes(mocker: EagerTensorProtocol) -> None:
    """Test function test_pad_modes."""
    mocker.patch("ml_switcheroo_compiler.ops.shape.pad_and_tile.pad", return_value="padded")
    assert pad_constant("arr", "pw") == "padded"
    assert pad_reflect("arr", "pw") == "padded"
    assert pad_replicate("arr", "pw") == "padded"
    assert pad_circular("arr", "pw") == "padded"


def test_compute_meshgrid_shape() -> None:
    """Test function test_compute_meshgrid_shape."""
    from ml_switcheroo_compiler.ops.shape.pad_and_tile import _compute_meshgrid_shape

    t1 = MockTensor((2,))
    t2 = MockTensor((3,))
    assert cast("Callable[..., EagerTensorProtocol]", _compute_meshgrid_shape)([t1], "ij") == (2,)
    assert _compute_meshgrid_shape([], "ij") == ()
    assert cast("Callable[..., EagerTensorProtocol]", _compute_meshgrid_shape)([t1, t2], "xy") == (3, 2)


def test_compute_pad_dim_extra() -> None:
    """Test function test_compute_pad_dim_extra."""
    assert _compute_pad_dim(5, 2) == 9
    assert _compute_pad_dim(5, "invalid") == 5


def test_pad_infer_shape_short_pw() -> None:
    """Test function test_pad_infer_shape_short_pw."""
    op = Pad()
    assert cast("Callable[..., EagerTensorProtocol]", op.infer_shape)(cast("Callable[..., EagerTensorProtocol]", MockTensor)((2, 3)), ((1, 1),)) == (4, 3)


def test_top_k_scalar(mocker: EagerTensorProtocol) -> None:
    """Test function test_top_k_scalar."""
    t = Tensor(MockTensor(()).data, TensorConfig((), "float32", "cpu"))
    config.eager_mode = False
    mocker.patch("ml_switcheroo_compiler.ops.shape.pad_and_tile._emit_shape_node", return_value=("val", "idx"))
    assert top_k(t, 2) == (("val", "idx"), ("val", "idx"))


def test_pad_and_tile_remaining_missing_branches() -> None:
    """Test DynamicShape, Rank, Size infer_shape methods, nested RavelMultiIndex, and multi-dim Searchsorted."""
    assert cast("Callable[..., EagerTensorProtocol]", DynamicShape)().infer_shape(cast("Callable[..., EagerTensorProtocol]", MockTensor)((2, 3, 4))) == (3,)
    assert cast("Callable[..., EagerTensorProtocol]", Rank)().infer_shape(cast("Callable[..., EagerTensorProtocol]", MockTensor)((2, 3))) == ()
    assert cast("Callable[..., EagerTensorProtocol]", Size)().infer_shape(cast("Callable[..., EagerTensorProtocol]", MockTensor)((2, 3))) == ()

    from ml_switcheroo_compiler.ops.shape.pad_and_tile import RavelMultiIndex, Searchsorted

    op_ravel = RavelMultiIndex()
    assert cast("Callable[..., EagerTensorProtocol]", op_ravel.infer_shape)([[cast("Callable[..., EagerTensorProtocol]", MockTensor)((2, 3))]]) == (2, 3)

    op_search = Searchsorted()
    seq_2d = MockTensor((2, 10))
    v_2d = MockTensor((2, 5))
    v_1d = MockTensor((5,))
    assert cast("Callable[..., EagerTensorProtocol]", op_search.infer_shape)(seq_2d, v_2d) == (2, 5)
    assert cast("Callable[..., EagerTensorProtocol]", op_search.infer_shape)(seq_2d, v_1d) == (2, 5)
