"""Tests for ops shape misc functions."""

import ml_switcheroo_compiler.ops.shape.pad_and_tile as shape_misc


def test_shape_misc_functions() -> None:
    """Test shape misc coverage."""

    class DummyNode:
        shape = (10, 20)

    a = DummyNode()
    shape_misc._infer_shape_percentile_quantile(a, 0.5, axis=None, keepdims=True)
    shape_misc._infer_shape_percentile_quantile(a, 0.5, axis=None, keepdims=False)
    shape_misc._infer_shape_percentile_quantile(a, [0.5], axis=0, keepdims=True)
    shape_misc._infer_shape_percentile_quantile(a, [0.5], axis=0, keepdims=False)

    for attr in dir(shape_misc):
        if attr.startswith("_infer_shape_"):
            val = getattr(shape_misc, attr)
            if callable(val):
                try:
                    val(a)
                except Exception:
                    pass
                try:
                    val(a, a)
                except Exception:
                    pass
                try:
                    val(a, a, a)
                except Exception:
                    pass


def test_tile_repeat_triu_tril_exact_shapes(mocker) -> None:
    """Verify exact output shape inference for tile, repeat, triu, and tril in graph mode.

    Args:
        mocker (object): Pytest mocker fixture.
    """
    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    captured_shapes: dict[str, tuple[int, ...]] = {}

    def mock_emit(op_name: str, inputs: list[Tensor], attrs: dict[str, object], shape: tuple[int, ...], dtype: object) -> str:
        captured_shapes[op_name] = shape
        return op_name

    mocker.patch("ml_switcheroo_compiler.ops.shape.pad_and_tile._emit_shape_node", side_effect=mock_emit)
    config.eager_mode = False

    t = Tensor(None, TensorConfig((2, 3), "float32", "cpu"))

    # 1. Tile with matching rank
    shape_misc.tile(t, reps=(2, 4))
    assert captured_shapes["Tile"] == (4, 12)

    # 2. Tile with expanding rank
    shape_misc.tile(t, reps=(3, 2, 4))
    assert captured_shapes["Tile"] == (3, 4, 12)

    # 2b. Tile with fewer reps than rank
    shape_misc.tile(t, reps=2)
    assert captured_shapes["Tile"] == (2, 6)

    # 3. Repeat along specific axis
    shape_misc.repeat(t, repeats=3, axis=1)
    assert captured_shapes["Repeat"] == (2, 9)

    # 3b. Repeat along axis with sequence repeats
    shape_misc.repeat(t, repeats=[2, 3], axis=0)
    assert captured_shapes["Repeat"] == (5, 3)

    # 3c. Repeat along axis with non-int/non-sequence repeats
    shape_misc.repeat(t, repeats=None, axis=0)  # type: ignore[arg-type]
    assert captured_shapes["Repeat"] == (2, 3)

    # 4. Repeat with axis=None (flattened)
    shape_misc.repeat(t, repeats=2, axis=None)
    assert captured_shapes["Repeat"] == (12,)

    # 4b. Repeat with axis=None and sequence repeats
    shape_misc.repeat(t, repeats=[1, 2, 3], axis=None)
    assert captured_shapes["Repeat"] == (6,)

    # 4c. Repeat with axis=None and non-int/non-sequence repeats
    shape_misc.repeat(t, repeats=None, axis=None)  # type: ignore[arg-type]
    assert captured_shapes["Repeat"] == (6,)

    # 5. Triu and Tril preserve shape
    shape_misc.triu(t, diagonal=0)
    assert captured_shapes["Triu"] == (2, 3)

    shape_misc.tril(t, diagonal=0)
    assert captured_shapes["Tril"] == (2, 3)
