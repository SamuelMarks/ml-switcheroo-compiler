"""Tests for ops shape misc functions."""

from __future__ import annotations

import typing

import pytest

import ml_switcheroo_compiler.ops.shape.pad_and_tile as shape_misc
from tests.testing_utils import as_callable

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue


def test_shape_misc_functions() -> None:
    """Test shape misc coverage."""

    class DummyNode:
        """Test class DummyNode."""

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
                with pytest.raises(TypeError):
                    val(a)
                val(a, a)
                with pytest.raises(AttributeError):
                    val(a, a, a)


def test_tile_repeat_triu_tril_exact_shapes(mocker: EagerTensorProtocol) -> None:
    """Verify exact output shape inference for tile, repeat, triu, and tril in graph mode.

    Args:
        mocker (EagerTensorProtocol): Pytest mocker fixture.
    """
    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    captured_shapes: dict[str, tuple[int, ...]] = {}

    def mock_emit(op_name: str, inputs: list[Tensor], attrs: dict[str, EagerValue], shape: tuple[int, ...], dtype: EagerValue) -> str:
        """Test function mock_emit.

        Args:
            op_name: Test parameter `op_name`.
            inputs: Test parameter `inputs`.
            attrs: Test parameter `attrs`.
            shape: Test parameter `shape`.
            dtype: Test parameter `dtype`.
        """
        captured_shapes[op_name] = shape
        return op_name

    mocker.patch("ml_switcheroo_compiler.ops.shape.pad_and_tile._emit_shape_node", side_effect=mock_emit)
    config.eager_mode = False
    t = Tensor(None, TensorConfig((2, 3), "float32", "cpu"))
    shape_misc.tile(t, reps=(2, 4))
    assert captured_shapes["Tile"] == (4, 12)
    shape_misc.tile(t, reps=(3, 2, 4))
    assert captured_shapes["Tile"] == (3, 4, 12)
    as_callable(shape_misc.tile)(t, reps=2)
    assert captured_shapes["Tile"] == (2, 6)
    shape_misc.repeat(t, repeats=3, axis=1)
    assert captured_shapes["Repeat"] == (2, 9)
    shape_misc.repeat(t, repeats=[2, 3], axis=0)
    assert captured_shapes["Repeat"] == (5, 3)
    as_callable(shape_misc.repeat)(t, repeats=None, axis=0)
    assert captured_shapes["Repeat"] == (2, 3)
    shape_misc.repeat(t, repeats=2, axis=None)
    assert captured_shapes["Repeat"] == (12,)
    shape_misc.repeat(t, repeats=[1, 2, 3], axis=None)
    assert captured_shapes["Repeat"] == (6,)
    as_callable(shape_misc.repeat)(t, repeats=None, axis=None)
    assert captured_shapes["Repeat"] == (6,)
    shape_misc.triu(t, diagonal=0)
    assert captured_shapes["Triu"] == (2, 3)
    shape_misc.tril(t, diagonal=0)
    assert captured_shapes["Tril"] == (2, 3)
