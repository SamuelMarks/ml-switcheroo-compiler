"""Module docstring."""

from ml_switcheroo_compiler.ops.shape import (
    DynamicIndexInDim,
    DynamicSliceInDim,
    DynamicUpdateIndexInDim,
    DynamicUpdateSliceInDim,
    ScatterApply,
    ScatterMax,
    ScatterMin,
    ScatterMul,
    SliceInDim,
)


def test_indexing_infer_shape() -> object:
    """Function docstring."""

    class DummyTensor:
        """Class docstring."""

        def __init__(self, shape: object) -> object:
            """Function docstring."""
            self.shape = shape

    t = DummyTensor((10, 20))

    assert DynamicSliceInDim().infer_shape(t, start_index=2, slice_size=5, axis=1) == (10, 5)
    assert DynamicUpdateSliceInDim().infer_shape(t, update=None, start_index=2, axis=1) == (10, 20)

    assert DynamicIndexInDim().infer_shape(t, index=5, axis=1, keepdims=True) == (10, 1)
    assert DynamicIndexInDim().infer_shape(t, index=5, axis=1, keepdims=False) == (10,)

    assert DynamicUpdateIndexInDim().infer_shape(t, update=None, index=5, axis=1) == (10, 20)

    assert SliceInDim().infer_shape(t, start_index=2, limit_index=12, stride=2, axis=1) == (10, 5)

    assert ScatterApply().infer_shape(t, indices=None, updates=None, func=None) == (10, 20)
    assert ScatterMax().infer_shape(t, indices=None, updates=None) == (10, 20)
    assert ScatterMin().infer_shape(t, indices=None, updates=None) == (10, 20)
    assert ScatterMul().infer_shape(t, indices=None, updates=None) == (10, 20)
