"""Unit tests for Dot and ApplyOverAxes shape inference across 1D, 2D, and ND arrays."""

from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.linalg.dot import Dot
from ml_switcheroo_compiler.ops.stats.descriptive import ApplyOverAxes


def _make_tensor(shape: tuple[int, ...]) -> Tensor:
    """Helper to create dummy tensor with specified shape.

    Args:
        shape (tuple[int, ...]): Shape of tensor.

    Returns:
        Tensor: Dummy tensor instance.
    """
    return Tensor(None, TensorConfig(shape, "float32", "cpu"))


def test_dot_infer_shape_1d_and_2d() -> None:
    """Verify Dot shape inference for 1D and 2D arrays."""
    op = Dot()

    # 1D x 1D -> scalar ()
    a_1d = _make_tensor((5,))
    b_1d = _make_tensor((5,))
    assert op.infer_shape(a_1d, b_1d) == ()

    # 2D x 2D -> (M, N)
    a_2d = _make_tensor((4, 6))
    b_2d = _make_tensor((6, 8))
    assert op.infer_shape(a_2d, b_2d) == (4, 8)

    # 0D scalar x ND
    scalar = _make_tensor(())
    assert op.infer_shape(scalar, a_2d) == (4, 6)
    assert op.infer_shape(a_2d, scalar) == (4, 6)


def test_dot_infer_shape_nd_contractions() -> None:
    """Verify Dot contraction rules for ND x 1D, 1D x ND, and ND x MD."""
    op = Dot()

    # ND x 1D -> a[:-1]
    a_3d = _make_tensor((2, 3, 5))
    v = _make_tensor((5,))
    assert op.infer_shape(a_3d, v) == (2, 3)

    # 1D x ND -> b[1:]
    b_3d = _make_tensor((5, 4, 6))
    assert op.infer_shape(v, b_3d) == (4, 6)

    # 3D x 2D: (2, 3, 5) x (5, 7) -> (2, 3, 7)
    w_2d = _make_tensor((5, 7))
    assert op.infer_shape(a_3d, w_2d) == (2, 3, 7)

    # 3D x 3D: (2, 3, 5) x (4, 5, 8) -> (2, 3, 4, 8)
    rhs_3d = _make_tensor((4, 5, 8))
    assert op.infer_shape(a_3d, rhs_3d) == (2, 3, 4, 8)

    # Fallbacks
    assert op.infer_shape(None, None) == ()

    # Test objects with shape_metadata and lists/tuples
    class DummyWithMetadata:
        """Dummy object with shape_metadata."""

        shape_metadata = (3, 4)

    class DummyWithEmptyMetadata:
        """Dummy object with empty shape_metadata."""

        shape_metadata = ()

    assert op.infer_shape(DummyWithMetadata(), DummyWithMetadata()) == (3, 4)
    assert op.infer_shape(DummyWithEmptyMetadata(), [2, 3]) == (2, 3)
    assert op.infer_shape("not_a_shape", "also_not") == ()


def test_apply_over_axes_shape_inference() -> None:
    """Verify ApplyOverAxes shape inference setting reduced axes to size 1."""
    op = ApplyOverAxes()

    t_3d = _make_tensor((4, 5, 6))

    # Single axis reduction
    assert op.infer_shape(None, t_3d, axes=0) == (1, 5, 6)
    assert op.infer_shape(None, t_3d, axes=1) == (4, 1, 6)

    # Multiple axes reduction
    assert op.infer_shape(None, t_3d, axes=[0, 2]) == (1, 5, 1)

    # Negative axis reduction (-1 -> last axis)
    assert op.infer_shape(None, t_3d, axes=[-1]) == (4, 5, 1)

    # Fallback
    assert op.infer_shape(None, None) == ()
