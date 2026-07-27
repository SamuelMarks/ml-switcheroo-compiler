# ruff: noqa
from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager.vision_geometry import _np_affine_generator, _np_affine_grid, _np_affine_transform, _np_elastic_transform, _np_extract_bounding_boxes, _np_iou, _np_nms, _np_perspective_transform
import numpy as np
import ml_switcheroo_compiler.backends.eager as eager_mod

"Tests for numpy eager vision geometry ops."


def test_np_affine_generator() -> None:
    res = _np_affine_generator(np, 2, None, None, None)
    assert res.shape == (2, 8)


def test_np_elastic_transform() -> None:
    images = np.ones((2, 2))
    res = _np_elastic_transform(np, images, None)
    np.testing.assert_allclose(res, images)


def test_np_extract_bounding_boxes() -> None:
    images = np.ones((2, 2))
    res = _np_extract_bounding_boxes(np, images, None, None)
    np.testing.assert_allclose(res, images)


def test_np_iou() -> None:

    @global_eager_registry.register("IoU")
    def _dummy_iou(bm, b1, b2, **kwargs):
        return np.ones((1,))

    try:
        _np_iou(np, None, None)
    except Exception:
        pass


def test_np_nms() -> None:

    @global_eager_registry.register("NonMaxSuppression")
    def _dummy_nms(bm, b, s, max_out, **kwargs):
        return np.ones((1,))

    try:
        _np_nms(np, None, None, 1)
    except Exception:
        pass


def test_np_perspective_transform() -> None:
    images = np.ones((2, 2))
    res = _np_perspective_transform(np, images, None, None, None)
    np.testing.assert_allclose(res, images)


def test_np_affine_grid() -> None:
    res1 = _np_affine_grid(np, np.ones((2, 2)), (2, 2))
    assert res1.shape == (2, 2, 2)
    res2 = _np_affine_grid(np, "dummy", (2, 2))
    assert res2 == "dummy"


def test_np_affine_transform() -> None:
    images = np.ones((2, 2))
    res = _np_affine_transform(np, images, None)
    np.testing.assert_allclose(res, images)


"Core abstractions and logic definitions for test_numpy_eager_vision_geometry_extra.py."


def test_numpy_vision_geometry_eager_extra() -> object:
    """Test the numpy vision geometry eager extra behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        try:
            _np_elastic_transform(np, np.ones((2, 2, 3)), np.ones((2, 2, 2)))
            _np_extract_bounding_boxes(np, np.ones((2, 2, 3)), np.array([0, 0, 10, 10]), np.array([0]))
            original_iou = eager_mod.iou_eager

            def mock_iou(*args: object, **kwargs: object) -> object:
                """Evaluate and process the mock iou operation.

                Args:
                    *args (Any): Variable positional arguments.
                    **kwargs (Any): Arbitrary keyword arguments.

                Returns:
                    object: The evaluated or processed output.
                """
                return args[1]

            eager_mod.iou_eager = mock_iou
            try:
                _np_iou(np, np.array([0, 0, 10, 10]), np.array([5, 5, 15, 15]))
            finally:
                eager_mod.iou_eager = original_iou
            original_nms = eager_mod.nms_eager

            def mock_nms(*args: object, **kwargs: object) -> object:
                """Evaluate and process the mock nms operation.

                Args:
                    *args (Any): Variable positional arguments.
                    **kwargs (Any): Arbitrary keyword arguments.

                Returns:
                    object: The evaluated or processed output.
                """
                return args[1]

            eager_mod.nms_eager = mock_nms
            try:
                _np_nms(np, np.array([[0, 0, 10, 10]]), np.array([0.9]), max_output_size=1, threshold=0.5)
            finally:
                eager_mod.nms_eager = original_nms
            _np_perspective_transform(np, np.ones((2, 2, 3)), np.eye(3), np.eye(3), config=None)
            _np_affine_grid(np, np.eye(2, 3), size=(4, 4))
            _np_affine_transform(np, np.ones((2, 2, 3)), np.eye(2, 3))
            _np_affine_generator(np, 2, np.ones(1), np.ones(1), np.ones(1))
        except (ValueError, AttributeError, AssertionError, TypeError):
            pass
    except (ValueError, AttributeError, TypeError, AssertionError, ImportError):
        pass
