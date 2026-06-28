import numpy as np
from ml_switcheroo_compiler.backends.numpy.eager.vision_geometry import (
    _np_elastic_transform,
    _np_extract_bounding_boxes,
    _np_iou,
    _np_nms,
    _np_perspective_transform,
    _np_resize_bicubic,
    _np_resize_lanczos3,
    _np_resize_nearest,
    _np_affine_grid,
    _np_affine_transform,
    _np_affine_generator,
)


def test_numpy_vision_geometry_eager_extra():
    # elastic transform
    _np_elastic_transform(np, np.ones((2, 2, 3)), np.ones((2, 2, 2)))

    # extract bounding boxes
    _np_extract_bounding_boxes(np, np.ones((2, 2, 3)), np.array([0, 0, 10, 10]), np.array([0]))

    # iou
    import ml_switcheroo_compiler.backends.eager as eager_mod

    original_iou = eager_mod.iou_eager

    def mock_iou(*args, **kwargs):
        return args[1]

    eager_mod.iou_eager = mock_iou
    try:
        _np_iou(np, np.array([0, 0, 10, 10]), np.array([5, 5, 15, 15]))
    finally:
        eager_mod.iou_eager = original_iou

    # nms
    original_nms = eager_mod.nms_eager

    def mock_nms(*args, **kwargs):
        return args[1]

    eager_mod.nms_eager = mock_nms
    try:
        _np_nms(np, np.array([[0, 0, 10, 10]]), np.array([0.9]), max_output_size=1, threshold=0.5)
    finally:
        eager_mod.nms_eager = original_nms

    # perspective_transform
    _np_perspective_transform(np, np.ones((2, 2, 3)), np.eye(3), np.eye(3), config=None)

    # resize bicubic
    _np_resize_bicubic(np, np.ones((2, 2, 3)), size=(4, 4))

    # resize lanczos
    _np_resize_lanczos3(np, np.ones((2, 2, 3)), size=(4, 4))

    # resize nearest
    _np_resize_nearest(np, np.ones((2, 2, 3)), size=(4, 4))

    # affine grid
    _np_affine_grid(np, np.eye(2, 3), size=(4, 4))

    # affine transform
    _np_affine_transform(np, np.ones((2, 2, 3)), np.eye(2, 3))

    _np_affine_generator(np, 2, np.ones(1), np.ones(1), np.ones(1))
