"""Tests for test_vision_geometry."""

from __future__ import annotations

import typing

import numpy as np
import pytest

import ml_switcheroo_compiler.backends.eager as eager_mod
from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager.vision_geometry import _np_affine_generator, _np_affine_grid, _np_affine_transform, _np_elastic_transform, _np_extract_bounding_boxes, _np_iou, _np_nms, _np_perspective_transform

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol, EagerTensorProtocol, EagerValue


def test_np_affine_generator() -> None:
    """Test function test_np_affine_generator."""
    res = _np_affine_generator(
        np, 2, typing.cast("np.ndarray[tuple[int, ...], np.dtype[np.generic]] | list[float] | float", None), typing.cast("np.ndarray[tuple[int, ...], np.dtype[np.generic]] | list[float] | float", None), typing.cast("np.ndarray[tuple[int, ...], np.dtype[np.generic]] | list[float] | float", None)
    )
    assert typing.cast("EagerTensorProtocol", getattr(res, "shape")) == (2, 8)


def test_np_elastic_transform() -> None:
    """Test function test_np_elastic_transform."""
    images = np.ones((2, 2))
    res = _np_elastic_transform(np, images, typing.cast("np.ndarray[tuple[int, ...], np.dtype[np.generic]]", None))
    typing.cast("EagerCallableProtocol", np.testing.assert_allclose)(res, images)


def test_np_extract_bounding_boxes() -> None:
    """Test function test_np_extract_bounding_boxes."""
    images = np.ones((2, 2))
    res = _np_extract_bounding_boxes(np, images, typing.cast("np.ndarray[tuple[int, ...], np.dtype[np.generic]]", None), typing.cast("np.ndarray[tuple[int, ...], np.dtype[np.generic]]", None))
    typing.cast("EagerCallableProtocol", np.testing.assert_allclose)(res, images)


def test_np_iou() -> None:
    """Test function test_np_iou."""

    @global_eager_registry.register("IoU")
    def _dummy_iou(bm: EagerValue, b1: EagerValue, b2: EagerValue, **kwargs: EagerValue) -> EagerValue:
        """Test function _dummy_iou.

        Args:
            bm: Test parameter `bm`.
            b1: Test parameter `b1`.
            b2: Test parameter `b2`.
            kwargs: Test parameter `kwargs`.
        """
        return np.ones((1,))

    with pytest.raises(IndexError):
        _np_iou(np, typing.cast("np.ndarray[tuple[int, ...], np.dtype[np.generic]]", None), typing.cast("np.ndarray[tuple[int, ...], np.dtype[np.generic]]", None))


def test_np_nms() -> None:
    """Test function test_np_nms."""

    @global_eager_registry.register("NonMaxSuppression")
    def _dummy_nms(bm: EagerValue, b: EagerValue, s: EagerValue, max_out: EagerValue, **kwargs: EagerValue) -> EagerValue:
        """Test function _dummy_nms.

        Args:
            bm: Test parameter `bm`.
            b: Test parameter `b`.
            s: Test parameter `s`.
            max_out: Test parameter `max_out`.
            kwargs: Test parameter `kwargs`.
        """
        return np.ones((1,))

    with pytest.raises(TypeError):
        _np_nms(np, typing.cast("np.ndarray[tuple[int, ...], np.dtype[np.generic]]", None), typing.cast("np.ndarray[tuple[int, ...], np.dtype[np.generic]]", None), 1)


def test_np_perspective_transform() -> None:
    """Test function test_np_perspective_transform."""
    images = np.ones((2, 2))
    res = _np_perspective_transform(np, images, typing.cast("np.ndarray[tuple[int, ...], np.dtype[np.generic]]", None), typing.cast("np.ndarray[tuple[int, ...], np.dtype[np.generic]]", None), None)
    typing.cast("EagerCallableProtocol", np.testing.assert_allclose)(res, images)


def test_np_affine_grid() -> None:
    """Test function test_np_affine_grid."""
    res1 = _np_affine_grid(np, np.ones((2, 2)), (2, 2))
    assert typing.cast("EagerTensorProtocol", getattr(res1, "shape")) == (2, 2, 2)
    res2 = _np_affine_grid(np, typing.cast("np.ndarray[tuple[int, ...], np.dtype[np.generic]] | float", "dummy"), (2, 2))
    assert res2 == "dummy"


def test_np_affine_transform() -> None:
    """Test function test_np_affine_transform."""
    images = np.ones((2, 2))
    res = _np_affine_transform(np, images, typing.cast("np.ndarray[tuple[int, ...], np.dtype[np.generic]]", None))
    typing.cast("EagerCallableProtocol", np.testing.assert_allclose)(res, images)


def test_numpy_vision_geometry_eager_extra() -> None:
    """Test the numpy vision geometry eager extra behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    with pytest.raises(TypeError):
        _np_elastic_transform(np, np.ones((2, 2, 3)), np.ones((2, 2, 2)))
        _np_extract_bounding_boxes(np, np.ones((2, 2, 3)), np.array([0, 0, 10, 10]), np.array([0]))
        original_iou = eager_mod.iou_eager

        def mock_iou(*args: EagerValue, **kwargs: EagerValue) -> EagerValue:
            """Evaluate and process the mock iou operation.

            Args:
                *args (EagerValue): Variable positional arguments.
                **kwargs (EagerValue): Arbitrary keyword arguments.

            Returns:
                EagerValue: The evaluated or processed output.
            """
            return args[1]

        eager_mod.iou_eager = mock_iou
        try:
            _np_iou(np, np.array([0, 0, 10, 10]), np.array([5, 5, 15, 15]))
        finally:
            eager_mod.iou_eager = original_iou
        original_nms = eager_mod.nms_eager

        def mock_nms(*args: EagerValue, **kwargs: EagerValue) -> EagerValue:
            """Evaluate and process the mock nms operation.

            Args:
                *args (EagerValue): Variable positional arguments.
                **kwargs (EagerValue): Arbitrary keyword arguments.

            Returns:
                EagerValue: The evaluated or processed output.
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
