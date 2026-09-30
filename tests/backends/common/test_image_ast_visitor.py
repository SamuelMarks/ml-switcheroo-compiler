"""Tests for test_image_ast_visitor."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

import ml_switcheroo_compiler.backends.common.mixins.image as img_mix
from ml_switcheroo_compiler.ir.core import IRNode


class _DummyGenerator:
    """Mock generator providing fallback prefix."""

    def get_fallback_prefix(self) -> str:
        """Return prefix for fallback ops.

        Returns:
            str: Prefix string.
        """
        return "np_eager"


class _MockArrayWithNumpy:
    """Mock object with numpy() method."""

    def __init__(self, arr: np.ndarray) -> None:
        """Initialize with numpy array.

        Args:
            arr (np.ndarray): Underlying array.
        """
        self.arr = arr

    def numpy(self) -> np.ndarray:
        """Return underlying numpy array.

        Returns:
            np.ndarray: Array.
        """
        return self.arr


class _MockDataWithNumpy:
    """Mock object with data.numpy() method."""

    def __init__(self, arr: np.ndarray) -> None:
        """Initialize with nested data object.

        Args:
            arr (np.ndarray): Underlying array.
        """
        self.data = _MockArrayWithNumpy(arr)


class _MockArrayWithToList:
    """Mock object with tolist() method."""

    def __init__(self, lst: list[float]) -> None:
        """Initialize with list.

        Args:
            lst (list[float]): List elements.
        """
        self.lst = lst

    def tolist(self) -> list[float]:
        """Return list representation.

        Returns:
            list[float]: The list.
        """
        return self.lst


class _NoShapeObject:
    """Mock object with dtype and tobytes, but missing shape."""

    dtype: str = "float32"

    def tobytes(self) -> bytes:
        """Return empty bytes.

        Returns:
            bytes: Empty bytes.
        """
        return b""


class _NoTobytesObject:
    """Mock object with dtype and shape, but missing tobytes."""

    dtype: str = "float32"
    shape: tuple[int, ...] = (2,)


class _MockBackendH5:
    """Mock backend implementing custom h5 save and load methods."""

    def __init__(self) -> None:
        """Initialize mock backend."""
        self.saved: tuple[dict[str, np.ndarray], str] | None = None

    def load_h5(self, path: str) -> dict[str, np.ndarray]:
        """Load h5 mock weights.

        Args:
            path (str): Filepath.

        Returns:
            dict[str, np.ndarray]: Dict of weights.
        """
        return {"backend_key": np.array([42.0])}

    def save_h5(self, weights: dict[str, np.ndarray], path: str) -> None:
        """Save h5 mock weights.

        Args:
            weights (dict[str, np.ndarray]): Weights dict.
            path (str): Filepath.
        """
        self.saved = (weights, path)


class _MockBackendNPZ:
    """Mock backend implementing custom npz save and load methods."""

    def __init__(self) -> None:
        """Initialize mock backend."""
        self.saved: tuple[dict[str, np.ndarray], str] | None = None

    def save_npz(self, weights: dict[str, np.ndarray], path: str) -> None:
        """Save npz mock weights.

        Args:
            weights (dict[str, np.ndarray]): Weights dict.
            path (str): Filepath.
        """
        self.saved = (weights, path)

    def load_npz(self, path: str) -> dict[str, np.ndarray]:
        """Load npz mock weights.

        Args:
            path (str): Filepath.

        Returns:
            dict[str, np.ndarray]: Dict of weights.
        """
        return {"backend_key": np.array([100.0])}


class _MockBackendSafetensors:
    """Mock backend implementing custom from_buffer method."""

    def from_buffer(self, buffer: bytes, dtype: str, shape: Sequence[int]) -> np.ndarray:
        """Reconstruct array from buffer.

        Args:
            buffer (bytes): Raw bytes.
            dtype (str): Dtype identifier.
            shape (Sequence[int]): Shape tuple or list.

        Returns:
            np.ndarray: Reconstructed array.
        """
        return np.frombuffer(buffer, dtype=np.float32).reshape(tuple(shape))


class _PlainBackend:
    """Plain backend without custom format hooks."""

    pass


def test_image_ast_visitor() -> None:
    """Verify ImageASTVisitor emission and parameter default handling.

    Returns:
        None
    """
    vis = img_mix.ImageASTVisitor(generator=_DummyGenerator())
    node = IRNode("Dummy", [], {})

    # With kwargs explicitly supplied
    assert vis.visit_AdjustBrightness(node, ["x"], delta=0.5) == "np_eager_adjust_brightness(x, 0.5)"
    assert vis.visit_AdjustContrast(node, ["x"], contrast_factor=1.5) == "np_eager_adjust_contrast(x, 1.5)"
    assert vis.visit_AdjustHue(node, ["x"], delta=0.2) == "np_eager_adjust_hue(x, 0.2)"
    assert vis.visit_AdjustSaturation(node, ["x"], saturation_factor=2.0) == "np_eager_adjust_saturation(x, 2.0)"
    assert vis.visit_AffineGenerator(node, ["th", "d"], batch_size=4) == "np_eager_affine_generator(4, th, d)"
    assert vis.visit_AffineGrid(node, ["th"], size=(2, 2), align_corners=True) == "np_eager_affine_grid(th, size=(2, 2), align_corners=True)"
    assert vis.visit_AffineTransform(node, ["img", "mat"], interpolation="bilinear") == "np_eager_affine_transform(img, mat, interpolation='bilinear')"
    assert vis.visit_AugMix(node, ["x"]) == "np_eager_augmix(x)"
    assert vis.visit_AutoContrast(node, ["x"]) == "np_eager_auto_contrast(x)"

    # With default kwargs omitted
    assert vis.visit_AdjustBrightness(node, ["x"]) == "np_eager_adjust_brightness(x, 0.0)"
    assert vis.visit_AdjustContrast(node, ["x"]) == "np_eager_adjust_contrast(x, 1.0)"
    assert vis.visit_AdjustHue(node, ["x"]) == "np_eager_adjust_hue(x, 0.0)"
    assert vis.visit_AdjustSaturation(node, ["x"]) == "np_eager_adjust_saturation(x, 1.0)"
    assert vis.visit_AffineGenerator(node, ["th"]) == "np_eager_affine_generator(1, th)"
    assert vis.visit_AffineGrid(node, ["th"]) == "np_eager_affine_grid(th, size=(), align_corners=False)"
    assert vis.visit_AffineTransform(node, ["img", "mat"]) == "np_eager_affine_transform(img, mat, interpolation='nearest')"
