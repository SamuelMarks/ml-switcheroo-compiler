"""Tests for test_ops_inits_coverage."""

from __future__ import annotations

import ml_switcheroo_compiler.ops._math_registry as math_reg
import ml_switcheroo_compiler.ops._nn_registry as nn_reg
import ml_switcheroo_compiler.ops._vision_registry as vision_reg
import ml_switcheroo_compiler.ops.audio as audio_pkg
import ml_switcheroo_compiler.ops.image as image_mod
import ml_switcheroo_compiler.ops.tensor as tensor_mod
from ml_switcheroo_compiler.core.shape import _has_attr


class _MockDimWithId:
    """Mock dimension exposing an id attribute."""

    def __init__(self, dim_id: str) -> None:
        """Initialize mock dimension.

        Args:
            dim_id (str): Identifier for dimension.
        """
        self.id = dim_id


class _MockShapeContainer:
    """Mock container exposing a shape attribute."""

    def __init__(self, shape: tuple[int, ...]) -> None:
        """Initialize mock shape container.

        Args:
            shape (tuple[int, ...]): Shape tuple.
        """
        self.shape = shape


class _MockShapeMetaContainer:
    """Mock container exposing a shape_metadata attribute."""

    def __init__(self, shape_metadata: tuple[int, ...]) -> None:
        """Initialize mock shape metadata container.

        Args:
            shape_metadata (tuple[int, ...]): Shape metadata tuple.
        """
        self.shape_metadata = shape_metadata


class _MockGenerator:
    """Mock generator for variable AST visitor."""

    def get_fallback_prefix(self) -> str:
        """Get fallback prefix for AST emissions.

        Returns:
            str: Fallback prefix string.
        """
        return "mock_backend"


def test_package_and_module_inits() -> None:
    """Verify symbol exports and clean imports for packages and registration modules."""
    # 1. ops.image exports
    assert _has_attr(image_mod, "resize")
    assert _has_attr(image_mod, "gaussian_blur")
    assert len(image_mod.__all__) > 0

    # 2. ops.tensor exports
    assert _has_attr(tensor_mod, "allclose")
    assert _has_attr(tensor_mod, "take_along_axis")
    assert "allclose" in tensor_mod.__all__

    # 3. ops.audio, _math_registry, _nn_registry, _vision_registry
    assert audio_pkg.__all__ == []
    assert math_reg.__all__ == []
    assert nn_reg.__all__ == []
    assert vision_reg.__all__ == []

    # 4. backends.pure_python
    import ml_switcheroo_compiler.backends.pure_python as pure_python_pkg

    assert _has_attr(pure_python_pkg, "PurePythonGenerator")
    assert _has_attr(pure_python_pkg, "PurePythonTensor")
    assert _has_attr(pure_python_pkg, "execute_op")

    # 5. benchmarks
    import ml_switcheroo_compiler.benchmarks as benchmarks_pkg

    assert _has_attr(benchmarks_pkg, "BenchmarkOrchestrator")
    assert _has_attr(benchmarks_pkg, "BenchmarkPlan")
    assert _has_attr(benchmarks_pkg, "BenchmarkRunResult")
