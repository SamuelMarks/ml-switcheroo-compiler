"""Mixins for MLX code generation."""

from __future__ import annotations

from ml_switcheroo_compiler.backends.mlx._mlx_nn_visitor import MLXNNOpsVisitor as MLXNNOpsVisitor
from ml_switcheroo_compiler.backends.mlx._mlx_op_registry_mixin import (
    MLXOpRegistryMixin as MLXOpRegistryMixin,
)
from ml_switcheroo_compiler.backends.mlx._mlx_shape_visitor import (
    MLXShapeOpsVisitor as MLXShapeOpsVisitor,
)
from ml_switcheroo_compiler.backends.mlx._mlx_vision_audio_visitor import (
    MLXAudioVisitor as MLXAudioVisitor,
)
from ml_switcheroo_compiler.backends.mlx._mlx_vision_audio_visitor import (
    MLXVisionVisitor as MLXVisionVisitor,
)


class MLXVisionAudioMixin:
    """Vision and Audio Mixin."""


class MLXShapeMixin:
    """Shape and Formatting Mixin."""


__all__ = [
    "MLXAudioVisitor",
    "MLXNNOpsVisitor",
    "MLXOpRegistryMixin",
    "MLXShapeMixin",
    "MLXShapeOpsVisitor",
    "MLXVisionAudioMixin",
    "MLXVisionVisitor",
]
