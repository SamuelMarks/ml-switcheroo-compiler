"""Tests for test_mlx_generator."""

from __future__ import annotations

from typing import cast
from typing import TYPE_CHECKING

# ruff: noqa
import pytest
from ml_switcheroo_compiler.backends.mlx.mlx_mixins import MLXAudioVisitor, MLXNNOpsVisitor, MLXOpRegistryMixin, MLXShapeOpsVisitor, MLXVisionVisitor
from ml_switcheroo_compiler.backends.mlx.generator import MLXCodeGenerator

if TYPE_CHECKING:
    from collections.abc import Callable
    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue

pytest.importorskip("mlx.core")


class DummyNode:
    """Test class DummyNode."""

    def __init__(self, attrs: EagerValue = None) -> None:
        """Test function __init__."""
        self.attributes = attrs or {}


def test_mlx_op_registry_mixin() -> None:
    """Test function test_mlx_op_registry_mixin."""
    vis = MLXOpRegistryMixin()
    ops = vis._SIMPLE_OPS_MAP
    assert ops["BroadcastInDim"] == "{0}.broadcast_in_dim({1}, {2})"
    assert ops["Matmul"] == "mx.matmul({0}, {1})"


def test_mlx_nn_ops_visitor() -> None:
    """Test function test_mlx_nn_ops_visitor."""
    vis = MLXNNOpsVisitor()
    assert cast("Callable[..., EagerTensorProtocol]", vis.visit_Rope)(cast("Callable[..., EagerTensorProtocol]", DummyNode)({"dim": 1, "traditional": True, "base": 2.0, "scale": 3.0, "offset": 4}), ["a"]) == "mx.fast.rope(a, 1, traditional=True, base=2.0, scale=3.0, offset=4)"
    assert cast("Callable[..., EagerTensorProtocol]", vis.visit_PowerIteration)(cast("Callable[..., EagerTensorProtocol]", DummyNode)({"num_iters": 2}), ["a", "u"]) == "mlx_power_iteration(a, 2, u)"
    assert cast("Callable[..., EagerTensorProtocol]", vis.visit_PowerIteration)(cast("Callable[..., EagerTensorProtocol]", DummyNode)(), ["a"]) == "mlx_power_iteration(a, 1, None)"


def test_mlx_vision_visitor() -> None:
    """Test function test_mlx_vision_visitor."""
    vis = MLXVisionVisitor()
    assert cast("Callable[..., EagerTensorProtocol]", vis.visit_ElasticTransform)(cast("Callable[..., EagerTensorProtocol]", DummyNode)({"data_format": "df", "interpolation": "bicubic"}), ["a", "b"]) == "mlx_elastic_transform(a, b, 'bicubic', 0.0, \"df\")"
    assert cast("Callable[..., EagerTensorProtocol]", vis.visit_ElasticTransform)(cast("Callable[..., EagerTensorProtocol]", DummyNode)(), ["a", "b"]) == "mlx_elastic_transform(a, b, 'bilinear', 0.0, None)"
    assert cast("Callable[..., EagerTensorProtocol]", vis.visit_GaussianBlur)(cast("Callable[..., EagerTensorProtocol]", DummyNode)({"kernel_size": 3, "sigma": 1.0, "data_format": "df"}), ["a"]) == "mlx_gaussian_blur(a, 3, 1.0, 'same', \"df\")"
    assert cast("Callable[..., EagerTensorProtocol]", vis.visit_GaussianBlur)(cast("Callable[..., EagerTensorProtocol]", DummyNode)(), ["a"]) == "mlx_gaussian_blur(a, None, None, 'same', None)"
    assert cast("Callable[..., EagerTensorProtocol]", vis.visit_MedianFilter)(cast("Callable[..., EagerTensorProtocol]", DummyNode)({"kernel_size": 3, "sigma": 1.0, "data_format": "df"}), ["a"]) == "mlx_median_filter(a, 3, 'same', \"df\")"
    assert cast("Callable[..., EagerTensorProtocol]", vis.visit_MedianFilter)(cast("Callable[..., EagerTensorProtocol]", DummyNode)(), ["a"]) == "mlx_median_filter(a, None, 'same', None)"
    assert cast("Callable[..., EagerTensorProtocol]", vis.visit_ExtractBoundingBoxes)(cast("Callable[..., EagerTensorProtocol]", DummyNode)({"crop_size": 10, "data_format": "df"}), ["a", "b", "c"]) == "mlx_extract_bounding_boxes(a, b, c, 10, 'bilinear', 0.0, \"df\")"
    assert cast("Callable[..., EagerTensorProtocol]", vis.visit_ExtractBoundingBoxes)(cast("Callable[..., EagerTensorProtocol]", DummyNode)(), ["a", "b", "c"]) == "mlx_extract_bounding_boxes(a, b, c, None, 'bilinear', 0.0, None)"
    assert cast("Callable[..., EagerTensorProtocol]", vis.visit_IoU)(cast("Callable[..., EagerTensorProtocol]", DummyNode)({"bounding_box_format": "cxywh"}), ["a", "b"]) == "mlx_iou(a, b, 'cxywh')"
    assert cast("Callable[..., EagerTensorProtocol]", vis.visit_IoU)(cast("Callable[..., EagerTensorProtocol]", DummyNode)(), ["a", "b"]) == "mlx_iou(a, b, 'xyxy')"
    assert cast("Callable[..., EagerTensorProtocol]", vis.visit_NonMaxSuppression)(cast("Callable[..., EagerTensorProtocol]", DummyNode)({"max_output_size": 10}), ["a", "b"]) == "mlx_nms(a, b, 10, 0.5, -inf)"
    assert cast("Callable[..., EagerTensorProtocol]", vis.visit_NonMaxSuppression)(cast("Callable[..., EagerTensorProtocol]", DummyNode)(), ["a", "b"]) == "mlx_nms(a, b, None, 0.5, -inf)"
    assert cast("Callable[..., EagerTensorProtocol]", vis.visit_ResizeBicubic)(cast("Callable[..., EagerTensorProtocol]", DummyNode)({"size": 10, "align_corners": True}), ["a"]) == "mlx_resize(a, 10, 'bicubic', True)"
    assert cast("Callable[..., EagerTensorProtocol]", vis.visit_ResizeBicubic)(cast("Callable[..., EagerTensorProtocol]", DummyNode)(), ["a"]) == "mlx_resize(a, None, 'bicubic', False)"
    assert cast("Callable[..., EagerTensorProtocol]", vis.visit_ResizeLanczos3)(cast("Callable[..., EagerTensorProtocol]", DummyNode)({"size": 10, "align_corners": True}), ["a"]) == "mlx_resize(a, 10, 'lanczos3', True)"
    assert cast("Callable[..., EagerTensorProtocol]", vis.visit_ResizeLanczos3)(cast("Callable[..., EagerTensorProtocol]", DummyNode)(), ["a"]) == "mlx_resize(a, None, 'lanczos3', False)"
    assert cast("Callable[..., EagerTensorProtocol]", vis.visit_PerspectiveTransform)(cast("Callable[..., EagerTensorProtocol]", DummyNode)({"data_format": "df"}), ["a", "b", "c"]) == "mlx_perspective_transform(a, b, c, 'bilinear', 0.0, \"df\")"
    assert cast("Callable[..., EagerTensorProtocol]", vis.visit_PerspectiveTransform)(cast("Callable[..., EagerTensorProtocol]", DummyNode)(), ["a", "b", "c"]) == "mlx_perspective_transform(a, b, c, 'bilinear', 0.0, None)"


def test_mlx_audio_visitor() -> None:
    """Test function test_mlx_audio_visitor."""
    vis = MLXAudioVisitor()
    assert cast("Callable[..., EagerTensorProtocol]", vis.visit_Istft)(cast("Callable[..., EagerTensorProtocol]", DummyNode)({"frame_length": 2048, "frame_step": 512, "center": False}), ["a"]) == "mlx_istft(a, STFTConfig(2048, 512, None, 'hann', False))"
    assert cast("Callable[..., EagerTensorProtocol]", vis.visit_MelFilterbank)(cast("Callable[..., EagerTensorProtocol]", DummyNode)({"num_mel_bins": 1, "num_spectrogram_bins": 2, "sample_rate": 3, "lower_edge_hertz": 4, "upper_edge_hertz": 5}), ["a"]) == "mlx_mel_filterbank(1, 2, 3, 4, 5)"
    assert cast("Callable[..., EagerTensorProtocol]", vis.visit_Mfcc)(cast("Callable[..., EagerTensorProtocol]", DummyNode)({"num_mel_bins": 1, "sample_rate": 2, "lower_edge_hertz": 3, "upper_edge_hertz": 4, "num_mfccs": 5}), ["a"]) == "mlx_mfcc(a, MFCCConfig(2, 1, 3, 4, 5))"
    assert cast("Callable[..., EagerTensorProtocol]", vis.visit_Mfcc)(cast("Callable[..., EagerTensorProtocol]", DummyNode)(), ["a"]) == "mlx_mfcc(a, MFCCConfig(None, 40, 20.0, 4000.0, 13))"


def test_mlx_shape_ops_visitor() -> None:
    """Test function test_mlx_shape_ops_visitor."""
    vis = MLXShapeOpsVisitor()
    assert cast("Callable[..., EagerTensorProtocol]", vis.visit_Concatenate)(cast("Callable[..., EagerTensorProtocol]", DummyNode)({"axis": 1}), ["a"]) == "mx.concatenate(a, axis=1)"
    assert cast("Callable[..., EagerTensorProtocol]", vis.visit_Stack)(cast("Callable[..., EagerTensorProtocol]", DummyNode)({"axis": 1}), ["a"]) == "mx.stack(a, axis=1)"
    assert cast("Callable[..., EagerTensorProtocol]", vis.visit_Partition)(cast("Callable[..., EagerTensorProtocol]", DummyNode)({"axis": 1, "kth": 2}), ["a"]) == "mx.partition(a, 2, axis=1)"
    assert cast("Callable[..., EagerTensorProtocol]", vis.visit_Argpartition)(cast("Callable[..., EagerTensorProtocol]", DummyNode)({"axis": 1, "kth": 2}), ["a"]) == "mx.argpartition(a, 2, axis=1)"
    assert cast("Callable[..., EagerTensorProtocol]", vis.visit_Repeat)(cast("Callable[..., EagerTensorProtocol]", DummyNode)({"repeats": 2, "axis": 1}), ["a"]) == "mx.repeat(a, 2, axis=1)"
    assert cast("Callable[..., EagerTensorProtocol]", vis.visit_Roll)(cast("Callable[..., EagerTensorProtocol]", DummyNode)({"shift": 2, "axis": 1}), ["a"]) == "mx.roll(a, 2, axis=1)"
    assert cast("Callable[..., EagerTensorProtocol]", vis.visit_Tile)(cast("Callable[..., EagerTensorProtocol]", DummyNode)({"reps": 2}), ["a"]) == "mx.tile(a, 2)"
    assert cast("Callable[..., EagerTensorProtocol]", vis.visit_TopK)(cast("Callable[..., EagerTensorProtocol]", DummyNode)({"k": 2, "axis": 1}), ["a"]) == "mx.topk(a, 2, axis=1)"
    assert cast("Callable[..., EagerTensorProtocol]", vis.visit_Moveaxis)(cast("Callable[..., EagerTensorProtocol]", DummyNode)({"source": 1, "destination": 2}), ["a"]) == "mx.moveaxis(a, 1, 2)"
    assert cast("Callable[..., EagerTensorProtocol]", vis.visit_RaggedDot)(cast("Callable[..., EagerTensorProtocol]", DummyNode)(), ["a", "b"]) == "mx.matmul(a, b)"
    assert cast("Callable[..., EagerTensorProtocol]", vis.visit_NanToNum)(cast("Callable[..., EagerTensorProtocol]", DummyNode)({"nan": 1.0, "posinf": 2.0, "neginf": -2.0}), ["a"]) == "mx.nan_to_num(a, nan=1.0, posinf=2.0, neginf=-2.0)"
    assert cast("Callable[..., EagerTensorProtocol]", vis.visit_Zeros)(cast("Callable[..., EagerTensorProtocol]", DummyNode)({"shape": [1, 2], "dtype": "int32"}), ["a"]) == "mx.zeros((1, 2), dtype=mx.int32)"
    assert cast("Callable[..., EagerTensorProtocol]", vis.visit_Zeros)(cast("Callable[..., EagerTensorProtocol]", DummyNode)({"shape": (1, 2), "dtype": None}), ["a"]) == "mx.zeros((1, 2), dtype=None)"
    assert cast("Callable[..., EagerTensorProtocol]", vis.visit_Ones)(cast("Callable[..., EagerTensorProtocol]", DummyNode)({"shape": [1, 2], "dtype": "int32"}), ["a"]) == "mx.ones((1, 2), dtype=mx.int32)"
    assert cast("Callable[..., EagerTensorProtocol]", vis.visit_Full)(cast("Callable[..., EagerTensorProtocol]", DummyNode)({"shape": [1, 2], "fill_value": 3.0, "dtype": "int32"}), ["a"]) == "mx.full((1, 2), 3.0, dtype=mx.int32)"
    assert cast("Callable[..., EagerTensorProtocol]", vis.visit_ConstantOfShape)(cast("Callable[..., EagerTensorProtocol]", DummyNode)({"shape": [1, 2], "fill_value": 3.0, "dtype": "int32"}), ["a"]) == "mx.full((1, 2), 3.0, dtype=mx.int32)"
    assert cast("Callable[..., EagerTensorProtocol]", vis.visit_Transpose)(cast("Callable[..., EagerTensorProtocol]", DummyNode)({"axes": (1, 0)}), ["a"]) == "mx.transpose(a, axes=(1, 0))"
    assert cast("Callable[..., EagerTensorProtocol]", vis.visit_Transpose)(cast("Callable[..., EagerTensorProtocol]", DummyNode)(), ["a"]) == "mx.transpose(a)"
    assert cast("Callable[..., EagerTensorProtocol]", vis.visit_RandomCategorical)(cast("Callable[..., EagerTensorProtocol]", DummyNode)({"num_samples": 2}), ["a"]) == "mx.random.categorical(a, num_samples=2)"


def test_mlx_shape_ops_visitor_branches() -> None:
    """Test function test_mlx_shape_ops_visitor_branches."""
    vis = MLXShapeOpsVisitor()
    assert cast("Callable[..., EagerTensorProtocol]", vis.visit_Full)(cast("Callable[..., EagerTensorProtocol]", DummyNode)({"shape": (1, 2), "fill_value": 3.0, "dtype": "int32"}), ["a"]) == "mx.full((1, 2), 3.0, dtype=mx.int32)"


class DummyGraph:
    """Test class DummyGraph."""

    def __init__(self) -> None:
        """Test function __init__."""
        self.nodes = cast("list[EagerTensorProtocol]", [])


def test_mlx_generator() -> None:
    """Test function test_mlx_generator."""
    g = DummyGraph()
    gen = cast("Callable[..., EagerTensorProtocol]", MLXCodeGenerator)(g)
    assert gen.get_fallback_prefix() == "mx"
    assert gen.get_fallback_prefix() == "mx"
    setattr(gen, "add_line", lambda x: None)
    gen._emit_constant_assignment("var", "42")
    prefix = gen._get_prefix_code()
    assert isinstance(prefix, list)
    assert "import mlx.core as mx" in prefix


def test_mlx_generator_classmethods(monkeypatch: EagerTensorProtocol) -> None:
    """Test function test_mlx_generator_classmethods."""
    pass

    class MockMetal:
        """Test class MockMetal."""

        def set_memory_limit(self, l: EagerValue) -> None:
            """Test function set_memory_limit."""
            pass

        def set_wired_limit(self, l: EagerValue) -> None:
            """Test function set_wired_limit."""
            pass

    class MockMx:
        """Test class MockMx."""

        metal = MockMetal()

        @staticmethod
        def save_gguf(f: EagerValue, a: EagerValue) -> None:
            """Test function save_gguf."""
            pass

        @staticmethod
        def set_default_stream(s: EagerValue) -> None:
            """Test function set_default_stream."""
            pass

    import mlx.core as real_mx

    monkeypatch.setattr(real_mx, "save_gguf", MockMx.save_gguf, raising=False)
    monkeypatch.setattr(real_mx, "set_default_stream", MockMx.set_default_stream, raising=False)
    cast("Callable[..., EagerTensorProtocol]", monkeypatch.setattr)(real_mx, "metal", cast("Callable[..., EagerTensorProtocol]", MockMetal)(), raising=False)
    MLXCodeGenerator.save_gguf("file", {})
    MLXCodeGenerator.set_default_stream("stream")
    MLXCodeGenerator.set_memory_limit(10)
    MLXCodeGenerator.set_wired_limit(10)
    assert callable(MLXCodeGenerator.set_wired_limit)


def test_mlx_generator_classmethods_no_attr(monkeypatch: EagerTensorProtocol) -> None:
    """Test function test_mlx_generator_classmethods_no_attr."""
    import mlx.core as real_mx

    if hasattr(real_mx, "save_gguf"):
        monkeypatch.delattr(real_mx, "save_gguf")
    if hasattr(real_mx, "set_default_stream"):
        monkeypatch.delattr(real_mx, "set_default_stream")
    if hasattr(real_mx, "metal"):
        monkeypatch.delattr(real_mx, "metal")
    MLXCodeGenerator.save_gguf("file", {})
    MLXCodeGenerator.set_default_stream("stream")
    MLXCodeGenerator.set_memory_limit(10)
    MLXCodeGenerator.set_wired_limit(10)
    assert callable(MLXCodeGenerator.set_wired_limit)
