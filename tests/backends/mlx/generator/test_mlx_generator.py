"""Tests for test_mlx_generator."""

from __future__ import annotations
from tests.testing_utils import as_callable, as_tensor, as_tensor_list
from ml_switcheroo_compiler.backends.eager_registry import EagerValue

import typing

# ruff: noqa
import pytest
from ml_switcheroo_compiler.backends.mlx.mlx_mixins import MLXAudioVisitor, MLXNNOpsVisitor, MLXOpRegistryMixin, MLXShapeOpsVisitor, MLXVisionVisitor
from ml_switcheroo_compiler.backends.mlx.generator import MLXCodeGenerator

if typing.TYPE_CHECKING:
    from collections.abc import Callable
    from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol, EagerTensorProtocol, EagerValue

pytest.importorskip("mlx.core")


class DummyNode:
    """Test class DummyNode."""

    def __init__(self, attrs: EagerValue = None) -> None:
        """Test function __init__.

        Args:
            attrs: Test parameter `attrs`.
        """
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
    assert as_callable(vis.visit_Rope)(as_callable(DummyNode)({"dim": 1, "traditional": True, "base": 2.0, "scale": 3.0, "offset": 4}), ["a"]) == "mx.fast.rope(a, 1, traditional=True, base=2.0, scale=3.0, offset=4)"
    assert as_callable(vis.visit_PowerIteration)(as_callable(DummyNode)({"num_iters": 2}), ["a", "u"]) == "mlx_power_iteration(a, 2, u)"
    assert as_callable(vis.visit_PowerIteration)(as_callable(DummyNode)(), ["a"]) == "mlx_power_iteration(a, 1, None)"


def test_mlx_vision_visitor() -> None:
    """Test function test_mlx_vision_visitor."""
    vis = MLXVisionVisitor()
    assert as_callable(vis.visit_ElasticTransform)(as_callable(DummyNode)({"data_format": "df", "interpolation": "bicubic"}), ["a", "b"]) == "mlx_elastic_transform(a, b, 'bicubic', 0.0, \"df\")"
    assert as_callable(vis.visit_ElasticTransform)(as_callable(DummyNode)(), ["a", "b"]) == "mlx_elastic_transform(a, b, 'bilinear', 0.0, None)"
    assert as_callable(vis.visit_GaussianBlur)(as_callable(DummyNode)({"kernel_size": 3, "sigma": 1.0, "data_format": "df"}), ["a"]) == "mlx_gaussian_blur(a, 3, 1.0, 'same', \"df\")"
    assert as_callable(vis.visit_GaussianBlur)(as_callable(DummyNode)(), ["a"]) == "mlx_gaussian_blur(a, None, None, 'same', None)"
    assert as_callable(vis.visit_MedianFilter)(as_callable(DummyNode)({"kernel_size": 3, "sigma": 1.0, "data_format": "df"}), ["a"]) == "mlx_median_filter(a, 3, 'same', \"df\")"
    assert as_callable(vis.visit_MedianFilter)(as_callable(DummyNode)(), ["a"]) == "mlx_median_filter(a, None, 'same', None)"
    assert as_callable(vis.visit_ExtractBoundingBoxes)(as_callable(DummyNode)({"crop_size": 10, "data_format": "df"}), ["a", "b", "c"]) == "mlx_extract_bounding_boxes(a, b, c, 10, 'bilinear', 0.0, \"df\")"
    assert as_callable(vis.visit_ExtractBoundingBoxes)(as_callable(DummyNode)(), ["a", "b", "c"]) == "mlx_extract_bounding_boxes(a, b, c, None, 'bilinear', 0.0, None)"
    assert as_callable(vis.visit_IoU)(as_callable(DummyNode)({"bounding_box_format": "cxywh"}), ["a", "b"]) == "mlx_iou(a, b, 'cxywh')"
    assert as_callable(vis.visit_IoU)(as_callable(DummyNode)(), ["a", "b"]) == "mlx_iou(a, b, 'xyxy')"
    assert as_callable(vis.visit_NonMaxSuppression)(as_callable(DummyNode)({"max_output_size": 10}), ["a", "b"]) == "mlx_nms(a, b, 10, 0.5, -inf)"
    assert as_callable(vis.visit_NonMaxSuppression)(as_callable(DummyNode)(), ["a", "b"]) == "mlx_nms(a, b, None, 0.5, -inf)"
    assert as_callable(vis.visit_ResizeBicubic)(as_callable(DummyNode)({"size": 10, "align_corners": True}), ["a"]) == "mlx_resize(a, 10, 'bicubic', True)"
    assert as_callable(vis.visit_ResizeBicubic)(as_callable(DummyNode)(), ["a"]) == "mlx_resize(a, None, 'bicubic', False)"
    assert as_callable(vis.visit_ResizeLanczos3)(as_callable(DummyNode)({"size": 10, "align_corners": True}), ["a"]) == "mlx_resize(a, 10, 'lanczos3', True)"
    assert as_callable(vis.visit_ResizeLanczos3)(as_callable(DummyNode)(), ["a"]) == "mlx_resize(a, None, 'lanczos3', False)"
    assert as_callable(vis.visit_PerspectiveTransform)(as_callable(DummyNode)({"data_format": "df"}), ["a", "b", "c"]) == "mlx_perspective_transform(a, b, c, 'bilinear', 0.0, \"df\")"
    assert as_callable(vis.visit_PerspectiveTransform)(as_callable(DummyNode)(), ["a", "b", "c"]) == "mlx_perspective_transform(a, b, c, 'bilinear', 0.0, None)"


def test_mlx_audio_visitor() -> None:
    """Test function test_mlx_audio_visitor."""
    vis = MLXAudioVisitor()
    assert as_callable(vis.visit_Istft)(as_callable(DummyNode)({"frame_length": 2048, "frame_step": 512, "center": False}), ["a"]) == "mlx_istft(a, STFTConfig(2048, 512, None, 'hann', False))"
    assert as_callable(vis.visit_MelFilterbank)(as_callable(DummyNode)({"num_mel_bins": 1, "num_spectrogram_bins": 2, "sample_rate": 3, "lower_edge_hertz": 4, "upper_edge_hertz": 5}), ["a"]) == "mlx_mel_filterbank(1, 2, 3, 4, 5)"
    assert as_callable(vis.visit_Mfcc)(as_callable(DummyNode)({"num_mel_bins": 1, "sample_rate": 2, "lower_edge_hertz": 3, "upper_edge_hertz": 4, "num_mfccs": 5}), ["a"]) == "mlx_mfcc(a, MFCCConfig(2, 1, 3, 4, 5))"
    assert as_callable(vis.visit_Mfcc)(as_callable(DummyNode)(), ["a"]) == "mlx_mfcc(a, MFCCConfig(None, 40, 20.0, 4000.0, 13))"


def test_mlx_shape_ops_visitor() -> None:
    """Test function test_mlx_shape_ops_visitor."""
    vis = MLXShapeOpsVisitor()
    assert as_callable(vis.visit_Concatenate)(as_callable(DummyNode)({"axis": 1}), ["a"]) == "mx.concatenate(a, axis=1)"
    assert as_callable(vis.visit_Stack)(as_callable(DummyNode)({"axis": 1}), ["a"]) == "mx.stack(a, axis=1)"
    assert as_callable(vis.visit_Partition)(as_callable(DummyNode)({"axis": 1, "kth": 2}), ["a"]) == "mx.partition(a, 2, axis=1)"
    assert as_callable(vis.visit_Argpartition)(as_callable(DummyNode)({"axis": 1, "kth": 2}), ["a"]) == "mx.argpartition(a, 2, axis=1)"
    assert as_callable(vis.visit_Repeat)(as_callable(DummyNode)({"repeats": 2, "axis": 1}), ["a"]) == "mx.repeat(a, 2, axis=1)"
    assert as_callable(vis.visit_Roll)(as_callable(DummyNode)({"shift": 2, "axis": 1}), ["a"]) == "mx.roll(a, 2, axis=1)"
    assert as_callable(vis.visit_Tile)(as_callable(DummyNode)({"reps": 2}), ["a"]) == "mx.tile(a, 2)"
    assert as_callable(vis.visit_TopK)(as_callable(DummyNode)({"k": 2, "axis": 1}), ["a"]) == "mx.topk(a, 2, axis=1)"
    assert as_callable(vis.visit_Moveaxis)(as_callable(DummyNode)({"source": 1, "destination": 2}), ["a"]) == "mx.moveaxis(a, 1, 2)"
    assert as_callable(vis.visit_RaggedDot)(as_callable(DummyNode)(), ["a", "b"]) == "mx.matmul(a, b)"
    assert as_callable(vis.visit_NanToNum)(as_callable(DummyNode)({"nan": 1.0, "posinf": 2.0, "neginf": -2.0}), ["a"]) == "mx.nan_to_num(a, nan=1.0, posinf=2.0, neginf=-2.0)"
    assert as_callable(vis.visit_Zeros)(as_callable(DummyNode)({"shape": [1, 2], "dtype": "int32"}), ["a"]) == "mx.zeros((1, 2), dtype=mx.int32)"
    assert as_callable(vis.visit_Zeros)(as_callable(DummyNode)({"shape": (1, 2), "dtype": None}), ["a"]) == "mx.zeros((1, 2), dtype=None)"
    assert as_callable(vis.visit_Ones)(as_callable(DummyNode)({"shape": [1, 2], "dtype": "int32"}), ["a"]) == "mx.ones((1, 2), dtype=mx.int32)"
    assert as_callable(vis.visit_Full)(as_callable(DummyNode)({"shape": [1, 2], "fill_value": 3.0, "dtype": "int32"}), ["a"]) == "mx.full((1, 2), 3.0, dtype=mx.int32)"
    assert as_callable(vis.visit_ConstantOfShape)(as_callable(DummyNode)({"shape": [1, 2], "fill_value": 3.0, "dtype": "int32"}), ["a"]) == "mx.full((1, 2), 3.0, dtype=mx.int32)"
    assert as_callable(vis.visit_Transpose)(as_callable(DummyNode)({"axes": (1, 0)}), ["a"]) == "mx.transpose(a, axes=(1, 0))"
    assert as_callable(vis.visit_Transpose)(as_callable(DummyNode)(), ["a"]) == "mx.transpose(a)"
    assert as_callable(vis.visit_RandomCategorical)(as_callable(DummyNode)({"num_samples": 2}), ["a"]) == "mx.random.categorical(a, num_samples=2)"


def test_mlx_shape_ops_visitor_branches() -> None:
    """Test function test_mlx_shape_ops_visitor_branches."""
    vis = MLXShapeOpsVisitor()
    assert as_callable(vis.visit_Full)(as_callable(DummyNode)({"shape": (1, 2), "fill_value": 3.0, "dtype": "int32"}), ["a"]) == "mx.full((1, 2), 3.0, dtype=mx.int32)"


class DummyGraph:
    """Test class DummyGraph."""

    def __init__(self) -> None:
        """Test function __init__."""
        self.nodes = as_tensor_list([])


def test_mlx_generator() -> None:
    """Test function test_mlx_generator."""
    g = DummyGraph()
    gen = as_callable(MLXCodeGenerator)(g)
    assert gen.get_fallback_prefix() == "mx"
    assert gen.get_fallback_prefix() == "mx"
    setattr(gen, "add_line", lambda x: None)
    gen._emit_constant_assignment("var", "42")
    prefix = gen._get_prefix_code()
    assert isinstance(as_tensor(prefix), list)
    assert "import mlx.core as mx" in prefix


def test_mlx_generator_classmethods(monkeypatch: EagerTensorProtocol) -> None:
    """Test function test_mlx_generator_classmethods.

    Args:
        monkeypatch: Test parameter `monkeypatch`.
    """
    pass

    class MockMetal:
        """Test class MockMetal."""

        def set_memory_limit(self, l: EagerValue) -> None:
            """Test function set_memory_limit.

            Args:
                l: Test parameter `l`.
            """
            pass

        def set_wired_limit(self, l: EagerValue) -> None:
            """Test function set_wired_limit.

            Args:
                l: Test parameter `l`.
            """
            pass

    class MockMx:
        """Test class MockMx."""

        metal = MockMetal()

        @staticmethod
        def save_gguf(f: EagerValue, a: EagerValue) -> None:
            """Test function save_gguf.

            Args:
                f: Test parameter `f`.
                a: Test parameter `a`.
            """
            pass

        @staticmethod
        def set_default_stream(s: EagerValue) -> None:
            """Test function set_default_stream.

            Args:
                s: Test parameter `s`.
            """
            pass

    import mlx.core as real_mx

    monkeypatch.setattr(real_mx, "save_gguf", MockMx.save_gguf, raising=False)
    monkeypatch.setattr(real_mx, "set_default_stream", MockMx.set_default_stream, raising=False)
    as_callable(monkeypatch.setattr)(real_mx, "metal", as_callable(MockMetal)(), raising=False)
    MLXCodeGenerator.save_gguf("file", {})
    MLXCodeGenerator.set_default_stream("stream")
    MLXCodeGenerator.set_memory_limit(10)
    MLXCodeGenerator.set_wired_limit(10)
    assert callable(MLXCodeGenerator.set_wired_limit)


def test_mlx_generator_classmethods_no_attr(monkeypatch: EagerTensorProtocol) -> None:
    """Test function test_mlx_generator_classmethods_no_attr.

    Args:
        monkeypatch: Test parameter `monkeypatch`.
    """
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
