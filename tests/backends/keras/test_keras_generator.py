"""Tests for test_keras_generator."""

from __future__ import annotations

import typing
from pathlib import Path
from unittest.mock import mock_open, patch

import pytest
from ml_switcheroo_ir import AttributeValue, IRGraph, NodeDict

from ml_switcheroo_compiler.backends.keras.generator import (
    KerasCodeGenerator,
    KerasSignatureBuilder,
    KerasTensorManipulator,
)
from ml_switcheroo_compiler.backends.keras.keras_mixins import KerasAudioVisitor, KerasVisionVisitor
from ml_switcheroo_compiler.backends.numpy.eager import ensure_ndarray
from tests.testing_utils import MockGraphStub, MockNodeStub, MockTestStub, NumpyTestArray, coerce_operand_list

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import BackendArg


class DummyNode(MockNodeStub):
    """Test class DummyNode."""

    def __init__(self, attrs: dict[str, AttributeValue] | None = None) -> None:
        """Test function __init__.

        Args:
            attrs (dict[str, AttributeValue] | None): Test parameter `attrs`.
        """
        self.attributes = attrs if attrs is not None else {}


def test_keras_vision_visitor() -> None:
    """Test function test_keras_vision_visitor."""
    vis = KerasVisionVisitor()
    assert (
        vis.visit_ElasticTransform(DummyNode({"data_format": "df", "interpolation": "bicubic"}), ["a", "b"])
        == "keras_elastic_transform(a, b, 'bicubic', 0.0, \"df\")"
    )
    assert vis.visit_ElasticTransform(DummyNode(), ["a", "b"]) == "keras_elastic_transform(a, b, 'bilinear', 0.0, None)"
    assert (
        vis.visit_GaussianBlur(DummyNode({"kernel_size": 3, "sigma": 1.0, "data_format": "df"}), ["a"])
        == "keras_gaussian_blur(a, 3, 1.0, 'same', \"df\")"
    )
    assert vis.visit_GaussianBlur(DummyNode(), ["a"]) == "keras_gaussian_blur(a, None, None, 'same', None)"
    assert (
        vis.visit_MedianFilter(DummyNode({"kernel_size": 3, "sigma": 1.0, "data_format": "df"}), ["a"])
        == "keras_median_filter(a, 3, 'same', \"df\")"
    )
    assert vis.visit_MedianFilter(DummyNode(), ["a"]) == "keras_median_filter(a, None, 'same', None)"
    assert vis.visit_IoU(DummyNode({"bounding_box_format": "cxywh"}), ["a", "b"]) == "keras_iou(a, b, 'cxywh')"
    assert vis.visit_IoU(DummyNode(), ["a", "b"]) == "keras_iou(a, b, 'xyxy')"
    assert (
        vis.visit_NonMaxSuppression(DummyNode({"max_output_size": 10}), ["a", "b"]) == "keras_nms(a, b, 10, 0.5, -inf)"
    )
    assert vis.visit_NonMaxSuppression(DummyNode(), ["a", "b"]) == "keras_nms(a, b, None, 0.5, -inf)"
    assert (
        vis.visit_ResizeBicubic(DummyNode({"size": 10, "align_corners": True}), ["a"])
        == "keras_resize(a, 10, 'bicubic', True)"
    )
    assert vis.visit_ResizeBicubic(DummyNode(), ["a"]) == "keras_resize(a, None, 'bicubic', False)"
    assert (
        vis.visit_ResizeLanczos3(DummyNode({"size": 10, "align_corners": True}), ["a"])
        == "keras_resize(a, 10, 'lanczos3', True)"
    )
    assert vis.visit_ResizeLanczos3(DummyNode(), ["a"]) == "keras_resize(a, None, 'lanczos3', False)"
    assert vis.visit_ExtractBoundingBoxes(DummyNode({"crop_size": 10, "data_format": "df"}), ["a", "b", "c"]) == (
        "keras_extract_bounding_boxes(a, b, c, BoundingBoxExtractionConfig(crop_size=10, "
        "interpolation='bilinear', extrapolation_value=0.0, data_format=\"df\"))"
    )
    assert vis.visit_ExtractBoundingBoxes(DummyNode(), ["a", "b", "c"]) == (
        "keras_extract_bounding_boxes(a, b, c, BoundingBoxExtractionConfig(crop_size=None, "
        "interpolation='bilinear', extrapolation_value=0.0, data_format=None))"
    )
    assert vis.visit_PerspectiveTransform(DummyNode({"data_format": "df"}), ["a", "b", "c"]) == (
        "keras.ops.image.perspective_transform(a, b, c, interpolation='bilinear', fill_value=0.0, data_format=\"df\")"
    )
    assert (
        vis.visit_PerspectiveTransform(DummyNode(), ["a", "b", "c"])
        == "keras.ops.image.perspective_transform(a, b, c, interpolation='bilinear', fill_value=0.0, data_format=None)"
    )


def test_keras_audio_visitor() -> None:
    """Test function test_keras_audio_visitor."""
    vis = KerasAudioVisitor()
    assert (
        vis.visit_Istft(DummyNode({"frame_length": 2048, "frame_step": 512, "center": False}), ["a"])
        == "keras_istft(a, STFTConfig(frame_length=2048, frame_step=512, fft_length=None, window='hann', center=False))"
    )
    assert (
        vis.visit_MelFilterbank(
            DummyNode(
                {
                    "num_mel_bins": 1,
                    "num_spectrogram_bins": 2,
                    "sample_rate": 3,
                    "lower_edge_hertz": 4,
                    "upper_edge_hertz": 5,
                }
            ),
            ["a"],
        )
        == "keras_mel_filterbank(1, 2, 3, 4.0, 5.0)"
    )
    assert (
        vis.visit_Mfcc(
            DummyNode(
                {"num_mel_bins": 1, "sample_rate": 2, "lower_edge_hertz": 3, "upper_edge_hertz": 4, "num_mfccs": 5}
            ),
            ["a"],
        )
        == "keras_mfcc(a, 2, 1, 3.0, 4.0, 5)"
    )


class DummyIRNode(MockNodeStub):
    """Test class DummyIRNode."""

    def __init__(
        self,
        id: str,
        shape_metadata: tuple[int, ...] | None = None,
    ) -> None:
        """Test function __init__.

        Args:
            id (str): Test parameter `id`.
            shape_metadata (tuple[int, ...] | None): Test parameter `shape_metadata`.
        """
        self.id = id
        if shape_metadata:
            self.shape_metadata = shape_metadata


class DummyGraph(MockGraphStub):
    """Test class DummyGraph."""

    nodes: NodeDict = NodeDict()


def test_keras_signature_builder() -> None:
    """Test function test_keras_signature_builder."""
    node = DummyIRNode("in_0", (2, 3))
    assert KerasSignatureBuilder.get_input_assignment("var", node) == "var = keras.Input(shape=(2, 3), name='in_0')"
    node2 = DummyIRNode("in_1")
    assert KerasSignatureBuilder.get_input_assignment("var2", node2) == "var2 = keras.Input(shape=(None,), name='in_1')"
    assert KerasSignatureBuilder.get_return_block(["i1"], ["o1"]) == "return keras.Model(inputs=[i1], outputs=[o1])"


def test_keras_tensor_manipulator() -> None:
    """Test function test_keras_tensor_manipulator."""
    assert KerasTensorManipulator.format_zeros_like("zeros", {}) == "keras.ops.zeros({shape})"
    assert (
        KerasTensorManipulator.format_zeros_like("ones", {"dtype": "int32"}) == "keras.ops.ones({shape}), dtype='int32'"
    )
    assert KerasTensorManipulator.format_full({}) == "keras.ops.full({shape}, {fill_value})"
    assert (
        KerasTensorManipulator.format_full({"dtype": "int32"}) == "keras.ops.full({shape}, {fill_value}), dtype='int32'"
    )
    assert KerasTensorManipulator.format_transpose({}) == "keras.ops.transpose({0})"
    assert KerasTensorManipulator.format_transpose({"axes": (1, 0)}) == "keras.ops.transpose({0}, {axes})"


def test_keras_generator() -> None:
    """Test function test_keras_generator."""
    gen = KerasCodeGenerator(DummyGraph())
    assert gen.get_fallback_prefix() == "keras.ops"
    assert gen.get_fallback_prefix() == "keras.ops"
    assert gen.visit_RaggedDot(DummyNode(), ["a", "b"]) == "keras_ragged_dot(a, b)"
    ops = gen.get_ops_map({})
    assert "Zeros" in ops
    assert "Transpose" in ops
    gen._emit_input_assignment("var1", DummyIRNode("in"), "inputs", 0)
    assert "var1 = keras.Input" in gen.code[-1]
    assert "var1" in gen.keras_input_vars
    gen._emit_body_return(["out1"])
    assert "out1" in gen.keras_output_vars
    gen._emit_output_assignment(DummyNode(), ["out2"], "out2")
    assert "out2" in gen.keras_output_vars
    assert gen._generate_file_header() == [gen.header.strip()]
    m_open = mock_open(read_data='imports: "import tmp"\nfunctions:\n  dummy: "def dummy(): pass"')
    with patch("builtins.open", m_open):
        imports = gen._resolve_imports()
        assert "import keras\n" in imports
        assert "import tmp" in imports
        assert "def dummy(): pass" in imports
    pytest.MonkeyPatch().setattr(gen, "code", [], raising=False)
    gen._generate_function_signature()
    assert "def get_model() -> object:" in gen.code[-1]
    assert gen.keras_input_vars == []
    pytest.MonkeyPatch().setattr(gen, "keras_input_vars", ["i"], raising=False)
    pytest.MonkeyPatch().setattr(gen, "keras_output_vars", ["o"], raising=False)
    from ml_switcheroo_compiler.backends.keras.generator import KerasSignatureBuilder

    gen.code.append(KerasSignatureBuilder.get_return_block(gen.keras_input_vars, gen.keras_output_vars))
    assert "return keras.Model(inputs=[i], outputs=[o])" in gen.code[-1]


def test_keras_generator_save_load(tmp_path: Path) -> None:
    """Test save and load methods of Keras generator.

    Args:
        tmp_path: Test parameter `tmp_path`.
    """
    arr = [1, 2, 3]
    filepath = str(tmp_path / "dummy.pkl")
    KerasCodeGenerator.save(filepath, arr)
    res = KerasCodeGenerator.load(filepath)
    assert res == arr
    filez = str(tmp_path / "dummy.npz")
    KerasCodeGenerator.savez(filez, arr, arg2=[4, 5])
    resz = KerasCodeGenerator.load(filez)
    assert isinstance(resz, dict)
    assert resz["arr_0"] == arr
    assert resz["arg2"] == [4, 5]
    filezc = str(tmp_path / "dummy_comp.npz")
    KerasCodeGenerator.savez_compressed(filezc, arr, arg2=[4, 5])
    import gzip
    import pickle

    with gzip.open(filezc, "rb") as f:
        reszc = pickle.load(f)
    assert reszc["arr_0"] == arr
    assert reszc["arg2"] == [4, 5]


def test_keras_generator_conv_transpose() -> None:
    """Test function test_keras_generator_conv_transpose."""
    from ml_switcheroo_ir import LogicalGraph

    gen = KerasCodeGenerator(LogicalGraph())

    class DummyNode(MockNodeStub):
        """Test class DummyNode."""

        pass

    assert gen.visit_ConvTranspose(DummyNode(), ["inp1", "inp2"]) == "keras_conv_transpose(inp1, inp2)"
    assert gen.visit_RaggedDot(DummyNode(), ["inp1", "inp2"]) == "keras_ragged_dot(inp1, inp2)"


def test_keras_generator_empty_prefix_yaml() -> None:
    """Test function test_keras_generator_empty_prefix_yaml."""
    from unittest.mock import patch

    from ml_switcheroo_ir import IRGraph

    from ml_switcheroo_compiler.backends.keras.generator import KerasCodeGenerator

    g = IRGraph()
    gen = KerasCodeGenerator(g)
    assert gen._get_backend_prefix() == "keras"
    with patch("yaml.safe_load", return_value={}):
        lines = gen._resolve_imports()
        assert lines == ["import keras\n"]


def test_keras_generator_generate_cst() -> None:
    """Test function test_keras_generator_generate_cst."""
    graph = IRGraph()
    gen = KerasCodeGenerator(graph)
    code = gen.generate()
    assert isinstance(code, str)
    assert "get_model(" in code


def test_keras_generator_return_only_outputs() -> None:
    """Test function test_keras_generator_return_only_outputs."""
    from ml_switcheroo_ir import IRNode

    graph = IRGraph()
    graph.nodes["const_0"] = IRNode(id="const_0", op_type="Constant", inputs=[])
    gen = KerasCodeGenerator(graph)

    def mock_generate_body(prefix: BackendArg) -> None:
        """Test function mock_generate_body.

        Args:
            prefix: Test parameter `prefix`.
        """
        gen.keras_input_vars = []
        gen.keras_output_vars = ["out_var"]

    pytest.MonkeyPatch().setattr(gen, "_generate_body", mock_generate_body, raising=False)
    code = gen.generate()
    assert "return out_var" in code


def test_keras_generator_return_block_both_inputs_and_outputs() -> None:
    """Test generate() when both keras_input_vars and keras_output_vars are present."""
    from ml_switcheroo_ir import IRNode

    graph = IRGraph()
    graph.nodes["const_0"] = IRNode(id="const_0", op_type="Constant", inputs=[])
    gen = KerasCodeGenerator(graph)

    def mock_generate_body(prefix: BackendArg) -> None:
        """Test function mock_generate_body.

        Args:
            prefix: Test parameter `prefix`.
        """
        gen.keras_input_vars = ["in_var"]
        gen.keras_output_vars = ["out_var"]

    pytest.MonkeyPatch().setattr(gen, "_generate_body", mock_generate_body, raising=False)
    code = gen.generate()
    assert "return keras.Model(inputs=[in_var], outputs=[out_var])" in code


def test_keras_generator_empty_inputs_and_outputs() -> None:
    """Test generate() when keras_input_vars is set but keras_output_vars is empty."""
    from ml_switcheroo_ir import IRNode

    graph = IRGraph()
    graph.nodes["dummy"] = IRNode(id="dummy", op_type="Identity", inputs=[])
    gen = KerasCodeGenerator(graph)

    def mock_generate_body(prefix: BackendArg) -> None:
        """Test function mock_generate_body.

        Args:
            prefix: Test parameter `prefix`.
        """
        gen.keras_input_vars = ["in0"]
        gen.keras_output_vars = []

    pytest.MonkeyPatch().setattr(gen, "_generate_body", mock_generate_body, raising=False)
    code = gen.generate()
    assert "pass" in code


def test_keras_generator_compile_aot() -> None:
    """Test KerasCodeGenerator._compile_aot_impl across successful exec and evaluator fallback."""
    import numpy as np
    from ml_switcheroo_ir import IRNode

    graph = IRGraph()
    gen = KerasCodeGenerator(graph)
    with patch.object(gen, "generate", return_value="def get_model(): return lambda x, **kw: x + 1"):
        artifact = gen.compile_aot(graph)
        assert artifact(41) == 42
        assert artifact.metadata is not None
        assert artifact.metadata["backend"] == "keras"
    gen_no_get_model = KerasCodeGenerator(graph)
    with patch.object(gen_no_get_model, "generate", return_value="x_const = 10"):
        artifact_no_gm = gen_no_get_model.compile_aot(graph)
        assert artifact_no_gm.metadata is not None
        assert artifact_no_gm.metadata["model"] is None
    g_single = IRGraph()
    node_in = IRNode(id="in0", op_type="Input", inputs=[], shape_metadata=[2])
    node_id = IRNode(id="out0", op_type="Identity", inputs=["in0"], shape_metadata=[2])
    g_single.nodes = NodeDict({"in0": node_in, "out0": node_id})
    g_single.inputs = ["in0"]
    g_single.outputs = ["out0"]

    class MockTensorWithData(MockTestStub):
        """Mock tensor wrapper holding underlying array data."""

        def __init__(self, data: NumpyTestArray) -> None:
            """Initialize mock tensor.

            Args:
                data (NumpyTestArray): Underlying array.
            """
            self.data: NumpyTestArray = data

    gen_single = KerasCodeGenerator(g_single)
    with patch.object(gen_single, "generate", return_value="invalid python syntax !!!"):
        art_single = gen_single.compile_aot(g_single)
        res_single = art_single(MockTensorWithData(np.array([1, 2])), "extra_arg")
        assert np.array_equal(ensure_ndarray(res_single), np.array([1, 2]))
    g_no_out = IRGraph()
    g_no_out.nodes = NodeDict({"in0": node_in})
    g_no_out.inputs = ["in0"]
    g_no_out.outputs = []
    gen_no_out = KerasCodeGenerator(g_no_out)
    with patch.object(gen_no_out, "generate", return_value="invalid python syntax !!!"):
        art_no_out = gen_no_out.compile_aot(g_no_out)
        res_no_out = art_no_out(np.array([1, 2]))
        assert isinstance(res_no_out, dict)
    g_multi = IRGraph()
    node_out1 = IRNode(id="out1", op_type="Identity", inputs=["in0"], shape_metadata=[2])
    node_out2 = IRNode(id="out2", op_type="Identity", inputs=["in0"], shape_metadata=[2])
    g_multi.nodes = NodeDict({"in0": node_in, "out1": node_out1, "out2": node_out2})
    g_multi.inputs = ["in0"]
    g_multi.outputs = ["out1", "out2"]
    gen_multi = KerasCodeGenerator(g_multi)
    with patch.object(gen_multi, "generate", return_value="invalid syntax"):
        art_multi = gen_multi.compile_aot(g_multi)
        res_multi = art_multi(np.array([5, 6]))
        assert len(coerce_operand_list(res_multi)) == 2
        assert np.array_equal(ensure_ndarray(coerce_operand_list(res_multi)[0]), np.array([5, 6]))
        assert np.array_equal(ensure_ndarray(coerce_operand_list(res_multi)[1]), np.array([5, 6]))
