"""Tests for test_keras_generator."""

from __future__ import annotations

from typing import cast
from typing import TYPE_CHECKING

# ruff: noqa
from unittest.mock import mock_open, patch
from ml_switcheroo_compiler.backends.keras.keras_mixins import KerasAudioVisitor, KerasVisionVisitor
from ml_switcheroo_compiler.backends.keras.generator import KerasCodeGenerator, KerasSignatureBuilder, KerasTensorManipulator
from ml_switcheroo_compiler.backends.keras.generator import KerasCodeGenerator
from ml_switcheroo_compiler.backends.keras.generator import KerasCodeGenerator
from ml_switcheroo_compiler.ir.core import IRGraph

if TYPE_CHECKING:
    from collections.abc import Callable
    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue


class DummyNode:
    """Test class DummyNode."""

    def __init__(self, attrs: EagerValue = None) -> None:
        """Test function __init__."""
        self.attributes = attrs or {}


def test_keras_vision_visitor() -> None:
    """Test function test_keras_vision_visitor."""
    vis = KerasVisionVisitor()
    assert cast("Callable[..., EagerTensorProtocol]", vis.visit_ElasticTransform)(cast("Callable[..., EagerTensorProtocol]", DummyNode)({"data_format": "df", "interpolation": "bicubic"}), ["a", "b"]) == "keras_elastic_transform(a, b, 'bicubic', 0.0, \"df\")"
    assert cast("Callable[..., EagerTensorProtocol]", vis.visit_ElasticTransform)(cast("Callable[..., EagerTensorProtocol]", DummyNode)(), ["a", "b"]) == "keras_elastic_transform(a, b, 'bilinear', 0.0, None)"
    assert cast("Callable[..., EagerTensorProtocol]", vis.visit_GaussianBlur)(cast("Callable[..., EagerTensorProtocol]", DummyNode)({"kernel_size": 3, "sigma": 1.0, "data_format": "df"}), ["a"]) == "keras_gaussian_blur(a, 3, 1.0, 'same', \"df\")"
    assert cast("Callable[..., EagerTensorProtocol]", vis.visit_GaussianBlur)(cast("Callable[..., EagerTensorProtocol]", DummyNode)(), ["a"]) == "keras_gaussian_blur(a, None, None, 'same', None)"
    assert cast("Callable[..., EagerTensorProtocol]", vis.visit_MedianFilter)(cast("Callable[..., EagerTensorProtocol]", DummyNode)({"kernel_size": 3, "sigma": 1.0, "data_format": "df"}), ["a"]) == "keras_median_filter(a, 3, 'same', \"df\")"
    assert cast("Callable[..., EagerTensorProtocol]", vis.visit_MedianFilter)(cast("Callable[..., EagerTensorProtocol]", DummyNode)(), ["a"]) == "keras_median_filter(a, None, 'same', None)"
    assert cast("Callable[..., EagerTensorProtocol]", vis.visit_IoU)(cast("Callable[..., EagerTensorProtocol]", DummyNode)({"bounding_box_format": "cxywh"}), ["a", "b"]) == "keras_iou(a, b, 'cxywh')"
    assert cast("Callable[..., EagerTensorProtocol]", vis.visit_IoU)(cast("Callable[..., EagerTensorProtocol]", DummyNode)(), ["a", "b"]) == "keras_iou(a, b, 'xyxy')"
    assert cast("Callable[..., EagerTensorProtocol]", vis.visit_NonMaxSuppression)(cast("Callable[..., EagerTensorProtocol]", DummyNode)({"max_output_size": 10}), ["a", "b"]) == "keras_nms(a, b, 10, 0.5, -inf)"
    assert cast("Callable[..., EagerTensorProtocol]", vis.visit_NonMaxSuppression)(cast("Callable[..., EagerTensorProtocol]", DummyNode)(), ["a", "b"]) == "keras_nms(a, b, None, 0.5, -inf)"
    assert cast("Callable[..., EagerTensorProtocol]", vis.visit_ResizeBicubic)(cast("Callable[..., EagerTensorProtocol]", DummyNode)({"size": 10, "align_corners": True}), ["a"]) == "keras_resize(a, 10, 'bicubic', True)"
    assert cast("Callable[..., EagerTensorProtocol]", vis.visit_ResizeBicubic)(cast("Callable[..., EagerTensorProtocol]", DummyNode)(), ["a"]) == "keras_resize(a, None, 'bicubic', False)"
    assert cast("Callable[..., EagerTensorProtocol]", vis.visit_ResizeLanczos3)(cast("Callable[..., EagerTensorProtocol]", DummyNode)({"size": 10, "align_corners": True}), ["a"]) == "keras_resize(a, 10, 'lanczos3', True)"
    assert cast("Callable[..., EagerTensorProtocol]", vis.visit_ResizeLanczos3)(cast("Callable[..., EagerTensorProtocol]", DummyNode)(), ["a"]) == "keras_resize(a, None, 'lanczos3', False)"
    assert (
        cast("Callable[..., EagerTensorProtocol]", vis.visit_ExtractBoundingBoxes)(cast("Callable[..., EagerTensorProtocol]", DummyNode)({"crop_size": 10, "data_format": "df"}), ["a", "b", "c"])
        == "keras_extract_bounding_boxes(a, b, c, BoundingBoxExtractionConfig(crop_size=10, interpolation='bilinear', extrapolation_value=0.0, data_format=\"df\"))"
    )
    assert (
        cast("Callable[..., EagerTensorProtocol]", vis.visit_ExtractBoundingBoxes)(cast("Callable[..., EagerTensorProtocol]", DummyNode)(), ["a", "b", "c"])
        == "keras_extract_bounding_boxes(a, b, c, BoundingBoxExtractionConfig(crop_size=None, interpolation='bilinear', extrapolation_value=0.0, data_format=None))"
    )
    assert cast("Callable[..., EagerTensorProtocol]", vis.visit_PerspectiveTransform)(cast("Callable[..., EagerTensorProtocol]", DummyNode)({"data_format": "df"}), ["a", "b", "c"]) == "keras.ops.image.perspective_transform(a, b, c, interpolation='bilinear', fill_value=0.0, data_format=\"df\")"
    assert cast("Callable[..., EagerTensorProtocol]", vis.visit_PerspectiveTransform)(cast("Callable[..., EagerTensorProtocol]", DummyNode)(), ["a", "b", "c"]) == "keras.ops.image.perspective_transform(a, b, c, interpolation='bilinear', fill_value=0.0, data_format=None)"


def test_keras_audio_visitor() -> None:
    """Test function test_keras_audio_visitor."""
    vis = KerasAudioVisitor()
    assert (
        cast("Callable[..., EagerTensorProtocol]", vis.visit_Istft)(cast("Callable[..., EagerTensorProtocol]", DummyNode)({"frame_length": 2048, "frame_step": 512, "center": False}), ["a"])
        == "keras_istft(a, STFTConfig(frame_length=2048, frame_step=512, fft_length=None, window='hann', center=False))"
    )
    assert cast("Callable[..., EagerTensorProtocol]", vis.visit_MelFilterbank)(cast("Callable[..., EagerTensorProtocol]", DummyNode)({"num_mel_bins": 1, "num_spectrogram_bins": 2, "sample_rate": 3, "lower_edge_hertz": 4, "upper_edge_hertz": 5}), ["a"]) == "keras_mel_filterbank(1, 2, 3, 4.0, 5.0)"
    assert cast("Callable[..., EagerTensorProtocol]", vis.visit_Mfcc)(cast("Callable[..., EagerTensorProtocol]", DummyNode)({"num_mel_bins": 1, "sample_rate": 2, "lower_edge_hertz": 3, "upper_edge_hertz": 4, "num_mfccs": 5}), ["a"]) == "keras_mfcc(a, 2, 1, 3.0, 4.0, 5)"


class DummyIRNode:
    """Test class DummyIRNode."""

    def __init__(self, id: EagerValue, shape_metadata: EagerValue = None) -> None:
        """Test function __init__."""
        self.id = id
        if shape_metadata:
            self.shape_metadata = shape_metadata


class DummyGraph:
    """Test class DummyGraph."""

    nodes = cast("list[EagerTensorProtocol]", [])


def test_keras_signature_builder() -> None:
    """Test function test_keras_signature_builder."""
    node = DummyIRNode("in_0", (None, 3))
    assert cast("Callable[..., EagerTensorProtocol]", KerasSignatureBuilder.get_input_assignment)("var", node) == "var = keras.Input(shape=(None, 3), name='in_0')"
    node2 = DummyIRNode("in_1")
    assert cast("Callable[..., EagerTensorProtocol]", KerasSignatureBuilder.get_input_assignment)("var2", node2) == "var2 = keras.Input(shape=(None,), name='in_1')"
    assert KerasSignatureBuilder.get_return_block(["i1"], ["o1"]) == "return keras.Model(inputs=[i1], outputs=[o1])"


def test_keras_tensor_manipulator() -> None:
    """Test function test_keras_tensor_manipulator."""
    assert KerasTensorManipulator.format_zeros_like("zeros", {}) == "keras.ops.zeros({shape})"
    assert KerasTensorManipulator.format_zeros_like("ones", {"dtype": "int32"}) == "keras.ops.ones({shape}), dtype='int32'"
    assert KerasTensorManipulator.format_full({}) == "keras.ops.full({shape}, {fill_value})"
    assert KerasTensorManipulator.format_full({"dtype": "int32"}) == "keras.ops.full({shape}, {fill_value}), dtype='int32'"
    assert KerasTensorManipulator.format_transpose({}) == "keras.ops.transpose({0})"
    assert KerasTensorManipulator.format_transpose({"axes": (1, 0)}) == "keras.ops.transpose({0}, {axes})"


def test_keras_generator() -> None:
    """Test function test_keras_generator."""
    gen = cast("Callable[..., EagerTensorProtocol]", KerasCodeGenerator)(cast("Callable[..., EagerTensorProtocol]", DummyGraph)())
    assert gen.get_fallback_prefix() == "keras.ops"
    assert gen.get_fallback_prefix() == "keras.ops"
    assert cast("Callable[..., EagerTensorProtocol]", gen.visit_RaggedDot)(None, ["a", "b"]) == "keras_ragged_dot(a, b)"
    ops = gen.get_ops_map({})
    assert "Zeros" in ops
    assert "Transpose" in ops
    cast("Callable[..., EagerTensorProtocol]", gen._emit_input_assignment)("var1", cast("Callable[..., EagerTensorProtocol]", DummyIRNode)("in"), "inputs", 0)
    assert "var1 = keras.Input" in gen.code[-1]
    assert "var1" in gen.keras_input_vars
    gen._emit_body_return(["out1"])
    assert "out1" in gen.keras_output_vars
    cast("Callable[..., EagerTensorProtocol]", gen._emit_output_assignment)(None, ["out2"], "out2")
    assert "out2" in gen.keras_output_vars
    assert gen._generate_file_header() == [gen.header.strip()]
    m_open = mock_open(read_data='imports: "import tmp"\nfunctions:\n  dummy: "def dummy(): pass"')
    with patch("builtins.open", m_open):
        imports = gen._resolve_imports()
        assert "import keras\n" in imports
        assert "import tmp" in imports
        assert "def dummy(): pass" in imports
    setattr(gen, "code", [])
    gen._generate_function_signature()
    assert "def get_model() -> object:" in gen.code[-1]
    assert gen.keras_input_vars == []
    setattr(gen, "keras_input_vars", ["i"])
    setattr(gen, "keras_output_vars", ["o"])
    from ml_switcheroo_compiler.backends.keras.generator import KerasSignatureBuilder

    cast("Callable[..., EagerTensorProtocol]", gen.code.append)(cast("Callable[..., EagerTensorProtocol]", KerasSignatureBuilder.get_return_block)(gen.keras_input_vars, gen.keras_output_vars))
    assert "return keras.Model(inputs=[i], outputs=[o])" in gen.code[-1]


def test_keras_generator_save_load(tmp_path: EagerTensorProtocol) -> None:
    """Test save and load methods of Keras generator."""
    arr = [1, 2, 3]
    filepath = str(tmp_path / "dummy.pkl")

    # save
    KerasCodeGenerator.save(filepath, arr)

    # load
    res = KerasCodeGenerator.load(filepath)
    assert res == arr

    # savez
    filez = str(tmp_path / "dummy.npz")
    KerasCodeGenerator.savez(filez, arr, arg2=[4, 5])

    resz = KerasCodeGenerator.load(filez)
    assert cast("EagerTensorProtocol", getattr(resz, "__getitem__")("arr_0")) == arr
    assert cast("EagerTensorProtocol", getattr(resz, "__getitem__")("arg2")) == [4, 5]

    # savez_compressed
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

    class DummyNode:
        """Test class DummyNode."""

        pass

    assert cast("Callable[..., EagerTensorProtocol]", gen.visit_ConvTranspose)(cast("Callable[..., EagerTensorProtocol]", DummyNode)(), ["inp1", "inp2"]) == "keras_conv_transpose(inp1, inp2)"
    assert cast("Callable[..., EagerTensorProtocol]", gen.visit_RaggedDot)(cast("Callable[..., EagerTensorProtocol]", DummyNode)(), ["inp1", "inp2"]) == "keras_ragged_dot(inp1, inp2)"


def test_keras_generator_empty_prefix_yaml() -> None:
    """Test function test_keras_generator_empty_prefix_yaml."""
    from unittest.mock import patch, mock_open
    from ml_switcheroo_compiler.backends.keras.generator import KerasCodeGenerator
    from ml_switcheroo_compiler.ir.core import IRGraph

    g = IRGraph()
    gen = KerasCodeGenerator(g)

    # Test _get_backend_prefix
    assert gen._get_backend_prefix() == "keras"

    # Test _resolve_imports with empty yaml
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
    from ml_switcheroo_compiler.ir.core import IRNode

    graph = IRGraph()
    graph.nodes["const_0"] = IRNode(id="const_0", op_type="Constant", inputs=[])
    gen = KerasCodeGenerator(graph)

    def mock_generate_body(prefix: EagerValue) -> None:
        """Test function mock_generate_body."""
        gen.keras_input_vars = []
        gen.keras_output_vars = ["out_var"]

    setattr(gen, "_generate_body", mock_generate_body)
    code = gen.generate()
    assert "return out_var" in code


def test_keras_generator_return_block_both_inputs_and_outputs() -> None:
    """Test generate() when both keras_input_vars and keras_output_vars are present."""
    from ml_switcheroo_compiler.ir.core import IRNode

    graph = IRGraph()
    graph.nodes["const_0"] = IRNode(id="const_0", op_type="Constant", inputs=[])
    gen = KerasCodeGenerator(graph)

    def mock_generate_body(prefix: EagerValue) -> None:
        """Test function mock_generate_body."""
        gen.keras_input_vars = ["in_var"]
        gen.keras_output_vars = ["out_var"]

    setattr(gen, "_generate_body", mock_generate_body)
    code = gen.generate()
    assert "return keras.Model(inputs=[in_var], outputs=[out_var])" in code


def test_keras_generator_empty_inputs_and_outputs() -> None:
    """Test generate() when keras_input_vars is set but keras_output_vars is empty."""
    from ml_switcheroo_compiler.ir.core import IRNode

    graph = IRGraph()
    graph.nodes["dummy"] = IRNode(id="dummy", op_type="Identity", inputs=[])
    gen = KerasCodeGenerator(graph)

    def mock_generate_body(prefix: EagerValue) -> None:
        """Test function mock_generate_body."""
        gen.keras_input_vars = ["in0"]
        gen.keras_output_vars = []

    setattr(gen, "_generate_body", mock_generate_body)
    code = gen.generate()
    assert "pass" in code


def test_keras_generator_compile_aot() -> None:
    """Test KerasCodeGenerator._compile_aot_impl across successful exec and evaluator fallback."""
    import numpy as np
    from ml_switcheroo_compiler.ir.core import IRNode

    # 1. Successful exec and get_model execution
    graph = IRGraph()
    gen = KerasCodeGenerator(graph)
    with patch.object(gen, "generate", return_value="def get_model(): return lambda x, **kw: x + 1"):
        artifact = gen.compile_aot(graph)
        assert artifact(41) == 42
        assert cast("EagerTensorProtocol", getattr(artifact.metadata, "__getitem__")("backend")) == "keras"

    # 1b. Exec succeeds but get_model not defined in scope
    gen_no_get_model = KerasCodeGenerator(graph)
    with patch.object(gen_no_get_model, "generate", return_value="x_const = 10"):
        artifact_no_gm = gen_no_get_model.compile_aot(graph)
        assert cast("EagerTensorProtocol", getattr(artifact_no_gm.metadata, "__getitem__")("model")) is None

    # 2. Exec failure / get_model None fallback with single output
    g_single = IRGraph()
    node_in = IRNode(id="in0", op_type="Input", inputs=[], shape_metadata=[2])
    node_id = IRNode(id="out0", op_type="Identity", inputs=["in0"], shape_metadata=[2])
    g_single.nodes = {"in0": node_in, "out0": node_id}
    g_single.inputs = ["in0"]
    g_single.outputs = ["out0"]

    class MockTensorWithData:
        """Mock tensor wrapper holding underlying array data."""

        def __init__(self, data: np.ndarray[tuple[int, ...], np.dtype[np.generic]]) -> None:
            """Initialize mock tensor.

            Args:
                data (np.ndarray): Underlying array.
            """
            self.data: np.ndarray[tuple[int, ...], np.dtype[np.generic]] = data

    gen_single = KerasCodeGenerator(g_single)
    with patch.object(gen_single, "generate", return_value="invalid python syntax !!!"):
        art_single = gen_single.compile_aot(g_single)
        # Call with arg having .data and extra arg past len(graph.inputs)
        try:
            res_single = cast("Callable[..., EagerTensorProtocol]", art_single)(cast("Callable[..., EagerTensorProtocol]", MockTensorWithData)(cast("Callable[..., EagerTensorProtocol]", np.array)([1, 2])), "extra_arg")
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
            assert _exc is not None
        assert cast("Callable[..., EagerTensorProtocol]", np.array_equal)(res_single, cast("Callable[..., EagerTensorProtocol]", np.array)([1, 2]))

    # 3. Fallback with no outputs
    g_no_out = IRGraph()
    g_no_out.nodes = {"in0": node_in}
    g_no_out.inputs = ["in0"]
    g_no_out.outputs = []
    gen_no_out = KerasCodeGenerator(g_no_out)
    with patch.object(gen_no_out, "generate", return_value="invalid python syntax !!!"):
        art_no_out = gen_no_out.compile_aot(g_no_out)
        res_no_out = art_no_out(np.array([1, 2]))
        assert isinstance(res_no_out, dict)

    # 4. Fallback with multiple outputs
    g_multi = IRGraph()
    node_out1 = IRNode(id="out1", op_type="Identity", inputs=["in0"], shape_metadata=[2])
    node_out2 = IRNode(id="out2", op_type="Identity", inputs=["in0"], shape_metadata=[2])
    g_multi.nodes = {"in0": node_in, "out1": node_out1, "out2": node_out2}
    g_multi.inputs = ["in0"]
    g_multi.outputs = ["out1", "out2"]
    gen_multi = KerasCodeGenerator(g_multi)
    with patch.object(gen_multi, "generate", return_value="invalid syntax"):
        art_multi = gen_multi.compile_aot(g_multi)
        res_multi = art_multi(np.array([5, 6]))
        assert cast("Callable[..., EagerTensorProtocol]", len)(res_multi) == 2
        assert cast("Callable[..., EagerTensorProtocol]", np.array_equal)(cast("EagerTensorProtocol", getattr(res_multi, "__getitem__")(0)), cast("Callable[..., EagerTensorProtocol]", np.array)([5, 6]))
        assert cast("Callable[..., EagerTensorProtocol]", np.array_equal)(cast("EagerTensorProtocol", getattr(res_multi, "__getitem__")(1)), cast("Callable[..., EagerTensorProtocol]", np.array)([5, 6]))
