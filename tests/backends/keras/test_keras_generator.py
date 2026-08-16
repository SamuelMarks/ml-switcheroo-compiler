# ruff: noqa
from unittest.mock import mock_open, patch
from ml_switcheroo_compiler.backends.keras.keras_mixins import KerasAudioVisitor, KerasVisionVisitor
from ml_switcheroo_compiler.backends.keras.generator import KerasCodeGenerator, KerasSignatureBuilder, KerasTensorManipulator

"Test module."


class DummyNode:
    def __init__(self, attrs=None):
        self.attributes = attrs or {}


def test_keras_vision_visitor():
    vis = KerasVisionVisitor()
    assert vis.visit_ElasticTransform(DummyNode({"data_format": "df", "interpolation": "bicubic"}), ["a", "b"]) == "keras_elastic_transform(a, b, 'bicubic', 0.0, \"df\")"
    assert vis.visit_ElasticTransform(DummyNode(), ["a", "b"]) == "keras_elastic_transform(a, b, 'bilinear', 0.0, None)"
    assert vis.visit_GaussianBlur(DummyNode({"kernel_size": 3, "sigma": 1.0, "data_format": "df"}), ["a"]) == "keras_gaussian_blur(a, 3, 1.0, 'same', \"df\")"
    assert vis.visit_GaussianBlur(DummyNode(), ["a"]) == "keras_gaussian_blur(a, None, None, 'same', None)"
    assert vis.visit_MedianFilter(DummyNode({"kernel_size": 3, "sigma": 1.0, "data_format": "df"}), ["a"]) == "keras_median_filter(a, 3, 'same', \"df\")"
    assert vis.visit_MedianFilter(DummyNode(), ["a"]) == "keras_median_filter(a, None, 'same', None)"
    assert vis.visit_IoU(DummyNode({"bounding_box_format": "cxywh"}), ["a", "b"]) == "keras_iou(a, b, 'cxywh')"
    assert vis.visit_IoU(DummyNode(), ["a", "b"]) == "keras_iou(a, b, 'xyxy')"
    assert vis.visit_NonMaxSuppression(DummyNode({"max_output_size": 10}), ["a", "b"]) == "keras_nms(a, b, 10, 0.5, -inf)"
    assert vis.visit_NonMaxSuppression(DummyNode(), ["a", "b"]) == "keras_nms(a, b, None, 0.5, -inf)"
    assert vis.visit_ResizeBicubic(DummyNode({"size": 10, "align_corners": True}), ["a"]) == "keras_resize(a, 10, 'bicubic', True)"
    assert vis.visit_ResizeBicubic(DummyNode(), ["a"]) == "keras_resize(a, None, 'bicubic', False)"
    assert vis.visit_ResizeLanczos3(DummyNode({"size": 10, "align_corners": True}), ["a"]) == "keras_resize(a, 10, 'lanczos3', True)"
    assert vis.visit_ResizeLanczos3(DummyNode(), ["a"]) == "keras_resize(a, None, 'lanczos3', False)"
    assert vis.visit_ExtractBoundingBoxes(DummyNode({"crop_size": 10, "data_format": "df"}), ["a", "b", "c"]) == "keras_extract_bounding_boxes(a, b, c, BoundingBoxExtractionConfig(crop_size=10, interpolation='bilinear', extrapolation_value=0.0, data_format=\"df\"))"
    assert vis.visit_ExtractBoundingBoxes(DummyNode(), ["a", "b", "c"]) == "keras_extract_bounding_boxes(a, b, c, BoundingBoxExtractionConfig(crop_size=None, interpolation='bilinear', extrapolation_value=0.0, data_format=None))"
    assert vis.visit_PerspectiveTransform(DummyNode({"data_format": "df"}), ["a", "b", "c"]) == "keras.ops.image.perspective_transform(a, b, c, interpolation='bilinear', fill_value=0.0, data_format=\"df\")"
    assert vis.visit_PerspectiveTransform(DummyNode(), ["a", "b", "c"]) == "keras.ops.image.perspective_transform(a, b, c, interpolation='bilinear', fill_value=0.0, data_format=None)"


def test_keras_audio_visitor():
    vis = KerasAudioVisitor()
    assert vis.visit_Istft(DummyNode({"frame_length": 2048, "frame_step": 512, "center": False}), ["a"]) == "keras_istft(a, STFTConfig(frame_length=2048, frame_step=512, fft_length=None, window='hann', center=False))"
    assert vis.visit_MelFilterbank(DummyNode({"num_mel_bins": 1, "num_spectrogram_bins": 2, "sample_rate": 3, "lower_edge_hertz": 4, "upper_edge_hertz": 5}), ["a"]) == "keras_mel_filterbank(1, 2, 3, 4, 5)"
    assert vis.visit_Mfcc(DummyNode({"num_mel_bins": 1, "sample_rate": 2, "lower_edge_hertz": 3, "upper_edge_hertz": 4, "num_mfccs": 5}), ["a"]) == "keras_mfcc(a, 2, 1, 3, 4, 5)"


"Test module."


class DummyIRNode:
    def __init__(self, id, shape_metadata=None):
        self.id = id
        if shape_metadata:
            self.shape_metadata = shape_metadata


class DummyGraph:
    nodes = []


def test_keras_signature_builder():
    node = DummyIRNode("in_0", (None, 3))
    assert KerasSignatureBuilder.get_input_assignment("var", node) == "var = keras.Input(shape=(None, 3), name='in_0')"
    node2 = DummyIRNode("in_1")
    assert KerasSignatureBuilder.get_input_assignment("var2", node2) == "var2 = keras.Input(shape=(None,), name='in_1')"
    assert KerasSignatureBuilder.get_return_block(["i1"], ["o1"]) == "return keras.Model(inputs=[i1], outputs=[o1])"


def test_keras_tensor_manipulator():
    assert KerasTensorManipulator.format_zeros_like("zeros", {}) == "keras.ops.zeros({shape})"
    assert KerasTensorManipulator.format_zeros_like("ones", {"dtype": "int32"}) == "keras.ops.ones({shape}), dtype='int32'"
    assert KerasTensorManipulator.format_full({}) == "keras.ops.full({shape}, {fill_value})"
    assert KerasTensorManipulator.format_full({"dtype": "int32"}) == "keras.ops.full({shape}, {fill_value}), dtype='int32'"
    assert KerasTensorManipulator.format_transpose({}) == "keras.ops.transpose({0})"
    assert KerasTensorManipulator.format_transpose({"axes": (1, 0)}) == "keras.ops.transpose({0}, {axes})"


def test_keras_generator():
    gen = KerasCodeGenerator(DummyGraph())
    assert gen.get_fallback_prefix() == "keras.ops"
    assert gen.get_fallback_prefix() == "keras.ops"
    assert gen.visit_RaggedDot(None, ["a", "b"]) == "keras_ragged_dot(a, b)"
    ops = gen.get_ops_map({})
    assert "Zeros" in ops
    assert "Transpose" in ops
    gen._emit_input_assignment("var1", DummyIRNode("in"), "inputs", 0)
    assert "var1 = keras.Input" in gen.code[-1]
    assert "var1" in gen.keras_input_vars
    gen._emit_body_return(["out1"])
    assert "out1" in gen.keras_output_vars
    gen._emit_output_assignment(None, ["out2"], "out2")
    assert "out2" in gen.keras_output_vars
    assert gen._generate_file_header() == [gen.header.strip()]
    m_open = mock_open(read_data='imports: "import tmp"\nfunctions:\n  dummy: "def dummy(): pass"')
    with patch("builtins.open", m_open):
        imports = gen._resolve_imports()
        assert "import keras\n" in imports
        assert "import tmp" in imports
        assert "def dummy(): pass" in imports
    gen.code = []
    gen._generate_function_signature()
    assert "def get_model():" in gen.code[-1]
    assert gen.keras_input_vars == []
    gen.keras_input_vars = ["i"]
    gen.keras_output_vars = ["o"]
    gen._generate_return_block()
    assert "return keras.Model(inputs=[i], outputs=[o])" in gen.code[-1]


"""Test Keras generator edge cases coverage."""

from ml_switcheroo_compiler.backends.keras.generator import KerasCodeGenerator


def test_keras_generator_save_load(tmp_path):
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
    assert resz["arr_0"] == arr
    assert resz["arg2"] == [4, 5]

    # savez_compressed
    filezc = str(tmp_path / "dummy_comp.npz")
    KerasCodeGenerator.savez_compressed(filezc, arr, arg2=[4, 5])

    import gzip
    import pickle

    with gzip.open(filezc, "rb") as f:
        reszc = pickle.load(f)
    assert reszc["arr_0"] == arr
    assert reszc["arg2"] == [4, 5]


def test_keras_generator_conv_transpose():
    from ml_switcheroo_ir import LogicalGraph

    gen = KerasCodeGenerator(LogicalGraph())

    class DummyNode:
        pass

    assert gen.visit_ConvTranspose(DummyNode(), ["inp1", "inp2"]) == "keras_conv_transpose(inp1, inp2)"
    assert gen.visit_RaggedDot(DummyNode(), ["inp1", "inp2"]) == "keras_ragged_dot(inp1, inp2)"
