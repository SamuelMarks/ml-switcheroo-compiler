"""Test module."""

from ml_switcheroo_compiler.backends.numpy.numpy_mixins import NumpyAudioVisitor, NumpyScatterVisitor, NumpyVisionVisitor


class DummyNode:
    def __init__(self, attrs=None):
        self.attributes = attrs or {}


def test_numpy_vision_visitor():
    vis = NumpyVisionVisitor()
    assert vis.visit_PerspectiveTransform(DummyNode({"data_format": "df", "interpolation": "bicubic"}), ["a", "b", "c"]) == "np_perspective_transform(a, b, c, PerspectiveConfig(interpolation='bicubic', fill_value=0.0, data_format=\"df\"))"
    assert vis.visit_PerspectiveTransform(DummyNode(), ["a", "b", "c"]) == "np_perspective_transform(a, b, c, PerspectiveConfig(interpolation='bilinear', fill_value=0.0, data_format=None))"

    assert vis.visit_ElasticTransform(DummyNode({"data_format": "df", "interpolation": "bicubic"}), ["a", "b"]) == "np_elastic_transform(a, b, 'bicubic', 0.0, \"df\")"
    assert vis.visit_ElasticTransform(DummyNode(), ["a", "b"]) == "np_elastic_transform(a, b, 'bilinear', 0.0, None)"

    assert vis.visit_GaussianBlur(DummyNode({"kernel_size": 3, "sigma": 1.0, "data_format": "df"}), ["a"]) == "np_gaussian_blur(a, 3, 1.0, 'same', \"df\")"
    assert vis.visit_GaussianBlur(DummyNode(), ["a"]) == "np_gaussian_blur(a, None, None, 'same', None)"

    assert vis.visit_MedianFilter(DummyNode({"kernel_size": 3, "sigma": 1.0, "data_format": "df"}), ["a"]) == "np_median_filter(a, 3, 'same', \"df\")"
    assert vis.visit_MedianFilter(DummyNode(), ["a"]) == "np_median_filter(a, None, 'same', None)"

    assert vis.visit_ExtractBoundingBoxes(DummyNode({"crop_size": 10, "data_format": "df"}), ["a", "b", "c"]) == "np_extract_bounding_boxes(a, b, c, 10, 'bilinear', 0.0, \"df\")"
    assert vis.visit_ExtractBoundingBoxes(DummyNode(), ["a", "b", "c"]) == "np_extract_bounding_boxes(a, b, c, None, 'bilinear', 0.0, None)"

    assert vis.visit_IoU(DummyNode({"bounding_box_format": "cxywh"}), ["a", "b"]) == "np_iou(a, b, 'cxywh')"
    assert vis.visit_IoU(DummyNode(), ["a", "b"]) == "np_iou(a, b, 'xyxy')"

    assert vis.visit_NonMaxSuppression(DummyNode({"max_output_size": 10}), ["a", "b"]) == "np_nms(a, b, 10, 0.5, -inf)"
    assert vis.visit_NonMaxSuppression(DummyNode(), ["a", "b"]) == "np_nms(a, b, None, 0.5, -inf)"

    assert vis.visit_ResizeBicubic(DummyNode({"size": 10, "align_corners": True}), ["a"]) == "np_resize(a, 10, 'bicubic', True)"
    assert vis.visit_ResizeBicubic(DummyNode(), ["a"]) == "np_resize(a, None, 'bicubic', False)"

    assert vis.visit_ResizeLanczos3(DummyNode({"size": 10, "align_corners": True}), ["a"]) == "np_resize(a, 10, 'lanczos3', True)"
    assert vis.visit_ResizeLanczos3(DummyNode(), ["a"]) == "np_resize(a, None, 'lanczos3', False)"


def test_numpy_audio_visitor():
    vis = NumpyAudioVisitor()
    assert vis.visit_Istft(DummyNode({"frame_length": 2048, "frame_step": 512, "center": False}), ["a"]) == "np_istft(a, 2048, 512, None, 'hann', False)"
    assert vis.visit_MelFilterbank(DummyNode({"num_mel_bins": 1, "num_spectrogram_bins": 2, "sample_rate": 3, "lower_edge_hertz": 4, "upper_edge_hertz": 5}), ["a"]) == "np_mel_filterbank(1, 2, 3, 4, 5)"
    assert vis.visit_Mfcc(DummyNode({"num_mel_bins": 1, "sample_rate": 2, "lower_edge_hertz": 3, "upper_edge_hertz": 4, "num_mfccs": 5}), ["a"]) == "np_mfcc(a, 2, 1, 3, 4, 5)"


def test_numpy_scatter_visitor():
    vis = NumpyScatterVisitor()
    node = DummyNode()
    assert vis.visit_TensorScatterUpdate(node, ["a", "i", "u"]) == "(lambda c, i, u: [c.__setitem__(tuple(np.moveaxis(np.asarray(i), -1, 0)), u), c][1])(np.copy(a), i, u)"
    assert vis.visit_TensorScatterAdd(node, ["a", "i", "u"]) == "(lambda c, i, u: [np.add.at(c, tuple(np.moveaxis(np.asarray(i), -1, 0)), u), c][1])(np.copy(a), i, u)"
    assert vis.visit_TensorScatterMax(node, ["a", "i", "u"]) == "(lambda c, i, u: [np.maximum.at(c, tuple(np.moveaxis(np.asarray(i), -1, 0)), u), c][1])(np.copy(a), i, u)"
    assert vis.visit_TensorScatterMin(node, ["a", "i", "u"]) == "(lambda c, i, u: [np.minimum.at(c, tuple(np.moveaxis(np.asarray(i), -1, 0)), u), c][1])(np.copy(a), i, u)"
