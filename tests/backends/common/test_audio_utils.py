"""Test module."""

from ml_switcheroo_compiler.backends.common.audio_utils import extract_mel_attributes, extract_stft_attributes


class DummyNode:
    def __init__(self, attrs):
        self.attributes = attrs


def test_extract_stft_attributes():
    node1 = DummyNode({"frame_length": 10, "frame_step": 5, "fft_length": 16, "window": "hamming", "center": False})
    assert extract_stft_attributes(node1) == (10, 5, 16, "hamming", False, "16")

    node2 = DummyNode({"frame_length": 10, "frame_step": 5})
    assert extract_stft_attributes(node2) == (10, 5, None, "hann", True, "None")


def test_extract_mel_attributes():
    node1 = DummyNode({"num_mel_bins": 50, "num_spectrogram_bins": 256, "sample_rate": 16000, "lower_edge_hertz": 50.0, "upper_edge_hertz": 8000.0, "num_mfccs": 20})
    assert extract_mel_attributes(node1) == (50, 256, 16000, 50.0, 8000.0, 20)

    node2 = DummyNode({"num_spectrogram_bins": 128, "sample_rate": 8000})
    assert extract_mel_attributes(node2) == (40, 128, 8000, 20.0, 4000.0, 13)
