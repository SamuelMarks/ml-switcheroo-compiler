# ruff: noqa: E501
import os
import tempfile

import pytest

from ml_switcheroo_compiler.utils.dataset_utils import (
    BatchConfig,
    NumpyDataset,
    _extract_timeseries_windows,
    _get_files_and_labels,
    _get_timeseries_indices,
    _is_valid_file,
    _parse_class_names,
    _walk_directory_and_filter,
    audio_dataset_from_directory,
    image_dataset_from_directory,
    pack_x_y_sample_weight,
    pad_sequences,
    split_dataset,
    text_dataset_from_directory,
    timeseries_dataset_from_array,
    unpack_x_y_sample_weight,
)


def test_numpy_dataset():
    ds = NumpyDataset([1, 2, 3], config=BatchConfig(batch_size=2, shuffle=False))
    assert len(ds) == 2
    it = iter(ds)
    assert next(it) == [1, 2]
    assert next(it) == [3]
    ds_no_y = NumpyDataset([1, 2], config=BatchConfig(batch_size=1, shuffle=False))
    it_no_y = iter(ds_no_y)
    assert next(it_no_y) == [1]
    for _ in ds:
        pass
    ds2 = NumpyDataset([1, 2], [3, 4], config=BatchConfig(batch_size=1, shuffle=True, seed=42))
    assert len(ds2) == 2
    ds3 = NumpyDataset(1, 2, config=BatchConfig(batch_size=2, shuffle=False))
    assert next(iter(ds3)) == ([1], [2])
    ds4 = NumpyDataset([], config=BatchConfig(batch_size=1))
    assert len(ds4) == 0


def test_parse_class_names():
    with tempfile.TemporaryDirectory() as td:
        os.mkdir(os.path.join(td, "c2"))
        os.mkdir(os.path.join(td, "c1"))
        assert _parse_class_names(td, None) == ["c1", "c2"]
        assert _parse_class_names(td, ["c3"]) == ["c3"]


def test_is_valid_file():
    with tempfile.TemporaryDirectory() as td:
        with open(os.path.join(td, "a.txt"), "w") as f:
            f.write("a")
        assert _is_valid_file("a.txt", td, (".txt",))
        assert not _is_valid_file("a.txt", td, (".jpg",))
        assert not _is_valid_file("b.txt", td, (".txt",))


def test_get_files_and_labels():
    with tempfile.TemporaryDirectory() as td:
        os.mkdir(os.path.join(td, "c1"))
        with open(os.path.join(td, "not_dir"), "w") as f:
            f.write("a")
        _walk_directory_and_filter(td, ["c1", "not_dir"], None)
    with tempfile.TemporaryDirectory() as td:
        os.mkdir(os.path.join(td, "c1"))
        with open(os.path.join(td, "c1", "a.txt"), "w") as f:
            f.write("a")
        with pytest.raises(ValueError):
            _get_files_and_labels("nonexistent")
        (paths, labels, names) = _get_files_and_labels(td, class_names=["c1"], valid_exts=[".txt"])
        assert len(paths) == 1
        assert labels == [0]
        assert names == ["c1"]
        with pytest.raises(ValueError):
            _get_files_and_labels(td, labels=[1, 2], class_names=["c1"], valid_exts=[".txt"])
        (paths, labels, names) = _get_files_and_labels(td, labels=[42], class_names=["c1"], valid_exts=[".txt"])
        assert labels == [42]
        (paths, labels, names) = _get_files_and_labels(td, labels="not_inferred_str", class_names=["c1"], valid_exts=[".txt"])


def test_dataset_from_directory():
    with tempfile.TemporaryDirectory() as td:
        os.mkdir(os.path.join(td, "c1"))
        with open(os.path.join(td, "c1", "a.wav"), "w") as f:
            f.write("a")
        with open(os.path.join(td, "c1", "a.jpg"), "w") as f:
            f.write("a")
        with open(os.path.join(td, "c1", "a.txt"), "w") as f:
            f.write("a")
        ds = audio_dataset_from_directory(td)
        assert len(ds) == 1
        ds2 = image_dataset_from_directory(td)
        assert len(ds2) == 1
        ds3 = text_dataset_from_directory(td)
        assert len(ds3) == 1


def test_timeseries():
    assert _get_timeseries_indices(10, {"start_index": None, "end_index": None, "sequence_length": 2, "sampling_rate": 2, "sequence_stride": 1}) == (0, 7, 1)
    (x, y) = _extract_timeseries_windows([1, 2, 3, 4], [10, 20, 30, 40], {"sequence_length": 2, "sampling_rate": 1}, (0, 2, 1))
    assert x == [[1, 2], [2, 3]]
    assert y == [20, 30]
    (x2, y2) = _extract_timeseries_windows([1, 2, 3], None, {"sequence_length": 2, "sampling_rate": 1}, (0, 2, 1))
    ds = timeseries_dataset_from_array([1, 2, 3], [10, 20, 30], 2)
    assert len(ds) > 0


def test_pack_unpack():
    assert pack_x_y_sample_weight(1) == 1
    assert pack_x_y_sample_weight(1, 2) == (1, 2)
    assert pack_x_y_sample_weight(1, 2, 3) == (1, 2, 3)
    assert unpack_x_y_sample_weight({"x": 1, "y": 2, "sample_weight": 3}) == (1, 2, 3)
    assert unpack_x_y_sample_weight((1,)) == (1, None, None)
    assert unpack_x_y_sample_weight((1, 2)) == (1, 2, None)
    assert unpack_x_y_sample_weight((1, 2, 3)) == (1, 2, 3)
    assert unpack_x_y_sample_weight((1, 2, 3, 4)) == ((1, 2, 3, 4), None, None)
    assert unpack_x_y_sample_weight(1) == (1, None, None)


def test_pad_sequences():
    assert pad_sequences([]) == []
    seqs = [[1, 2], [3]]
    res = pad_sequences(seqs)
    assert res == [[1, 2], [0.0, 3]]
    res2 = pad_sequences(seqs, maxlen=1, truncating="pre")
    assert res2 == [[2], [3]]
    res3 = pad_sequences(seqs, maxlen=1, truncating="post")
    assert res3 == [[1], [3]]
    res4 = pad_sequences(seqs, maxlen=3, padding="post")
    assert res4 == [[1, 2, 0.0], [3, 0.0, 0.0]]
    res5 = pad_sequences([[1]], maxlen=1)
    assert res5 == [[1]]


def test_split_dataset():
    assert split_dataset("ds") == ("ds", "ds")
