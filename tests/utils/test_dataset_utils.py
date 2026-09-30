"""Tests for dataset utilities."""

from __future__ import annotations

import os
import tempfile

import pytest

from ml_switcheroo_compiler.utils.dataset_utils import (
    BatchConfig,
    DataLoaderConfig,
    DatasetConfig,
    IOConfig,
    NumpyDataset,
    TimeseriesConfig,
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


def test_pack_x_y_sample_weight() -> None:
    """Test pack."""
    assert pack_x_y_sample_weight(1) == 1
    assert pack_x_y_sample_weight(1, 2) == (1, 2)
    assert pack_x_y_sample_weight(1, 2, 3) == (1, 2, 3)


def test_unpack_x_y_sample_weight() -> None:
    """Test unpack."""
    assert unpack_x_y_sample_weight(1) == (1, None, None)
    assert unpack_x_y_sample_weight((1,)) == (1, None, None)
    assert unpack_x_y_sample_weight((1, 2)) == (1, 2, None)
    assert unpack_x_y_sample_weight((1, 2, 3)) == (1, 2, 3)
    assert unpack_x_y_sample_weight((1, 2, 3, 4)) == ((1, 2, 3, 4), None, None)
    assert unpack_x_y_sample_weight({"x": 1, "y": 2, "sample_weight": 3}) == (1, 2, 3)


def test_split_dataset() -> None:
    """Test split."""
    assert split_dataset(1, 0.5) == (1, 1)


def test_pad_sequences() -> None:
    """Test pad sequences."""
    assert pad_sequences([]) == []
    seqs = [[1, 2], [1, 2, 3, 4], [1]]

    # maxlen inferred
    res1 = pad_sequences(seqs, value=0)
    assert res1 == [[0, 0, 1, 2], [1, 2, 3, 4], [0, 0, 0, 1]]

    # maxlen set, pre padding, pre trunc
    res2 = pad_sequences(seqs, maxlen=3, padding="pre", truncating="pre", value=0)
    assert res2 == [[0, 1, 2], [2, 3, 4], [0, 0, 1]]

    # post padding, post trunc
    res3 = pad_sequences(seqs, maxlen=3, padding="post", truncating="post", value=0)
    assert res3 == [[1, 2, 0], [1, 2, 3], [1, 0, 0]]

    # sequence equal to maxlen
    seqs_equal = [[10, 20]]
    res_eq = pad_sequences(seqs_equal, maxlen=2)
    assert res_eq == [[10, 20]]


def test_numpy_dataset_iteration_and_length() -> None:
    """Test NumpyDataset batching, shuffling, length, and iteration with and without targets."""
    # Empty dataset
    ds_empty = NumpyDataset([], config=BatchConfig(batch_size=2, shuffle=False))
    assert len(ds_empty) == 0
    assert list(ds_empty) == []

    # Non-iterable scalar input
    ds_scalar = NumpyDataset(42, y=99, config=BatchConfig(batch_size=1, shuffle=False))
    assert len(ds_scalar) == 1
    assert list(ds_scalar) == [([42], [99])]

    # With targets and shuffle
    x = [1, 2, 3, 4, 5]
    y = [10, 20, 30, 40, 50]
    ds = NumpyDataset(x, y, config=BatchConfig(batch_size=2, shuffle=True, seed=42))
    assert len(ds) == 3
    batches = list(ds)
    assert len(batches) == 3
    assert len(batches[0][0]) == 2
    assert len(batches[0][1]) == 2
    assert len(batches[2][0]) == 1

    # Without targets
    ds_no_y = NumpyDataset(x, config=BatchConfig(batch_size=3, shuffle=False))
    assert len(ds_no_y) == 2
    batches_no_y = list(ds_no_y)
    assert batches_no_y == [[1, 2, 3], [4, 5]]


def test_directory_dataset_parsers_and_loaders() -> None:
    """Test directory parsing, file validation, and datasets from directory."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create class folders and files
        cat_dir = os.path.join(tmp_dir, "cats")
        dog_dir = os.path.join(tmp_dir, "dogs")
        empty_dir = os.path.join(tmp_dir, "not_a_dir.txt")
        os.makedirs(cat_dir)
        os.makedirs(dog_dir)
        with open(empty_dir, "w") as f:
            f.write("dummy")

        cat1_img = os.path.join(cat_dir, "cat1.jpg")
        cat2_png = os.path.join(cat_dir, "cat2.png")
        cat_txt = os.path.join(cat_dir, "cat.txt")
        cat_audio = os.path.join(cat_dir, "cat.wav")

        dog1_img = os.path.join(dog_dir, "dog1.bmp")
        dog_txt = os.path.join(dog_dir, "dog.txt")
        dog_audio = os.path.join(dog_dir, "dog.flac")

        for p in [cat1_img, cat2_png, cat_txt, cat_audio, dog1_img, dog_txt, dog_audio]:
            with open(p, "w") as f:
                f.write("content")

        # Create broken symlink in dog_dir if symlinks supported
        try:
            broken_sym = os.path.join(dog_dir, "broken.jpg")
            os.symlink(os.path.join(dog_dir, "nonexistent.jpg"), broken_sym)
        except (OSError, NotImplementedError):
            pass

        # Test _parse_class_names
        classes = _parse_class_names(tmp_dir, None)
        assert classes == ["cats", "dogs"]
        classes_explicit = _parse_class_names(tmp_dir, ["dogs", "cats"])
        assert classes_explicit == ["dogs", "cats"]

        # Test _is_valid_file extension filtering
        assert _is_valid_file("cat1.jpg", cat_dir, [".jpg"])
        assert not _is_valid_file("cat1.jpg", cat_dir, [".png"])

        # Test _walk_directory_and_filter skipping non-directory
        fps, lbls = _walk_directory_and_filter(tmp_dir, ["cats", "nonexistent_dir"], [".jpg"])
        assert len(fps) == 1
        assert lbls == [0]

        # Test _get_files_and_labels error on non-existent directory
        with pytest.raises(ValueError, match="does not exist"):
            _get_files_and_labels(os.path.join(tmp_dir, "missing_folder"))

        # Test _get_files_and_labels with custom labels
        fps, lbls, cls_names = _get_files_and_labels(tmp_dir, labels=[10, 20, 30], class_names=["cats", "dogs"], valid_exts=(".jpg", ".png", ".bmp"))
        assert lbls == [10, 20, 30]

        # Test _get_files_and_labels with labels as string not equal to 'inferred'
        fps_str, lbls_str, _ = _get_files_and_labels(tmp_dir, labels="custom_mode", class_names=["cats", "dogs"], valid_exts=(".jpg", ".png", ".bmp"))
        assert len(lbls_str) == 3

        # Test _get_files_and_labels label length mismatch
        with pytest.raises(ValueError, match="Length of labels does not match"):
            _get_files_and_labels(tmp_dir, labels=[1], class_names=["cats", "dogs"], valid_exts=(".jpg", ".png", ".bmp"))

        # Test audio_dataset_from_directory
        cfg_audio = DatasetConfig(io_config=IOConfig(), batch_config=BatchConfig(batch_size=1, seed=123))
        audio_ds = audio_dataset_from_directory(tmp_dir, cfg_audio)
        assert len(audio_ds) == 2

        # Test default config for audio_dataset_from_directory
        audio_ds_default = audio_dataset_from_directory(tmp_dir)
        assert len(audio_ds_default) >= 1

        # Test image_dataset_from_directory
        img_ds = image_dataset_from_directory(tmp_dir)
        assert len(img_ds) >= 1

        # Test text_dataset_from_directory
        txt_ds = text_dataset_from_directory(tmp_dir)
        assert len(txt_ds) >= 1
        items = list(txt_ds)
        assert items[0][0][0] == "content"


def test_timeseries_dataset_from_array() -> None:
    """Test timeseries window extraction, index calculations, and timeseries_dataset_from_array."""
    data = list(range(20))
    targets = [i * 10 for i in range(20)]

    # Test _get_timeseries_indices defaults and overrides
    start, stop, stride = _get_timeseries_indices(
        20,
        {
            "start_index": None,
            "end_index": None,
            "sequence_length": None,
            "sampling_rate": None,
            "sequence_stride": None,
        },
    )
    assert start == 0
    assert stop == 20
    assert stride == 1

    # Test timeseries_dataset_from_array without targets
    ds_no_target = timeseries_dataset_from_array(data, None, sequence_length=3)
    windows_no_target = list(ds_no_target)
    assert len(windows_no_target) > 0
    assert len(windows_no_target[0][0]) == 3

    # Test timeseries_dataset_from_array with targets and loader configs
    cfg = DatasetConfig(
        loader=DataLoaderConfig(
            sequence_stride=2,
            sampling_rate=2,
            start_index=1,
            end_index=18,
        ),
        batch_config=BatchConfig(batch_size=2, shuffle=False),
    )
    ds = timeseries_dataset_from_array(data, targets, sequence_length=3, config=cfg)
    batches = list(ds)
    assert len(batches) > 0
    assert len(batches[0]) == 2  # batch_x, batch_y


def test_timeseries_config_dataclass() -> None:
    """Test TimeseriesConfig initialization."""
    tc = TimeseriesConfig(
        sequence_length=10,
        sampling_rate=1,
        sequence_stride=1,
        start_index=0,
        end_index=100,
    )
    assert tc.sequence_length == 10
    assert tc.end_index == 100


def test_file_utils_exists() -> None:
    """Test file_utils.exists helper."""
    from ml_switcheroo_compiler.utils.file_utils import exists

    assert exists(__file__)
    assert not exists("non_existent_path_xyz_123")
