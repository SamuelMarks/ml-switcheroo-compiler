"""Tests for dataset utilities."""

from __future__ import annotations

import os
import tempfile
import typing

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
from tests.testing_utils import ensure_callable, ensure_eager_tensor, ensure_tensor_dict

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol


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
    """Verify `split_dataset` produces disjoint subsets, preserves target alignment, and validates `left_size`."""
    ds = NumpyDataset([10, 20, 30, 40], y=[1, 2, 3, 4], config=BatchConfig(batch_size=1, shuffle=False, seed=7))
    left_ds, right_ds = split_dataset(ds, left_size=0.5, shuffle=False)
    assert left_ds.x == [10, 20] and left_ds.y == [1, 2]
    assert right_ds.x == [30, 40] and right_ds.y == [3, 4]

    ds_no_y = NumpyDataset([10, 20, 30, 40], config=BatchConfig(batch_size=2, shuffle=False, seed=7))
    shuf_l, shuf_r = split_dataset(ds_no_y, left_size=0.25, shuffle=True)
    assert len(shuf_l.x) == 1 and len(shuf_r.x) == 3
    assert set(ensure_eager_tensor(shuf_l.x)) | set(ensure_eager_tensor(shuf_r.x)) == {10, 20, 30, 40}

    with pytest.raises(ValueError, match="left_size"):
        split_dataset(ds, left_size=1.5)


def test_pad_sequences() -> None:
    """Test pad sequences."""
    assert pad_sequences([]) == []
    seqs = [[1, 2], [1, 2, 3, 4], [1]]
    res1 = pad_sequences(seqs, value=0)
    assert res1 == [[0, 0, 1, 2], [1, 2, 3, 4], [0, 0, 0, 1]]
    res2 = pad_sequences(seqs, maxlen=3, padding="pre", truncating="pre", value=0)
    assert res2 == [[0, 1, 2], [2, 3, 4], [0, 0, 1]]
    res3 = pad_sequences(seqs, maxlen=3, padding="post", truncating="post", value=0)
    assert res3 == [[1, 2, 0], [1, 2, 3], [1, 0, 0]]
    seqs_equal = [[10, 20]]
    res_eq = pad_sequences(seqs_equal, maxlen=2)
    assert res_eq == [[10, 20]]


def test_numpy_dataset_iteration_and_length() -> None:
    """Test NumpyDataset batching, shuffling, length, and iteration with and without targets."""
    ds_empty = NumpyDataset([], config=BatchConfig(batch_size=2, shuffle=False))
    assert len(ds_empty) == 0
    assert list(ds_empty) == []
    ds_scalar = NumpyDataset(42, y=99, config=BatchConfig(batch_size=1, shuffle=False))
    assert len(ds_scalar) == 1
    assert list(ds_scalar) == [([42], [99])]
    x = [1, 2, 3, 4, 5]
    y = [10, 20, 30, 40, 50]
    ds = NumpyDataset(x, y, config=BatchConfig(batch_size=2, shuffle=True, seed=42))
    assert len(ds) == 3
    batches = list(ds)
    assert len(batches) == 3
    assert ensure_callable(len)(batches[0][0]) == 2
    assert ensure_callable(len)(batches[0][1]) == 2
    assert ensure_callable(len)(batches[2][0]) == 1
    ds_no_y = NumpyDataset(x, config=BatchConfig(batch_size=3, shuffle=False))
    assert len(ds_no_y) == 2
    batches_no_y = list(ds_no_y)
    assert batches_no_y == [[1, 2, 3], [4, 5]]


def _test_directory_dataset_parsers_and_loaders_sub_1(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute sub-part 1 of test_directory_dataset_parsers_and_loaders.

    Args:
        _state: Shared test state dictionary.
    """
    tmp_dir = _state["tmp_dir"]
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
    _state.update(ensure_tensor_dict({"cat1_img": cat1_img, "cat2_png": cat2_png, "cat_audio": cat_audio, "cat_dir": cat_dir}))
    _state.update(ensure_tensor_dict({"cat_txt": cat_txt, "dog1_img": dog1_img, "dog_audio": dog_audio, "dog_dir": dog_dir}))
    _state.update(ensure_tensor_dict({"dog_txt": dog_txt, "empty_dir": empty_dir, "f": f}))
    _state.update(ensure_tensor_dict({"tmp_dir": tmp_dir}))


def _test_directory_dataset_parsers_and_loaders_sub_2(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute sub-part 2 of test_directory_dataset_parsers_and_loaders.

    Args:
        _state: Shared test state dictionary.
    """
    (cat_dir, dog_dir, tmp_dir) = (_state["cat_dir"], _state["dog_dir"], _state["tmp_dir"])
    broken_sym = os.path.join(dog_dir, "broken.jpg")
    os.symlink(os.path.join(dog_dir, "nonexistent.jpg"), broken_sym)
    classes = ensure_callable(_parse_class_names)(tmp_dir, None)
    assert classes == ["cats", "dogs"]
    classes_explicit = ensure_callable(_parse_class_names)(tmp_dir, ["dogs", "cats"])
    assert classes_explicit == ["dogs", "cats"]
    assert ensure_callable(_is_valid_file)("cat1.jpg", cat_dir, [".jpg"])
    assert not ensure_callable(_is_valid_file)("cat1.jpg", cat_dir, [".png"])
    (fps, lbls) = ensure_callable(_walk_directory_and_filter)(tmp_dir, ["cats", "nonexistent_dir"], [".jpg"])
    assert len(fps) == 1
    assert lbls == [0]
    with pytest.raises(ValueError, match="does not exist"):
        _get_files_and_labels(os.path.join(tmp_dir, "missing_folder"))
    (fps, lbls, cls_names) = ensure_callable(_get_files_and_labels)(tmp_dir, labels=[10, 20, 30], class_names=["cats", "dogs"], valid_exts=(".jpg", ".png", ".bmp"))
    assert lbls == [10, 20, 30]
    (fps_str, lbls_str, _) = ensure_callable(_get_files_and_labels)(tmp_dir, labels="custom_mode", class_names=["cats", "dogs"], valid_exts=(".jpg", ".png", ".bmp"))
    _state.update(ensure_tensor_dict({"_": _, "classes": classes, "classes_explicit": classes_explicit, "cls_names": cls_names}))
    _state.update(ensure_tensor_dict({"fps": fps, "fps_str": fps_str, "lbls": lbls, "lbls_str": lbls_str}))
    _state.update(ensure_tensor_dict({"cat_dir": cat_dir, "dog_dir": dog_dir, "tmp_dir": tmp_dir}))


def _test_directory_dataset_parsers_and_loaders_sub_3(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute sub-part 3 of test_directory_dataset_parsers_and_loaders.

    Args:
        _state: Shared test state dictionary.
    """
    (lbls_str, tmp_dir) = (_state["lbls_str"], _state["tmp_dir"])
    assert len(lbls_str) == 3
    with pytest.raises(ValueError, match="Length of labels does not match"):
        ensure_callable(_get_files_and_labels)(tmp_dir, labels=[1], class_names=["cats", "dogs"], valid_exts=(".jpg", ".png", ".bmp"))
    cfg_audio = DatasetConfig(io_config=IOConfig(), batch_config=BatchConfig(batch_size=1, seed=123))
    audio_ds = ensure_callable(audio_dataset_from_directory)(tmp_dir, cfg_audio)
    assert len(audio_ds) == 2
    audio_ds_default = ensure_callable(audio_dataset_from_directory)(tmp_dir)
    assert len(audio_ds_default) >= 1
    img_ds = ensure_callable(image_dataset_from_directory)(tmp_dir)
    assert len(img_ds) >= 1
    txt_ds = ensure_callable(text_dataset_from_directory)(tmp_dir)
    assert len(txt_ds) >= 1
    items = list(txt_ds)
    assert ensure_eager_tensor(items)[0][0][0] == "content"
    _state.update(ensure_tensor_dict({"audio_ds": audio_ds, "audio_ds_default": audio_ds_default, "cfg_audio": cfg_audio, "img_ds": img_ds}))
    _state.update(ensure_tensor_dict({"items": items, "txt_ds": txt_ds}))
    _state.update(ensure_tensor_dict({"lbls_str": lbls_str, "tmp_dir": tmp_dir}))


def test_directory_dataset_parsers_and_loaders() -> None:
    """Test directory parsing, file validation, and datasets from directory."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        _state: dict[str, EagerTensorProtocol] = {}
        _state.update(ensure_tensor_dict({"tmp_dir": tmp_dir}))
        for _sub_fn in (_test_directory_dataset_parsers_and_loaders_sub_1, _test_directory_dataset_parsers_and_loaders_sub_2, _test_directory_dataset_parsers_and_loaders_sub_3):
            _sub_fn(_state)
        assert len(_state) > 0


def test_timeseries_dataset_from_array() -> None:
    """Test timeseries window extraction, index calculations, and timeseries_dataset_from_array."""
    data = list(range(20))
    targets = [i * 10 for i in range(20)]
    (start, stop, stride) = _get_timeseries_indices(20, {"start_index": None, "end_index": None, "sequence_length": None, "sampling_rate": None, "sequence_stride": None})
    assert start == 0
    assert stop == 20
    assert stride == 1
    ds_no_target = timeseries_dataset_from_array(data, None, sequence_length=3)
    windows_no_target = list(ds_no_target)
    assert len(windows_no_target) > 0
    assert ensure_callable(len)(windows_no_target[0][0]) == 3
    cfg = DatasetConfig(loader=DataLoaderConfig(sequence_stride=2, sampling_rate=2, start_index=1, end_index=18), batch_config=BatchConfig(batch_size=2, shuffle=False))
    ds = timeseries_dataset_from_array(data, targets, sequence_length=3, config=cfg)
    batches = list(ds)
    assert len(batches) > 0
    assert len(batches[0]) == 2


def test_timeseries_config_dataclass() -> None:
    """Test TimeseriesConfig initialization."""
    tc = TimeseriesConfig(sequence_length=10, sampling_rate=1, sequence_stride=1, start_index=0, end_index=100)
    assert tc.sequence_length == 10
    assert tc.end_index == 100


def test_file_utils_exists() -> None:
    """Test file_utils.exists helper."""
    from ml_switcheroo_compiler.utils.file_utils import exists

    assert exists(__file__)
    assert not exists("non_existent_path_xyz_123")
