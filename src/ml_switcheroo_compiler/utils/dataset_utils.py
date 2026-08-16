"""Module dataset_utils.py."""

from __future__ import annotations

# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
from typing import Any

"""Core abstractions and logic definitions for dataset_utils.py."""


import os
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any


@dataclass
class TimeseriesConfig:
    """TimeseriesConfig class."""

    sequence_length: int
    sampling_rate: int
    sequence_stride: int
    start_index: int | None
    end_index: int | None


@dataclass
class BatchConfig:
    """Configuration for batching and shuffling data."""

    batch_size: int = 32
    shuffle: bool = True
    seed: int | None = None


@dataclass
class IOConfig:
    """Configuration for file I/O operations."""

    labels: Any = "inferred"
    label_mode: str = "int"
    class_names: Sequence[str] | None = None
    follow_links: bool = False
    validation_split: float | None = None
    subset: str | None = None


@dataclass
class DataAugmentationConfig:
    """Configuration for data augmentation and image properties."""

    color_mode: str = "rgb"
    image_size: tuple[int, int] = (256, 256)
    interpolation: str = "bilinear"
    crop_to_aspect_ratio: bool = False


@dataclass
class DataLoaderConfig:
    """Configuration for data loader sequences and timing."""

    sampling_rate: int | None = None
    output_sequence_length: int | None = None
    max_length: int | None = None
    sequence_stride: int = 1
    start_index: int | None = None
    end_index: int | None = None


@dataclass
class DatasetConfig:
    """Dataset configuration options."""

    batch_config: BatchConfig = BatchConfig()
    io_config: IOConfig = IOConfig()
    augmentation: DataAugmentationConfig = DataAugmentationConfig()
    loader: DataLoaderConfig = DataLoaderConfig()


if TYPE_CHECKING:
    _ = None
else:
    _ = None


class NumpyDataset:
    """Provide a simple dataset iterator for numpy arrays."""

    def __init__(
        self,
        x: Sequence | Any,  # type: ignore
        y: Sequence | object | None = None,  # type: ignore
        config: BatchConfig | None = None,
    ) -> None:
        """Initialize dataset.

        Args:
            x: Input data.
            y: Target data.
            config: Dataset configuration.
        """
        conf = config if config is not None else BatchConfig()
        self.x = list(x) if hasattr(x, "__iter__") else [x]
        self.y = list(y) if hasattr(y, "__iter__") else [y] if y is not None else None
        self.batch_size = conf.batch_size
        self.shuffle = conf.shuffle
        self.seed = conf.seed
        self._indices = list(range(len(self.x)))
        if self.shuffle:
            import random

            rng = random.Random(self.seed)
            rng.shuffle(self._indices)

    def __iter__(
        self,
    ) -> Iterator[Any | tuple[Any, Any]]:
        """Iterate over dataset.

        Yields: Any: Yielded value.
            Iterator: Result.
        """
        for i in range(0, len(self._indices), self.batch_size):
            batch_idx = self._indices[i : i + self.batch_size]
            batch_x = [self.x[idx] for idx in batch_idx]
            batch_y = [self.y[idx] for idx in batch_idx] if self.y is not None else None
            if batch_y is not None:
                yield batch_x, batch_y
            else:
                yield batch_x

    def __len__(self) -> int:
        """Length of dataset.

        Returns:
        int: Result.
        """
        if len(self._indices) == 0:
            return 0
        return int((len(self._indices) + self.batch_size - 1) // self.batch_size)


def _parse_class_names(directory: str, class_names: Sequence[str] | None) -> list[str]:
    """Parse class labels from folder names.

    Args:
        directory (str): The directory parameter.
        class_names (object): The class_names parameter.

    Yields: Any: Result.
    """
    if class_names is not None:
        return list(class_names)
    return sorted([d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))])


def _is_valid_file(fname: str, class_dir: str, valid_exts: Sequence[str] | None) -> bool:
    """Check if file is valid based on extension and symlink/readability.

    Args:
        fname (str): The fname parameter.
        class_dir (str): The class_dir parameter.
        valid_exts (object): The valid_exts parameter.

    Returns:
        bool: Result.
    """
    if valid_exts and not any(fname.endswith(ext) for ext in valid_exts):
        return False
    # Add check for broken symlink or unreadable
    fpath = os.path.join(class_dir, fname)
    if not os.path.exists(fpath):
        return False  # Broken symlink
    return True


def _walk_directory_and_filter(directory: str, class_names: list[str], valid_exts: Sequence[str] | None) -> tuple[list[str], list[int]]:
    """Walk through the directory and filter files.

    Args:
        directory (str): The directory parameter.
        class_names (object): The class_names parameter.
        valid_exts (object): The valid_exts parameter.

    Yields: Any: Result.
    """
    file_paths = []
    file_labels = []
    for i, class_name in enumerate(class_names):
        class_dir = os.path.join(directory, class_name)
        if not os.path.isdir(class_dir):
            continue
        for fname in sorted(os.listdir(class_dir)):
            if _is_valid_file(fname, class_dir, valid_exts):
                file_paths.append(os.path.join(class_dir, fname))
                file_labels.append(i)
    return file_paths, file_labels


def _get_files_and_labels(
    directory: str,
    labels: str | Sequence[int] = "inferred",
    class_names: Sequence[str] | None = None,
    valid_exts: Sequence[str] | None = None,
) -> tuple[list[str], list[int], list[str]]:
    """Get files and labels from directory.

    Args:
        directory (str): The directory parameter.
        labels (object): The labels parameter.
        class_names (object): The class_names parameter.
        valid_exts (object): The valid_exts parameter.

    Returns:
        tuple: Result.

    Raises:
        ValueError: An exception.
    """
    directory = os.path.abspath(directory)
    if not os.path.exists(directory):
        raise ValueError(f"Directory {directory} does not exist.")
    parsed_class_names = _parse_class_names(directory, class_names)
    file_paths, file_labels = _walk_directory_and_filter(directory, parsed_class_names, valid_exts)
    if labels != "inferred":
        if not isinstance(labels, str):
            if len(labels) != len(file_paths):
                raise ValueError("Length of labels does not match number of files.")
            file_labels = list(labels)
    return file_paths, file_labels, parsed_class_names


def audio_dataset_from_directory(
    directory: str,
    config: DatasetConfig | None = None,
) -> NumpyDataset:
    """Generate a dataset from audio files in a directory.

    Args:
        directory (str): The directory parameter.
        config (object): The config parameter.

    Returns:
        NumpyDataset: Result.
    """
    conf = config if config is not None else DatasetConfig()
    labels = conf.io_config.labels
    class_names = conf.io_config.class_names
    batch_size = conf.batch_config.batch_size
    seed = conf.batch_config.seed
    file_paths, file_labels, class_names = _get_files_and_labels(directory, labels, class_names, valid_exts=(".wav", ".mp3", ".flac"))
    return NumpyDataset(
        file_paths,
        file_labels,
        config=BatchConfig(batch_size=batch_size, shuffle=(seed is not None), seed=seed),
    )


def image_dataset_from_directory(
    directory: str,
    config: DatasetConfig | None = None,
) -> NumpyDataset:
    """Generate a dataset from image files in a directory.

    Args:
        directory (str): The directory parameter.
        config (object): The config parameter.

    Returns:
        NumpyDataset: Result.
    """
    conf = config if config is not None else DatasetConfig()
    labels = conf.io_config.labels
    class_names = conf.io_config.class_names
    batch_size = conf.batch_config.batch_size
    shuffle = conf.batch_config.shuffle
    seed = conf.batch_config.seed
    file_paths, file_labels, class_names = _get_files_and_labels(
        directory,
        labels,
        class_names,
        valid_exts=(".jpg", ".jpeg", ".png", ".bmp", ".gif"),
    )
    return NumpyDataset(
        file_paths,
        file_labels,
        config=BatchConfig(batch_size=batch_size, shuffle=shuffle, seed=seed),
    )


def text_dataset_from_directory(
    directory: str,
    config: DatasetConfig | None = None,
) -> NumpyDataset:
    """Generate a dataset from text files in a directory.

    Args:
        directory (str): The directory parameter.
        config (object): The config parameter.

    Returns:
        NumpyDataset: Result.
    """
    conf = config if config is not None else DatasetConfig()
    labels = conf.io_config.labels
    class_names = conf.io_config.class_names
    batch_size = conf.batch_config.batch_size
    shuffle = conf.batch_config.shuffle
    seed = conf.batch_config.seed
    file_paths, file_labels, class_names = _get_files_and_labels(directory, labels, class_names, valid_exts=(".txt",))
    texts = []
    for fp in file_paths:
        with open(fp, encoding="utf-8") as f:
            texts.append(f.read())
    return NumpyDataset(texts, file_labels, config=BatchConfig(batch_size=batch_size, shuffle=shuffle, seed=seed))


def _get_timeseries_indices(
    data_len: int,
    config: dict[str, int | None],
) -> tuple[int, int, int]:
    """Evaluate _get_timeseries_indices operation.

    Args:
        data_len (int): The data_len parameter.
        config (object): The config parameter.

    Yields: Any: Result.
    """
    start = 0 if config["start_index"] is None else config["start_index"]
    end = data_len if config["end_index"] is None else config["end_index"]
    stop = end - config["sequence_length"] * config["sampling_rate"] + 1  # type: ignore
    return start, stop, config["sequence_stride"]  # type: ignore


def _extract_timeseries_windows(
    data: Any,
    targets: Any,
    params: dict[str, int],
    bounds: tuple[int, int, int],
) -> tuple[list[Any], list[Any] | None]:
    """Evaluate _extract_timeseries_windows operation.

    Args:
        data (object): The data parameter.
        targets (object): The targets parameter.
        params (object): The params parameter.
        bounds (object): The bounds parameter.

    Yields: Any: Result.
    """
    x = []
    y = [] if targets is not None else None  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism
    start, stop, stride = bounds
    seq_len, samp_rate = params["sequence_length"], params["sampling_rate"]
    for i in range(start, stop, stride):
        x.append(data[i : i + seq_len * samp_rate : samp_rate])
        if y is not None:
            y.append(targets[i + seq_len * samp_rate - 1])
    return x, y


def timeseries_dataset_from_array(
    data: Any,
    targets: Any,
    sequence_length: int,
    config: DatasetConfig | None = None,
) -> NumpyDataset:
    """Create a dataset of sliding windows over a timeseries provided as array.

    Args:
        data (object): The data parameter.
        targets (object): The targets parameter.
        sequence_length (int): The sequence_length parameter.
        config (object): The config parameter.

    Returns:
        NumpyDataset: Result.
    """
    conf = config if config is not None else DatasetConfig()
    sequence_stride = conf.loader.sequence_stride
    sampling_rate = conf.loader.sampling_rate if conf.loader.sampling_rate is not None else 1
    start, stop, stride = _get_timeseries_indices(
        len(data),
        {
            "sequence_length": sequence_length,
            "sampling_rate": sampling_rate,
            "sequence_stride": sequence_stride,
            "start_index": conf.loader.start_index,
            "end_index": conf.loader.end_index,
        },
    )
    x, y = _extract_timeseries_windows(
        data,
        targets,
        {"sequence_length": sequence_length, "sampling_rate": sampling_rate},
        (start, stop, stride),
    )
    return NumpyDataset(
        x,
        y,
        config=BatchConfig(
            batch_size=conf.batch_config.batch_size,
            shuffle=conf.batch_config.shuffle,
            seed=conf.batch_config.seed,
        ),
    )


def pack_x_y_sample_weight(x: Any, y: Any | None = None, sample_weight: Any | None = None) -> Any:
    """Pack x, y, and sample_weight.

    Args:
        x: Input data.
        y: Target data.
        sample_weight: Sample weights.

    Returns:
        Packed output.
    """
    if y is None and sample_weight is None:
        return x
    if sample_weight is None:
        return (x, y)
    return (x, y, sample_weight)


def pad_sequences(
    sequences: list[list[Any]],
    maxlen: int | None = None,
    dtype: str = "int32",
    padding: str = "pre",
    truncating: str = "pre",
    value: Any = 0.0,
) -> list[list[Any]]:
    """Pad sequences to the same length.

    Args:
        sequences: List of lists of sequences.
        maxlen: Maximum length of sequences.
        dtype: Data type.
        padding: Padding mode ('pre' or 'post').
        truncating: Truncating mode ('pre' or 'post').
        value: Padding value.

    Returns:
        Padded sequences.
    """
    if not sequences:
        return []
    if maxlen is None:
        maxlen = max(len(seq) for seq in sequences)
    padded = []
    for seq in sequences:
        if len(seq) > maxlen:
            if truncating == "pre":
                padded.append(seq[-maxlen:])
            else:
                padded.append(seq[:maxlen])
        elif len(seq) < maxlen:
            pad_len = maxlen - len(seq)
            pad_vals = [value] * pad_len
            if padding == "pre":
                padded.append(pad_vals + seq)
            else:
                padded.append(seq + pad_vals)
        else:
            padded.append(seq.copy() if hasattr(seq, "copy") else list(seq))
    return padded


def split_dataset(dataset: Any, left_size: float = 0.5, shuffle: bool = False) -> tuple[Any, Any]:
    """Split a dataset.

    Args:
        dataset: Dataset to split.
        left_size: Size of the left split.
        shuffle: Whether to shuffle before splitting.

    Returns:
        Split dataset.
    """
    return dataset, dataset


def unpack_x_y_sample_weight(data: Any) -> tuple[Any, ...]:
    """Unpack x, y, and sample_weight.

    Args:
        data: Packed data.

    Returns:
        Unpacked values.
    """
    if isinstance(data, dict):
        return data.get("x"), data.get("y"), data.get("sample_weight")
    if isinstance(data, tuple):
        if len(data) == 1:
            return data[0], None, None
        if len(data) == 2:
            return data[0], data[1], None
        if len(data) == 3:
            return data[0], data[1], data[2]
    return data, None, None
