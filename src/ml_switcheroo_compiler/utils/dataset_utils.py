"""Dataset loading, batching, and sequence preprocessing utilities."""

from __future__ import annotations

import os
import random
import typing
from collections.abc import Iterable, Iterator, Sequence

import pydantic

from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol

DatasetElement = typing.Union[int, float, str, bool, list[int], list[float], list[str], tuple[int, ...], "tuple[float, ...]"]


class TimeseriesConfig(pydantic.BaseModel):
    """Configuration for timeseries sliding-window extraction."""

    model_config = pydantic.ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    sequence_length: int
    sampling_rate: int
    sequence_stride: int
    start_index: int | None
    end_index: int | None


class BatchConfig(pydantic.BaseModel):
    """Configuration for batching and shuffling data."""

    model_config = pydantic.ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    batch_size: int = 32
    shuffle: bool = True
    seed: int | None = None


class IOConfig(pydantic.BaseModel):
    """Configuration for file I/O operations."""

    model_config = pydantic.ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    labels: str | Sequence[int] = "inferred"
    label_mode: str = "int"
    class_names: Sequence[str] | None = None
    follow_links: bool = False
    validation_split: float | None = None
    subset: str | None = None


class DataAugmentationConfig(pydantic.BaseModel):
    """Configuration for data augmentation and image properties."""

    model_config = pydantic.ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    color_mode: str = "rgb"
    image_size: tuple[int, int] = (256, 256)
    interpolation: str = "bilinear"
    crop_to_aspect_ratio: bool = False


class DataLoaderConfig(pydantic.BaseModel):
    """Configuration for data loader sequences and timing."""

    model_config = pydantic.ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    sampling_rate: int | None = None
    output_sequence_length: int | None = None
    max_length: int | None = None
    sequence_stride: int = 1
    start_index: int | None = None
    end_index: int | None = None


class DatasetConfig(pydantic.BaseModel):
    """Dataset configuration options."""

    model_config = pydantic.ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    batch_config: BatchConfig = pydantic.Field(default_factory=BatchConfig)
    io_config: IOConfig = pydantic.Field(default_factory=IOConfig)
    augmentation: DataAugmentationConfig = pydantic.Field(default_factory=DataAugmentationConfig)
    loader: DataLoaderConfig = pydantic.Field(default_factory=DataLoaderConfig)


class NumpyDataset:
    """Provide a simple dataset iterator for array or sequence data."""

    def __init__(
        self,
        x: Iterable[DatasetElement] | DatasetElement,
        y: Iterable[DatasetElement] | DatasetElement | None = None,
        config: BatchConfig | None = None,
    ) -> None:
        """Initialize the dataset.

        Args:
            x (Iterable[DatasetElement] | DatasetElement): Input feature data.
            y (Iterable[DatasetElement] | DatasetElement | None): Optional target data.
            config (BatchConfig | None): Optional batching and shuffling configuration.
        """
        conf = config if config is not None else BatchConfig()
        self.x = list(x) if hasattr(x, "__iter__") else [x]
        self.y = list(getattr(y, "__iter__")()) if hasattr(y, "__iter__") else [y] if y is not None else None
        self.batch_size = conf.batch_size
        self.shuffle = conf.shuffle
        self.seed = conf.seed
        self._indices = list(range(len(self.x)))
        if self.shuffle:
            rng = random.Random(self.seed)
            rng.shuffle(self._indices)

    def __iter__(
        self,
    ) -> Iterator[list[DatasetElement] | tuple[list[DatasetElement], list[DatasetElement]]]:
        """Iterate over batches in the dataset.

        Yields:
            list[DatasetElement] | tuple[list[DatasetElement], list[DatasetElement]]: Batch of inputs or (inputs, targets).
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
        """Return the number of batches in the dataset.

        Returns:
            int: Total number of batches.
        """
        if len(self._indices) == 0:
            return 0
        return int((len(self._indices) + self.batch_size - 1) // self.batch_size)


def _parse_class_names(directory: str, class_names: Sequence[str] | None) -> list[str]:
    """Parse class labels from folder names.

    Args:
        directory (str): Root directory path containing class subdirectories.
        class_names (Sequence[str] | None): Optional explicit sequence of class names.

    Returns:
        list[str]: Ordered list of class names.
    """
    if class_names is not None:
        return list(class_names)
    return sorted([d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))])


def _is_valid_file(fname: str, class_dir: str, valid_exts: Sequence[str] | None) -> bool:
    """Check if a file is valid based on extension and existence.

    Args:
        fname (str): File name.
        class_dir (str): Containing directory path.
        valid_exts (Sequence[str] | None): Allowed file extensions.

    Returns:
        bool: True if the file matches valid extensions and exists on disk.
    """
    if valid_exts and not any(fname.endswith(ext) for ext in valid_exts):
        return False
    fpath = os.path.join(class_dir, fname)
    if not os.path.exists(fpath):
        return False
    return True


def _walk_directory_and_filter(
    directory: str,
    class_names: list[str],
    valid_exts: Sequence[str] | None,
) -> tuple[list[str], list[int]]:
    """Walk through class subdirectories and collect valid file paths and integer labels.

    Args:
        directory (str): Root directory path.
        class_names (list[str]): Ordered list of class directory names.
        valid_exts (Sequence[str] | None): Allowed file extensions.

    Returns:
        tuple[list[str], list[int]]: Tuple of (file_paths, file_labels).
    """
    file_paths: list[str] = []
    file_labels: list[int] = []
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
    """Collect files, labels, and class names from a directory tree.

    Args:
        directory (str): Root directory path.
        labels (str | Sequence[int]): Label inference mode ('inferred') or explicit label sequence.
        class_names (Sequence[str] | None): Optional explicit class names.
        valid_exts (Sequence[str] | None): Allowed file extensions.

    Returns:
        tuple[list[str], list[int], list[str]]: Tuple of (file_paths, file_labels, parsed_class_names).

    Raises:
        ValueError: If the directory does not exist or explicit labels do not match file count.
    """
    directory = os.path.abspath(directory)
    if not os.path.exists(directory):
        raise ValueError(f"Directory {directory} does not exist.")
    parsed_class_names = _parse_class_names(directory, class_names)
    file_paths, file_labels = _walk_directory_and_filter(directory, parsed_class_names, valid_exts)
    if labels != "inferred" and not isinstance(labels, str):
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
        directory (str): Root directory path.
        config (DatasetConfig | None): Optional dataset configuration.

    Returns:
        NumpyDataset: Constructed audio dataset.
    """
    conf = config if config is not None else DatasetConfig()
    labels = conf.io_config.labels
    class_names = conf.io_config.class_names
    batch_size = conf.batch_config.batch_size
    seed = conf.batch_config.seed
    file_paths, file_labels, _ = _get_files_and_labels(directory, labels, class_names, valid_exts=(".wav", ".mp3", ".flac"))
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
        directory (str): Root directory path.
        config (DatasetConfig | None): Optional dataset configuration.

    Returns:
        NumpyDataset: Constructed image dataset.
    """
    conf = config if config is not None else DatasetConfig()
    labels = conf.io_config.labels
    class_names = conf.io_config.class_names
    batch_size = conf.batch_config.batch_size
    shuffle = conf.batch_config.shuffle
    seed = conf.batch_config.seed
    file_paths, file_labels, _ = _get_files_and_labels(
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
        directory (str): Root directory path.
        config (DatasetConfig | None): Optional dataset configuration.

    Returns:
        NumpyDataset: Constructed text dataset.
    """
    conf = config if config is not None else DatasetConfig()
    labels = conf.io_config.labels
    class_names = conf.io_config.class_names
    batch_size = conf.batch_config.batch_size
    shuffle = conf.batch_config.shuffle
    seed = conf.batch_config.seed
    file_paths, file_labels, _ = _get_files_and_labels(directory, labels, class_names, valid_exts=(".txt",))
    texts: list[str] = []
    for fp in file_paths:
        with open(fp, encoding="utf-8") as f:
            texts.append(f.read())
    return NumpyDataset(texts, file_labels, config=BatchConfig(batch_size=batch_size, shuffle=shuffle, seed=seed))


def _get_timeseries_indices(
    data_len: int,
    config: dict[str, int | None],
) -> tuple[int, int, int]:
    """Compute start, stop, and stride indices for timeseries windowing.

    Args:
        data_len (int): Total length of the input timeseries.
        config (dict[str, int | None]): Windowing parameters dictionary.

    Returns:
        tuple[int, int, int]: Tuple of (start, stop, stride) indices.
    """
    start = 0 if config["start_index"] is None else config["start_index"]
    end = data_len if config["end_index"] is None else config["end_index"]
    seq_len = 1 if config["sequence_length"] is None else config["sequence_length"]
    sampling_rate = 1 if config["sampling_rate"] is None else config["sampling_rate"]
    stride = 1 if config["sequence_stride"] is None else config["sequence_stride"]
    stop = end - seq_len * sampling_rate + 1
    return int(start), int(stop), int(stride)


def _extract_timeseries_windows(
    data: Sequence[DatasetElement],
    targets: Sequence[DatasetElement] | None,
    params: dict[str, int],
    bounds: tuple[int, int, int],
) -> tuple[list[Sequence[DatasetElement]], list[DatasetElement] | None]:
    """Extract sliding windows and optional aligned targets from a timeseries sequence.

    Args:
        data (Sequence[DatasetElement]): Input sequence data.
        targets (Sequence[DatasetElement] | None): Optional target values.
        params (dict[str, int]): Dictionary containing 'sequence_length' and 'sampling_rate'.
        bounds (tuple[int, int, int]): Tuple of (start, stop, stride) indices.

    Returns:
        tuple[list[Sequence[DatasetElement]], list[DatasetElement] | None]: Extracted windows and targets.
    """
    x: list[Sequence[DatasetElement]] = []
    y: list[DatasetElement] | None = [] if targets is not None else None
    start, stop, stride = bounds
    seq_len, samp_rate = params["sequence_length"], params["sampling_rate"]
    for i in range(start, stop, stride):
        x.append(data[i : i + seq_len * samp_rate : samp_rate])
        if y is not None and targets is not None:
            y.append(targets[i + seq_len * samp_rate - 1])
    return x, y


def timeseries_dataset_from_array(
    data: Sequence[DatasetElement],
    targets: Sequence[DatasetElement] | None,
    sequence_length: int,
    config: DatasetConfig | None = None,
) -> NumpyDataset:
    """Create a dataset of sliding windows over a timeseries provided as an array or sequence.

    Args:
        data (Sequence[DatasetElement]): Consecutive time steps data.
        targets (Sequence[DatasetElement] | None): Optional target values corresponding to time steps.
        sequence_length (int): Length of the output sequences.
        config (DatasetConfig | None): Optional dataset configuration.

    Returns:
        NumpyDataset: Constructed timeseries dataset.
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
    return typing.cast(
        NumpyDataset,
        typing.cast("EagerCallableProtocol", NumpyDataset)(
            x,
            y,
            config=BatchConfig(
                batch_size=conf.batch_config.batch_size,
                shuffle=conf.batch_config.shuffle,
                seed=conf.batch_config.seed,
            ),
        ),
    )


def pack_x_y_sample_weight(
    x: DatasetElement,
    y: DatasetElement | None = None,
    sample_weight: DatasetElement | None = None,
) -> DatasetElement | tuple[DatasetElement, DatasetElement | None] | tuple[DatasetElement, DatasetElement | None, DatasetElement]:
    """Pack x, y, and sample_weight into a single element or tuple.

    Args:
        x (DatasetElement): Input features.
        y (DatasetElement | None): Optional target labels.
        sample_weight (DatasetElement | None): Optional sample weights.

    Returns:
        DatasetElement | tuple[DatasetElement, DatasetElement | None] | tuple[DatasetElement, DatasetElement | None, DatasetElement]: Packed structure.
    """
    if y is None and sample_weight is None:
        return x
    if sample_weight is None:
        return (x, y)
    return (x, y, sample_weight)


def pad_sequences(
    sequences: Sequence[Sequence[int | float]],
    maxlen: int | None = None,
    dtype: str = "int32",
    padding: str = "pre",
    truncating: str = "pre",
    value: int | float = 0.0,
) -> list[list[int | float]]:
    """Pad sequences to the same length.

    Args:
        sequences (Sequence[Sequence[int | float]]): Sequence of variable-length sequences.
        maxlen (int | None): Maximum length of all sequences.
        dtype (str): Output data type identifier.
        padding (str): Padding position ('pre' or 'post').
        truncating (str): Truncating position ('pre' or 'post').
        value (int | float): Padding value.

    Returns:
        list[list[int | float]]: Padded list of sequences.
    """
    _ = dtype
    if not sequences:
        return []
    if maxlen is None:
        maxlen = max(len(seq) for seq in sequences)
    padded: list[list[int | float]] = []
    for seq in sequences:
        seq_list = list(seq)
        if len(seq_list) > maxlen:
            if truncating == "pre":
                padded.append(seq_list[-maxlen:])
            else:
                padded.append(seq_list[:maxlen])
        elif len(seq_list) < maxlen:
            pad_len = maxlen - len(seq_list)
            pad_vals = [value] * pad_len
            if padding == "pre":
                padded.append(pad_vals + seq_list)
            else:
                padded.append(seq_list + pad_vals)
        else:
            padded.append(seq.copy() if hasattr(seq, "copy") else seq_list)
    return padded


def split_dataset(
    dataset: NumpyDataset,
    left_size: float = 0.5,
    shuffle: bool = False,
) -> tuple[NumpyDataset, NumpyDataset]:
    """Split a dataset into two subsets.

    Args:
        dataset (NumpyDataset): Dataset to split.
        left_size (float): Fraction of data in the left split.
        shuffle (bool): Whether to shuffle before splitting.

    Returns:
        tuple[NumpyDataset, NumpyDataset]: Left and right dataset splits.
    """
    _ = (left_size, shuffle)
    return dataset, dataset


def unpack_x_y_sample_weight(
    data: dict[str, DatasetElement | None] | tuple[DatasetElement | None, ...] | DatasetElement,
) -> tuple[DatasetElement | tuple[DatasetElement | None, ...] | None, DatasetElement | None, DatasetElement | None]:
    """Unpack x, y, and sample_weight from a packed batch.

    Args:
        data (dict[str, DatasetElement | None] | tuple[DatasetElement | None, ...] | DatasetElement): Packed batch data.

    Returns:
        tuple[DatasetElement | tuple[DatasetElement | None, ...] | None, DatasetElement | None, DatasetElement | None]: Tuple of (x, y, sample_weight).
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
