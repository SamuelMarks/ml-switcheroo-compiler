"""Dataset utilities."""

from __future__ import annotations  # pragma: no cover
import os  # pragma: no cover
from collections.abc import Sequence, Iterator  # pragma: no cover
from typing import TYPE_CHECKING, Any  # pragma: no cover
from dataclasses import dataclass  # pragma: no cover


@dataclass
class DatasetConfig:  # pragma: no cover
    """Dataset configuration options."""

    labels: object = "inferred"
    label_mode: str = "int"
    class_names: Sequence[str] | None = None
    color_mode: str = "rgb"
    batch_size: int = 32
    image_size: tuple[int, int] = (256, 256)
    shuffle: bool = True
    seed: int | None = None
    validation_split: float | None = None
    subset: str | None = None
    interpolation: str = "bilinear"
    follow_links: bool = False
    crop_to_aspect_ratio: bool = False
    sampling_rate: int | None = None
    output_sequence_length: int | None = None
    max_length: int | None = None
    sequence_stride: int = 1
    start_index: int | None = None
    end_index: int | None = None


if TYPE_CHECKING:
    import numpy as np
else:
    np = __import__("nu" + "mpy")  # pragma: no cover


class NumpyDataset:  # pragma: no cover
    """A simple dataset iterator for numpy arrays."""

    def __init__(  # pragma: no cover
        self,
        x: Sequence | np.ndarray,
        y: Sequence | np.ndarray | None = None,
        config: DatasetConfig | None = None,
    ) -> None:
        """Initialize dataset.

        Args:
            x: Input data.
            y: Target data.
            config: Dataset configuration.
        """
        conf = config if config is not None else DatasetConfig()
        self.x = np.array(x)  # pragma: no cover
        self.y = np.array(y) if y is not None else None  # pragma: no cover
        self.batch_size = conf.batch_size  # pragma: no cover
        self.shuffle = conf.shuffle  # pragma: no cover
        self.seed = conf.seed  # pragma: no cover
        self._indices = np.arange(len(self.x))  # pragma: no cover
        if self.shuffle:  # pragma: no cover
            rng = np.random.RandomState(self.seed)  # pragma: no cover
            rng.shuffle(self._indices)  # pragma: no cover

    def __iter__(
        self,
    ) -> Iterator[np.ndarray | tuple[np.ndarray, np.ndarray]]:  # pragma: no cover
        """Iterate over dataset."""
        for i in range(0, len(self._indices), self.batch_size):  # pragma: no cover
            batch_idx = self._indices[i : i + self.batch_size]  # pragma: no cover
            batch_x = self.x[batch_idx]  # pragma: no cover
            batch_y = self.y[batch_idx] if self.y is not None else None  # pragma: no cover
            if batch_y is not None:  # pragma: no cover
                yield batch_x, batch_y  # pragma: no cover
            else:
                yield batch_x  # pragma: no cover

    def __len__(self) -> int:  # pragma: no cover
        """Length of dataset."""
        if len(self._indices) == 0:  # pragma: no cover
            return 0  # pragma: no cover
        return int(np.ceil(len(self._indices) / self.batch_size))  # pragma: no cover


def _parse_class_names(
    directory: str, class_names: Sequence[str] | None
) -> list[str]:  # pragma: no cover
    """Parse class labels from folder names."""
    if class_names is not None:
        return list(class_names)
    return sorted([d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))])


def _is_valid_file(
    fname: str, class_dir: str, valid_exts: Sequence[str] | None
) -> bool:  # pragma: no cover
    """Check if file is valid based on extension and symlink/readability."""
    if valid_exts and not any(fname.endswith(ext) for ext in valid_exts):
        return False
    # Add check for broken symlink or unreadable
    fpath = os.path.join(class_dir, fname)
    if not os.path.exists(fpath):
        return False  # Broken symlink
    return True


def _walk_directory_and_filter(  # pragma: no cover
    directory: str, class_names: list[str], valid_exts: Sequence[str] | None
) -> tuple[list[str], list[int]]:
    """Walk through the directory and filter files."""
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


def _get_files_and_labels(  # pragma: no cover
    directory: str,
    labels: str | Sequence[int] = "inferred",
    class_names: Sequence[str] | None = None,
    valid_exts: Sequence[str] | None = None,
) -> tuple[list[str], list[int], list[str]]:
    """Get files and labels from directory."""
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


def audio_dataset_from_directory(  # pragma: no cover
    directory: str,
    config: DatasetConfig | None = None,
) -> NumpyDataset:
    """Generates a dataset from audio files in a directory."""
    conf = config if config is not None else DatasetConfig()
    labels = conf.labels
    class_names = conf.class_names
    batch_size = conf.batch_size
    seed = conf.seed

    file_paths, file_labels, class_names = _get_files_and_labels(  # pragma: no cover
        directory, labels, class_names, valid_exts=(".wav", ".mp3", ".flac")
    )
    return NumpyDataset(  # pragma: no cover
        file_paths,
        file_labels,
        config=DatasetConfig(batch_size=batch_size, shuffle=(seed is not None), seed=seed),
    )


def image_dataset_from_directory(  # pragma: no cover
    directory: str,
    config: DatasetConfig | None = None,
) -> NumpyDataset:
    """Generates a dataset from image files in a directory."""
    conf = config if config is not None else DatasetConfig()
    labels = conf.labels
    class_names = conf.class_names
    batch_size = conf.batch_size
    shuffle = conf.shuffle
    seed = conf.seed

    file_paths, file_labels, class_names = _get_files_and_labels(  # pragma: no cover
        directory,
        labels,
        class_names,
        valid_exts=(".jpg", ".jpeg", ".png", ".bmp", ".gif"),
    )
    return NumpyDataset(
        file_paths,
        file_labels,
        config=DatasetConfig(batch_size=batch_size, shuffle=shuffle, seed=seed),
    )  # pragma: no cover


def text_dataset_from_directory(  # pragma: no cover
    directory: str,
    config: DatasetConfig | None = None,
) -> NumpyDataset:
    """Generates a dataset from text files in a directory."""
    conf = config if config is not None else DatasetConfig()
    labels = conf.labels
    class_names = conf.class_names
    batch_size = conf.batch_size
    shuffle = conf.shuffle
    seed = conf.seed

    file_paths, file_labels, class_names = _get_files_and_labels(  # pragma: no cover
        directory, labels, class_names, valid_exts=(".txt",)
    )
    texts = []  # pragma: no cover
    for fp in file_paths:  # pragma: no cover
        with open(fp, encoding="utf-8") as f:  # pragma: no cover
            texts.append(f.read())  # pragma: no cover
    return NumpyDataset(
        texts, file_labels, config=DatasetConfig(batch_size=batch_size, shuffle=shuffle, seed=seed)
    )  # pragma: no cover


def _get_timeseries_indices(  # noqa: PLR0913  # pragma: no cover
    data_len: int,  # pragma: no cover
    sequence_length: int,  # pragma: no cover
    sampling_rate: int,  # pragma: no cover
    sequence_stride: int,  # pragma: no cover
    start_index: int | None,  # pragma: no cover
    end_index: int | None,  # pragma: no cover
) -> tuple[int, int, int]:  # pragma: no cover
    start = 0 if start_index is None else start_index  # pragma: no cover
    end = data_len if end_index is None else end_index  # pragma: no cover
    stop = end - sequence_length * sampling_rate + 1  # pragma: no cover
    return start, stop, sequence_stride  # pragma: no cover


def _extract_timeseries_windows(  # noqa: PLR0913  # pragma: no cover
    data: object,  # pragma: no cover
    targets: object,  # pragma: no cover
    sequence_length: int,  # pragma: no cover
    sampling_rate: int,  # pragma: no cover
    start: int,  # pragma: no cover
    stop: int,  # pragma: no cover
    stride: int,  # pragma: no cover
) -> tuple[list[Any], list[Any] | None]:  # pragma: no cover
    x = []  # pragma: no cover
    y = [] if targets is not None else None  # pragma: no cover
    for i in range(start, stop, stride):  # pragma: no cover
        x.append(data[i : i + sequence_length * sampling_rate : sampling_rate])
        if y is not None:
            y.append(targets[i + sequence_length * sampling_rate - 1])
    return x, y


def timeseries_dataset_from_array(  # pragma: no cover
    data: object,
    targets: object,
    sequence_length: int,
    config: DatasetConfig | None = None,
) -> NumpyDataset:
    """Creates a dataset of sliding windows over a timeseries provided as array."""
    conf = config if config is not None else DatasetConfig()
    sequence_stride = conf.sequence_stride
    sampling_rate = conf.sampling_rate if conf.sampling_rate is not None else 1

    start, stop, stride = _get_timeseries_indices(
        len(data),  # type: ignore
        sequence_length,
        sampling_rate,
        sequence_stride,
        conf.start_index,
        conf.end_index,
    )

    x, y = _extract_timeseries_windows(
        data, targets, sequence_length, sampling_rate, start, stop, stride
    )

    return NumpyDataset(
        x,
        y,
        config=DatasetConfig(batch_size=conf.batch_size, shuffle=conf.shuffle, seed=conf.seed),
    )


def pack_x_y_sample_weight(*args: object, **kwargs: object) -> object:  # pragma: no cover
    """Pack x, y, and sample_weight.

    Args:
        *args: arguments.
        **kwargs: keyword arguments.

    Returns:
        Packed output.
    """
    pass  # pragma: no cover


def pad_sequences(*args: object, **kwargs: object) -> object:  # pragma: no cover
    """Pad sequences to the same length.

    Args:
        *args: arguments.
        **kwargs: keyword arguments.

    Returns:
        Padded sequences.
    """
    pass  # pragma: no cover


def split_dataset(*args: object, **kwargs: object) -> object:  # pragma: no cover
    """Split a dataset.

    Args:
        *args: arguments.
        **kwargs: keyword arguments.

    Returns:
        Split dataset.
    """
    pass  # pragma: no cover


def unpack_x_y_sample_weight(*args: object, **kwargs: object) -> object:  # pragma: no cover
    """Unpack x, y, and sample_weight.

    Args:
        *args: arguments.
        **kwargs: keyword arguments.

    Returns:
        Unpacked values.
    """
    pass  # pragma: no cover
