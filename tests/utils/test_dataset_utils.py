"""Tests for dataset utilities."""

from ml_switcheroo_compiler.utils.dataset_utils import (
    pack_x_y_sample_weight,
    pad_sequences,
    split_dataset,
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
