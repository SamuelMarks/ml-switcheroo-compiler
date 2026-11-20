"""Tests for test_loss_ops."""

from __future__ import annotations

import numpy as np
import pytest

from ml_switcheroo_compiler.backends.eager.nn_ops.losses import (
    _np_categorical_generalized_cross_entropy,
    _np_circle_loss,
    _np_ctc_loss,
    _np_ctc_loss_single,
)
from ml_switcheroo_compiler.backends.eager_registry import BackendArg
from tests.testing_utils import MockTestStub, coerce_int_list, extract_test_shape


class DummyBackend(MockTestStub):
    """Test class DummyBackend."""

    @staticmethod
    def zeros(shape: BackendArg) -> BackendArg:
        """Test function zeros.

        Args:
            shape: Test parameter `shape`.

        Returns:
            BackendArg: Computed output value from `zeros`.
        """
        return np.zeros(coerce_int_list(shape))


def test_loss_ops() -> None:
    """Test function test_loss_ops."""
    labels = np.array([[1, 2]])
    logits = np.ones((5, 1, 3))
    label_length = np.array([2])
    logit_length = np.array([5])
    logit_length_3 = np.array([3])
    assert extract_test_shape(_np_ctc_loss(DummyBackend(), labels, logits, label_length, logit_length)) == (1,)
    logits2 = np.ones((5, 3))
    assert extract_test_shape(_np_ctc_loss(DummyBackend(), labels, logits2, label_length, logit_length)) == (1,)
    logits1 = np.ones((3,))
    assert extract_test_shape(_np_ctc_loss(DummyBackend(), labels, logits1, label_length, logit_length_3)) == (1,)
    logits_batch_major = np.ones((1, 5, 3))
    assert extract_test_shape(
        _np_ctc_loss(DummyBackend(), labels, logits_batch_major, label_length, logit_length, logits_time_major=False)
    ) == (1,)
    labels_1d = np.array([1, 2])
    assert extract_test_shape(_np_ctc_loss(DummyBackend(), labels_1d, logits, label_length, logit_length)) == (1,)
    assert extract_test_shape(
        _np_ctc_loss(DummyBackend(), np.array([[]], dtype=int), logits, np.array([0]), logit_length)
    ) == (1,)
    y_true = np.array([1, 0])
    y_pred = np.array([0.9, 0.1])
    assert extract_test_shape(_np_circle_loss(DummyBackend(), y_true, y_pred)) == ()
    assert _np_circle_loss(DummyBackend(), 1) == 0
    y_true_cce = np.array([[1, 0], [0, 1]])
    y_pred_cce = np.array([[0.9, 0.1], [0.2, 0.8]])
    assert extract_test_shape(_np_categorical_generalized_cross_entropy(DummyBackend(), y_true_cce, y_pred_cce)) == ()
    assert _np_categorical_generalized_cross_entropy(DummyBackend(), 1) == 0


def test_ctc_loss_missing_branches() -> None:
    """Test function test_ctc_loss_missing_branches."""
    probs = np.ones((2, 2))
    labels = np.array([])
    res = _np_ctc_loss_single(probs, labels, 2, 0)
    assert isinstance(res, float)
    probs = np.ones((0, 2))
    labels = np.array([0, 1])
    with pytest.raises(IndexError):
        _np_ctc_loss_single(probs, labels, 0, 2)
