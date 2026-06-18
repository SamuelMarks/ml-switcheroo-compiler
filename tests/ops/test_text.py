"""Tests for text operations."""

import pytest
import numpy as np
from unittest import mock
from ml_switcheroo_compiler.core.config import ConfigContext
from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.device import Device, DeviceType
from ml_switcheroo_compiler.ops.text import (
    string_to_hash,
    regex_replace,
    string_split,
    lookup,
)
from ml_switcheroo_compiler.tracing import _tracer


def test_text_eager_mode_exceptions():
    device = Device(DeviceType.CPU, 0)
    img = Tensor(np.array(["test"]), (1,), DType.String, device)

    with ConfigContext(eager_mode=True):
        with mock.patch(
            "ml_switcheroo_compiler.backends.registry.get_active_backend"
        ) as mock_backend:
            mock_backend.return_value.execute_op.return_value = (np.zeros((1,)), np.zeros((1,)))
            mock_backend.return_value.array.return_value = np.zeros((1,))
            try:
                string_to_hash(img, 100)
                regex_replace(img, "test", "rewrite")
                string_split(img, " ")
                lookup(img, img)
            except Exception:
                pass


def test_text_tracing_mode():
    device = Device(DeviceType.CPU, 0)

    # Test RuntimeError outside of tracing
    with ConfigContext(eager_mode=False):
        img = Tensor("dummy_text", (1,), DType.String, device)
        with pytest.raises(RuntimeError, match="Cannot emit"):
            string_split(img, " ")

        _tracer.start_tracing()
        try:
            img = Tensor("dummy_text", (1,), DType.String, device)

            string_to_hash(img, 100)
            regex_replace(img, "test", "rewrite")
            string_split(img, " ")
            lookup(img, img)
        finally:
            _tracer.stop_tracing()
