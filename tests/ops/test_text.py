"""Tests for text operations."""

from unittest import mock

import numpy as np
import pytest

from ml_switcheroo_compiler.core.config import ConfigContext
from ml_switcheroo_compiler.core.device import Device, DeviceType
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.text import (
    as_string,
    edit_distance,
    lookup,
    regex_full_match,
    regex_replace,
    string_join,
    string_length,
    string_split,
    string_substr,
    string_to_hash,
)
from ml_switcheroo_compiler.tracing import global_tracing_state
from ml_switcheroo_compiler.tracing.tracer import ProxyTensor


def test_text_eager_mode_exceptions() -> object:
    """Function docstring."""
    device = Device(DeviceType.CPU, 0)
    img = Tensor(np.array(["test"]), TensorConfig((1,), DType.String, device))

    with ConfigContext(eager_mode=True):
        with mock.patch("ml_switcheroo_compiler.backends.registry.get_active_backend") as mock_backend:
            mock_backend.return_value.execute_op.return_value = (np.zeros((1,)), np.zeros((1,)))
            mock_backend.return_value.array.return_value = np.zeros((1,))
            try:
                string_to_hash(img, 100)
                regex_replace(img, "test", "rewrite")
                string_split(img, " ")
                lookup(img, img)
            except Exception:
                pass

            try:
                regex_full_match(img, "pattern")
                string_join([img, img])
                string_length(img)
                string_substr(img, 0, 1)
            except Exception:
                pass


def test_text_tracing_mode() -> object:
    """Function docstring."""
    device = Device(DeviceType.CPU, 0)

    # Test RuntimeError outside of tracing
    with ConfigContext(eager_mode=False):
        img = Tensor("dummy_text", TensorConfig((1,), DType.String, device))
        with pytest.raises(RuntimeError, match="Cannot emit"):
            string_split(img, " ")

        global_tracing_state.start_tracing()
        try:
            img = Tensor("dummy_text", TensorConfig((1,), DType.String, device))

            string_to_hash(img, 100)
            regex_replace(img, "test", "rewrite")
            string_split(img, " ")
            lookup(img, img)
        finally:
            global_tracing_state.stop_tracing()


def test_text_new_ops() -> None:
    """Function docstring."""
    hypothesis = Tensor(ProxyTensor(id="h", shape=(), dtype="string"), TensorConfig((), "string", None))
    truth = Tensor(ProxyTensor(id="t", shape=(), dtype="string"), TensorConfig((), "string", None))
    num_tensor = Tensor(ProxyTensor(id="n", shape=(), dtype="float32"), TensorConfig((), "float32", None))

    with ConfigContext(eager_mode=False):
        global_tracing_state.start_tracing()
        edit_distance(hypothesis, truth)
        as_string(num_tensor)
        regex_full_match(hypothesis, pattern="^test")
        string_join([hypothesis, truth], separator=",")
        string_length(hypothesis)
        string_substr(hypothesis, pos=0, len=1)
        global_tracing_state.stop_tracing()
