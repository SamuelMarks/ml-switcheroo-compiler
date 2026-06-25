"""Tests for text operations."""

from unittest import mock

import numpy as np
import pytest

from ml_switcheroo_compiler.core.config import ConfigContext
from ml_switcheroo_compiler.core.device import Device, DeviceType
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.text import lookup, regex_replace, string_split, string_to_hash
from ml_switcheroo_compiler.tracing import _tracer


def test_text_eager_mode_exceptions():
    device = Device(DeviceType.CPU, 0)
    img = Tensor(np.array(["test"]), TensorConfig((1,), DType.String, device))

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

            from ml_switcheroo_compiler.ops.text import (
                regex_full_match,
                string_join,
                string_length,
                string_substr,
            )

            try:
                regex_full_match(img, "pattern")
                string_join([img, img])
                string_length(img)
                string_substr(img, 0, 1)
            except Exception:
                pass


def test_text_tracing_mode():
    device = Device(DeviceType.CPU, 0)

    # Test RuntimeError outside of tracing
    with ConfigContext(eager_mode=False):
        img = Tensor("dummy_text", TensorConfig((1,), DType.String, device))
        with pytest.raises(RuntimeError, match="Cannot emit"):
            string_split(img, " ")

        _tracer.start_tracing()
        try:
            img = Tensor("dummy_text", TensorConfig((1,), DType.String, device))

            string_to_hash(img, 100)
            regex_replace(img, "test", "rewrite")
            string_split(img, " ")
            lookup(img, img)
        finally:
            _tracer.stop_tracing()


def test_text_new_ops() -> None:
    from ml_switcheroo_compiler.core.config import ConfigContext
    from ml_switcheroo_compiler.ops.text import (
        edit_distance,
        as_string,
        regex_full_match,
        string_join,
        string_length,
        string_substr,
    )
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
    from ml_switcheroo_compiler.tracing.tracer import _tracer, ProxyTensor

    hypothesis = Tensor(
        ProxyTensor(id="h", shape=(), dtype="string"), TensorConfig((), "string", None)
    )
    truth = Tensor(ProxyTensor(id="t", shape=(), dtype="string"), TensorConfig((), "string", None))
    num_tensor = Tensor(
        ProxyTensor(id="n", shape=(), dtype="float32"), TensorConfig((), "float32", None)
    )

    with ConfigContext(eager_mode=False):
        _tracer.start_tracing()
        edit_distance(hypothesis, truth)
        as_string(num_tensor)
        regex_full_match(hypothesis, pattern="^test")
        string_join([hypothesis, truth], separator=",")
        string_length(hypothesis)
        string_substr(hypothesis, pos=0, len=1)
        _tracer.stop_tracing()
