# ruff: noqa: E501
"""Core abstractions and logic definitions for test_dropout2d.py."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.nn.dropout import dropout2d
from ml_switcheroo_compiler.tracing import global_tracing_state

if TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol


def test_dropout2d_shape() -> None:
    """Test the dropout2d shape behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        try:
            config.eager_mode = False
            setattr(config, "default_device", None)
            global_tracing_state.start_tracing()
            try:
                a = Tensor(None, TensorConfig((2, 3, 64, 64), DType.Float32, None))
                out = dropout2d(a, p=0.5)
                assert cast("EagerTensorProtocol", getattr(out, "shape")) == (2, 3, 64, 64)
                assert cast("EagerTensorProtocol", getattr(out, "dtype")) == DType.Float32
            finally:
                global_tracing_state.stop_tracing()
        except Exception as e:
            raise e
            pass
    except Exception as e:
        raise e
        pass


def test_dropout2d_eager() -> None:
    """Test the dropout2d eager behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        try:
            config.eager_mode = True
            setattr(config, "default_device", None)
            config.backend = "numpy"
            a_data = np.random.randn(2, 3, 64, 64).astype(np.float32)
            a = Tensor(a_data, TensorConfig((2, 3, 64, 64), DType.Float32, None))
            out = dropout2d(a, p=0.5)
            assert cast("EagerTensorProtocol", getattr(out, "shape")) == (2, 3, 64, 64)
        except Exception as e:
            raise e
            pass
    except Exception as e:
        raise e
        pass
