# ruff: noqa: E501
"""Core abstractions and logic definitions for test_block_masked_mm.py."""

from __future__ import annotations

import typing

import numpy as np
import pytest

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.linalg.matmul import block_masked_mm
from ml_switcheroo_compiler.tracing import global_tracing_state
from tests.testing_utils import extract_test_dtype, extract_test_shape

if typing.TYPE_CHECKING:
    pass


def test_block_masked_mm_shape() -> None:
    """Test the block masked mm shape behavior.

    Returns:
        OpValue: The inferred shape or computed result.

    Raises:
        e: If execution or validation fails in `test_block_masked_mm_shape`.
    """
    try:
        config.eager_mode = False
        pytest.MonkeyPatch().setattr(config, "default_device", None, raising=False)
        global_tracing_state.start_tracing()
        try:
            a = Tensor(None, TensorConfig((2, 64, 128), DType.Float32, None))
            b = Tensor(None, TensorConfig((2, 128, 64), DType.Float32, None))
            out = block_masked_mm(a, b, block_size=64)
            assert extract_test_shape(out) == (2, 64, 64)
            assert extract_test_dtype(out) == DType.Float32
        finally:
            global_tracing_state.stop_tracing()
    except Exception as e:
        raise e


def test_block_masked_mm_eager() -> None:
    """Test the block masked mm eager behavior.

    Returns:
        OpValue: The inferred shape or computed result.

    Raises:
        e: If execution or validation fails in `test_block_masked_mm_eager`.
    """
    try:
        config.eager_mode = True
        pytest.MonkeyPatch().setattr(config, "default_device", None, raising=False)
        config.backend = "numpy"
        a_data = np.random.randn(2, 64, 128).astype(np.float32)
        b_data = np.random.randn(2, 128, 64).astype(np.float32)
        a = Tensor(a_data, TensorConfig((2, 64, 128), DType.Float32, None))
        b = Tensor(b_data, TensorConfig((2, 128, 64), DType.Float32, None))
        out = block_masked_mm(a, b, block_size=64)
        assert extract_test_shape(out) == (2, 64, 64)
    except Exception as e:
        raise e
