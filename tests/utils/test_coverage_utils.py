"""Docstring module."""

import pytest
from ml_switcheroo.core.config import EagerMode, config, ConfigContext
from ml_switcheroo.core.tensor import Tensor
from ml_switcheroo.core.dtype import DType
from ml_switcheroo.core.device import Device, DeviceType
from ml_switcheroo.ops.shape import squeeze
from ml_switcheroo.core.type_promotion import promote_types
from ml_switcheroo.core.errors import DTypePromotionError
from ml_switcheroo.tracing import _tracer


def test_config_eager_mode() -> None:
    """Docstring."""
    with EagerMode():
        assert config.eager_mode


def test_config_invalid_key() -> None:
    """Docstring."""
    with pytest.raises(ValueError):
        with ConfigContext(invalid_key=True):
            pass


def test_shape_squeeze_no_dim_tracing() -> None:
    """Docstring."""
    config.eager_mode = False
    _tracer.start_tracing()
    try:

        class MockProxy:
            """Docstring."""

            def __init__(self) -> None:
                """Docstring."""
                self.id = "mock_id"

        t = Tensor(MockProxy(), (1, 2, 1, 3), DType.Float32, Device(DeviceType.CPU, 0))
        out = squeeze(t)
        assert out.shape == (2, 3)
    finally:
        _tracer.stop_tracing()


def test_type_promotion_complex128() -> None:
    """Docstring."""
    assert promote_types(DType.Complex128, DType.Float32) == DType.Complex128
    assert promote_types(DType.Float32, DType.Complex128) == DType.Complex128


def test_type_promotion_float64() -> None:
    """Docstring."""
    assert promote_types(DType.Float64, DType.Int32) == DType.Float64
    assert promote_types(DType.Int32, DType.Float64) == DType.Float64


def test_type_promotion_float_rank() -> None:
    """Docstring."""
    assert promote_types(DType.Float16, DType.Int32) == DType.Float16
    assert promote_types(DType.Int32, DType.Float16) == DType.Float16
    assert promote_types(DType.Float16, DType.BFloat16) == DType.Float32
    assert promote_types(DType.BFloat16, DType.Float16) == DType.Float32


def test_type_promotion_int_rank() -> None:
    """Docstring."""
    assert promote_types(DType.Int64, DType.Int32) == DType.Int64
    assert promote_types(DType.Int32, DType.Int64) == DType.Int64


def test_type_promotion_errors() -> None:
    """Docstring."""
    with pytest.raises(DTypePromotionError):
        promote_types("invalid", DType.Int32)


def test_type_promotion_coverage_gaps() -> None:
    """Docstring."""
    assert promote_types(DType.Float32, DType.Float32) == DType.Float32
    assert promote_types(DType.Complex64, DType.Float64) == DType.Complex128
    assert promote_types(DType.Float64, DType.Complex64) == DType.Complex128
    assert promote_types(DType.Complex64, DType.Float32) == DType.Complex64
    assert promote_types(DType.Float32, DType.Int32) == DType.Float32

    class MockDTypeBase:
        """Docstring."""

        def __init__(self, target: object, rank_call_limit: object = 0) -> None:
            """Docstring."""
            self.target = target
            self.calls = 0
            self.rank_call_limit = rank_call_limit

        def __eq__(self, other: object) -> object:
            """Docstring."""
            if other is self.target:
                self.calls += 1
                if self.calls <= self.rank_call_limit:
                    return True
                return False
            return other is self

        def __hash__(self) -> object:
            """Docstring."""
            return hash(self.target)

    m1 = MockDTypeBase(DType.Bool, 2)
    m2 = MockDTypeBase(DType.UInt8, 2)
    assert promote_types(m2, m1) is m2

    m3 = MockDTypeBase(DType.UInt8, 2)
    m4 = MockDTypeBase(DType.Bool, 2)
    assert promote_types(m4, m3) is m3
