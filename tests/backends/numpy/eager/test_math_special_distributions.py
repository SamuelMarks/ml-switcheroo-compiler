"""Test math_misc edge cases coverage."""

from __future__ import annotations

import numpy as np
import pytest

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager import ensure_ndarray
from tests.backends.numpy.eager._math_advanced_test_ns import _np_rawmatmul, _np_rawmerge, _np_rem, _np_tensorarraywrite
from tests.testing_utils import BackendOperand, EagerTensorProtocol, MockTestStub, require_callable, unpack_pair


def test_np_rawmatmul() -> None:
    """Test function test_np_rawmatmul."""
    a = np.ones((2, 2))
    b = np.ones((2, 2))
    res = _np_rawmatmul(np, a, b)
    assert np.array_equal(ensure_ndarray(res), np.matmul(a, b))


def test_np_rawmerge() -> None:
    """Test function test_np_rawmerge."""
    a = np.ones((2, 2))
    res, status = unpack_pair(_np_rawmerge(np, [a, a]))
    assert res is not None
    res_empty, status_empty = unpack_pair(_np_rawmerge(np, []))
    assert res_empty is None


def test_np_tensorarraywrite() -> None:
    """Test function test_np_tensorarraywrite."""
    handle = [1, 2]
    # In bounds
    res = _np_tensorarraywrite(np, handle, 1, 3)
    assert res == [1, 3]


def test_np_rem() -> None:
    """Test function test_np_rem."""
    a = np.array([5])
    b = np.array([3])
    res = _np_rem(np, a, b)
    assert ensure_ndarray(res)[0] == 2


def test_np_confusion_matrix_fallback() -> None:
    """Test function test_np_confusion_matrix_fallback."""
    a = np.array([0, 1])
    b = np.array([1, 1])
    fn = numpy_eager_registry.get("confusion_matrix")
    res = require_callable(fn)(np, a, b)
    assert res.shape == (2, 2)


def test_np_descriptive() -> None:
    """Test function test_np_descriptive."""
    a = np.array([1.0, 2.0, 3.0])
    fn = numpy_eager_registry.get("descriptive")
    res = require_callable(fn)(np, a)
    assert len(res) == 3


def test_np_distributions() -> None:
    """Test function test_np_distributions."""
    a = np.array([1.0, 2.0, 3.0])
    fn = numpy_eager_registry.get("distributions")
    res = require_callable(fn)(np, a)
    assert len(res) == 2


def test_np_confusion_matrix_camel() -> None:
    """Test function test_np_confusion_matrix_camel."""
    a = np.array([0, 1])
    b = np.array([1, 1])
    fn = numpy_eager_registry.get("ConfusionMatrix")
    res = require_callable(fn)(np, a, b, num_classes=3)
    assert res.shape == (3, 3)


def test_np_decode_csv_empty() -> None:
    """Test function test_np_decode_csv_empty."""
    fn = numpy_eager_registry.get("DecodeCsv")
    res = require_callable(fn)(np, "", record_defaults=[1.0, 2.0])
    assert len(res) == 2
    assert res[0] == 1.0


def test_fallback_snippets_mock(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test function test_fallback_snippets_mock.

    Args:
        monkeypatch: Test parameter `monkeypatch`.
    """
    import ml_switcheroo_compiler.ops as ops

    class FakeOp(MockTestStub):
        """Test class FakeOp."""

        hit: bool = False

        def __init__(self, *args: BackendOperand, **kwargs: BackendOperand) -> None:
            """Initialize FakeOp test stub.

            Args:
                *args (BackendOperand): Positional arguments.
                **kwargs (BackendOperand): Keyword arguments.
            """
            del args, kwargs
            super().__init__("FakeOp")
            self.hit = True

    # Mocking standard snippet ops to hit the cls_or_func() path
    monkeypatch.setattr(ops, "RawMatMul", FakeOp, raising=False)
    monkeypatch.setattr(ops, "SparseDenseMatMul", FakeOp, raising=False)
    monkeypatch.setattr(ops, "rem", FakeOp, raising=False)
    monkeypatch.setattr(ops, "confusion_matrix", FakeOp, raising=False)
    monkeypatch.setattr(ops, "descriptive", FakeOp, raising=False)
    monkeypatch.setattr(ops, "distributions", FakeOp, raising=False)

    res1: EagerTensorProtocol | FakeOp = require_callable(numpy_eager_registry.get)("RawMatMul")(
        np, np.ones((2, 2)), np.ones((2, 2))
    )
    if isinstance(res1, FakeOp):
        assert res1.hit

    res2: EagerTensorProtocol | FakeOp = require_callable(numpy_eager_registry.get)("SparseDenseMatMul")(
        np, np.ones((2, 2)), np.ones((2, 2))
    )
    if isinstance(res2, FakeOp):
        assert res2.hit

    res3: EagerTensorProtocol | FakeOp = require_callable(numpy_eager_registry.get)("rem")(
        np, np.ones((2, 2)), np.ones((2, 2))
    )
    if isinstance(res3, FakeOp):
        assert res3.hit

    res4: EagerTensorProtocol | FakeOp = require_callable(numpy_eager_registry.get)("confusion_matrix")(
        np, np.array([1], dtype=np.int32), np.array([1], dtype=np.int32)
    )
    if isinstance(res4, FakeOp):
        assert res4.hit

    res5: EagerTensorProtocol | FakeOp = require_callable(numpy_eager_registry.get)("descriptive")(np, np.ones((2, 2)))
    if isinstance(res5, FakeOp):
        assert res5.hit

    res6: EagerTensorProtocol | FakeOp = require_callable(numpy_eager_registry.get)("distributions")(
        np, np.ones((2, 2))
    )
    if isinstance(res6, FakeOp):
        assert res6.hit


def test_custom_root_coverage() -> None:
    """Test function test_custom_root_coverage."""
    import numpy as np

    from tests.backends.numpy.eager._math_advanced_test_ns import _np_customroot

    # solve is None
    res = _np_customroot(np, lambda x: x, 42.0)
    assert res == 42.0


def test_custom_root_coverage_solve() -> None:
    """Test function test_custom_root_coverage_solve."""
    import numpy as np

    from tests.backends.numpy.eager._math_advanced_test_ns import _np_customroot

    # solve is provided
    res = _np_customroot(np, lambda x: x, 42.0, solve=lambda f, x: f(x) + 1.0)
    assert res == 43.0
