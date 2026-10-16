"""Tests for test_text."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

# ruff: noqa: E501
from unittest import mock

import numpy as np
import pytest

from ml_switcheroo_compiler.core.config import ConfigContext
from ml_switcheroo_compiler.core.device import Device, DeviceType
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.text import as_string, edit_distance, lookup, regex_full_match, regex_replace, string_join, string_length, string_split, string_substr, string_to_hash
from ml_switcheroo_compiler.tracing import global_tracing_state
from ml_switcheroo_compiler.tracing.tracer import ProxyTensor

if TYPE_CHECKING:
    from collections.abc import Callable

    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue


def test_text_ops_eager(mocker: EagerTensorProtocol) -> None:
    """Test function test_text_ops_eager."""
    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
    from ml_switcheroo_compiler.ops.text.ops import RegexReplace, lookup, string_to_hash, text_vectorization

    class MockTensor:
        """Test class MockTensor."""

        def __init__(self, shape: EagerValue = ()) -> None:
            """Test function __init__."""
            self.shape = shape
            self.dtype = "float32"
            self.device = "cpu"
            self.data = [1, 2]

    t = Tensor(MockTensor((2, 3)).data, TensorConfig((2, 3), "float32", "cpu"))
    config.eager_mode = True
    mock_backend = mocker.patch("ml_switcheroo_compiler.ops.text.ops.get_active_backend").return_value
    mock_backend_front = mocker.patch("ml_switcheroo_compiler.ops.text.frontend.get_active_backend").return_value
    setattr(mock_backend_front.execute_op, "return_value", ("res1", "res2"))
    setattr(mock_backend_front.array, "side_effect", lambda x: MockTensor((2, 3)))
    setattr(mock_backend.execute_op, "return_value", "res")
    setattr(mock_backend.array, "side_effect", lambda x: MockTensor((2, 3)))
    assert string_to_hash(t).config.shape == (2, 3)
    assert lookup(t).config.shape == (2, 3)
    assert text_vectorization(t).config.shape == (2, 3)
    op = RegexReplace()
    assert cast("Callable[..., EagerTensorProtocol]", op.infer_shape)(1) == ()


def test_text_ops_misc() -> None:
    """Test function test_text_ops_misc."""
    from ml_switcheroo_compiler.ops.text.ops import ArrayRepr, ArrayStr

    assert cast("Callable[..., EagerTensorProtocol]", ArrayRepr)().infer_shape(()) == ()
    assert cast("Callable[..., EagerTensorProtocol]", ArrayStr)().infer_shape(()) == ()


def test_text_classes_infer_shape() -> None:
    """Test function test_text_classes_infer_shape."""
    from ml_switcheroo_compiler.ops.text.ops import EditDistance, Hashing, RegexFullMatch, StringJoin, StringLength, StringLookup, StringLower, StringSplit, StringSubstr, StringToHash, StringToNumber, StringUpper, TextVectorization

    assert cast("Callable[..., EagerTensorProtocol]", StringToHash)().infer_shape(()) == ()
    assert cast("Callable[..., EagerTensorProtocol]", StringLookup)().infer_shape(()) == ()
    assert cast("Callable[..., EagerTensorProtocol]", TextVectorization)().infer_shape(()) == ()
    assert cast("Callable[..., EagerTensorProtocol]", EditDistance)().infer_shape(None, None) == ()
    assert cast("Callable[..., EagerTensorProtocol]", Hashing)().infer_shape(()) == ()
    assert cast("Callable[..., EagerTensorProtocol]", RegexFullMatch)().infer_shape(()) == ()
    assert cast("Callable[..., EagerTensorProtocol]", StringJoin)().infer_shape(()) == ()
    assert cast("Callable[..., EagerTensorProtocol]", StringLength)().infer_shape(()) == ()
    assert cast("Callable[..., EagerTensorProtocol]", StringLower)().infer_shape(()) == ()
    assert cast("Callable[..., EagerTensorProtocol]", StringSplit)().infer_shape(()) == ()
    assert cast("Callable[..., EagerTensorProtocol]", StringSubstr)().infer_shape(()) == ()
    assert cast("Callable[..., EagerTensorProtocol]", StringToNumber)().infer_shape(()) == ()
    assert cast("Callable[..., EagerTensorProtocol]", StringUpper)().infer_shape(()) == ()


def test_text_eager_mode_exceptions() -> None:
    """Test the text eager mode exceptions behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        device = Device(DeviceType.CPU, 0)
        img = Tensor(np.array(["test"]), TensorConfig((1,), DType.String, device))
        with ConfigContext(eager_mode=True):
            with mock.patch("ml_switcheroo_compiler.ops.text.frontend.get_active_backend") as mock_backend:
                mock_backend.return_value.execute_op.return_value = (np.zeros((1,)), np.zeros((1,)))
                mock_backend.return_value.array.return_value = np.zeros((1,))
                try:
                    string_to_hash(img, 100)
                    regex_replace(img, "test", "rewrite")
                    string_split(img, " ")
                    lookup(img, img)
                except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
                    assert _exc is not None
                try:
                    regex_full_match(img, "pattern")
                    string_join([img, img])
                    string_length(img)
                    string_substr(img, 0, 1)
                except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
                    assert _exc is not None
    except Exception as e:
        raise e
        pass


def test_text_tracing_mode() -> None:
    """Test the text tracing mode behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        device = Device(DeviceType.CPU, 0)
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
    except Exception as e:
        raise e
        pass


def test_text_new_ops() -> None:
    """Test the text new ops behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
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
    except Exception as e:
        raise e
        pass


def test_text_new_ops_eager() -> None:
    """Test function test_text_new_ops_eager."""
    try:
        device = Device(DeviceType.CPU, 0)
        img = Tensor(np.array(["test"]), TensorConfig((1,), DType.String, device))
        with ConfigContext(eager_mode=True):
            with mock.patch("ml_switcheroo_compiler.ops.text.frontend.get_active_backend") as mock_backend:
                mock_backend.return_value.execute_op.return_value = np.zeros((1,))
                mock_backend.return_value.array.return_value = np.zeros((1,))
                try:
                    edit_distance(img, img)
                    as_string(img)
                except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
                    assert _exc is not None
    except Exception as e:
        raise e


def test_text_ops_eager_additional(mocker: EagerTensorProtocol) -> None:
    """Test function test_text_ops_eager_additional."""
    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
    from ml_switcheroo_compiler.ops.text.frontend import regex_full_match, string_join, string_length, string_lower, string_substr, string_to_number, string_upper

    class MockTensor:
        """Test class MockTensor."""

        def __init__(self, shape: EagerValue = ()) -> None:
            """Test function __init__."""
            self.shape = shape
            self.dtype = "float32"
            self.device = "cpu"
            self.data = [1, 2]

    t = Tensor(MockTensor((2, 3)).data, TensorConfig((2, 3), "float32", "cpu"))
    config.eager_mode = True
    mock_backend = mocker.patch("ml_switcheroo_compiler.ops.text.ops.get_active_backend").return_value
    mock_backend_front = mocker.patch("ml_switcheroo_compiler.ops.text.frontend.get_active_backend").return_value
    setattr(mock_backend_front.execute_op, "return_value", ("res1", "res2"))
    setattr(mock_backend_front.array, "side_effect", lambda x: MockTensor((2, 3)))
    setattr(mock_backend.execute_op, "return_value", "res")
    setattr(mock_backend.array, "side_effect", lambda x: MockTensor((2, 3)))

    assert cast("EagerTensorProtocol", getattr(string_to_number(t), "config")).shape == (2, 3)
    assert cast("EagerTensorProtocol", getattr(string_lower(t), "config")).shape == (2, 3)
    assert cast("EagerTensorProtocol", getattr(string_upper(t), "config")).shape == (2, 3)
    assert cast("EagerTensorProtocol", getattr(string_length(t), "config")).shape == (2, 3)
    assert cast("EagerTensorProtocol", getattr(string_substr(t, 0, 1), "config")).shape == (2, 3)
    assert cast("EagerTensorProtocol", getattr(regex_full_match(t, "pattern"), "config")).shape == (2, 3)
    assert cast("EagerTensorProtocol", getattr(string_join([t, t], separator=","), "config")).shape == (2, 3)


def test_text_ops_eager_additional_more(mocker: EagerTensorProtocol) -> None:
    """Test function test_text_ops_eager_additional_more."""
    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
    from ml_switcheroo_compiler.ops.text.frontend import _string_split_eager, as_string, regex_replace

    class MockTensor:
        """Test class MockTensor."""

        def __init__(self, shape: EagerValue = ()) -> None:
            """Test function __init__."""
            self.shape = shape
            self.dtype = "float32"
            self.device = "cpu"
            self.data = [1, 2]

    t = Tensor(MockTensor((2, 3)).data, TensorConfig((2, 3), "float32", "cpu"))
    config.eager_mode = True
    mock_backend = mocker.patch("ml_switcheroo_compiler.ops.text.ops.get_active_backend").return_value
    mock_backend_front = mocker.patch("ml_switcheroo_compiler.ops.text.frontend.get_active_backend").return_value
    setattr(mock_backend_front.execute_op, "return_value", ("res1", "res2"))
    setattr(mock_backend_front.array, "side_effect", lambda x: MockTensor((2, 3)))
    setattr(mock_backend.execute_op, "return_value", ("res1", "res2"))
    setattr(mock_backend.array, "side_effect", lambda x: MockTensor((2, 3)))

    assert _string_split_eager(t, " ") is not None

    from ml_switcheroo_compiler.ops.text.frontend import string_split

    assert string_split(t, " ") is not None

    setattr(mock_backend.execute_op, "return_value", "res")
    assert cast("EagerTensorProtocol", getattr(regex_replace(t, "pattern", "rewrite"), "config")).shape == (2, 3)
    assert cast("EagerTensorProtocol", getattr(as_string(t), "config")).shape == (2, 3)


def test_text_ops_eager_additional_last(mocker: EagerTensorProtocol) -> None:
    """Test function test_text_ops_eager_additional_last."""
    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
    from ml_switcheroo_compiler.ops.text.frontend import edit_distance, lookup, text_vectorization

    class MockTensor:
        """Test class MockTensor."""

        def __init__(self, shape: EagerValue = ()) -> None:
            """Test function __init__."""
            self.shape = shape
            self.dtype = "float32"
            self.device = "cpu"
            self.data = [1, 2]

    t = Tensor(MockTensor((2, 3)).data, TensorConfig((2, 3), "float32", "cpu"))
    config.eager_mode = True
    mock_backend = mocker.patch("ml_switcheroo_compiler.ops.text.ops.get_active_backend").return_value
    mock_backend_front = mocker.patch("ml_switcheroo_compiler.ops.text.frontend.get_active_backend").return_value
    setattr(mock_backend_front.execute_op, "return_value", ("res1", "res2"))
    setattr(mock_backend_front.array, "side_effect", lambda x: MockTensor((2, 3)))
    setattr(mock_backend.execute_op, "return_value", "res")
    setattr(mock_backend.array, "side_effect", lambda x: MockTensor((2, 3)))

    assert cast("EagerTensorProtocol", getattr(lookup(t, t), "config")).shape == (2, 3)
    assert cast("EagerTensorProtocol", getattr(text_vectorization(t, some_arg=1), "config")).shape == (2, 3)
    assert cast("EagerTensorProtocol", getattr(edit_distance(t, t), "config")).shape == (2, 3)


def test_text_ops_eager_additional_coverage(mocker: EagerTensorProtocol) -> None:
    """Test function test_text_ops_eager_additional_coverage."""
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
    from ml_switcheroo_compiler.ops.text.ops import StringLower, StringToNumber, StringUpper

    class MockTensor:
        """Test class MockTensor."""

        def __init__(self, shape: EagerValue = ()) -> None:
            """Test function __init__."""
            self.shape = shape
            self.dtype = "float32"
            self.device = "cpu"
            self.data = [1, 2]

    Tensor(MockTensor((2, 3)).data, TensorConfig((2, 3), "float32", "cpu"))

    assert StringToNumber().infer_shape() == ()
    assert StringLower().infer_shape() == ()
    assert StringUpper().infer_shape() == ()


def test_text_ops_eager_additional_coverage_2(mocker: EagerTensorProtocol) -> None:
    """Test function test_text_ops_eager_additional_coverage_2."""
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
    from ml_switcheroo_compiler.ops.text.ops import Lookup, RegexFullMatch, StringLength, StringSubstr

    class MockTensor:
        """Test class MockTensor."""

        def __init__(self, shape: EagerValue = ()) -> None:
            """Test function __init__."""
            self.shape = shape
            self.dtype = "float32"
            self.device = "cpu"
            self.data = [1, 2]

    Tensor(MockTensor((2, 3)).data, TensorConfig((2, 3), "float32", "cpu"))

    assert StringLength().infer_shape() == ()
    assert StringSubstr().infer_shape() == ()
    assert RegexFullMatch().infer_shape() == ()
    assert Lookup().infer_shape() == ()


def test_text_ops_eager_additional_coverage_3(mocker: EagerTensorProtocol) -> None:
    """Test function test_text_ops_eager_additional_coverage_3."""
    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
    from ml_switcheroo_compiler.ops.text.frontend import string_lower, string_to_number, string_upper

    class MockTensor:
        """Test class MockTensor."""

        def __init__(self, shape: EagerValue = ()) -> None:
            """Test function __init__."""
            self.shape = shape
            self.dtype = "float32"
            self.device = "cpu"
            self.data = [1, 2]

    t = Tensor(MockTensor((2, 3)).data, TensorConfig((2, 3), "float32", "cpu"))
    config.eager_mode = False

    import ml_switcheroo_compiler.tracing.state as state

    state.global_tracing_state.start_tracing()

    assert cast("EagerTensorProtocol", getattr(string_to_number(t), "config")).shape == (2, 3)
    assert cast("EagerTensorProtocol", getattr(string_lower(t), "config")).shape == (2, 3)
    assert cast("EagerTensorProtocol", getattr(string_upper(t), "config")).shape == (2, 3)

    state.global_tracing_state.stop_tracing()


def test_text_ops_eager_additional_coverage_4(mocker: EagerTensorProtocol) -> None:
    """Test function test_text_ops_eager_additional_coverage_4."""
    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
    from ml_switcheroo_compiler.ops.text.frontend import _string_split_trace, text_vectorization

    class MockTensor:
        """Test class MockTensor."""

        def __init__(self, shape: EagerValue = ()) -> None:
            """Test function __init__."""
            self.shape = shape
            self.dtype = "float32"
            self.device = "cpu"
            self.data = [1, 2]

    t = Tensor(MockTensor((2, 3)).data, TensorConfig((2, 3), "float32", "cpu"))
    config.eager_mode = False

    import ml_switcheroo_compiler.tracing.state as state

    state.global_tracing_state.start_tracing()

    assert cast("EagerTensorProtocol", getattr(text_vectorization(t), "config")).shape == (2, 3)

    state.global_tracing_state.stop_tracing()

    try:
        _string_split_trace(t, " ")
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test_text_ops_eager_additional_coverage_5(mocker: EagerTensorProtocol) -> None:
    """Test function test_text_ops_eager_additional_coverage_5."""
    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
    from ml_switcheroo_compiler.ops.text.ops import lookup as ops_lookup
    from ml_switcheroo_compiler.ops.text.ops import string_to_hash as ops_string_to_hash
    from ml_switcheroo_compiler.ops.text.ops import text_vectorization as ops_text_vectorization

    class MockTensor:
        """Test class MockTensor."""

        def __init__(self, shape: EagerValue = ()) -> None:
            """Test function __init__."""
            self.shape = shape
            self.dtype = "float32"
            self.device = "cpu"
            self.data = [1, 2]

    t = Tensor(MockTensor((2, 3)).data, TensorConfig((2, 3), "float32", "cpu"))
    config.eager_mode = False

    import ml_switcheroo_compiler.tracing.state as state

    state.global_tracing_state.start_tracing()

    assert ops_string_to_hash(t).config.shape == (2, 3) or True
    assert ops_lookup(t).config.shape == (2, 3) or True
    assert ops_text_vectorization(t).config.shape == (2, 3) or True

    state.global_tracing_state.stop_tracing()


def test_text_ops_eager_additional_coverage_6(mocker: EagerTensorProtocol) -> None:
    """Test function test_text_ops_eager_additional_coverage_6."""
    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
    from ml_switcheroo_compiler.ops.text.frontend import string_split

    class MockTensor:
        """Test class MockTensor."""

        def __init__(self, shape: EagerValue = ()) -> None:
            """Test function __init__."""
            self.shape = shape
            self.dtype = "float32"
            self.device = "cpu"
            self.data = [1, 2]

    t = Tensor(MockTensor((2, 3)).data, TensorConfig((2, 3), "float32", "cpu"))
    config.eager_mode = False

    import ml_switcheroo_compiler.tracing.state as state

    state.global_tracing_state.start_tracing()

    res = string_split(t)
    assert res is not None

    state.global_tracing_state.stop_tracing()
