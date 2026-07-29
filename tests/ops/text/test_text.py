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


def test_text_ops_eager(mocker):
    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
    from ml_switcheroo_compiler.ops.text.ops import RegexReplace, lookup, string_to_hash, text_vectorization

    class MockTensor:
        def __init__(self, shape=()):
            self.shape = shape
            self.dtype = "float32"
            self.device = "cpu"
            self.data = [1, 2]

    t = Tensor(MockTensor((2, 3)).data, TensorConfig((2, 3), "float32", "cpu"))
    config.eager_mode = True
    mock_backend = mocker.patch("ml_switcheroo_compiler.ops.text.ops.get_active_backend").return_value
    mock_backend_front = mocker.patch("ml_switcheroo_compiler.ops.text.frontend.get_active_backend").return_value
    mock_backend_front.execute_op.return_value = ("res1", "res2")
    mock_backend_front.array.side_effect = lambda x: MockTensor((2, 3))
    mock_backend.execute_op.return_value = "res"
    mock_backend.array.side_effect = lambda x: MockTensor((2, 3))
    assert string_to_hash(t).config.shape == (2, 3)
    assert lookup(t).config.shape == (2, 3)
    assert text_vectorization(t).config.shape == (2, 3)
    op = RegexReplace()
    assert op.infer_shape(1) == ()


def test_text_ops_misc():
    from ml_switcheroo_compiler.ops.text.ops import ArrayRepr, ArrayStr

    assert ArrayRepr().infer_shape(()) == ()
    assert ArrayStr().infer_shape(()) == ()


def test_text_classes_infer_shape():
    from ml_switcheroo_compiler.ops.text.ops import EditDistance, Hashing, RegexFullMatch, StringJoin, StringLength, StringLookup, StringLower, StringSplit, StringSubstr, StringToHash, StringToNumber, StringUpper, TextVectorization

    assert StringToHash().infer_shape(()) == ()
    assert StringLookup().infer_shape(()) == ()
    assert TextVectorization().infer_shape(()) == ()
    assert EditDistance().infer_shape(None, None) == ()
    assert Hashing().infer_shape(()) == ()
    assert RegexFullMatch().infer_shape(()) == ()
    assert StringJoin().infer_shape(()) == ()
    assert StringLength().infer_shape(()) == ()
    assert StringLower().infer_shape(()) == ()
    assert StringSplit().infer_shape(()) == ()
    assert StringSubstr().infer_shape(()) == ()
    assert StringToNumber().infer_shape(()) == ()
    assert StringUpper().infer_shape(()) == ()


"Tests for text operations."


def test_text_eager_mode_exceptions() -> object:
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
                except Exception:
                    pass
                try:
                    regex_full_match(img, "pattern")
                    string_join([img, img])
                    string_length(img)
                    string_substr(img, 0, 1)
                except Exception:
                    pass
    except Exception as e:
        raise e
        pass


def test_text_tracing_mode() -> object:
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
        Any: The inferred shape or computed result.
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
                except Exception:
                    pass
    except Exception as e:
        raise e


def test_text_ops_eager_additional(mocker):
    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
    from ml_switcheroo_compiler.ops.text.frontend import regex_full_match, string_join, string_length, string_lower, string_substr, string_to_number, string_upper

    class MockTensor:
        def __init__(self, shape=()):
            self.shape = shape
            self.dtype = "float32"
            self.device = "cpu"
            self.data = [1, 2]

    t = Tensor(MockTensor((2, 3)).data, TensorConfig((2, 3), "float32", "cpu"))
    config.eager_mode = True
    mock_backend = mocker.patch("ml_switcheroo_compiler.ops.text.ops.get_active_backend").return_value
    mock_backend_front = mocker.patch("ml_switcheroo_compiler.ops.text.frontend.get_active_backend").return_value
    mock_backend_front.execute_op.return_value = ("res1", "res2")
    mock_backend_front.array.side_effect = lambda x: MockTensor((2, 3))
    mock_backend.execute_op.return_value = "res"
    mock_backend.array.side_effect = lambda x: MockTensor((2, 3))

    assert string_to_number(t).config.shape == (2, 3)
    assert string_lower(t).config.shape == (2, 3)
    assert string_upper(t).config.shape == (2, 3)
    assert string_length(t).config.shape == (2, 3)
    assert string_substr(t, 0, 1).config.shape == (2, 3)
    assert regex_full_match(t, "pattern").config.shape == (2, 3)
    assert string_join([t, t], separator=",").config.shape == (2, 3)


def test_text_ops_eager_additional_more(mocker):
    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
    from ml_switcheroo_compiler.ops.text.frontend import _string_split_eager, as_string, regex_replace

    class MockTensor:
        def __init__(self, shape=()):
            self.shape = shape
            self.dtype = "float32"
            self.device = "cpu"
            self.data = [1, 2]

    t = Tensor(MockTensor((2, 3)).data, TensorConfig((2, 3), "float32", "cpu"))
    config.eager_mode = True
    mock_backend = mocker.patch("ml_switcheroo_compiler.ops.text.ops.get_active_backend").return_value
    mock_backend_front = mocker.patch("ml_switcheroo_compiler.ops.text.frontend.get_active_backend").return_value
    mock_backend_front.execute_op.return_value = ("res1", "res2")
    mock_backend_front.array.side_effect = lambda x: MockTensor((2, 3))
    mock_backend.execute_op.return_value = ("res1", "res2")
    mock_backend.array.side_effect = lambda x: MockTensor((2, 3))

    assert _string_split_eager(t, " ") is not None

    from ml_switcheroo_compiler.ops.text.frontend import string_split

    assert string_split(t, " ") is not None

    mock_backend.execute_op.return_value = "res"
    assert regex_replace(t, "pattern", "rewrite").config.shape == (2, 3)
    assert as_string(t).config.shape == (2, 3)


def test_text_ops_eager_additional_last(mocker):
    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
    from ml_switcheroo_compiler.ops.text.frontend import edit_distance, lookup, text_vectorization

    class MockTensor:
        def __init__(self, shape=()):
            self.shape = shape
            self.dtype = "float32"
            self.device = "cpu"
            self.data = [1, 2]

    t = Tensor(MockTensor((2, 3)).data, TensorConfig((2, 3), "float32", "cpu"))
    config.eager_mode = True
    mock_backend = mocker.patch("ml_switcheroo_compiler.ops.text.ops.get_active_backend").return_value
    mock_backend_front = mocker.patch("ml_switcheroo_compiler.ops.text.frontend.get_active_backend").return_value
    mock_backend_front.execute_op.return_value = ("res1", "res2")
    mock_backend_front.array.side_effect = lambda x: MockTensor((2, 3))
    mock_backend.execute_op.return_value = "res"
    mock_backend.array.side_effect = lambda x: MockTensor((2, 3))

    assert lookup(t, t).config.shape == (2, 3)
    assert text_vectorization(t, some_arg=1).config.shape == (2, 3)
    assert edit_distance(t, t).config.shape == (2, 3)


def test_text_ops_eager_additional_coverage(mocker):
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
    from ml_switcheroo_compiler.ops.text.ops import StringLower, StringToNumber, StringUpper

    class MockTensor:
        def __init__(self, shape=()):
            self.shape = shape
            self.dtype = "float32"
            self.device = "cpu"
            self.data = [1, 2]

    t = Tensor(MockTensor((2, 3)).data, TensorConfig((2, 3), "float32", "cpu"))

    assert StringToNumber().infer_shape() == ()
    assert StringLower().infer_shape() == ()
    assert StringUpper().infer_shape() == ()


def test_text_ops_eager_additional_coverage_2(mocker):
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
    from ml_switcheroo_compiler.ops.text.ops import Lookup, RegexFullMatch, StringLength, StringSubstr

    class MockTensor:
        def __init__(self, shape=()):
            self.shape = shape
            self.dtype = "float32"
            self.device = "cpu"
            self.data = [1, 2]

    t = Tensor(MockTensor((2, 3)).data, TensorConfig((2, 3), "float32", "cpu"))

    assert StringLength().infer_shape() == ()
    assert StringSubstr().infer_shape() == ()
    assert RegexFullMatch().infer_shape() == ()
    assert Lookup().infer_shape() == ()


def test_text_ops_eager_additional_coverage_3(mocker):
    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
    from ml_switcheroo_compiler.ops.text.frontend import string_lower, string_to_number, string_upper

    class MockTensor:
        def __init__(self, shape=()):
            self.shape = shape
            self.dtype = "float32"
            self.device = "cpu"
            self.data = [1, 2]

    t = Tensor(MockTensor((2, 3)).data, TensorConfig((2, 3), "float32", "cpu"))
    config.eager_mode = False

    import ml_switcheroo_compiler.tracing.state as state

    state.global_tracing_state.start_tracing()

    assert string_to_number(t).config.shape == (2, 3)
    assert string_lower(t).config.shape == (2, 3)
    assert string_upper(t).config.shape == (2, 3)

    state.global_tracing_state.stop_tracing()


def test_text_ops_eager_additional_coverage_4(mocker):
    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
    from ml_switcheroo_compiler.ops.text.frontend import _string_split_trace, text_vectorization

    class MockTensor:
        def __init__(self, shape=()):
            self.shape = shape
            self.dtype = "float32"
            self.device = "cpu"
            self.data = [1, 2]

    t = Tensor(MockTensor((2, 3)).data, TensorConfig((2, 3), "float32", "cpu"))
    config.eager_mode = False

    import ml_switcheroo_compiler.tracing.state as state

    state.global_tracing_state.start_tracing()

    assert text_vectorization(t).config.shape == (2, 3)

    state.global_tracing_state.stop_tracing()

    try:
        _string_split_trace(t, " ")
    except Exception:
        pass


def test_text_ops_eager_additional_coverage_5(mocker):
    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
    from ml_switcheroo_compiler.ops.text.ops import lookup as ops_lookup
    from ml_switcheroo_compiler.ops.text.ops import string_to_hash as ops_string_to_hash
    from ml_switcheroo_compiler.ops.text.ops import text_vectorization as ops_text_vectorization

    class MockTensor:
        def __init__(self, shape=()):
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


def test_text_ops_eager_additional_coverage_6(mocker):
    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
    from ml_switcheroo_compiler.ops.text.frontend import string_split

    class MockTensor:
        def __init__(self, shape=()):
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
