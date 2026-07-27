"""Test module."""

from ml_switcheroo_compiler.backends.backend_utils import format_shape_metadata, resolve_input_vars


class DummyNode:
    def __init__(self, inputs):
        self.inputs = inputs


class DummyDim:
    def __init__(self, id_val):
        self.id = id_val


def test_resolve_input_vars():
    node = DummyNode(inputs=["a", "b", "c"])
    var_names = {"a": "var_a", "b": "var_b"}
    res = resolve_input_vars(node, var_names)
    assert res == ["var_a", "var_b", "c"]


def test_format_shape_metadata():
    node = DummyNode(inputs=[])

    # 1. No shape metadata
    assert format_shape_metadata(node, {}) is None

    # 2. empty shape metadata
    node.shape_metadata = ()
    assert format_shape_metadata(node, {}) is None

    # 3. various dimension types
    node.shape_metadata = (DummyDim("dim1"), "str_dim", 42)
    var_names = {"dim1": "resolved_dim1"}
    res = format_shape_metadata(node, var_names)
    assert res == "(resolved_dim1, 'str_dim', 42)"

    # 4. single element tuple formatting
    node.shape_metadata = (42,)
    res = format_shape_metadata(node, {})
    assert res == "(42,)"

    # 5. dummy dim missing from var_names
    node.shape_metadata = (DummyDim("missing"),)
    res = format_shape_metadata(node, {})
    assert res == "(missing,)"
