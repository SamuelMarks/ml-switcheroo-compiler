"""Test torch fx ingest."""

import pytest

from ml_switcheroo_compiler.transforms.foreign import ingest_torch_fx


class MockNode:
    """Mock Node."""

    def __init__(self, op, name, target, args=(), kwargs=None):
        """Init."""
        self.op = op
        self.name = name
        self.target = target
        self.args = args
        self.kwargs = kwargs or {}


class MockGraph:
    """Mock graph."""

    def __init__(self, nodes):
        """Init."""
        self.nodes = nodes


class MockGraphModule:
    """Mock graph module."""

    def __init__(self, nodes=None, modules=None, no_graph=False, empty_nodes=False):
        """Init."""
        if not no_graph:
            if empty_nodes:
                self.graph = MockGraph([])
                del self.graph.nodes
            else:
                self.graph = MockGraph(nodes or [])
        self.modules = modules or {}

    def named_modules(self):
        """Named modules."""
        return self.modules.items()


def test_ingest_torch_fx_empty():
    """Test empty handling."""
    with pytest.raises(ValueError):
        ingest_torch_fx(None)

    gm = MockGraphModule(no_graph=True)
    g = ingest_torch_fx(gm)
    assert len(g.nodes) == 0

    gm2 = MockGraphModule(empty_nodes=True)
    g2 = ingest_torch_fx(gm2)
    assert len(g2.nodes) == 0


def test_fx_call_method():
    """Test."""
    arg_node = MockNode("placeholder", "x", "x")
    view_node = MockNode("call_method", "view_1", "view", args=(arg_node, -1, 10))
    reshape_node = MockNode("call_method", "reshape_1", "reshape", args=(arg_node, 10, -1))
    permute_node = MockNode("call_method", "permute_1", "permute", args=(arg_node, 1, 0))
    transpose_node = MockNode("call_method", "transpose_1", "transpose", args=(arg_node, 0, 1))
    contiguous_node = MockNode("call_method", "contiguous_1", "contiguous", args=(arg_node,))
    relu_node = MockNode("call_method", "relu_1", "relu", args=(arg_node,))
    flatten_node = MockNode("call_method", "flatten_1", "flatten", args=(arg_node, 1))
    unsqueeze_node = MockNode("call_method", "unsqueeze_1", "unsqueeze", args=(arg_node, 0))
    squeeze_node = MockNode("call_method", "squeeze_1", "squeeze", args=(arg_node, 0))
    unknown_node = MockNode("call_method", "unknown_1", "unknown_target", args=(arg_node, "other"))
    kwarg_node = MockNode("call_method", "kwarg_1", "view", args=(arg_node,), kwargs={"dim": 1, "tup": (1, 2), "obj": object()})

    gm = MockGraphModule([arg_node, view_node, reshape_node, permute_node, transpose_node, contiguous_node, relu_node, flatten_node, unsqueeze_node, squeeze_node, unknown_node, kwarg_node])

    g = ingest_torch_fx(gm)

    assert g.nodes["view_1"].op_type == "Reshape"
    assert g.nodes["reshape_1"].op_type == "Reshape"
    assert g.nodes["permute_1"].op_type == "Transpose"
    assert g.nodes["transpose_1"].op_type == "Transpose"
    assert g.nodes["contiguous_1"].op_type == "Identity"
    assert g.nodes["relu_1"].op_type == "Relu"
    assert g.nodes["flatten_1"].op_type == "Flatten"
    assert g.nodes["unsqueeze_1"].op_type == "Unsqueeze"
    assert g.nodes["squeeze_1"].op_type == "Squeeze"
    assert g.nodes["unknown_1"].op_type == "Unknown"
    assert g.nodes["unknown_1"].inputs[1] == "other"
    assert g.nodes["kwarg_1"].attributes["dim"] == 1
    assert g.nodes["kwarg_1"].attributes["tup"] == [1, 2]
    assert "obj" not in g.nodes["kwarg_1"].attributes


class Conv2d:
    """Mock."""


class BatchNorm2d:
    """Mock."""


class LayerNorm:
    """Mock."""


class GroupNorm:
    """Mock."""


class Linear:
    """Mock."""


class Dropout:
    """Mock."""


class ReLU:
    """Mock."""


class MaxPool2d:
    """Mock."""


class AvgPool2d:
    """Mock."""


class UnknownModule:
    """Mock."""


def test_fx_call_module():
    """Test."""
    arg_node = MockNode("placeholder", "x", "x")
    conv_node = MockNode("call_module", "conv_1", "my_conv", args=(arg_node,))
    batchnorm_node = MockNode("call_module", "bn_1", "my_bn", args=(arg_node,))
    layernorm_node = MockNode("call_module", "ln_1", "my_ln", args=(arg_node,))
    groupnorm_node = MockNode("call_module", "gn_1", "my_gn", args=(arg_node,))
    linear_node = MockNode("call_module", "lin_1", "my_lin", args=(arg_node,))
    dropout_node = MockNode("call_module", "drop_1", "my_drop", args=(arg_node,))
    relu_node = MockNode("call_module", "relu_1", "my_relu", args=(arg_node,))
    maxpool_node = MockNode("call_module", "mp_1", "my_mp", args=(arg_node,))
    avgpool_node = MockNode("call_module", "ap_1", "my_ap", args=(arg_node,))
    unknown_node = MockNode("call_module", "unk_1", "my_unknown", args=(arg_node, "other"))
    kwarg_node = MockNode("call_module", "kwarg_1", "my_conv", args=(arg_node,), kwargs={"dim": 1, "tup": (1, 2), "obj": object()})
    no_target_node = MockNode("call_module", "notgt_1", "missing_tgt", args=(arg_node,))

    modules = {
        "my_conv": Conv2d(),
        "my_bn": BatchNorm2d(),
        "my_ln": LayerNorm(),
        "my_gn": GroupNorm(),
        "my_lin": Linear(),
        "my_drop": Dropout(),
        "my_relu": ReLU(),
        "my_mp": MaxPool2d(),
        "my_ap": AvgPool2d(),
        "my_unknown": UnknownModule(),
    }

    gm = MockGraphModule(
        [arg_node, conv_node, batchnorm_node, layernorm_node, groupnorm_node, linear_node, dropout_node, relu_node, maxpool_node, avgpool_node, unknown_node, kwarg_node, no_target_node],
        modules,
    )

    # We test with modules and without named_modules
    class NoNamedModulesMock:
        """Mock."""

        def __init__(self, nodes, modules):
            """Init."""
            self.graph = MockGraph(nodes or [])
            self.modules = modules or {}

    # Conv2d would map to 'Conv' if named_modules were present, but since it's not, it falls back to 'CallModule' if modules dictionary isn't correctly processed.
    # Ah! The code in `ingest_torch_fx` does:
    # modules = dict(fx_graph_module.named_modules()) if hasattr(fx_graph_module, "named_modules") else {}
    # So if there is no named_modules, `modules` is empty dict.
    # Thus, "my_conv" is not in modules, so op_type remains "CallModule"
    g_no_mod = ingest_torch_fx(NoNamedModulesMock([arg_node, conv_node], modules))
    assert "conv_1" in g_no_mod.nodes
    assert g_no_mod.nodes["conv_1"].op_type == "CallModule"

    g = ingest_torch_fx(gm)

    assert g.nodes["conv_1"].op_type == "Conv"
    assert g.nodes["bn_1"].op_type == "BatchNormalization"
    assert g.nodes["ln_1"].op_type == "LayerNormalization"
    assert g.nodes["gn_1"].op_type == "GroupNormalization"
    assert g.nodes["lin_1"].op_type == "Gemm"
    assert g.nodes["drop_1"].op_type == "Dropout"
    assert g.nodes["relu_1"].op_type == "Relu"
    assert g.nodes["mp_1"].op_type == "MaxPool"
    assert g.nodes["ap_1"].op_type == "AveragePool"
    assert g.nodes["unk_1"].op_type == "CallModule"
    assert g.nodes["unk_1"].inputs[1] == "other"
    assert g.nodes["kwarg_1"].attributes["dim"] == 1
    assert g.nodes["kwarg_1"].attributes["tup"] == [1, 2]
    assert "obj" not in g.nodes["kwarg_1"].attributes
    assert g.nodes["notgt_1"].op_type == "CallModule"


def test_fx_get_attr():
    """Test."""
    attr_node = MockNode("get_attr", "w", "weight")

    gm = MockGraphModule([attr_node])
    g = ingest_torch_fx(gm)

    assert g.nodes["w"].op_type == "Constant"
    assert g.nodes["w"].attributes["target"] == "weight"


def test_fx_call_function():
    """Test."""

    class FunctionMock:
        """Mock func."""

        def __init__(self, name):
            """Init."""
            self.__name__ = name

    arg_node = MockNode("placeholder", "x", "x")

    func_names = [
        "add",
        "sub",
        "mul",
        "div",
        "neg",
        "pow",
        "abs",
        "exp",
        "log",
        "sqrt",
        "sin",
        "cos",
        "tan",
        "tanh",
        "sigmoid",
        "matmul",
        "bmm",
        "linear",
        "mm",
        "conv1d",
        "conv2d",
        "conv3d",
        "max_pool2d",
        "avg_pool2d",
        "batch_norm",
        "layer_norm",
        "group_norm",
        "sum",
        "mean",
        "max",
        "min",
        "argmax",
        "argmin",
        "slice",
        "gather",
        "scatter",
        "cat",
        "stack",
        "relu",
    ]

    nodes = [arg_node]
    for name in func_names:
        nodes.append(MockNode("call_function", f"{name}_1", FunctionMock(name), args=(arg_node,)))

    unknown_node = MockNode("call_function", "unk_1", FunctionMock("unknown"), args=(arg_node, "other"))
    kwarg_node = MockNode("call_function", "kw_1", FunctionMock("add"), args=(arg_node,), kwargs={"dim": 1, "tup": (1, 2), "obj": object()})
    nodes.extend([unknown_node, kwarg_node])

    gm = MockGraphModule(nodes)
    g = ingest_torch_fx(gm)

    assert g.nodes["add_1"].op_type == "Add"
    assert g.nodes["conv2d_1"].op_type == "Conv"
    assert g.nodes["mean_1"].op_type == "ReduceMean"
    assert g.nodes["unk_1"].op_type == "Unknown"
    assert g.nodes["unk_1"].inputs[1] == "other"
    assert g.nodes["kw_1"].attributes["dim"] == 1
    assert g.nodes["kw_1"].attributes["tup"] == [1, 2]
    assert "obj" not in g.nodes["kw_1"].attributes


class MissingGraphNode:
    """Mock empty."""

    pass


def test_fx_output():
    """Test."""
    arg_node = MockNode("placeholder", "x", "x")
    out_node = MockNode("output", "out", "out", args=([arg_node],))
    out_node2 = MockNode("output", "out2", "out2", args=(arg_node,))

    gm1 = MockGraphModule([arg_node, out_node])
    g1 = ingest_torch_fx(gm1)
    assert g1.outputs == ["x"]

    gm2 = MockGraphModule([arg_node, out_node2])
    g2 = ingest_torch_fx(gm2)
    assert g2.outputs == ["x"]


def test_fx_unknown_op():
    """Test unknown node op type."""
    arg_node = MockNode("placeholder", "x", "x")
    unk_node = MockNode("some_unknown_op", "unk_1", "unk")
    gm = MockGraphModule([arg_node, unk_node])
    g = ingest_torch_fx(gm)
    assert "unk_1" not in g.nodes


def test_fx_call_function_fallback():
    """Test fallback op_type matching logic."""

    class FunctionMock:
        """Mock func."""

        def __init__(self, name):
            """Init."""
            self.__name__ = name

        def __str__(self):
            """Str."""
            return self.__name__

    arg_node = MockNode("placeholder", "x", "x")

    # Matches target_name.startswith(key + "_") -> "add_something" -> "Add"
    fallback_node1 = MockNode("call_function", "f1", FunctionMock("add_something"), args=(arg_node,))
    # Matches target_name.endswith("." + key) -> "my_module.add" -> "Add"
    fallback_node2 = MockNode("call_function", "f2", FunctionMock("my_module.add"), args=(arg_node,))

    gm = MockGraphModule([arg_node, fallback_node1, fallback_node2])
    g = ingest_torch_fx(gm)

    assert g.nodes["f1"].op_type == "Add"
    assert g.nodes["f2"].op_type == "Add"


def test_fx_call_module_dead_branch():
    """Test dead branch in call_module for missing module dict."""
    arg_node = MockNode("placeholder", "x", "x")
    no_target_node = MockNode("call_module", "notgt_1", "missing_tgt", args=(arg_node,))

    gm = MockGraphModule([arg_node, no_target_node], modules=None)
    g = ingest_torch_fx(gm)

    assert g.nodes["notgt_1"].op_type == "CallModule"
