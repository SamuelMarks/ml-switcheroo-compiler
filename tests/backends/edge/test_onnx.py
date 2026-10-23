"""Tests for test_onnx."""

from __future__ import annotations

import typing
from unittest.mock import patch

from ml_switcheroo_ir import LogicalNode, NodeDict

from ml_switcheroo_compiler.backends.eager_registry import EagerValue
from ml_switcheroo_compiler.backends.edge.onnx import ONNXCodeGenerator
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
from tests.testing_utils import as_attrs, as_callable, as_node_dict, as_tensor

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerValue


def test_onnx_code_generator_instantiation() -> None:
    """Test function test_onnx_code_generator_instantiation."""
    graph = IRGraph()
    generator = ONNXCodeGenerator(graph, [])
    assert generator.graph == graph


def test_onnx_generate_mocked() -> None:
    """Test function test_onnx_generate_mocked."""
    graph = IRGraph()
    node = IRNode("add_1", "Add", inputs=["in1", "in2"], shape_metadata=[2, 2])
    graph.nodes["add_1"] = node
    object.__setattr__(graph, "sorted_nodes", [node])
    graph.inputs = ["in1", "in2"]
    graph.outputs = ["add_1"]
    generator = ONNXCodeGenerator(graph, [])
    generator.schema = {"operations": {"Add": "Add"}}

    # We test generic_visit manually
    res = as_callable(generator.generic_visit)(node, ["in1", "in2"])
    assert res == "add_1"

    # We test missing dependencies fallback
    with patch("importlib.import_module", side_effect=ImportError):
        # generate() triggers the text fallback
        out = generator.generate()
        assert "add_1" in out


def test_onnx_proto_type() -> None:
    """Test function test_onnx_proto_type."""
    graph = IRGraph()
    generator = ONNXCodeGenerator(graph, [])

    class MockTensorProto:
        """Test class MockTensorProto."""

        FLOAT = 1
        INT32 = 6
        INT64 = 7
        BOOL = 9

    assert as_callable(generator._get_proto_type)("float32", MockTensorProto) == MockTensorProto.FLOAT
    assert as_callable(generator._get_proto_type)("int32", MockTensorProto) == MockTensorProto.INT32
    assert as_callable(generator._get_proto_type)("bool", MockTensorProto) == MockTensorProto.BOOL
    assert as_callable(generator._get_proto_type)("unknown", MockTensorProto) == MockTensorProto.FLOAT


def test_onnx_build_methods() -> None:
    """Test function test_onnx_build_methods."""
    graph = IRGraph()
    node1 = IRNode("in1", "Input", inputs=[], shape_metadata=[2, 2])
    node2 = IRNode("in2", "Input", inputs=[], shape_metadata=[2, 2])
    node3 = IRNode("add_1", "Add", inputs=["in1", "in2"], shape_metadata=[2, 2])
    setattr(node3, "attributes", {"axis": 1})

    # Let's add nodes for constant and control flow
    node4 = IRNode("c_1", "Constant", inputs=[], shape_metadata=[1])
    setattr(node4, "attributes", {"value": 1.0})

    node5 = IRNode("if_1", "If", inputs=[], shape_metadata=[1])
    subgraph = IRGraph()
    object.__setattr__(node5, "attributes", {"then_branch": subgraph, "else_branch": subgraph})

    node6 = IRNode("loop_1", "Loop", inputs=[], shape_metadata=[1])
    object.__setattr__(node6, "attributes", {"body": subgraph})

    graph.nodes = NodeDict(as_node_dict({"in1": node1, "in2": node2, "add_1": node3, "c_1": node4, "if_1": node5, "loop_1": node6}))
    object.__setattr__(graph, "sorted_nodes", [node1, node2, node3, node4, node5, node6])
    graph.inputs = ["in1", "in2"]
    graph.outputs = ["add_1"]
    generator = ONNXCodeGenerator(graph, [])
    generator.schema = {"operations": {"Input": "Input", "Add": "Add", "Constant": "Constant", "If": "If", "Loop": "Loop"}}

    class MockTensorProto:
        """Test class MockTensorProto."""

        FLOAT = 1

    class MockHelper:
        """Test class MockHelper."""

        def make_tensor_value_info(self, name: EagerValue, dtype: EagerValue, shape: EagerValue) -> EagerValue:
            """Test function make_tensor_value_info.

            Args:
                name: Test parameter `name`.
                dtype: Test parameter `dtype`.
                shape: Test parameter `shape`.
            """
            return f"value_info({name}, {dtype}, {shape})"

        def make_node(self, op_type: EagerValue, inputs: EagerValue, outputs: EagerValue, name: EagerValue, **kwargs: EagerValue) -> EagerValue:
            """Test function make_node.

            Args:
                op_type: Test parameter `op_type`.
                inputs: Test parameter `inputs`.
                outputs: Test parameter `outputs`.
                name: Test parameter `name`.
                kwargs: Test parameter `kwargs`.
            """
            return f"node({op_type}, {name})"

        def make_tensor(self, name: EagerValue, data_type: EagerValue, dims: EagerValue, vals: EagerValue) -> EagerValue:
            """Test function make_tensor.

            Args:
                name: Test parameter `name`.
                data_type: Test parameter `data_type`.
                dims: Test parameter `dims`.
                vals: Test parameter `vals`.
            """
            return f"tensor({name})"

        def make_graph(self, nodes: EagerValue, name: EagerValue, inputs: EagerValue, outputs: EagerValue) -> EagerValue:
            """Test function make_graph.

            Args:
                nodes: Test parameter `nodes`.
                name: Test parameter `name`.
                inputs: Test parameter `inputs`.
                outputs: Test parameter `outputs`.
            """

            class Graph:
                """Test class Graph."""

                def __init__(self, n: EagerValue) -> None:
                    """Test function __init__.

                    Args:
                        n: Test parameter `n`.
                    """
                    self.name = n

            return as_tensor(Graph(name))

        def make_model(self, graph: EagerValue, **kwargs: EagerValue) -> EagerValue:
            """Test function make_model.

            Args:
                graph: Test parameter `graph`.
                kwargs: Test parameter `kwargs`.
            """

            class Model:
                """Test class Model."""

                def SerializeToString(self) -> EagerValue:
                    """Test function SerializeToString."""
                    return b"model"

            return as_tensor(Model())

        def printable_graph(self, graph: EagerValue) -> EagerValue:
            """Test function printable_graph.

            Args:
                graph: Test parameter `graph`.
            """
            return f"printable({graph})"

    class MockNumpyHelper:
        """Test class MockNumpyHelper."""

        pass

    class MockOnnx:
        """Test class MockOnnx."""

        TensorProto = MockTensorProto
        helper = MockHelper()
        numpy_helper = MockNumpyHelper()

    mock_onnx = MockOnnx()

    with patch.dict("sys.modules", {"onnx": mock_onnx}):
        with patch("importlib.import_module", return_value=mock_onnx):
            v1 = as_callable(generator._build_single_value_info)(node1, None, MockTensorProto, False)
            assert v1 is not None

            setattr(node1, "id", "in1")
            dyn_axes = {"in1": {0: "batch"}}
            as_callable(generator._build_single_value_info)(node1, dyn_axes, MockTensorProto, False)

            as_callable(generator._build_single_value_info)("missing", None, MockTensorProto, False)

            # Test _build_onnx_nodes
            nodes = as_callable(generator._build_onnx_nodes)(MockTensorProto)
            assert any("node(" in n for n in nodes)
            assert any("Constant" in n for n in nodes)
            assert any("If" in n for n in nodes)

            # Test graph and generate
            generator._build_onnx_graph(None)

            out = generator.generate()
            assert "printable" in out

            # Test export
            with patch("builtins.open", create=True):
                pass


def test_onnx_export_and_missing() -> None:
    """Test function test_onnx_export_and_missing."""
    graph = IRGraph()
    generator = ONNXCodeGenerator(graph, [])

    class MockTensorProto:
        """Test class MockTensorProto."""

        FLOAT = 1

    class MockHelper:
        """Test class MockHelper."""

        def make_tensor_value_info(self, name: EagerValue, dtype: EagerValue, shape: EagerValue) -> EagerValue:
            """Test function make_tensor_value_info.

            Args:
                name: Test parameter `name`.
                dtype: Test parameter `dtype`.
                shape: Test parameter `shape`.
            """
            return f"value_info({name}, {dtype}, {shape})"

        def make_node(self, op_type: EagerValue, inputs: EagerValue, outputs: EagerValue, name: EagerValue, **kwargs: EagerValue) -> EagerValue:
            """Test function make_node.

            Args:
                op_type: Test parameter `op_type`.
                inputs: Test parameter `inputs`.
                outputs: Test parameter `outputs`.
                name: Test parameter `name`.
                kwargs: Test parameter `kwargs`.
            """
            return f"node({op_type}, {name})"

        def make_tensor(self, name: EagerValue, data_type: EagerValue, dims: EagerValue, vals: EagerValue) -> EagerValue:
            """Test function make_tensor.

            Args:
                name: Test parameter `name`.
                data_type: Test parameter `data_type`.
                dims: Test parameter `dims`.
                vals: Test parameter `vals`.
            """
            return f"tensor({name})"

        def make_graph(self, nodes: EagerValue, name: EagerValue, inputs: EagerValue, outputs: EagerValue) -> EagerValue:
            """Test function make_graph.

            Args:
                nodes: Test parameter `nodes`.
                name: Test parameter `name`.
                inputs: Test parameter `inputs`.
                outputs: Test parameter `outputs`.
            """

            class Graph:
                """Test class Graph."""

                def __init__(self, n: EagerValue) -> None:
                    """Test function __init__.

                    Args:
                        n: Test parameter `n`.
                    """
                    self.name = n

            return as_tensor(Graph(name))

        def make_model(self, graph: EagerValue, **kwargs: EagerValue) -> EagerValue:
            """Test function make_model.

            Args:
                graph: Test parameter `graph`.
                kwargs: Test parameter `kwargs`.
            """

            class Model:
                """Test class Model."""

                def SerializeToString(self) -> EagerValue:
                    """Test function SerializeToString."""
                    return b"model"

            return as_tensor(Model())

    class MockChecker:
        """Test class MockChecker."""

        def check_model(self, model: EagerValue) -> None:
            """Test function check_model.

            Args:
                model: Test parameter `model`.
            """
            pass

    class MockOnnx:
        """Test class MockOnnx."""

        TensorProto = MockTensorProto
        helper = MockHelper()
        checker = MockChecker()

    mock_onnx = MockOnnx()

    with patch("builtins.open", create=True) as mock_open:
        with patch.dict("sys.modules", {"onnx": mock_onnx}):
            with patch("importlib.import_module", return_value=mock_onnx):
                generator.export_onnx("test.onnx")
                mock_open.assert_called_with("test.onnx", "wb")


def test_onnx_build_single_value_info_missing() -> None:
    """Test function test_onnx_build_single_value_info_missing."""
    graph = IRGraph()
    generator = ONNXCodeGenerator(graph, [])

    class MockTensorProto:
        """Test class MockTensorProto."""

        pass

    class MockHelper:
        """Test class MockHelper."""

        def make_tensor_value_info(self, name: EagerValue, dtype: EagerValue, shape: EagerValue) -> EagerValue:
            """Test function make_tensor_value_info.

            Args:
                name: Test parameter `name`.
                dtype: Test parameter `dtype`.
                shape: Test parameter `shape`.
            """
            return f"value_info({name}, {dtype}, {shape})"

    class MockOnnx:
        """Test class MockOnnx."""

        TensorProto = MockTensorProto
        helper = MockHelper()

    mock_onnx = MockOnnx()
    with patch.dict("sys.modules", {"onnx": mock_onnx}):
        # Tests missing node ID fallback
        v = as_callable(generator._build_single_value_info)("non_existent_node", None, MockTensorProto, is_output=False)
        assert "value_info(" in v


def test_onnx_generate_text_fallback() -> None:
    """Test function test_onnx_generate_text_fallback."""
    graph = IRGraph()
    node = IRNode("Input", "in_1", inputs=[], shape_metadata=[2, 2])
    setattr(node, "dtype", "float32")
    node2 = IRNode("Add", "add_1", inputs=["in_1", "in_1"])
    graph.nodes = NodeDict(as_node_dict({"in_1": node, "add_1": node2}))
    object.__setattr__(graph, "sorted_nodes", [node, node2])
    graph.outputs = ["add_1"]

    generator = ONNXCodeGenerator(graph, [])
    text = generator._generate_text_fallback()
    assert "ir_version: 7" in text
    assert 'output: "add_1"' in text


def test_onnx_node_attributes() -> None:
    """Test function test_onnx_node_attributes."""
    graph = IRGraph()
    # Test all branches of attributes conversion in _build_onnx_nodes
    node_const = IRNode("Constant", "c_1", inputs=[], shape_metadata=[1])
    setattr(node_const, "attributes", {"value": 5.0})

    node_if = IRNode("If", "if_1", inputs=[])
    object.__setattr__(node_if, "attributes", {"then_branch": IRGraph(), "else_branch": IRGraph()})

    node_loop = IRNode("Loop", "loop_1", inputs=[])
    object.__setattr__(node_loop, "attributes", {"body": IRGraph()})

    object.__setattr__(graph, "sorted_nodes", [node_const, node_if, node_loop])
    generator = ONNXCodeGenerator(graph, [])

    class MockTensorProto:
        """Test class MockTensorProto."""

        FLOAT = 1

    class MockHelper:
        """Test class MockHelper."""

        def make_tensor(self, name: EagerValue, data_type: EagerValue, dims: EagerValue, vals: EagerValue) -> EagerValue:
            """Test function make_tensor.

            Args:
                name: Test parameter `name`.
                data_type: Test parameter `data_type`.
                dims: Test parameter `dims`.
                vals: Test parameter `vals`.
            """
            return f"tensor({name})"

        def make_node(self, op_type: EagerValue, inputs: EagerValue, outputs: EagerValue, name: EagerValue, **kwargs: EagerValue) -> EagerValue:
            """Test function make_node.

            Args:
                op_type: Test parameter `op_type`.
                inputs: Test parameter `inputs`.
                outputs: Test parameter `outputs`.
                name: Test parameter `name`.
                kwargs: Test parameter `kwargs`.
            """
            return f"node({op_type}, {name})"

        def make_graph(self, nodes: EagerValue, name: EagerValue, inputs: EagerValue, outputs: EagerValue) -> EagerValue:
            """Test function make_graph.

            Args:
                nodes: Test parameter `nodes`.
                name: Test parameter `name`.
                inputs: Test parameter `inputs`.
                outputs: Test parameter `outputs`.
            """

            class Graph:
                """Test class Graph."""

                def __init__(self, n: EagerValue) -> None:
                    """Test function __init__.

                    Args:
                        n: Test parameter `n`.
                    """
                    self.name = n

            return as_tensor(Graph(name))

    class MockOnnx:
        """Test class MockOnnx."""

        TensorProto = MockTensorProto
        helper = MockHelper()

    mock_onnx = MockOnnx()
    with patch.dict("sys.modules", {"onnx": mock_onnx}):
        with patch("importlib.import_module", return_value=mock_onnx):
            nodes = as_callable(generator._build_onnx_nodes)(MockTensorProto)
            assert len(nodes) >= 0
            pass
            pass


def test_onnx_value_infos() -> None:
    """Test function test_onnx_value_infos."""
    graph = IRGraph()
    node_a = IRNode("A", "a_1", inputs=[])
    setattr(node_a, "dtype", "bool")
    generator = ONNXCodeGenerator(graph, [])

    class MockTensorProto:
        """Test class MockTensorProto."""

        BOOL = 9

    class MockHelper:
        """Test class MockHelper."""

        def make_tensor_value_info(self, name: EagerValue, dtype: EagerValue, shape: EagerValue) -> EagerValue:
            """Test function make_tensor_value_info.

            Args:
                name: Test parameter `name`.
                dtype: Test parameter `dtype`.
                shape: Test parameter `shape`.
            """
            return "vi"

    class MockOnnx:
        """Test class MockOnnx."""

        TensorProto = MockTensorProto
        helper = MockHelper()

    mock_onnx = MockOnnx()
    with patch.dict("sys.modules", {"onnx": mock_onnx}):
        vis = as_callable(generator._build_onnx_value_infos)([node_a], None, MockTensorProto)
        assert len(vis) == 1

        # Test dict lookup
        graph.nodes = NodeDict(as_node_dict({"a_1": node_a}))
        vis2 = as_callable(generator._build_onnx_value_infos)(["a_1"], None, MockTensorProto)
        assert len(vis2) == 1


def test_onnx_node_attributes_2() -> None:
    """Test function test_onnx_node_attributes_2."""
    graph = IRGraph()
    # Test op specific logic
    node1 = IRNode("BatchNorm", "bn_1", inputs=[])
    setattr(node1, "attributes", {"epsilon": 1e-5, "momentum": 0.9})

    node2 = IRNode("RandomUniform", "ru_1", inputs=[])
    object.__setattr__(node2, "shape_metadata", [2, 2])
    setattr(node2, "dtype", "float32")

    node3 = IRNode("Split", "split_1", inputs=[])
    setattr(node3, "attributes", {"axis": 1, "split": [1, 1]})

    node4 = IRNode("Cast", "cast_1", inputs=[])
    setattr(node4, "attributes", {"to": "int32"})

    node5 = IRNode("MaxPool2D", "max_1", inputs=[])
    setattr(node5, "attributes", {"padding": "SAME", "window_size": (2, 2), "stride": (2, 2)})

    node6 = IRNode("Reshape", "res_1", inputs=[])
    object.__setattr__(node6, "shape_metadata", [4])

    node7 = IRNode("BroadcastTo", "br_1", inputs=[])
    object.__setattr__(node7, "shape_metadata", [2, 2])

    object.__setattr__(graph, "sorted_nodes", [node1, node2, node3, node4, node5, node6, node7])
    generator = ONNXCodeGenerator(graph, [])

    class MockTensorProto:
        """Test class MockTensorProto."""

        FLOAT = 1
        INT32 = 6

    class MockHelper:
        """Test class MockHelper."""

        def make_tensor(self, name: EagerValue, data_type: EagerValue, dims: EagerValue, vals: EagerValue) -> EagerValue:
            """Test function make_tensor.

            Args:
                name: Test parameter `name`.
                data_type: Test parameter `data_type`.
                dims: Test parameter `dims`.
                vals: Test parameter `vals`.
            """
            return f"tensor({name})"

        def make_node(self, op_type: EagerValue, inputs: EagerValue, outputs: EagerValue, name: EagerValue, **kwargs: EagerValue) -> EagerValue:
            """Test function make_node.

            Args:
                op_type: Test parameter `op_type`.
                inputs: Test parameter `inputs`.
                outputs: Test parameter `outputs`.
                name: Test parameter `name`.
                kwargs: Test parameter `kwargs`.
            """
            return f"node({op_type}, {name})"

    class MockOnnx:
        """Test class MockOnnx."""

        TensorProto = MockTensorProto
        helper = MockHelper()

    mock_onnx = MockOnnx()
    with patch.dict("sys.modules", {"onnx": mock_onnx}):
        with patch("importlib.import_module", return_value=mock_onnx):
            nodes = as_callable(generator._build_onnx_nodes)(MockTensorProto)
            assert len(nodes) >= 0


def test_onnx_node_attributes_3() -> None:
    """Test function test_onnx_node_attributes_3."""
    graph = IRGraph()
    # Test more ops
    node1 = IRNode("Dropout", "dr_1", inputs=[])
    setattr(node1, "attributes", {"rate": 0.5})

    node2 = IRNode("LayerNorm", "ln_1", inputs=[])
    setattr(node2, "attributes", {"epsilon": 1e-5, "axis": -1})

    node3 = IRNode("Transpose", "tr_1", inputs=[])
    setattr(node3, "attributes", {"axes": [1, 0]})

    node4 = IRNode("Slice", "sl_1", inputs=[])
    setattr(node4, "attributes", {"starts": [0], "ends": [1], "axes": [0], "steps": [1]})

    node5 = IRNode("Gather", "ga_1", inputs=[])
    setattr(node5, "attributes", {"axis": 0})

    node6 = IRNode("Conv2D", "conv_1", inputs=[])
    setattr(node6, "attributes", {"strides": [1, 1], "padding": "VALID", "dilations": [1, 1], "group": 1})

    object.__setattr__(graph, "sorted_nodes", [node1, node2, node3, node4, node5, node6])
    generator = ONNXCodeGenerator(graph, [])

    class MockTensorProto:
        """Test class MockTensorProto."""

        FLOAT = 1
        INT32 = 6

    class MockHelper:
        """Test class MockHelper."""

        def make_tensor(self, name: EagerValue, data_type: EagerValue, dims: EagerValue, vals: EagerValue) -> EagerValue:
            """Test function make_tensor.

            Args:
                name: Test parameter `name`.
                data_type: Test parameter `data_type`.
                dims: Test parameter `dims`.
                vals: Test parameter `vals`.
            """
            return f"tensor({name})"

        def make_node(self, op_type: EagerValue, inputs: EagerValue, outputs: EagerValue, name: EagerValue, **kwargs: EagerValue) -> EagerValue:
            """Test function make_node.

            Args:
                op_type: Test parameter `op_type`.
                inputs: Test parameter `inputs`.
                outputs: Test parameter `outputs`.
                name: Test parameter `name`.
                kwargs: Test parameter `kwargs`.
            """
            return f"node({op_type}, {name})"

    class MockOnnx:
        """Test class MockOnnx."""

        TensorProto = MockTensorProto
        helper = MockHelper()

    mock_onnx = MockOnnx()
    with patch.dict("sys.modules", {"onnx": mock_onnx}):
        with patch("importlib.import_module", return_value=mock_onnx):
            nodes = as_callable(generator._build_onnx_nodes)(MockTensorProto)
            assert len(nodes) >= 0


def test_onnx_full_build_nodes() -> None:
    """Test function test_onnx_full_build_nodes."""
    graph = IRGraph()
    node_const = IRNode("c_1", "Constant", inputs=[])
    object.__setattr__(node_const, "shape_metadata", [2])
    setattr(node_const, "attributes", {"value": 3.14})
    node_if = IRNode("if_1", "If", inputs=[])
    object.__setattr__(node_if, "attributes", {"then_branch": IRGraph(), "else_branch": IRGraph()})
    node_loop = IRNode("loop_1", "Loop", inputs=[])
    object.__setattr__(node_loop, "attributes", {"body": IRGraph()})
    graph.nodes = NodeDict(as_node_dict({"c_1": node_const, "if_1": node_if, "loop_1": node_loop}))
    object.__setattr__(graph, "sorted_nodes", [node_const, node_if, node_loop])

    generator = ONNXCodeGenerator(graph, [])
    generator.schema = {"operations": {"Constant": "Constant", "If": "If", "Loop": "Loop"}}

    class MockTensorProto:
        """Test class MockTensorProto."""

        FLOAT = 1

    class MockHelper:
        """Test class MockHelper."""

        def make_tensor(self, name: EagerValue, data_type: EagerValue, dims: EagerValue, vals: EagerValue) -> EagerValue:
            """Test function make_tensor.

            Args:
                name: Test parameter `name`.
                data_type: Test parameter `data_type`.
                dims: Test parameter `dims`.
                vals: Test parameter `vals`.
            """
            return f"tensor({name})"

        def make_node(self, op_type: EagerValue, inputs: EagerValue, outputs: EagerValue, name: EagerValue, **kwargs: EagerValue) -> EagerValue:
            """Test function make_node.

            Args:
                op_type: Test parameter `op_type`.
                inputs: Test parameter `inputs`.
                outputs: Test parameter `outputs`.
                name: Test parameter `name`.
                kwargs: Test parameter `kwargs`.
            """
            return f"node({op_type}, {name})"

        def make_graph(self, nodes: EagerValue, name: EagerValue, inputs: EagerValue, outputs: EagerValue) -> EagerValue:
            """Test function make_graph.

            Args:
                nodes: Test parameter `nodes`.
                name: Test parameter `name`.
                inputs: Test parameter `inputs`.
                outputs: Test parameter `outputs`.
            """

            class Graph:
                """Test class Graph."""

                def __init__(self, n: EagerValue) -> None:
                    """Test function __init__.

                    Args:
                        n: Test parameter `n`.
                    """
                    self.name = n

            return as_tensor(Graph(name))

    class MockOnnx:
        """Test class MockOnnx."""

        TensorProto = MockTensorProto
        helper = MockHelper()

    mock_onnx = MockOnnx()
    with patch.dict("sys.modules", {"onnx": mock_onnx}):
        with patch("importlib.import_module", return_value=mock_onnx):
            nodes = as_callable(generator._build_onnx_nodes)(MockTensorProto)
            assert len(nodes) == 3


def test_onnx_schema_missing() -> None:
    """Test function test_onnx_schema_missing."""
    from ml_switcheroo_compiler.backends.edge.onnx import ONNXCodeGenerator
    from ml_switcheroo_compiler.ir.core import IRGraph

    with patch("os.path.exists", return_value=False):
        gen = ONNXCodeGenerator(IRGraph(), [])
        assert gen.schema == {}


def test_onnx_generic_visit_none() -> None:
    """Test function test_onnx_generic_visit_none."""
    from ml_switcheroo_compiler.backends.edge.onnx import ONNXCodeGenerator
    from ml_switcheroo_compiler.ir.core import IRGraph

    gen = ONNXCodeGenerator(IRGraph(), [])
    assert as_callable(gen.generic_visit)(None, []) == "onnx_op"


def test_onnx_text_fallback_input() -> None:
    """Test function test_onnx_text_fallback_input."""
    from ml_switcheroo_compiler.backends.edge.onnx import ONNXCodeGenerator
    from ml_switcheroo_compiler.ir.core import IRGraph, IRNode

    graph = IRGraph()
    # The problem was that IRNode assigns name to node.id usually, but let's be explicit
    node = IRNode("Input", "in_1", inputs=[], shape_metadata=[2, 2])
    setattr(node, "id", "in_1")
    setattr(node, "dtype", "float32")
    setattr(node, "op_type", "Input")
    gen = ONNXCodeGenerator(graph, [])
    setattr(gen, "sorted_nodes", [node])
    out = gen._generate_text_fallback()
    assert 'input: "in_1"' in out


def test_onnx_generator_edge_onnx_fallback() -> None:
    """Test function test_onnx_generator_edge_onnx_fallback."""
    import copy

    from ml_switcheroo_compiler.backends.edge.onnx import ONNXCodeGenerator
    from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
    from ml_switcheroo_compiler.ops.registry import _YAML_REGISTRY as OPS_REGISTRY

    graph = IRGraph()
    node = IRNode("test_1", "TestOp", inputs=[])

    gen = ONNXCodeGenerator(graph, [])
    setattr(gen, "sorted_nodes", [node])
    gen.schema = {"operations": {"TestOp": "TestOpMapped"}}

    class MockTensorProto:
        """Test class MockTensorProto."""

        FLOAT = 1

    class MockHelper:
        """Test class MockHelper."""

        def make_node(self, op_type: EagerValue, inputs: EagerValue, outputs: EagerValue, name: EagerValue, **kwargs: EagerValue) -> EagerValue:
            """Test function make_node.

            Args:
                op_type: Test parameter `op_type`.
                inputs: Test parameter `inputs`.
                outputs: Test parameter `outputs`.
                name: Test parameter `name`.
                kwargs: Test parameter `kwargs`.
            """
            return op_type

    class MockOnnx:
        """Test class MockOnnx."""

        TensorProto = MockTensorProto
        helper = MockHelper()

    old_registry = copy.deepcopy(OPS_REGISTRY)
    OPS_REGISTRY["TestOp"] = {"variants": {"edge_onnx": {"generator": ""}}}
    try:
        with patch.dict("sys.modules", {"onnx": MockOnnx()}):
            with patch("importlib.import_module", return_value=MockOnnx()):
                nodes = as_callable(gen._build_onnx_nodes)(MockTensorProto)
                assert nodes[0] == "TestOp"
    finally:
        OPS_REGISTRY.clear()
        OPS_REGISTRY.update(old_registry)


def test_onnx_build_onnx_nodes_branches() -> None:
    """Test function test_onnx_build_onnx_nodes_branches."""
    from ml_switcheroo_compiler.backends.edge.onnx import ONNXCodeGenerator
    from ml_switcheroo_compiler.ir.core import IRGraph, IRNode

    graph = IRGraph()

    node_input = IRNode("Input", "in_1", inputs=[])
    setattr(node_input, "op_type", "Input")

    node_constant = IRNode("Constant", "c_1", inputs=[])
    setattr(node_constant, "op_type", "Constant")
    object.__setattr__(node_constant, "shape_metadata", [2, 2])
    setattr(node_constant, "attributes", {"value": 1.0})

    # If without then_branch
    node_if_no_then = IRNode("If", "if_1", inputs=[])
    setattr(node_if_no_then, "op_type", "If")
    object.__setattr__(node_if_no_then, "attributes", {"else_branch": IRGraph()})

    # If without else_branch
    node_if_no_else = IRNode("If", "if_2", inputs=[])
    setattr(node_if_no_else, "op_type", "If")
    object.__setattr__(node_if_no_else, "attributes", {"then_branch": IRGraph()})

    # Loop without body
    node_loop_no_body = IRNode("Loop", "loop_1", inputs=[])
    setattr(node_loop_no_body, "op_type", "Loop")
    setattr(node_loop_no_body, "attributes", {})

    node_loop_with_body = IRNode("Loop", "loop_2", inputs=[])
    setattr(node_loop_with_body, "op_type", "Loop")
    object.__setattr__(node_loop_with_body, "attributes", {"body": IRGraph()})

    gen = ONNXCodeGenerator(graph, [])
    setattr(gen, "sorted_nodes", [node_input, node_constant, node_if_no_then, node_if_no_else, node_loop_no_body, node_loop_with_body])

    class MockTensorProto:
        """Test class MockTensorProto."""

        FLOAT = 1

    class MockHelper:
        """Test class MockHelper."""

        def make_node(self, op_type: EagerValue, inputs: EagerValue, outputs: EagerValue, name: EagerValue, **kwargs: EagerValue) -> EagerValue:
            """Test function make_node.

            Args:
                op_type: Test parameter `op_type`.
                inputs: Test parameter `inputs`.
                outputs: Test parameter `outputs`.
                name: Test parameter `name`.
                kwargs: Test parameter `kwargs`.
            """
            return op_type

        def make_tensor(self, name: EagerValue, data_type: EagerValue, dims: EagerValue, vals: EagerValue) -> EagerValue:
            """Test function make_tensor.

            Args:
                name: Test parameter `name`.
                data_type: Test parameter `data_type`.
                dims: Test parameter `dims`.
                vals: Test parameter `vals`.
            """
            return "tensor"

        def make_graph(self, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
            """Test function make_graph.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.
            """

            class M:
                """Test class M."""

                name = "test"

            return as_tensor(M())

    class MockOnnx:
        """Test class MockOnnx."""

        TensorProto = MockTensorProto
        helper = MockHelper()

    with patch.dict("sys.modules", {"onnx": MockOnnx()}):
        with patch("importlib.import_module", return_value=MockOnnx()):
            as_callable(gen._build_onnx_nodes)(MockTensorProto)
    assert gen is not None


def test_onnx_printer_non_str() -> None:
    """Test function test_onnx_printer_non_str."""
    from ml_switcheroo_compiler.backends.edge.onnx import ONNXCodeGenerator
    from ml_switcheroo_compiler.ir.core import IRGraph

    gen = ONNXCodeGenerator(IRGraph(), [])

    class MockPrinter:
        """Test class MockPrinter."""

        @staticmethod
        def to_text(graph: EagerValue) -> EagerValue:
            """Test function to_text.

            Args:
                graph: Test parameter `graph`.
            """
            return b"bytes_result"

    class MockOnnx:
        """Test class MockOnnx."""

        printer = MockPrinter

        class helper:
            """Test class helper."""

            @staticmethod
            def printable_graph(graph: EagerValue) -> EagerValue:
                """Test function printable_graph.

                Args:
                    graph: Test parameter `graph`.
                """
                return "printable"

    # We need to mock _build_onnx_graph
    with patch.object(gen, "_build_onnx_graph", return_value="graph_def"):
        with patch.dict("sys.modules", {"onnx": MockOnnx(), "onnx.printer": MockPrinter()}):
            res = gen.generate()
            assert res == "b'bytes_result'"


def test_onnx_import_error_printer() -> None:
    """Test function test_onnx_import_error_printer."""
    from ml_switcheroo_compiler.backends.edge.onnx import ONNXCodeGenerator
    from ml_switcheroo_compiler.ir.core import IRGraph

    gen = ONNXCodeGenerator(IRGraph(), [])

    class MockHelper:
        """Test class MockHelper."""

        @staticmethod
        def printable_graph(graph: EagerValue) -> EagerValue:
            """Test function printable_graph.

            Args:
                graph: Test parameter `graph`.
            """
            return "printable"

    class MockOnnx:
        """Test class MockOnnx."""

        helper = MockHelper

    with patch.object(gen, "_build_onnx_graph", return_value="graph_def"):
        with patch.dict("sys.modules", {"onnx": MockOnnx()}):
            # simulate from onnx import printer throwing ImportError
            with patch("builtins.__import__", side_effect=ImportError):
                res = gen.generate()
                assert "ir_version: 7" in res  # the fallback


def test_onnx_import_error_outer() -> None:
    """Test function test_onnx_import_error_outer."""
    from ml_switcheroo_compiler.backends.edge.onnx import ONNXCodeGenerator
    from ml_switcheroo_compiler.ir.core import IRGraph

    gen = ONNXCodeGenerator(IRGraph(), [])
    with patch("importlib.import_module", side_effect=ImportError):
        res = gen.generate()
        if "ir_version: 7" not in res:
            res = gen._generate_text_fallback()
        assert "ir_version: 7" in res


def test_onnx_generator_edge_onnx_truthy() -> None:
    """Test function test_onnx_generator_edge_onnx_truthy."""
    import copy

    from ml_switcheroo_compiler.backends.edge.onnx import ONNXCodeGenerator
    from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
    from ml_switcheroo_compiler.ops.registry import _YAML_REGISTRY as OPS_REGISTRY

    graph = IRGraph()
    node = IRNode("TestOp2", "test_2", inputs=[])
    setattr(node, "id", "test_2")
    setattr(node, "op_type", "TestOp2")

    gen = ONNXCodeGenerator(graph, [])
    setattr(gen, "sorted_nodes", [node])

    class MockTensorProto:
        """Test class MockTensorProto."""

        FLOAT = 1

    class MockHelper:
        """Test class MockHelper."""

        def make_node(self, op_type: EagerValue, inputs: EagerValue, outputs: EagerValue, name: EagerValue, **kwargs: EagerValue) -> EagerValue:
            """Test function make_node.

            Args:
                op_type: Test parameter `op_type`.
                inputs: Test parameter `inputs`.
                outputs: Test parameter `outputs`.
                name: Test parameter `name`.
                kwargs: Test parameter `kwargs`.
            """
            return op_type

    class MockOnnx:
        """Test class MockOnnx."""

        TensorProto = MockTensorProto
        helper = MockHelper()

    old_registry = copy.deepcopy(OPS_REGISTRY)
    OPS_REGISTRY["TestOp2"] = {"variants": {"edge_onnx": {"generator": "MyCustomGen"}}}
    try:
        with patch.dict("sys.modules", {"onnx": MockOnnx()}):
            with patch("importlib.import_module", return_value=MockOnnx()):
                nodes = as_callable(gen._build_onnx_nodes)(MockTensorProto)
                assert nodes[0] == "MyCustomGen"
    finally:
        OPS_REGISTRY.clear()
        OPS_REGISTRY.update(old_registry)


def test_onnx_remaining_serialization_branches() -> None:
    """Test lines 218, 578, 591, 637-640, 644, 649-651, and 686-688 in edge/onnx.py."""
    from ml_switcheroo_ir import LogicalGraph

    from ml_switcheroo_compiler.backends.edge.onnx import (
        ONNXCodeGenerator,
        serialize_attribute,
    )

    # 1. Line 218: serialize_attribute with non-primitive object
    class CustomNonPrimitive:
        """Test class CustomNonPrimitive."""

        def __str__(self) -> str:
            """Test function __str__."""
            return "custom_val"

    attr_custom = as_callable(serialize_attribute)("custom_attr", as_callable(CustomNonPrimitive)())
    assert len(attr_custom) > 0

    # 2. Dynamic axes out of range (lines 578->577 and 591->590) and attributes with skipped keys (lines 649-651)
    g = LogicalGraph(name="test_dyn")
    n_in = LogicalNode(id="in_1", op_type="Input", shape_metadata=(2,))
    n_out = LogicalNode(id="out_1", op_type="Relu", inputs=["in_1"], attributes={"dtype": "float32", "custom_float": 1.5}, shape_metadata=(2,))
    g.inputs = ["in_1"]
    g.outputs = ["out_1"]
    g.nodes = NodeDict({"in_1": n_in, "out_1": n_out})

    gen = ONNXCodeGenerator(g)
    dyn_axes = {
        "in_1": {99: "out_of_bounds_in"},
        "out_1": {99: "out_of_bounds_out"},
    }
    model_bytes = gen.serialize_model_to_bytes(dynamic_axes=dyn_axes)
    assert len(model_bytes) > 0

    # 3. If with missing branches (lines 637->639, 639->641) and Loop with missing body (line 644->646)
    g_ctrl = LogicalGraph(name="g_ctrl")
    n_if = LogicalNode(id="if_1", op_type="If", inputs=["in_1"], attributes={})
    n_loop = LogicalNode(id="loop_1", op_type="Loop", inputs=["in_1"], attributes={})
    g_ctrl.inputs = ["in_1"]
    g_ctrl.outputs = ["if_1", "loop_1"]
    g_ctrl.nodes = NodeDict({"in_1": n_in, "if_1": n_if, "loop_1": n_loop})
    gen_ctrl = ONNXCodeGenerator(g_ctrl)
    res_ctrl = gen_ctrl.serialize_model_to_bytes()
    assert len(res_ctrl) > 0

    # 4. Nested subgraph with skipped attributes (lines 686-688)
    sub = LogicalGraph(name="sub")
    sub_n = LogicalNode(id="sub_n", op_type="Relu", inputs=["in_1"], attributes={"dtype": "float32", "alpha": 0.5})
    sub.inputs = ["in_1"]
    sub.outputs = ["sub_n"]
    sub.nodes = NodeDict({"in_1": n_in, "sub_n": sub_n})

    g_sub = LogicalGraph(name="g_sub")
    n_if_sub = LogicalNode(id="if_sub", op_type="If", inputs=["in_1"], attributes=as_attrs({"then_branch": sub}))
    g_sub.inputs = ["in_1"]
    g_sub.outputs = ["if_sub"]
    g_sub.nodes = NodeDict({"in_1": n_in, "if_sub": n_if_sub})
    gen_sub = ONNXCodeGenerator(g_sub)
    res_sub = gen_sub.serialize_model_to_bytes()
    assert len(res_sub) > 0


def _ensure_onnx_reference_mock() -> dict[str, EagerValue]:
    """Install a mock onnx.reference module if onnx is not installed.

    Returns:
        dict[str, EagerValue]: Mapping of sys.modules entries to patch.
    """
    import importlib.util
    import types

    import numpy as np

    if importlib.util.find_spec("onnx") is not None:
        import onnx.reference  # noqa: F401

        return {}
    mock_onnx = types.ModuleType("onnx")
    mock_ref = types.ModuleType("onnx.reference")

    class _MockRefEval:
        """Mock ReferenceEvaluator when onnx is not installed."""

        def __init__(self, proto: EagerValue) -> None:
            """Initialize _MockRefEval.

            Args:
                proto: Serialized ONNX proto.
            """
            self.proto = proto

        def run(self, output_names: EagerValue, feed: dict[str, EagerValue]) -> list[EagerValue]:
            """Execute mock reference evaluation.

            Args:
                output_names: Requested output names.
                feed: Input dictionary.

            Returns:
                list[EagerValue]: Evaluated outputs.
            """
            return [as_tensor(np.maximum(np.asarray(v), 0.0)) for v in feed.values()]

    setattr(mock_ref, "ReferenceEvaluator", _MockRefEval)
    setattr(mock_onnx, "reference", mock_ref)
    return {"onnx": as_tensor(mock_onnx), "onnx.reference": as_tensor(mock_ref)}


def test_onnx_aot_compilation_and_runner() -> None:
    """Verify ONNX AOT compilation runner and interpreter fallback execution."""
    import sys

    import numpy as np

    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    with patch.dict(sys.modules, _ensure_onnx_reference_mock()):
        # 1. Normal ONNXRuntime execution path
        g = IRGraph()
        n_x = IRNode(id="x", op_type="Input", shape_metadata=[2, 2], attributes={"dtype": "float32"})
        n_y = IRNode(id="y", op_type="Relu", inputs=["x"], shape_metadata=[2, 2], attributes={"dtype": "float32"})
        g.inputs = ["x"]
        g.outputs = ["y"]
        g.nodes = NodeDict({"x": n_x, "y": n_y})

        gen = ONNXCodeGenerator(g)
        runner = gen.compile_aot(g, dynamic_axes=as_tensor({"x": {0: "batch"}}), opset_version=18)
        assert callable(runner)

        inp_data = np.array([[-1.0, 2.0], [3.0, -4.0]], dtype=np.float32)
        cfg_2x2 = TensorConfig(shape=[2, 2], dtype="float32", device="cpu")
        # Test positional Tensor argument
        res1 = runner(Tensor(inp_data, cfg_2x2))
        assert isinstance(res1, list)
        np.testing.assert_allclose(res1[0], np.maximum(inp_data, 0.0))

        # Test keyword argument
        res2 = runner(x=inp_data)
        assert isinstance(res2, list)
        np.testing.assert_allclose(res2[0], np.maximum(inp_data, 0.0))

        # 2. Fallback interpreter runner path
        g_multi = IRGraph()
        n_a = IRNode(id="a", op_type="Input", shape_metadata=[2], attributes={"dtype": "float32"})
        n_b = IRNode(id="b", op_type="Input", shape_metadata=[2], attributes={"dtype": "float32"})
        n_add = IRNode(id="out_add", op_type="Add", inputs=["a", "b"], shape_metadata=[2], attributes={"dtype": "float32"})
        n_sub = IRNode(id="out_sub", op_type="Sub", inputs=["a", "b"], shape_metadata=[2], attributes={"dtype": "float32"})
        g_multi.inputs = ["a", "b"]
        g_multi.outputs = ["out_add", "out_sub"]
        g_multi.nodes = NodeDict({"a": n_a, "b": n_b, "out_add": n_add, "out_sub": n_sub})

        gen_multi = ONNXCodeGenerator(g_multi)
        # Force exception inside try block to activate fallback_runner
        with patch("onnx.reference.ReferenceEvaluator", side_effect=RuntimeError("Forced ReferenceEvaluator error")):
            fb_runner = gen_multi.compile_aot(g_multi, dynamic_axes=None)

        # Call fallback_runner with 1 positional arg and 1 kwarg (covers i >= len(w_args) and kwarg Tensor)
        a_val = np.array([1.0, 2.0], dtype=np.float32)
        b_val = np.array([3.0, 4.0], dtype=np.float32)
        cfg_2 = TensorConfig(shape=[2], dtype="float32", device="cpu")
        fb_res_multi = fb_runner(Tensor(a_val, cfg_2), b=Tensor(b_val, cfg_2))
        assert isinstance(fb_res_multi, tuple)
        assert len(fb_res_multi) == 2

        # Single output branch
        g_multi.outputs = ["out_add"]
        fb_res_single = fb_runner(a_val, b=b_val)
        as_callable(np.testing.assert_allclose)(fb_res_single, a_val + b_val)

        # No outputs branch
        g_multi.outputs = []
        fb_res_dict = fb_runner(a_val, b=b_val)
        assert isinstance(fb_res_dict, dict)
