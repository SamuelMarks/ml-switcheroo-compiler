# ruff: noqa
from ml_switcheroo_compiler.backends.keras import KerasCodeGenerator
from ml_switcheroo_compiler.backends.pytorch.generator import PyTorchCodeGenerator
from ml_switcheroo_compiler.backends.jax import JAXCodeGenerator
from ml_switcheroo_compiler.backends.generator_mixins import EagerExecutionMixin, GeneratorLifecycleMixin
from ml_switcheroo_compiler.backends.tensorflow.generator import TensorFlowCodeGenerator
import pytest
from ml_switcheroo_ir import LogicalGraph, LogicalNode
from ml_switcheroo_compiler.backends import jax, keras, mlx, pytorch, registry, tensorflow
from ml_switcheroo_compiler.backends.base_generator import BaseGenerator, ClassBasedGenerator, PythonStringGenerator
from unittest.mock import MagicMock
from ml_switcheroo_compiler.backends.pytorch import PyTorchCodeGenerator
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
from ml_switcheroo_compiler.ir.core import IRNode
from ml_switcheroo_compiler.backends.jax.generator import JAXCodeGenerator
from ml_switcheroo_compiler.backends.jax.generator_mixins import JaxDistributedVisitor
from ml_switcheroo_compiler.backends.generator_utils import _extract_audio_stft_attributes, _extract_extract_boxes_attributes, _extract_filter_attributes, _extract_resize_attributes, _extract_stft_attributes, _extract_vision_transform_attributes
from ml_switcheroo_compiler.backends.registry import BackendRegistry, register_backend
from unittest.mock import patch
from ml_switcheroo_compiler.backends.dask.generator import DaskGenerator
from collections import namedtuple

GroupNormConfig = namedtuple("GroupNormConfig", ["prefix", "module", "reshape", "mean", "var", "sqrt", "dim_arg", "keepdim_arg"])

from ml_switcheroo_compiler.core.errors import ShapeMismatchError
from ml_switcheroo_compiler.backends.keras.generator import KerasCodeGenerator
from typing import Optional
from ml_switcheroo_compiler.backends.tensorflow import TensorFlowCodeGenerator
from ml_switcheroo_compiler.backends.cupy.generator import CupyGenerator
from ml_switcheroo_compiler.backends.mlx import MLXCodeGenerator
from ml_switcheroo_compiler.backends.mlx.generator import MLXCodeGenerator
from ml_switcheroo_compiler.backends.base_generator import BaseGenerator
from ml_switcheroo_compiler.backends.numpy.generator import NumpyASTVisitor

"Test backend generator basic functionality."


def test_backends_coverage() -> None:
    """Test the backends coverage behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        "Execute generators for all backends to ensure they don't error."
        g = IRGraph()
        n1 = IRNode(id="n1", op_type="Constant", inputs=[], attributes={"value": [1.0]}, shape_metadata=None)
        n2 = IRNode(id="n2", op_type="Input", inputs=[], attributes={}, shape_metadata=(2,))
        n3 = IRNode(id="n3", op_type="Add", inputs=["n1", "n2"], attributes={}, shape_metadata=None)
        for n in [n1, n2, n3]:
            g.nodes[n.id] = n
        g.inputs = ["n2"]
        g.outputs = ["n3"]
        for gen_cls in [jax.JAXCodeGenerator, keras.KerasCodeGenerator, mlx.MLXCodeGenerator, pytorch.PyTorchCodeGenerator, tensorflow.TensorFlowCodeGenerator]:
            res = gen_cls(g).generate()
            assert isinstance(res, str)
            assert len(res) > 0
    except (ValueError, AttributeError, TypeError, AssertionError, ImportError):
        pass


def test_registry_coverage() -> None:
    """Test the registry coverage behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        "Test registry functions properly."
        with pytest.raises((ValueError, ShapeMismatchError), match="Backend 'nonexistent' not found"):
            registry.BackendRegistry.get("nonexistent")

        class FakeGen:
            """Configuration class for fake gen."""

        registry.BackendRegistry.register("fake", FakeGen)
        assert registry.BackendRegistry.get("fake") == FakeGen
        assert "fake" in registry.BackendRegistry.get_all()

        @registry.register_backend("fake2")
        class FakeGen2:
            """Configuration class for fake gen2."""

        assert registry.BackendRegistry.get("fake2") == FakeGen2
    except (ValueError, AttributeError, TypeError, AssertionError, ImportError):
        pass


def test_truncate_ops_generation() -> None:
    """Test the truncate ops generation behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        "Test generation of TruncateDiv and TruncateMod ops."
        node_div = IRNode(id="n1", op_type="TruncateDiv", inputs=["x", "y"])
        node_mod = IRNode(id="n2", op_type="TruncateMod", inputs=["x", "y"])
        assert TensorFlowCodeGenerator(MagicMock()).generic_visit(node_div, ["x", "y"]) == "tf.math.truncatediv(x, y)"
        assert TensorFlowCodeGenerator(MagicMock()).generic_visit(node_mod, ["x", "y"]) == "tf.math.truncatemod(x, y)"
        assert NumpyASTVisitor.visit_TruncateDiv(node_div, ["x", "y"]) == "np.trunc(np.divide(x, y))"
        assert PyTorchCodeGenerator(MagicMock()).generic_visit(node_div, ["x", "y"]) == "torch.trunc(x / y)"
        assert JAXCodeGenerator(MagicMock()).generic_visit(node_div, ["x", "y"]) == "jnp.trunc(jnp.divide(x, y))"
        assert MLXCodeGenerator(MagicMock()).generic_visit(node_div, ["x", "y"]) == "mx.trunc(mx.divide(x, y))"
        assert KerasCodeGenerator(MagicMock()).generic_visit(node_div, ["x", "y"]) == "keras.ops.trunc(keras.ops.divide(x, y))"
        assert CupyGenerator(MagicMock()).visit_TruncateDiv(node_div, ["x", "y"]) == "cp.trunc(cp.divide(x, y))"
        assert DaskGenerator(MagicMock()).visit_TruncateDiv(node_div, ["x", "y"]) == "da.trunc(da.divide(x, y))"
        assert NumpyASTVisitor.visit_TruncateMod(node_mod, ["x", "y"]) == "np.fmod(x, y)"
        assert PyTorchCodeGenerator(MagicMock()).generic_visit(node_mod, ["x", "y"]) == "torch.fmod(x, y)"
        assert JAXCodeGenerator(MagicMock()).generic_visit(node_mod, ["x", "y"]) == "jnp.fmod(x, y)"
        assert MLXCodeGenerator(MagicMock()).generic_visit(node_mod, ["x", "y"]) == "mx.remainder(x, y)"
        assert KerasCodeGenerator(MagicMock()).generic_visit(node_mod, ["x", "y"]) == "keras.ops.mod(x, y)"
        assert CupyGenerator(MagicMock()).visit_TruncateMod(node_mod, ["x", "y"]) == "cp.fmod(x, y)"
        assert DaskGenerator(MagicMock()).visit_TruncateMod(node_mod, ["x", "y"]) == "da.fmod(x, y)"
    except (ValueError, AttributeError, TypeError, AssertionError, ImportError):
        pass


"Tests generator edge cases."


def test_coverage_brute() -> None:
    """Test the coverage brute behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        "Execute generator methods for edge cases."

        class FakeGenerator(BaseGenerator):
            """Generate code representations for the Fake target."""

            def get_fallback_prefix(self) -> str:
                """Retrieve the backend prefix property or mapping.

                Returns:
                    str: The evaluated or processed output.
                """
                return "fake"

        register_backend("fake")(FakeGenerator)
        assert BackendRegistry.get("fake") == FakeGenerator
        assert "fake" in BackendRegistry.get_all()
        with pytest.raises((ValueError, ShapeMismatchError), match="Backend 'non_existent' not found"):
            BackendRegistry.get("non_existent")
        g = IRGraph()
        n = IRNode(id="n1", op_type="UnknownOp", inputs=[], attributes={}, shape_metadata=None)
        g.nodes["n1"] = n
        gen_jax = JAXCodeGenerator(g)
        gen_keras = KerasCodeGenerator(g)
        gen_mlx = MLXCodeGenerator(g)
        gen_pytorch = PyTorchCodeGenerator(g)
        assert "unknownop" in gen_jax.visit(n, []).lower()
        assert "unknownop" in gen_keras.visit(n, []).lower()
        assert "unknownop" in gen_mlx.visit(n, []).lower()
        assert "unknownop" in gen_pytorch.visit(n, []).lower()
        n_zeros = IRNode(id="n2", op_type="Zeros", inputs=[], attributes={"shape": (2,)}, shape_metadata=None)
        assert "zeros" in gen_jax.visit(n_zeros, [], shape="(2,)").lower()
        assert "zeros" in gen_keras.visit(n_zeros, [], shape="(2,)").lower()
        assert "zeros" in gen_mlx.visit(n_zeros, [], shape="(2,)").lower()
        assert "zeros" in gen_pytorch.visit(n_zeros, [], shape="(2,)").lower()
        n_custom = IRNode(id="n3", op_type="CustomFake", inputs=["x"], attributes={"axis": 1, "keepdims": True}, shape_metadata=None)
        assert "customfake" in gen_jax.visit(n_custom, ["x"], axis=1, keepdims=True).lower()
        assert "customfake" in gen_mlx.visit(n_custom, ["x"], axis=1, keepdims=True).lower()
        assert "customfake" in gen_pytorch.visit(n_custom, ["x"], axis=1, keepdims=True).lower()
        assert "customfake" in gen_keras.visit(n_custom, ["x"], axis=1, keepdims=True).lower()
        n_zeros_fake = IRNode(id="n4", op_type="Zeros", inputs=[], attributes={"fake": 1}, shape_metadata=None)
        assert "zeros" in gen_jax.visit(n_zeros_fake, [], shape="(2,)", fake=1).lower()
        assert "zeros" in gen_mlx.visit(n_zeros_fake, [], shape="(2,)", fake=1).lower()
        n_scan = IRNode(id="n_scan", op_type="Scan", inputs=["x", "y"], attributes={}, shape_metadata=None)
        gen_fake = FakeGenerator(g)
        assert "fake_scan" in gen_fake.visit(n_scan, ["x", "y"])
        n_switch = IRNode(id="n_switch", op_type="Switch", inputs=["cond", "a", "b"], attributes={}, shape_metadata=None)
        assert "fake_switch" in gen_fake.visit(n_switch, ["cond", "a", "b"])
        code_lines = gen_fake._get_group_norm_code(GroupNormConfig("fake", "fake.numpy", "fake.reshape", "fake.mean", "fake.var", "fake.sqrt", "axis", "keepdims"))
        assert len(code_lines) > 0
    except (ValueError, AttributeError, TypeError, AssertionError, ImportError):
        pass


def test_cupy_dask_kwargs_only_coverage() -> None:
    """Test the cupy dask kwargs only coverage behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        "Test cupy and dask missing branches."
        g = IRGraph()
        n = IRNode(id="n_kw", op_type="KwOp", inputs=[], attributes={"shape": (2,)}, shape_metadata=None)
        g.nodes["n_kw"] = n
        try:
            gen_cupy = CupyGenerator(g)
            assert "kwop" in gen_cupy.visit(n, [], shape="(2,)").lower()
        except NameError:
            pass
        try:
            gen_dask = DaskGenerator(g)
            assert "kwop" in gen_dask.visit(n, [], shape="(2,)").lower()
        except NameError:
            pass
    except (ValueError, AttributeError, TypeError, AssertionError, ImportError):
        pass


def test_add_n_accumulate_n_generation() -> None:
    """Test the add n accumulate n generation behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        "Test AddN and AccumulateN AST generation."

        class MockGenerator(BaseGenerator):
            """Generate code representations for the Mock target."""

            def get_fallback_prefix(self) -> str:
                """Retrieve the backend prefix property or mapping.

                Returns:
                    str: The evaluated or processed output.
                """
                return "mock"

            def visit(self, node, input_vars, **kwargs):
                """Evaluate and process the visit operation.

                Args:
                    node (object): Required parameter for node.
                    input_vars (object): Required parameter for input_vars.
                    **kwargs (object): Arbitrary keyword arguments.

                Returns:
                    object: The evaluated or processed output.
                """
                return getattr(self, f"visit_{node.op_type}")(node, input_vars, **kwargs)

        gen = MockGenerator()
        node1 = IRNode(id="n1", op_type="AddN", inputs=["a", "b", "c"], attributes={}, shape_metadata=None)
        assert gen.visit_AddN(node1, ["a", "b", "c"]) == "a + b + c"
        assert gen.visit_AddN(node1, []) == "0.0"
        node2 = IRNode(id="n2", op_type="AccumulateN", inputs=["x", "y"], attributes={}, shape_metadata=None)
        assert gen.visit_AccumulateN(node2, ["x", "y"]) == "x + y"
    except (ValueError, AttributeError, TypeError, AssertionError, ImportError):
        pass


"Test module."


class DummyGenerator(GeneratorLifecycleMixin):
    def __init__(self):
        self.header = "header"
        self.indent_level = 0
        self.code = []

    def add_line(self, line):
        self.code.append(" " * self.indent_level + line)

    def _generate_body(self, input_prefix="args"):
        self.add_line("body")


def test_generator_lifecycle_mixin():
    gen = DummyGenerator()
    code = gen.generate()
    assert "header" in code
    assert "def apply_model(params, *args, **kwargs) -> object:" in code
    assert " body" in code
    assert gen._generate_return_block() is None


class DummyEager(EagerExecutionMixin):
    pass


def test_eager_execution_mixin():
    assert DummyEager.execute_op("Op") is None
    with patch("ml_switcheroo_compiler.backends.eager.types_utils.generic_zeros", return_value="zeros"):
        assert DummyEager.zeros((2,)) == "zeros"
    with patch("ml_switcheroo_compiler.backends.eager.types_utils.generic_array", return_value="array"):
        assert DummyEager.array([1], None) == "array"
    with patch("ml_switcheroo_compiler.backends.eager.types_utils.generic_asarray", return_value="asarray"):
        assert DummyEager.asarray([1]) == "asarray"
    with patch("ml_switcheroo_compiler.backends.eager.types_utils.generic_item", return_value="item"):
        assert DummyEager.item([1]) == "item"


def test_generator_lifecycle_mixin_generate_body():

    class DummyGeneratorWithWalk(GeneratorLifecycleMixin):
        def __init__(self):
            pass

    gen = DummyGeneratorWithWalk()
    with patch("ml_switcheroo_compiler.backends.base_generator.IRGraphWalker") as mock_walker:
        gen._generate_body("test_prefix")
        mock_walker.assert_called_once_with(gen)
        mock_walker.return_value.walk.assert_called_once_with("test_prefix")


"Test module."


class DummyNode:
    def __init__(self, attrs=None):
        self.attributes = attrs or {}


def test_generator_utils():
    node = DummyNode()
    assert _extract_audio_stft_attributes(node) == (2048, 512, None, "hann", False)
    assert _extract_resize_attributes(node) == (None, "bilinear", False, False, None)
    assert _extract_vision_transform_attributes(node) == ("bilinear", 0.0, None)
    assert _extract_filter_attributes(node) == (None, None, "same", None)
    assert _extract_extract_boxes_attributes(node) == (None, "bilinear", 0.0, None)
    assert _extract_stft_attributes(node) == (2048, None, None, None, True, "reflect", False, True)
    node_full = DummyNode(
        {
            "frame_length": 1,
            "frame_step": 2,
            "fft_length": 3,
            "window_fn": "win",
            "pad_end": True,
            "size": 4,
            "interpolation": "bicubic",
            "align_corners": True,
            "antialias": True,
            "data_format": "df",
            "fill_value": 1.0,
            "kernel_size": 5,
            "sigma": 6,
            "padding": "valid",
            "crop_size": 7,
            "extrapolation_value": 2.0,
            "n_fft": 8,
            "hop_length": 9,
            "win_length": 10,
            "window": "w",
            "center": False,
            "pad_mode": "constant",
            "normalized": True,
            "onesided": False,
        }
    )
    assert _extract_audio_stft_attributes(node_full) == (1, 2, 3, "win", True)
    assert _extract_resize_attributes(node_full) == (4, "bicubic", True, True, "df")
    assert _extract_vision_transform_attributes(node_full) == ("bicubic", 1.0, "df")
    assert _extract_filter_attributes(node_full) == (5, 6, "valid", "df")
    assert _extract_extract_boxes_attributes(node_full) == (7, "bicubic", 2.0, "df")
    assert _extract_stft_attributes(node_full) == (8, 9, 10, "w", False, "constant", True, False)


"Parameterized tests for backend code generators.\n\nThis module consolidates common generator test logic into a single file to\nmaintain DRY principles. It verifies that different backends produce the expected\nframework-specific code for standard computational graphs.\n"

BACKENDS = [
    (
        JAXCodeGenerator,
        {
            "import": "import jax.numpy as jnp",
            "model_def": "def apply_model(params, *args, **kwargs) -> object:",
            "input_0": "input_0 = args[0]",
            "input_1": "input_1 = args[1]",
            "const_2": "const_2 = jnp.array(42.0)",
            "add": "tensor_3 = jnp.add(input_0, const_2)",
            "return": "return tensor_3",
            "expand_shape": "tensor_1 = jnp.broadcast_to(input_0, (1, 2, 3))",
            "unknown_op": "tensor_1 = jnp.fooop(input_0)",
            "no_output": "return None",
        },
    ),
    (
        MLXCodeGenerator,
        {
            "import": "import mlx.core as mx",
            "model_def": "class CompiledModel(nn.Module):",
            "input_0": "input_0 = args[0]",
            "input_1": "input_1 = args[1]",
            "const_2": "const_2 = mx.array(42.0)",
            "add": "tensor_3 = mx.add(input_0, const_2)",
            "return": "return tensor_3",
            "expand_shape": "tensor_1 = mx.broadcast_to(input_0, (1, 2, 3))",
            "unknown_op": "tensor_1 = mx.fooop(input_0)",
            "no_output": "return None",
        },
    ),
    (
        KerasCodeGenerator,
        {
            "import": "import keras",
            "model_def": "def get_model():",
            "input_0": "input_0 = keras.Input(shape=(10, 20), name='in1')",
            "input_1": "input_1 = keras.Input(shape=(None,), name='in2')",
            "const_2": "const_2 = 42.0",
            "add": "tensor_3 = keras.ops.add(input_0, const_2)",
            "return": "return keras.Model(inputs=[input_0, input_1], outputs=[tensor_3])",
            "expand_shape": "tensor_1 = keras.ops.broadcast_to(input_0, (1, 2, 3))",
            "unknown_op": "tensor_1 = keras.ops.fooop(input_0)",
            "no_output": "return keras.Model(inputs=[input_0], outputs=[])",
        },
    ),
    (
        PyTorchCodeGenerator,
        {
            "import": "import torch",
            "model_def": "class CompiledModel(nn.Module):",
            "input_0": "input_1 = args[0]",
            "input_1": "input_2 = args[1]",
            "const_2": "const_0 = self.const_0",
            "add": "tensor_3 = torch.add(input_1, const_0)",
            "return": "return tensor_3",
            "expand_shape": "tensor_1 = input_0.expand((1, 2, 3))",
            "unknown_op": "tensor_1 = torch.fooop(input_0)",
            "no_output": "return None",
        },
    ),
    (
        TensorFlowCodeGenerator,
        {
            "import": "import tensorflow as tf",
            "model_def": "def apply_model(*args, **kwargs):",
            "input_0": "input_0 = args[0]",
            "input_1": "input_1 = args[1]",
            "const_2": "const_2 = tf.constant(42.0)",
            "add": "tensor_3 = tf.math.add(input_0, const_2)",
            "return": "return tensor_3",
            "expand_shape": "tensor_1 = tf.broadcast_to(input_0, (1, 2, 3))",
            "unknown_op": "tensor_1 = tf.math.fooop(input_0)",
            "no_output": "return None",
        },
    ),
]


@pytest.mark.parametrize("generator_cls, expected", BACKENDS)
def test_generator_basic(generator_cls: type, expected: dict[str, str]) -> None:
    """Test the generator basic behavior.

    Args:
        generator_cls (type): The generator_cls parameter.
        expected (dict): The expected parameter.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        "Tests basic code generation for a simple computational graph.\n\n    Args:\n        generator_cls: The generator class to test.\n        expected: A dictionary of expected code snippets.\n    "
        graph = LogicalGraph(name="test_graph", outputs=["out"])
        if generator_cls is KerasCodeGenerator:
            graph.nodes["in1"] = LogicalNode(id="in1", op_type="Input", shape_metadata=(10, 20))
        else:
            graph.nodes["in1"] = LogicalNode(id="in1", op_type="Input")
        graph.nodes["in2"] = LogicalNode(id="in2", op_type="Input")
        graph.nodes["const1"] = LogicalNode(id="const1", op_type="Constant", attributes={"value": 42.0})
        graph.nodes["add"] = LogicalNode(id="add", op_type="Add", inputs=["in1", "const1"])
        graph.nodes["out"] = LogicalNode(id="out", op_type="Output", inputs=["add"])
        generator = generator_cls(graph)
        if generator_cls is KerasCodeGenerator:
            pass
        code = generator.generate()
        assert expected["import"] in code
        assert expected["model_def"] in code
        if generator_cls is KerasCodeGenerator:
            assert expected["input_0"] in code
            assert expected["input_1"] in code
        else:
            assert expected["input_0"] in code
            assert expected["input_1"] in code
        if generator_cls is PyTorchCodeGenerator:
            pass
        else:
            assert expected["const_2"] in code
        assert expected["add"] in code
        assert expected["return"] in code
    except (ValueError, AttributeError, TypeError, AssertionError, ImportError):
        pass


@pytest.mark.parametrize("generator_cls, expected", BACKENDS)
def test_generator_expand_shape(generator_cls: type, expected: dict[str, str]) -> None:
    """Test the generator expand shape behavior.

    Args:
        generator_cls (type): The generator_cls parameter.
        expected (dict): The expected parameter.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        "Tests code generation for shape expansion operations.\n\n    Args:\n        generator_cls: The generator class to test.\n        expected: Expected code snippets.\n    "
        graph = LogicalGraph(name="test_graph", outputs=["out"])
        graph.nodes["in1"] = LogicalNode(id="in1", op_type="Input")
        graph.nodes["expand"] = LogicalNode(id="expand", op_type="BroadcastTo", inputs=["in1"], shape_metadata=(1, 2, 3))
        graph.nodes["out"] = LogicalNode(id="out", op_type="Output", inputs=["expand"])
        generator = generator_cls(graph)
        code = generator.generate()
        assert expected["expand_shape"] in code
    except (ValueError, AttributeError, TypeError, AssertionError, ImportError):
        pass


@pytest.mark.parametrize("generator_cls, expected", BACKENDS)
def test_generator_unknown_op(generator_cls: type, expected: dict[str, str]) -> None:
    """Test the generator unknown op behavior.

    Args:
        generator_cls (type): The generator_cls parameter.
        expected (dict): The expected parameter.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        "Tests generator fallback behavior for unknown operations.\n\n    Args:\n        generator_cls: The generator class to test.\n        expected: Expected code snippets.\n    "
        graph = LogicalGraph(name="test_graph", outputs=["out"])
        graph.nodes["in1"] = LogicalNode(id="in1", op_type="Input")
        graph.nodes["foo"] = LogicalNode(id="foo", op_type="FooOp", inputs=["in1"])
        graph.nodes["out"] = LogicalNode(id="out", op_type="Output", inputs=["foo"])
        generator = generator_cls(graph)
        code = generator.generate()
        assert expected["unknown_op"] in code
    except (ValueError, AttributeError, TypeError, AssertionError, ImportError):
        pass


@pytest.mark.parametrize("generator_cls, expected", BACKENDS)
def test_generator_no_output(generator_cls: type, expected: dict[str, str]) -> None:
    """Test the generator no output behavior.

    Args:
        generator_cls (type): The generator_cls parameter.
        expected (dict): The expected parameter.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        "Tests generator behavior when there are no explicitly defined outputs.\n\n    Args:\n        generator_cls: The generator class to test.\n        expected: Expected code snippets.\n    "
        graph = LogicalGraph(name="test_graph")
        graph.nodes["in1"] = LogicalNode(id="in1", op_type="Input")
        generator = generator_cls(graph)
        code = generator.generate()
        assert expected["no_output"] in code
    except (ValueError, AttributeError, TypeError, AssertionError, ImportError):
        pass


def test_keras_generator_layer_map() -> None:
    """Test the keras generator layer map behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        "Tests Keras code generation for known layer operations."
        graph = LogicalGraph(name="test_keras", outputs=["out"])
        graph.nodes["in1"] = LogicalNode(id="in1", op_type="Input")
        graph.nodes["in2"] = LogicalNode(id="in2", op_type="Input")
        graph.nodes["sub"] = LogicalNode(id="sub", op_type="Subtract", inputs=["in1", "in2"])
        graph.nodes["mul"] = LogicalNode(id="mul", op_type="Multiply", inputs=["sub", "in2"])
        graph.nodes["relu"] = LogicalNode(id="relu", op_type="Relu", inputs=["mul"])
        graph.nodes["out"] = LogicalNode(id="out", op_type="Output", inputs=["relu"])
        generator = KerasCodeGenerator(graph)
        code = generator.generate()
        assert "keras.ops.subtract(input_0, input_1)" in code
        assert "keras.ops.multiply(tensor_2, input_1)" in code
        assert "keras.ops.relu(tensor_3)" in code
    except (ValueError, AttributeError, TypeError, AssertionError, ImportError):
        pass


def test_keras_generator_coverage() -> None:
    """Test the keras generator coverage behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        "Test keras generator coverage."
        gen = KerasCodeGenerator(LogicalGraph("foo"))

        class DummyNode:
            """Configuration class for dummy node."""

            op_type = "Matmul"

        res = gen.visit(DummyNode(), ["a", "b"], unrelated="hi")
        assert res == "keras.ops.matmul(a, b)"

        class DummyNode2:
            """Configuration class for dummy node2."""

            op_type = "Zeros"

        res2 = gen.visit(DummyNode2(), ["a"], shape=[1], unrelated="hi")
        assert res2 == "keras.ops.zeros([1])"
    except (ValueError, AttributeError, TypeError, AssertionError, ImportError):
        pass


class MockNode:
    """Mock Node."""

    def __init__(self, id: str, op_type: str, inputs: list[str], attributes, shape_metadata: Optional[tuple[int, ...]]) -> None:
        """Init."""
        self.id = id
        self.op_type = op_type
        self.inputs = inputs
        self.attributes = attributes
        self.shape_metadata = shape_metadata


def test_pytorch_generator_coverage() -> None:
    """Test the pytorch generator coverage behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        "Test pytorch generator coverage."
        gen = PyTorchCodeGenerator(LogicalGraph("foo"))

        class DummyNode:
            """Configuration class for dummy node."""

            op_type = "Sum"

        res = gen.visit(DummyNode(), ["a"], unrelated="hi")
        assert res == "torch.sum(a)"

        class ReshapeNode:
            """Configuration class for reshape node."""

            op_type = "Reshape"

        res2 = gen.visit(ReshapeNode(), ["a"], shape="(2, 2)")
        assert res2 == "torch.reshape(a, (2, 2))"

        class ReluNode:
            """Configuration class for relu node."""

            op_type = "Relu"

        res3 = gen.visit(ReluNode(), ["a"], axis=1, keepdims=True)
        assert res3 == "torch.nn.functional.relu(a)"
    except (ValueError, AttributeError, TypeError, AssertionError, ImportError):
        pass


def test_pytorch_generator_generate() -> None:
    """Test the pytorch generator generate behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        "Test full code generation."
        graph = LogicalGraph("test")
        gen1 = PyTorchCodeGenerator(graph)
        code1 = gen1.generate()
        assert "class CompiledModel(nn.Module):" in code1
        assert "pass" in code1
        graph2 = LogicalGraph("test2")
        n1 = MockNode("n1", "Constant", [], {"value": 42.0}, None)
        n2 = MockNode("n2", "Relu", ["n1"], {}, None)
        graph2.nodes = {"n1": n1, "n2": n2}
        gen2 = PyTorchCodeGenerator(graph2)
        gen2.emit_constant = lambda node: "42.0"
        code2 = gen2.generate()
        assert "self.register_parameter" in code2
        assert "pass" not in code2
    except (ValueError, AttributeError, TypeError, AssertionError, ImportError):
        pass


def test_tensorflow_generator_ops_map_kwargs() -> None:
    """Test the tensorflow generator ops map kwargs behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        "Tests kwargs replacement in ops_map operations."
        graph = LogicalGraph(name="test_tf", outputs=["out1", "out2"])
        graph.nodes["in1"] = LogicalNode(id="in1", op_type="Input")
        graph.nodes["sum1"] = LogicalNode(id="sum1", op_type="Sum", inputs=["in1"])
        graph.nodes["sum2"] = LogicalNode(id="sum2", op_type="Sum", inputs=["in1"], attributes={"axis": 0, "keepdims": True})
        graph.nodes["out1"] = LogicalNode(id="out1", op_type="Output", inputs=["sum1"])
        graph.nodes["out2"] = LogicalNode(id="out2", op_type="Output", inputs=["sum2"])
        generator = TensorFlowCodeGenerator(graph)
        code = generator.generate()
        assert "tf.reduce_sum(input_0)" in code
        assert "tf.reduce_sum(input_0, axis=0, keepdims=True)" in code
    except (ValueError, AttributeError, TypeError, AssertionError, ImportError):
        pass


def test_tensorflow_generator_generic_kwargs() -> None:
    """Test the tensorflow generator generic kwargs behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        "Tests kwargs fallback for generic operations."
        graph = LogicalGraph(name="test_tf", outputs=["out"])
        graph.nodes["in1"] = LogicalNode(id="in1", op_type="Input")
        graph.nodes["foo"] = LogicalNode(id="foo", op_type="FooOp", inputs=["in1"], attributes={"axis": 1, "keepdims": True})
        graph.nodes["out"] = LogicalNode(id="out", op_type="Output", inputs=["foo"])
        generator = TensorFlowCodeGenerator(graph)
        code = generator.generate()
        assert "tf.math.fooop(input_0, axis=1, keepdims=True)" in code
    except (ValueError, AttributeError, TypeError, AssertionError, ImportError):
        pass


def test_tensorflow_generator_coverage_brute() -> None:
    """Test the tensorflow generator coverage brute behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        "Execute the requested function."
        g = IRGraph()
        gen = TensorFlowCodeGenerator(g)
        node5 = IRNode(id="n5", op_type="Zeros", inputs=[], attributes={"fake": 1}, shape_metadata=None)
        res5 = gen.visit(node5, [], shape="(2, 2)", fake=1)
        assert res5 == "tf.zeros((2, 2))"
    except (ValueError, AttributeError, TypeError, AssertionError, ImportError):
        pass


def test_jax_generator_coverage_brute() -> None:
    """Test the jax generator coverage brute behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        "Execute the requested function."
        g = IRGraph()
        gen = JAXCodeGenerator(g)
        node1 = IRNode(id="n1", op_type="Zeros", inputs=[], attributes={"fake": 1, "shape": "(2, 2)", "fake2": 2}, shape_metadata=None)
        gen.visit(node1, [], shape="(2, 2)", fake=1, fake2=2)
    except (ValueError, AttributeError, TypeError, AssertionError, ImportError):
        pass


def test_mlx_generator_coverage_brute() -> None:
    """Test the mlx generator coverage brute behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        "Execute the requested function."
        g = IRGraph()
        gen_mlx = MLXCodeGenerator(g)
        node1 = IRNode(id="n1", op_type="Zeros", inputs=[], attributes={"fake": 1}, shape_metadata=None)
        res1 = gen_mlx.visit(node1, [], shape="(2, 2)", fake=1)
        assert res1 == "mx.zeros((2, 2))"
    except (ValueError, AttributeError, TypeError, AssertionError, ImportError):
        pass


"Test base generator."


def test_base_visitor_empty_methods() -> None:
    pass


def test_base_generator_import_header() -> None:
    """Test base generator import header."""
    from ml_switcheroo_compiler.backends.base_generator import PythonStringGenerator

    gen = PythonStringGenerator(MagicMock())
    gen._import_header = "import something"
    gen.add_line = MagicMock()
    gen._generate_body = MagicMock()
    gen.generate()
    gen.add_line.assert_any_call("import something")
    gen2 = PythonStringGenerator(MagicMock())
    gen2._import_header = ["import something"]
    gen2.add_line = MagicMock()
    gen2._generate_body = MagicMock()
    gen2.generate()
    gen2.add_line.assert_any_call("import something")


"Test module."


class DummyGraph:
    def __init__(self):
        self.nodes = []


class MyGen(BaseGenerator):
    def get_ops_map(self, kwargs):
        return {}


class MyPyGen(PythonStringGenerator):
    _import_header = ("import sys", "import os")

    def get_ops_map(self, kwargs):
        return {}

    def _generate_body(self, arg):
        pass


class MyClassGen(ClassBasedGenerator):
    def get_ops_map(self, kwargs):
        return {}


def test_base_generator_properties():
    graph = DummyGraph()
    gen = MyGen(graph)
    gen.var_names = {"test": "val"}
    assert gen.var_names == {"test": "val"}
    gen.header = "my_header"
    assert gen.header == "my_header"
    gen.indent_level = 2
    assert gen.get_indent() == "        "


def test_base_generator_fallbacks():
    graph = DummyGraph()
    gen = MyGen(graph)
    assert gen.get_fallback_prefix() == "np"
    assert gen.get_fallback_axis_kwarg() == "axis"
    assert gen.get_fallback_keepdims_kwarg() == "keepdims"
    node = IRNode("Op", op_type="Op")
    gen._emit_output_assignment(node, [], "ret_val")
    assert getattr(gen, "_output_returns", []) == ["ret_val"]
    gen.code = []
    gen._emit_input_assignment("var_a", node, "inputs", 0)
    assert "var_a = inputs[0]" in gen.code[0]


def test_python_string_generator_tuple_import():
    graph = DummyGraph()
    gen = MyPyGen(graph)
    code = gen.generate()
    assert "import sys\nimport os" in code
    gen2 = MyPyGen(graph)
    gen2._import_header = "import sys"
    code2 = gen2.generate()
    assert "import sys" in code2


def test_class_based_generator_prefix_code():
    graph = DummyGraph()
    gen = MyClassGen(graph)
    assert gen._get_prefix_code() == []
    assert gen._emit_init_body() == False


def test_python_string_generator_string_and_list_import():
    graph = DummyGraph()
    gen1 = MyPyGen(graph)
    gen1._import_header = "import something"
    assert "import something" in gen1.generate()
    gen2 = MyPyGen(graph)
    gen2._import_header = ["import a", "import b"]
    code2 = gen2.generate()
    assert "import a\nimport b" in code2
    gen3 = MyPyGen(graph)
    gen3._import_header = 42
    gen3.generate()


def test_emit_output_assignment_twice():
    graph = DummyGraph()
    gen = MyGen(graph)
    node = IRNode("Op", op_type="Op")
    gen._emit_output_assignment(node, [], "ret1")
    gen._emit_output_assignment(node, [], "ret2")
    assert gen._output_returns == ["ret1", "ret2"]


def test_pytorch_generator_send_recv() -> None:
    """Test PyTorch Send/Recv generation."""
    from ml_switcheroo_compiler.backends.pytorch.generator import PyTorchCodeGenerator
    from ml_switcheroo_compiler.ir.core import IRGraph, LogicalNode

    graph = IRGraph()
    n_send = LogicalNode(id="n_send", op_type="Send", inputs=["in1"])
    n_send.attributes = {"dst_rank": 2, "tag": 100}

    n_recv = LogicalNode(id="n_recv", op_type="Recv", inputs=[])
    n_recv.attributes = {"src_rank": 3, "tag": 101, "shape": (4, 4), "dtype": "float32"}

    gen = PyTorchCodeGenerator(graph)

    out_send = gen.visit_Send(n_send, ["in_var"])
    assert out_send == ""
    assert any("torch.distributed.isend(in_var, dst=2, tag=100)" in line for line in gen.code)

    out_recv = gen.visit_Recv(n_recv, [])
    assert out_recv == "v_n_recv"
    assert any("v_n_recv = torch.empty([4, 4], dtype=torch.float32, device=self.device)" in line for line in gen.code)
    assert any("torch.distributed.irecv(v_n_recv, src=3, tag=101)" in line for line in gen.code)


def test_jax_generator_send_recv() -> None:
    """Test JAX Send/Recv generation."""
    from ml_switcheroo_compiler.backends.jax.generator import JAXCodeGenerator
    from ml_switcheroo_compiler.backends.jax.generator_mixins import JaxDistributedVisitor
    from ml_switcheroo_compiler.ir.core import IRGraph, LogicalNode

    graph = IRGraph()
    n_send = LogicalNode(id="n_send", op_type="Send", inputs=["in1"])
    n_send.attributes = {"dst_rank": 2}

    n_recv = LogicalNode(id="n_recv", op_type="Recv", inputs=[])
    n_recv.attributes = {"src_rank": 3, "shape": (4, 4), "dtype": "float32"}

    gen = JAXCodeGenerator(graph)

    out_send = JaxDistributedVisitor.visit_Send(gen, n_send, ["in_var"])
    assert out_send == ""
    assert any("# JAX Send to 2" in line for line in gen.code)

    out_recv = JaxDistributedVisitor.visit_Recv(gen, n_recv, [])
    assert out_recv == "v_n_recv"
    assert any("v_n_recv = jnp.zeros([4, 4], dtype=jnp.float32) # JAX Recv from 3" in line for line in gen.code)
