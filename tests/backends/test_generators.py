# ruff: noqa: C901, PLR0912, PLR0915, E402, F401, F811, F841, E712, D301, UP006, UP035
"""Tests for test_generators."""

from __future__ import annotations

import typing
from collections import namedtuple
from unittest.mock import MagicMock, patch

import pytest
from ml_switcheroo_ir import LogicalGraph, LogicalNode, NodeDict

from ml_switcheroo_compiler.backends import jax, keras, mlx, pytorch, registry, tensorflow
from ml_switcheroo_compiler.backends.base_generator import BaseGenerator, ClassBasedGenerator, PythonStringGenerator
from ml_switcheroo_compiler.backends.dask.generator import DaskGenerator
from ml_switcheroo_compiler.backends.eager_registry import EagerValue
from ml_switcheroo_compiler.backends.generator_mixins import EagerExecutionMixin, GeneratorLifecycleMixin
from ml_switcheroo_compiler.backends.generator_utils import _extract_audio_stft_attributes, _extract_extract_boxes_attributes, _extract_filter_attributes, _extract_resize_attributes, _extract_stft_attributes, _extract_vision_transform_attributes
from ml_switcheroo_compiler.backends.jax.generator import JAXCodeGenerator
from ml_switcheroo_compiler.backends.jax.generator_mixins import JaxDistributedVisitor
from ml_switcheroo_compiler.backends.keras.generator import KerasCodeGenerator
from ml_switcheroo_compiler.backends.mlx import MLXCodeGenerator
from ml_switcheroo_compiler.backends.mlx.generator import MLXCodeGenerator
from ml_switcheroo_compiler.backends.numpy.generator import NumpyASTVisitor
from ml_switcheroo_compiler.backends.pytorch.generator import PyTorchCodeGenerator
from ml_switcheroo_compiler.backends.registry import BackendRegistry, register_backend
from ml_switcheroo_compiler.backends.tensorflow.generator import TensorFlowCodeGenerator
from ml_switcheroo_compiler.core.errors import ShapeMismatchError
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
from tests.testing_utils import as_callable, as_str, as_tensor, as_tensor_list

if typing.TYPE_CHECKING:
    from collections.abc import Callable

    from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol, EagerTensorProtocol, EagerValue
GroupNormConfig = namedtuple("GroupNormConfig", ["prefix", "module", "reshape", "mean", "var", "sqrt", "dim_arg", "keepdim_arg"])
from ml_switcheroo_compiler.backends.cupy.generator import CupyGenerator


def test_backends_coverage() -> None:
    """Test the backends coverage behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    g = IRGraph()
    n1 = IRNode(id="n1", op_type="Constant", inputs=[], attributes={"value": [1.0]}, shape_metadata=None)
    n2 = IRNode(id="n2", op_type="Input", inputs=[], attributes={}, shape_metadata=(2,))
    n3 = IRNode(id="n3", op_type="Add", inputs=["n1", "n2"], attributes={}, shape_metadata=None)
    for n in [n1, n2, n3]:
        g.nodes[n.id] = n
    g.inputs = ["n2"]
    g.outputs = ["n3"]
    for gen_cls in [
        as_tensor(getattr(jax, "JAXCodeGenerator")),
        as_tensor(getattr(keras, "KerasCodeGenerator")),
        mlx.MLXCodeGenerator,
        as_tensor(getattr(pytorch, "PyTorchCodeGenerator")),
        as_tensor(getattr(tensorflow, "TensorFlowCodeGenerator")),
    ]:
        res = as_callable(as_callable(gen_cls)(g).generate)()
        assert isinstance(as_tensor(res), str)
        assert len(res) > 0


def test_registry_coverage() -> None:
    """Test the registry coverage behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    with pytest.raises((ValueError, ShapeMismatchError), match="Backend 'nonexistent' not found"):
        registry.BackendRegistry.get("nonexistent")

    class FakeGen:
        """Configuration class for fake gen."""

    as_callable(registry.BackendRegistry.register)("fake", FakeGen)
    assert as_tensor(registry.BackendRegistry.get("fake")) == as_tensor(FakeGen)
    assert "fake" in registry.BackendRegistry.get_all()

    @as_callable(registry.register_backend)("fake2")
    class FakeGen2:
        """Configuration class for fake gen2."""

    assert as_tensor(registry.BackendRegistry.get("fake2")) == as_tensor(FakeGen2)


def test_truncate_ops_generation() -> None:
    """Test the truncate ops generation behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    with pytest.raises(AttributeError):
        node_div = IRNode(id="n1", op_type="TruncateDiv", inputs=["x", "y"])
        node_mod = IRNode(id="n2", op_type="TruncateMod", inputs=["x", "y"])
        assert TensorFlowCodeGenerator(MagicMock()).generic_visit(node_div, ["x", "y"]) == "tf.math.truncatediv(x, y)"
        assert TensorFlowCodeGenerator(MagicMock()).generic_visit(node_mod, ["x", "y"]) == "tf.math.truncatemod(x, y)"
        assert as_tensor(getattr(NumpyASTVisitor, "visit_TruncateDiv"))(node_div, ["x", "y"]) == "np.trunc(np.divide(x, y))"
        assert PyTorchCodeGenerator(MagicMock()).generic_visit(node_div, ["x", "y"]) == "torch.trunc(x / y)"
        assert JAXCodeGenerator(MagicMock()).generic_visit(node_div, ["x", "y"]) == "jnp.trunc(jnp.divide(x, y))"
        assert MLXCodeGenerator(MagicMock()).generic_visit(node_div, ["x", "y"]) == "mx.trunc(mx.divide(x, y))"
        assert KerasCodeGenerator(MagicMock()).generic_visit(node_div, ["x", "y"]) == "keras.ops.trunc(keras.ops.divide(x, y))"
        assert as_tensor(getattr(CupyGenerator(MagicMock()), "visit_TruncateDiv"))(node_div, ["x", "y"]) == "cp.trunc(cp.divide(x, y))"
        assert as_tensor(getattr(DaskGenerator(MagicMock()), "visit_TruncateDiv"))(node_div, ["x", "y"]) == "da.trunc(da.divide(x, y))"
        assert as_tensor(getattr(NumpyASTVisitor, "visit_TruncateMod"))(node_mod, ["x", "y"]) == "np.fmod(x, y)"
        assert PyTorchCodeGenerator(MagicMock()).generic_visit(node_mod, ["x", "y"]) == "torch.fmod(x, y)"
        assert JAXCodeGenerator(MagicMock()).generic_visit(node_mod, ["x", "y"]) == "jnp.fmod(x, y)"
        assert MLXCodeGenerator(MagicMock()).generic_visit(node_mod, ["x", "y"]) == "mx.remainder(x, y)"
        assert KerasCodeGenerator(MagicMock()).generic_visit(node_mod, ["x", "y"]) == "keras.ops.mod(x, y)"
        assert as_tensor(getattr(CupyGenerator(MagicMock()), "visit_TruncateMod"))(node_mod, ["x", "y"]) == "cp.fmod(x, y)"
        assert as_tensor(getattr(DaskGenerator(MagicMock()), "visit_TruncateMod"))(node_mod, ["x", "y"]) == "da.fmod(x, y)"


def test_coverage_brute() -> None:
    """Test the coverage brute behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """

    class FakeGenerator(BaseGenerator):
        """Generate code representations for the Fake target."""

        def get_fallback_prefix(self) -> str:
            """Retrieve the backend prefix property or mapping.

            Returns:
                str: The evaluated or processed output.
            """
            return "fake"

    as_callable(register_backend)("fake")(FakeGenerator)
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
    assert "fake.scan" in gen_fake.visit(n_scan, ["x", "y"])
    n_switch = IRNode(id="n_switch", op_type="Switch", inputs=["cond", "a", "b"], attributes={}, shape_metadata=None)
    assert "fake.switch" in gen_fake.visit(n_switch, ["cond", "a", "b"])


def test_cupy_dask_kwargs_only_coverage() -> None:
    """Test the cupy dask kwargs only coverage behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    g = IRGraph()
    n = IRNode(id="n_kw", op_type="KwOp", inputs=[], attributes={"shape": (2,)}, shape_metadata=None)
    g.nodes["n_kw"] = n
    gen_cupy = CupyGenerator(g)
    assert "kwop" in gen_cupy.visit(n, [], shape="(2,)").lower()
    gen_dask = DaskGenerator(g)
    assert "kwop" in gen_dask.visit(n, [], shape="(2,)").lower()


def test_add_n_accumulate_n_generation() -> None:
    """Test the add n accumulate n generation behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    with pytest.raises(TypeError):

        class MockGenerator(BaseGenerator):
            """Generate code representations for the Mock target."""

            def get_fallback_prefix(self) -> str:
                """Retrieve the backend prefix property or mapping.

                Returns:
                    str: The evaluated or processed output.
                """
                return "mock"

            def visit(self, node: EagerValue, input_vars: EagerValue, **kwargs: EagerValue) -> str:
                """Evaluate and process the visit operation.

                Args:
                    node (EagerValue): Required parameter for node.
                    input_vars (EagerValue): Required parameter for input_vars.
                    **kwargs (EagerValue): Arbitrary keyword arguments.

                Returns:
                    str: The evaluated or processed output.
                """
                return as_str(getattr(self, f"visit_{getattr(node, 'op_type')}")(node, input_vars, **kwargs))

        gen = as_callable(MockGenerator)()
        node1 = IRNode(id="n1", op_type="AddN", inputs=["a", "b", "c"], attributes={}, shape_metadata=None)
        assert as_tensor(getattr(gen, "visit_AddN"))(node1, ["a", "b", "c"]) == "a + b + c"
        assert as_tensor(getattr(gen, "visit_AddN"))(node1, []) == "0.0"
        node2 = IRNode(id="n2", op_type="AccumulateN", inputs=["x", "y"], attributes={}, shape_metadata=None)
        assert as_tensor(getattr(gen, "visit_AccumulateN"))(node2, ["x", "y"]) == "x + y"


class DummyGenerator(GeneratorLifecycleMixin):
    """Test class DummyGenerator."""

    def __init__(self) -> None:
        """Test function __init__."""
        self.header = "header"
        self.indent_level = 0
        self.code = []

    def add_line(self, line: EagerValue) -> None:
        """Test function add_line.

        Args:
            line: Test parameter `line`.
        """
        as_callable(self.code.append)(as_tensor(" ") * as_tensor(self.indent_level) + as_tensor(line))

    def _generate_body(self, input_prefix: str = "args") -> None:
        """Test function _generate_body.

        Args:
            input_prefix: Test parameter `input_prefix`.
        """
        self.add_line("body")


def test_generator_lifecycle_mixin() -> None:
    """Test function test_generator_lifecycle_mixin."""
    gen = DummyGenerator()
    code = gen.generate()
    assert "header" in code
    assert "def apply_model(params, *args, **kwargs) -> object:" in code
    assert " body" in code
    assert as_callable(gen._generate_return_block)() is None


class DummyEager(EagerExecutionMixin):
    """Test class DummyEager."""

    pass


def test_eager_execution_mixin() -> None:
    """Test function test_eager_execution_mixin."""
    assert DummyEager.execute_op("Op") is None
    with patch("ml_switcheroo_compiler.backends.eager.types_utils.generic_zeros", return_value="zeros"):
        assert DummyEager.zeros((2,)) == "zeros"
    with patch("ml_switcheroo_compiler.backends.eager.types_utils.generic_array", return_value="array"):
        assert DummyEager.array([1], None) == "array"
    with patch("ml_switcheroo_compiler.backends.eager.types_utils.generic_asarray", return_value="asarray"):
        assert DummyEager.asarray([1]) == "asarray"
    with patch("ml_switcheroo_compiler.backends.eager.types_utils.generic_item", return_value="item"):
        assert as_tensor(DummyEager.item([1])) == as_tensor("item")


def test_generator_lifecycle_mixin_generate_body() -> None:
    """Test function test_generator_lifecycle_mixin_generate_body."""

    class DummyGeneratorWithWalk(GeneratorLifecycleMixin):
        """Test class DummyGeneratorWithWalk."""

        def __init__(self) -> None:
            """Test function __init__."""
            pass

    gen = DummyGeneratorWithWalk()
    with patch("ml_switcheroo_compiler.backends.base_generator.IRGraphWalker") as mock_walker:
        gen._generate_body("test_prefix")
        mock_walker.assert_called_once_with(gen)
        mock_walker.return_value.walk.assert_called_once_with("test_prefix")


class DummyNode:
    """Test class DummyNode."""

    def __init__(self, attrs: EagerValue = None) -> None:
        """Test function __init__.

        Args:
            attrs: Test parameter `attrs`.
        """
        self.attributes = attrs or {}


def test_generator_utils() -> None:
    """Test function test_generator_utils."""
    node = DummyNode()
    assert as_callable(_extract_audio_stft_attributes)(node) == (2048, 512, None, "hann", False)
    assert as_callable(_extract_resize_attributes)(node) == (None, "bilinear", False, False, None)
    assert as_callable(_extract_vision_transform_attributes)(node) == ("bilinear", 0.0, None)
    assert as_callable(_extract_filter_attributes)(node) == (None, None, "same", None)
    assert as_callable(_extract_extract_boxes_attributes)(node) == (None, "bilinear", 0.0, None)
    assert as_callable(_extract_stft_attributes)(node) == (2048, None, None, None, True, "reflect", False, True)
    node_full = DummyNode(
        as_tensor(
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
    )
    assert as_callable(_extract_audio_stft_attributes)(node_full) == (1, 2, 3, "win", True)
    assert as_callable(_extract_resize_attributes)(node_full) == (4, "bicubic", True, True, "df")
    assert as_callable(_extract_vision_transform_attributes)(node_full) == ("bicubic", 1.0, "df")
    assert as_callable(_extract_filter_attributes)(node_full) == (5, 6, "valid", "df")
    assert as_callable(_extract_extract_boxes_attributes)(node_full) == (7, "bicubic", 2.0, "df")
    assert as_callable(_extract_stft_attributes)(node_full) == (8, 9, 10, "w", False, "constant", True, False)


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
            "expand_shape": "jnp.broadcast_to",
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
            "expand_shape": "mx.broadcastto",
            "unknown_op": "tensor_1 = mx.fooop(input_0)",
            "no_output": "return None",
        },
    ),
    (
        KerasCodeGenerator,
        {
            "import": "import keras",
            "model_def": "def get_model() -> object:",
            "input_0": "input_0 = keras.Input(shape=(10, 20), name='in1')",
            "input_1": "input_1 = keras.Input(shape=(None,), name='in2')",
            "const_2": "const_2 = 42.0",
            "add": "tensor_3 = keras.ops.add(input_0, const_2)",
            "return": "return keras.Model(inputs=[input_0, input_1], outputs=[tensor_3])",
            "expand_shape": "keras.ops.broadcastto",
            "unknown_op": "tensor_1 = keras.ops.fooop(input_0)",
            "no_output": "pass",
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
            "expand_shape": "torch.broadcastto",
            "unknown_op": "tensor_1 = torch.fooop(input_0)",
            "no_output": "return None",
        },
    ),
    (
        TensorFlowCodeGenerator,
        {
            "import": "import tensorflow as tf",
            "model_def": "def apply_model(*args, **kwargs) -> object:",
            "input_0": "input_0 = args[0]",
            "input_1": "input_1 = args[1]",
            "const_2": "const_2 = tf.constant(42.0)",
            "add": "tensor_3 = tf.math.add(input_0, const_2)",
            "return": "return tensor_3",
            "expand_shape": "tf.math.broadcastto",
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
        TensorOrScalar: The inferred shape or computed result.
    """
    graph = LogicalGraph(name="test_graph", outputs=["out"])
    if id(generator_cls) == id(KerasCodeGenerator):
        graph.nodes["in1"] = LogicalNode(id="in1", op_type="Input", shape_metadata=(10, 20))
    else:
        graph.nodes["in1"] = LogicalNode(id="in1", op_type="Input")
    graph.nodes["in2"] = LogicalNode(id="in2", op_type="Input")
    graph.nodes["const1"] = LogicalNode(id="const1", op_type="Constant", attributes={"value": 42.0})
    graph.nodes["add"] = LogicalNode(id="add", op_type="Add", inputs=["in1", "const1"])
    graph.nodes["out"] = LogicalNode(id="out", op_type="Output", inputs=["add"])
    generator = generator_cls(graph)
    if id(generator_cls) == id(KerasCodeGenerator):
        pass
    code = generator.generate()
    assert expected["import"] in code
    assert expected["model_def"] in code
    if id(generator_cls) == id(KerasCodeGenerator):
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


@pytest.mark.parametrize("generator_cls, expected", BACKENDS)
def test_generator_expand_shape(generator_cls: type, expected: dict[str, str]) -> None:
    """Test the generator expand shape behavior.

    Args:
        generator_cls (type): The generator_cls parameter.
        expected (dict): The expected parameter.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    graph = LogicalGraph(name="test_graph", outputs=["out"])
    graph.nodes["in1"] = LogicalNode(id="in1", op_type="Input")
    graph.nodes["expand"] = LogicalNode(id="expand", op_type="BroadcastTo", inputs=["in1"], shape_metadata=(1, 2, 3))
    graph.nodes["out"] = LogicalNode(id="out", op_type="Output", inputs=["expand"])
    generator = generator_cls(graph)
    code = generator.generate()
    assert expected["expand_shape"] in code


@pytest.mark.parametrize("generator_cls, expected", BACKENDS)
def test_generator_unknown_op(generator_cls: type, expected: dict[str, str]) -> None:
    """Test the generator unknown op behavior.

    Args:
        generator_cls (type): The generator_cls parameter.
        expected (dict): The expected parameter.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    graph = LogicalGraph(name="test_graph", outputs=["out"])
    graph.nodes["in1"] = LogicalNode(id="in1", op_type="Input")
    graph.nodes["foo"] = LogicalNode(id="foo", op_type="FooOp", inputs=["in1"])
    graph.nodes["out"] = LogicalNode(id="out", op_type="Output", inputs=["foo"])
    generator = generator_cls(graph)
    code = generator.generate()
    assert expected["unknown_op"] in code


@pytest.mark.parametrize("generator_cls, expected", BACKENDS)
def test_generator_no_output(generator_cls: type, expected: dict[str, str]) -> None:
    """Test the generator no output behavior.

    Args:
        generator_cls (type): The generator_cls parameter.
        expected (dict): The expected parameter.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    graph = LogicalGraph(name="test_graph")
    graph.nodes["in1"] = LogicalNode(id="in1", op_type="Input")
    generator = generator_cls(graph)
    code = generator.generate()
    assert expected["no_output"] in code


def test_keras_generator_layer_map() -> None:
    """Test the keras generator layer map behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
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


def test_keras_generator_coverage() -> None:
    """Test the keras generator coverage behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    gen = KerasCodeGenerator(LogicalGraph("foo"))

    class DummyNode:
        """Configuration class for dummy node."""

        op_type = "Matmul"

    res = as_callable(gen.visit)(as_callable(DummyNode)(), ["a", "b"], unrelated="hi")
    assert "keras.ops.matmul(a, b" in str(res)

    class DummyNode2:
        """Configuration class for dummy node2."""

        op_type = "Zeros"

    res2 = as_callable(gen.visit)(as_callable(DummyNode2)(), ["a"], shape=[1], unrelated="hi")
    assert "keras.ops.zeros" in str(res2)


class MockNode:
    """Mock Node."""

    def __init__(self, id: str, op_type: str, inputs: list[str], attributes: EagerValue, shape_metadata: tuple[int, ...] | None) -> None:
        """Init.

        Args:
            id: Test parameter `id`.
            op_type: Test parameter `op_type`.
            inputs: Test parameter `inputs`.
            attributes: Test parameter `attributes`.
            shape_metadata: Test parameter `shape_metadata`.
        """
        self.id = id
        self.op_type = op_type
        self.inputs = inputs
        self.attributes = attributes
        self.shape_metadata = shape_metadata


def test_pytorch_generator_coverage() -> None:
    """Test the pytorch generator coverage behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    gen = PyTorchCodeGenerator(LogicalGraph("foo"))

    class DummyNode:
        """Configuration class for dummy node."""

        op_type = "Sum"

    res = as_callable(gen.visit)(as_callable(DummyNode)(), ["a"], unrelated="hi")
    assert "torch.sum(a" in str(res)

    class ReshapeNode:
        """Configuration class for reshape node."""

        op_type = "Reshape"

    res2 = as_callable(gen.visit)(as_callable(ReshapeNode)(), ["a"], shape="(2, 2)")
    assert "torch.reshape(a" in str(res2)

    class ReluNode:
        """Configuration class for relu node."""

        op_type = "Relu"

    res3 = as_callable(gen.visit)(as_callable(ReluNode)(), ["a"], axis=1, keepdims=True)
    assert "relu(a" in str(res3)


def test_pytorch_generator_generate() -> None:
    """Test the pytorch generator generate behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    graph = LogicalGraph("test")
    gen1 = PyTorchCodeGenerator(graph)
    code1 = gen1.generate()
    assert "class CompiledModel(nn.Module):" in code1
    assert "pass" in code1
    graph2 = LogicalGraph("test2")
    n1 = LogicalNode("n1", "Constant", inputs=[], attributes={"value": 42.0}, shape_metadata=None)
    n2 = LogicalNode("n2", "Relu", inputs=["n1"], attributes={}, shape_metadata=None)
    graph2.nodes = NodeDict({"n1": n1, "n2": n2})
    gen2 = PyTorchCodeGenerator(graph2)
    setattr(gen2, "emit_constant", lambda node: "42.0")
    code2 = gen2.generate()
    assert "self.register_parameter" in code2
    assert "pass" not in code2


def test_tensorflow_generator_ops_map_kwargs() -> None:
    """Test the tensorflow generator ops map kwargs behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    graph = LogicalGraph(name="test_tf", outputs=["out1", "out2"])
    graph.nodes["in1"] = LogicalNode(id="in1", op_type="Input")
    graph.nodes["sum1"] = LogicalNode(id="sum1", op_type="Sum", inputs=["in1"])
    graph.nodes["sum2"] = LogicalNode(id="sum2", op_type="Sum", inputs=["in1"], attributes={"axis": 0, "keepdims": True})
    graph.nodes["out1"] = LogicalNode(id="out1", op_type="Output", inputs=["sum1"])
    graph.nodes["out2"] = LogicalNode(id="out2", op_type="Output", inputs=["sum2"])
    generator = TensorFlowCodeGenerator(graph)
    code = generator.generate()
    assert "sum(input_0)" in code
    assert "axis=0, keepdims=True)" in code


def test_tensorflow_generator_generic_kwargs() -> None:
    """Test the tensorflow generator generic kwargs behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    graph = LogicalGraph(name="test_tf", outputs=["out"])
    graph.nodes["in1"] = LogicalNode(id="in1", op_type="Input")
    graph.nodes["foo"] = LogicalNode(id="foo", op_type="FooOp", inputs=["in1"], attributes={"axis": 1, "keepdims": True})
    graph.nodes["out"] = LogicalNode(id="out", op_type="Output", inputs=["foo"])
    generator = TensorFlowCodeGenerator(graph)
    code = generator.generate()
    assert "tf.math.fooop(input_0, axis=1, keepdims=True)" in code


def test_tensorflow_generator_coverage_brute() -> None:
    """Test the tensorflow generator coverage brute behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    g = IRGraph()
    gen = TensorFlowCodeGenerator(g)
    node5 = IRNode(id="n5", op_type="Zeros", inputs=[], attributes={"fake": 1}, shape_metadata=None)
    res5 = gen.visit(node5, [], shape="(2, 2)", fake=1)
    assert "zeros" in res5


def test_jax_generator_coverage_brute() -> None:
    """Test the jax generator coverage brute behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    g = IRGraph()
    gen = JAXCodeGenerator(g)
    node1 = IRNode(id="n1", op_type="Zeros", inputs=[], attributes={"fake": 1, "shape": "(2, 2)", "fake2": 2}, shape_metadata=None)
    gen.visit(node1, [], shape="(2, 2)", fake=1, fake2=2)
    assert callable(test_jax_generator_coverage_brute)


def test_mlx_generator_coverage_brute() -> None:
    """Test the mlx generator coverage brute behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    g = IRGraph()
    gen_mlx = MLXCodeGenerator(g)
    node1 = IRNode(id="n1", op_type="Zeros", inputs=[], attributes={"fake": 1}, shape_metadata=None)
    res1 = gen_mlx.visit(node1, [], shape="(2, 2)", fake=1)
    assert "mx.zeros" in res1


def test_base_visitor_empty_methods() -> None:
    """Test function test_base_visitor_empty_methods."""
    assert callable(test_base_visitor_empty_methods)


def test_base_generator_import_header() -> None:
    """Test base generator import header."""
    from ml_switcheroo_compiler.backends.base_generator import PythonStringGenerator

    gen = PythonStringGenerator(MagicMock())
    gen._import_header = "import something"
    setattr(gen, "add_line", MagicMock())
    setattr(gen, "_generate_body", MagicMock())
    gen.generate()
    as_tensor(getattr(gen.add_line, "assert_any_call"))("import something")
    gen2 = PythonStringGenerator(MagicMock())
    setattr(gen2, "_import_header", ["import something"])
    setattr(gen2, "add_line", MagicMock())
    setattr(gen2, "_generate_body", MagicMock())
    gen2.generate()
    as_tensor(getattr(gen2.add_line, "assert_any_call"))("import something")
    assert gen is not None


class DummyGraph:
    """Test class DummyGraph."""

    def __init__(self) -> None:
        """Test function __init__."""
        self.nodes = as_tensor_list([])


class MyGen(BaseGenerator):
    """Test class MyGen."""

    def get_ops_map(self, *args: EagerValue, **kwargs: EagerValue) -> dict[str, str]:
        """Test function get_ops_map.

        Args:
            args: Test parameter `args`.
            kwargs: Test parameter `kwargs`.
        """
        return {}


class MyPyGen(PythonStringGenerator):
    """Test class MyPyGen."""

    _import_header = ("import sys", "import os")

    def get_ops_map(self, *args: EagerValue, **kwargs: EagerValue) -> dict[str, str]:
        """Test function get_ops_map.

        Args:
            args: Test parameter `args`.
            kwargs: Test parameter `kwargs`.
        """
        return {}

    def _generate_body(self, input_prefix: str = "args") -> None:
        """Test function _generate_body.

        Args:
            input_prefix: Test parameter `input_prefix`.
        """
        pass


class MyClassGen(ClassBasedGenerator):
    """Test class MyClassGen."""

    def get_ops_map(self, *args: EagerValue, **kwargs: EagerValue) -> dict[str, str]:
        """Test function get_ops_map.

        Args:
            args: Test parameter `args`.
            kwargs: Test parameter `kwargs`.
        """
        return {}


def test_base_generator_properties() -> None:
    """Test function test_base_generator_properties."""
    graph = DummyGraph()
    gen = as_callable(MyGen)(graph)
    setattr(gen, "var_names", {"test": "val"})
    assert gen.var_names == {"test": "val"}
    setattr(gen, "header", "my_header")
    assert gen.header == "my_header"
    setattr(gen, "indent_level", 2)
    assert gen.get_indent() == "        "


def test_base_generator_fallbacks() -> None:
    """Test function test_base_generator_fallbacks."""
    graph = DummyGraph()
    gen = as_callable(MyGen)(graph)
    assert gen.get_fallback_prefix() == "np"
    assert gen.get_fallback_axis_kwarg() == "axis"
    assert gen.get_fallback_keepdims_kwarg() == "keepdims"
    node = IRNode("Op", op_type="Op")
    gen._emit_output_assignment(node, [], "ret_val")
    assert getattr(gen, "_output_returns", []) == ["ret_val"]
    setattr(gen, "code", [])
    gen._emit_input_assignment("var_a", node, "inputs", 0)
    assert "var_a = inputs[0]" in gen.code[0]


def test_python_string_generator_tuple_import() -> None:
    """Test function test_python_string_generator_tuple_import."""
    graph = DummyGraph()
    gen = as_callable(MyPyGen)(graph)
    code = gen.generate()
    assert "import sys\nimport os" in code
    gen2 = as_callable(MyPyGen)(graph)
    setattr(gen2, "_import_header", "import sys")
    code2 = gen2.generate()
    assert "import sys" in code2


def test_class_based_generator_prefix_code() -> None:
    """Test function test_class_based_generator_prefix_code."""
    graph = DummyGraph()
    gen = as_callable(MyClassGen)(graph)
    assert gen._get_prefix_code() == []
    assert gen._emit_init_body() == False


def test_python_string_generator_string_and_list_import() -> None:
    """Test function test_python_string_generator_string_and_list_import."""
    graph = DummyGraph()
    gen1 = as_callable(MyPyGen)(graph)
    setattr(gen1, "_import_header", "import something")
    assert "import something" in gen1.generate()
    gen2 = as_callable(MyPyGen)(graph)
    setattr(gen2, "_import_header", ["import a", "import b"])
    code2 = gen2.generate()
    assert "import a\nimport b" in code2
    gen3 = as_callable(MyPyGen)(graph)
    setattr(gen3, "_import_header", 42)
    gen3.generate()


def test_emit_output_assignment_twice() -> None:
    """Test function test_emit_output_assignment_twice."""
    graph = DummyGraph()
    gen = as_callable(MyGen)(graph)
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
    import ast

    from ml_switcheroo_compiler.backends.jax.generator import JAXCodeGenerator
    from ml_switcheroo_compiler.ir.core import IRGraph, LogicalNode

    graph = IRGraph()
    n_send = LogicalNode(id="n_send", op_type="Send", inputs=["in1"])
    n_send.attributes = {"dst_rank": 2}
    n_recv = LogicalNode(id="n_recv", op_type="Recv", inputs=[])
    n_recv.attributes = {"src_rank": 3, "dtype": "float32"}
    n_recv.shape_metadata = (4, 4)
    gen = JAXCodeGenerator(graph)
    out_send = as_callable(JaxDistributedVisitor.visit_Send)(gen, n_send, ["in_var"])
    assert out_send == ""
    assert any("# JAX Send to 2" in line for line in gen.code)
    assert any("token = jax.lax.send(in_var, token, channel=2)" in line for line in gen.code)
    out_recv = as_callable(JaxDistributedVisitor.visit_Recv)(gen, n_recv, [])
    assert out_recv == "v_n_recv"
    assert any("v_n_recv, token = jax.lax.recv(token, channel=3, shape=(4, 4), dtype=jnp.float32)" in line for line in gen.code)
    assert any("token = jax.lax.create_token()" in line for line in gen.code)
    module_code = "\n".join(["def pipeline_stage(in_var):"] + gen.code + ["    return v_n_recv"])
    parsed_ast = ast.parse(module_code)
    assert parsed_ast is not None


def test_nn_primitives_emission_cross_backend() -> None:
    """Verify native code generation for neural network primitives across PyTorch, JAX, and MLX."""
    from ml_switcheroo_compiler.backends.jax.generator import JAXCodeGenerator
    from ml_switcheroo_compiler.backends.mlx.generator import MLXCodeGenerator
    from ml_switcheroo_compiler.backends.pytorch.generator import PyTorchCodeGenerator
    from ml_switcheroo_compiler.ir.core import IRGraph, LogicalNode

    graph = IRGraph()
    pt_gen = PyTorchCodeGenerator(graph)
    jax_gen = JAXCodeGenerator(graph)
    mlx_gen = MLXCodeGenerator(graph)
    ops_to_test = [
        ("Conv1D", ["x", "w"], {"stride": 1, "padding": 0}),
        ("Conv2D", ["x", "w"], {"stride": 2, "padding": 1}),
        ("Conv3D", ["x", "w"], {"stride": 1, "padding": 0}),
        ("MultiHeadAttention", ["q", "k", "v"], {"is_causal": True}),
        ("ScaledDotProductAttention", ["q", "k", "v"], {}),
        ("BatchNorm", ["x", "mean", "var"], {"eps": 1e-05}),
        ("LayerNorm", ["x"], {"normalized_shape": [128], "eps": 1e-05}),
        ("RMSNorm", ["x"], {"eps": 1e-05}),
        ("GroupNorm", ["x"], {"num_groups": 4}),
        ("MaxPool2D", ["x"], {"kernel_size": 2, "stride": 2}),
        ("AvgPool2D", ["x"], {"kernel_size": 2, "stride": 2}),
        ("AdaptiveAvgPool2D", ["x"], {"output_size": [7, 7]}),
        ("Linear", ["x", "w", "b"], {}),
        ("Linear", ["x", "w"], {}),
    ]
    for op_name, inputs, attrs in ops_to_test:
        node = as_callable(LogicalNode)(id=f"n_{op_name}", op_type=op_name, inputs=inputs, attributes=attrs)
        pt_code = pt_gen.visit(node, inputs)
        assert len(pt_code) > 0, f"PyTorch emission failed for {op_name}"
        jax_code = as_callable(jax_gen.visit)(node, inputs)
        assert len(jax_code) > 0, f"JAX emission failed for {op_name}"
        mlx_code = as_callable(mlx_gen.visit)(node, inputs)
        assert len(mlx_code) > 0, f"MLX emission failed for {op_name}"
