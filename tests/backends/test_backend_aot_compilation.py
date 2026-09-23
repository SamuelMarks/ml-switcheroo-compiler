"""Comprehensive unit tests for ahead-of-time (AOT) backend compilation infrastructure."""

from __future__ import annotations

import importlib
import sys
from typing import Callable
from unittest.mock import MagicMock, patch

import numpy as np

from ml_switcheroo_compiler.backends.cupy.generator import CuPyGenerator, CupyGenerator
from ml_switcheroo_compiler.backends.dask.generator import DaskGenerator
from ml_switcheroo_compiler.backends.edge.onnx import ONNXCodeGenerator
from ml_switcheroo_compiler.backends.edge.stablehlo import StableHLOCodeGenerator
from ml_switcheroo_compiler.backends.jax.generator import JAXCodeGenerator, JaxGenerator
from ml_switcheroo_compiler.backends.jax.generator_mixins import JaxDistributedVisitor
from ml_switcheroo_compiler.backends.mlx.generator import MLXCodeGenerator, MLXGenerator
from ml_switcheroo_compiler.backends.pytorch.generator import PyTorchCodeGenerator, PyTorchGenerator
from ml_switcheroo_compiler.backends.tensorflow.generator import TensorFlowCodeGenerator, TensorFlowGenerator
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode, LogicalNode


def _to_tensor(arr: np.ndarray) -> Tensor[np.ndarray]:
    """Convert numpy array into a Tensor instance with valid TensorConfig.

    Args:
        arr (np.ndarray): Input numpy ndarray.

    Returns:
        Tensor[np.ndarray]: Wrapped Tensor.
    """
    cfg = TensorConfig(shape=tuple(arr.shape), dtype="float32", device="cpu")
    return Tensor(arr, cfg)


def _create_sample_graph() -> IRGraph:
    """Create a sample computation graph: out = exp(x).

    Returns:
        IRGraph: Test computational graph.
    """
    g = IRGraph()
    n_in = IRNode(id="x", op_type="Input", inputs=[], shape_metadata=[2, 2])
    n_exp = IRNode(id="exp_out", op_type="Exp", inputs=["x"], shape_metadata=[2, 2])
    g.nodes = {"x": n_in, "exp_out": n_exp}
    g.inputs = ["x"]
    g.outputs = ["exp_out"]
    return g


def _create_multi_output_graph() -> IRGraph:
    """Create a sample computation graph with multiple outputs: y1 = exp(x), y2 = exp(x).

    Returns:
        IRGraph: Test computational graph with two outputs.
    """
    g = IRGraph()
    n_in = IRNode(id="x", op_type="Input", inputs=[], shape_metadata=[2, 2])
    n_exp1 = IRNode(id="exp1", op_type="Exp", inputs=["x"], shape_metadata=[2, 2])
    n_exp2 = IRNode(id="exp2", op_type="Exp", inputs=["x"], shape_metadata=[2, 2])
    g.nodes = {"x": n_in, "exp1": n_exp1, "exp2": n_exp2}
    g.inputs = ["x"]
    g.outputs = ["exp1", "exp2"]
    return g


def _create_no_output_graph() -> IRGraph:
    """Create a sample computation graph without explicit outputs.

    Returns:
        IRGraph: Test computational graph with empty outputs list.
    """
    g = IRGraph()
    n_in = IRNode(id="x", op_type="Input", inputs=[], shape_metadata=[2, 2])
    g.nodes = {"x": n_in}
    g.inputs = ["x"]
    g.outputs = []
    return g


def _create_multi_input_graph() -> IRGraph:
    """Create a sample computation graph with multiple inputs: out = exp(x).

    Returns:
        IRGraph: Test computational graph with two inputs.
    """
    g = IRGraph()
    n_x = IRNode(id="x", op_type="Input", inputs=[], shape_metadata=[2, 2])
    n_y = IRNode(id="y", op_type="Input", inputs=[], shape_metadata=[2, 2])
    n_exp = IRNode(id="exp_out", op_type="Exp", inputs=["x"], shape_metadata=[2, 2])
    g.nodes = {"x": n_x, "y": n_y, "exp_out": n_exp}
    g.inputs = ["x", "y"]
    g.outputs = ["exp_out"]
    return g


def test_generator_aliases() -> None:
    """Verify backend generator alias exports."""
    assert JaxGenerator is JAXCodeGenerator
    assert PyTorchGenerator is PyTorchCodeGenerator
    assert MLXGenerator is MLXCodeGenerator
    assert CuPyGenerator is CupyGenerator
    assert TensorFlowGenerator is TensorFlowCodeGenerator


def test_jax_compile_aot_impl() -> None:
    """Verify JAX AOT compilation and execution with raw and Tensor arguments."""
    graph = _create_sample_graph()
    gen = JAXCodeGenerator(graph)

    inp = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

    # 1. JIT compile fallback or execution
    callable_jit: Callable[..., object] = gen._compile_aot_impl(graph)  # type: ignore[assignment]
    res_jit = callable_jit(inp)
    assert res_jit is not None
    np.testing.assert_allclose(np.asarray(res_jit), np.exp(inp), rtol=1e-4)

    # Call JIT executable with Tensor argument
    res_jit_tensor = callable_jit(_to_tensor(inp))
    assert res_jit_tensor is not None
    np.testing.assert_allclose(np.asarray(res_jit_tensor), np.exp(inp), rtol=1e-4)

    # 2. AOT compilation with sample inputs (both raw and Tensor)
    callable_aot: Callable[..., object] = gen._compile_aot_impl(graph, sample_inputs=[_to_tensor(inp)])  # type: ignore[assignment]
    res_aot = callable_aot(inp)
    assert res_aot is not None
    np.testing.assert_allclose(np.asarray(res_aot), np.exp(inp), rtol=1e-4)

    res_aot_tensor = callable_aot(_to_tensor(inp))
    assert res_aot_tensor is not None
    np.testing.assert_allclose(np.asarray(res_aot_tensor), np.exp(inp), rtol=1e-4)

    # 3. Load / save / savez / savez_compressed methods
    with (
        patch("jax.numpy.load", return_value=inp, create=True),
        patch("jax.numpy.save", create=True),
        patch("jax.numpy.savez", create=True),
        patch("jax.numpy.savez_compressed", create=True),
    ):
        assert JAXCodeGenerator.load("dummy.npy") is not None
        JAXCodeGenerator.save("dummy.npy", inp)
        JAXCodeGenerator.savez("dummy.npz", inp, k=inp)
        JAXCodeGenerator.savez_compressed("dummy.npz", inp, k=inp)


def test_jax_compile_aot_fallback_and_missing_model() -> None:
    """Verify JAX AOT compilation exception fallback, multi-output, and missing apply_model."""
    graph = _create_sample_graph()
    gen = JAXCodeGenerator(graph)
    inp = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

    # Missing apply_model triggers ValueError, handled by except Exception fallback
    with patch.object(gen, "generate", return_value=""):
        callable_fallback: Callable[..., object] = gen._compile_aot_impl(graph)  # type: ignore[assignment]
        res_tensor = callable_fallback(_to_tensor(inp))
        np.testing.assert_allclose(np.asarray(res_tensor), np.exp(inp), rtol=1e-4)

    # Multi-output fallback
    multi_graph = _create_multi_output_graph()
    gen_multi = JAXCodeGenerator(multi_graph)
    with patch.object(gen_multi, "generate", return_value=""):
        callable_multi: Callable[..., object] = gen_multi._compile_aot_impl(multi_graph)  # type: ignore[assignment]
        res_multi = callable_multi(inp)
        assert isinstance(res_multi, tuple)
        assert len(res_multi) == 2
        np.testing.assert_allclose(np.asarray(res_multi[0]), np.exp(inp), rtol=1e-4)
        np.testing.assert_allclose(np.asarray(res_multi[1]), np.exp(inp), rtol=1e-4)

    # No-output fallback
    no_out_graph = _create_no_output_graph()
    gen_no_out = JAXCodeGenerator(no_out_graph)
    with patch.object(gen_no_out, "generate", return_value=""):
        callable_no_out: Callable[..., object] = gen_no_out._compile_aot_impl(no_out_graph)  # type: ignore[assignment]
        res_no_out = callable_no_out(inp)
        assert isinstance(res_no_out, dict)

    # Multi-input graph with fewer arguments provided
    multi_in_graph = _create_multi_input_graph()
    gen_multi_in = JAXCodeGenerator(multi_in_graph)
    with (
        patch.object(gen_multi_in, "generate", return_value=""),
        patch("ml_switcheroo_compiler.interpreter.evaluator.evaluate_graph", return_value={"exp_out": np.exp(inp)}),
    ):
        callable_few: Callable[..., object] = gen_multi_in._compile_aot_impl(multi_in_graph)  # type: ignore[assignment]
        res_few = callable_few(inp)
        assert res_few is not None


def test_jax_distributed_visitor_coverage() -> None:
    """Verify code lines extraction, token creation deduplication, shapes, and collective operations."""

    class DummyGenWithCode:
        """Dummy generator having code lines list."""

        code: list[str] = ["# existing"]

    class DummyVisitorWithGen:
        """Dummy visitor wrapping generator with code lines."""

        generator: DummyGenWithCode = DummyGenWithCode()

    class DummyEmpty:
        """Dummy target lacking code and generator."""

    class DummyTargetNonListCode:
        """Dummy target whose generator code attribute is not a list."""

        class FakeGen:
            code: str = "not a list"

        generator: FakeGen = FakeGen()

    # Test _extract_code_lines target traversal
    assert JaxDistributedVisitor._extract_code_lines(DummyVisitorWithGen()) == ["# existing"]
    assert JaxDistributedVisitor._extract_code_lines(DummyTargetNonListCode()) == []
    assert JaxDistributedVisitor._extract_code_lines(DummyEmpty()) == []

    gen = JAXCodeGenerator(IRGraph())

    # Test visit_Send when token already exists
    gen.code.append("    token = jax.lax.create_token()")
    n_send = LogicalNode(id="send_node", op_type="Send", inputs=["x"], attributes={"dst_rank": 1})
    JaxDistributedVisitor.visit_Send(gen, n_send, ["x"])
    token_creates = [line for line in gen.code if "jax.lax.create_token()" in line]
    assert len(token_creates) == 1

    # Test visit_Recv when token does NOT exist yet (fresh generator)
    gen_fresh = JAXCodeGenerator(IRGraph())
    n_recv0 = LogicalNode(id="recv_fresh", op_type="Recv", inputs=[], attributes={"src_rank": 1})
    JaxDistributedVisitor.visit_Recv(gen_fresh, n_recv0, [])
    assert any("token = jax.lax.create_token()" in line for line in gen_fresh.code)

    # Test visit_Recv when token already exists
    n_recv1 = LogicalNode(
        id="recv_shape_dict",
        op_type="Recv",
        inputs=[],
        attributes={"src_rank": 0, "shape": (3, 3), "dtype": "jnp.bfloat16"},
    )
    res_recv1 = JaxDistributedVisitor.visit_Recv(gen, n_recv1, [])
    assert res_recv1 == "v_recv_shape_dict"
    assert any("shape=(3, 3), dtype=jnp.bfloat16" in line for line in gen.code)

    # Test visit_Recv with scalar integer shape metadata
    n_recv2 = LogicalNode(id="recv_scalar_shape", op_type="Recv", inputs=[], attributes={"src_rank": 2})
    n_recv2.shape_metadata = 8  # type: ignore[assignment]
    res_recv2 = JaxDistributedVisitor.visit_Recv(gen, n_recv2, [])
    assert res_recv2 == "v_recv_scalar_shape"
    assert any("shape=(8,)" in line for line in gen.code)

    # Test visit_AllGather
    n_ag = LogicalNode(id="ag", op_type="AllGather", inputs=["x"], attributes={"axis_name": "'data'"})
    assert JaxDistributedVisitor.visit_AllGather(gen, n_ag, ["x"]) == "jax.lax.all_gather(x, axis_name='data')"

    # Test visit_ReduceScatter
    n_rs = LogicalNode(id="rs", op_type="ReduceScatter", inputs=["x"], attributes={"axis": 1, "axis_name": "'proc'", "op": "jax.lax.pmax"})
    assert JaxDistributedVisitor.visit_ReduceScatter(gen, n_rs, ["x"]) == "jax.lax.reduce_scatter(x, jax.lax.pmax, scatter_dimension=1, axis_name='proc')"

    # Test visit_AllReduce
    n_ar = LogicalNode(id="ar", op_type="AllReduce", inputs=["x"], attributes={"axis_name": "'mesh'", "op": "pmean"})
    assert JaxDistributedVisitor.visit_AllReduce(gen, n_ar, ["x"]) == "jax.lax.pmean(x, axis_name='mesh')"


def test_pytorch_compile_aot_impl() -> None:
    """Verify PyTorch AOT compilation via torch.compile and torch.jit."""
    graph = _create_sample_graph()
    gen = PyTorchCodeGenerator(graph)

    inp = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)

    # Mock torch.compile
    mock_torch = MagicMock()
    mock_compiled_fn = MagicMock(return_value=np.exp(inp))
    mock_torch.compile.return_value = mock_compiled_fn
    mock_torch.as_tensor.side_effect = lambda x: x

    with patch.dict("sys.modules", {"torch": mock_torch}):
        runner: Callable[..., object] = gen._compile_aot_impl(graph)  # type: ignore[assignment]
        res = runner(inp)
        assert res is not None

        # Call with Tensor instance
        res_tensor = runner(_to_tensor(inp))
        assert res_tensor is not None

    # Traced mode
    mock_traced_fn = MagicMock(return_value=np.exp(inp))
    mock_torch_no_compile = MagicMock(spec=["jit", "as_tensor"])
    mock_torch_no_compile.jit.trace.return_value = mock_traced_fn
    mock_torch_no_compile.as_tensor.side_effect = lambda x: x

    with patch.dict("sys.modules", {"torch": mock_torch_no_compile}):
        runner_traced: Callable[..., object] = gen._compile_aot_impl(graph, sample_inputs=[inp])  # type: ignore[assignment]
        res_traced = runner_traced(inp)
        assert res_traced is not None

        res_traced_tensor = runner_traced(_to_tensor(inp))
        assert res_traced_tensor is not None


def test_pytorch_compile_aot_fallbacks() -> None:
    """Verify PyTorch GraphModule execution, torch.compile failure fallback, and multi-output."""
    inp = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)

    # Fallback to GraphModule when torch.compile fails and sample_inputs is provided
    graph = _create_sample_graph()
    gen = PyTorchCodeGenerator(graph)

    mock_torch = MagicMock()
    mock_torch.compile.side_effect = RuntimeError("Inductor unsupported")
    mock_traced_fn = MagicMock(return_value=np.exp(inp))
    mock_torch.jit.trace.return_value = mock_traced_fn
    mock_torch.as_tensor.side_effect = lambda x: x

    with patch.dict("sys.modules", {"torch": mock_torch}):
        runner_traced: Callable[..., object] = gen._compile_aot_impl(graph, sample_inputs=[inp])  # type: ignore[assignment]
        res = runner_traced(inp)
        assert res is not None
        assert mock_torch.jit.trace.called

    # When torch has no compile and sample_inputs is None, returns GraphModule directly
    mock_torch_bare = MagicMock(spec=["as_tensor"])
    with patch.dict("sys.modules", {"torch": mock_torch_bare}):
        mod: Callable[..., object] = gen._compile_aot_impl(graph)  # type: ignore[assignment]
        res_mod = mod(_to_tensor(inp))
        np.testing.assert_allclose(np.asarray(res_mod), np.exp(inp), rtol=1e-4)

    # Outer exception fallback returns GraphModule
    with patch.dict("sys.modules", {"torch": None}):
        mod_fallback: Callable[..., object] = gen._compile_aot_impl(graph)  # type: ignore[assignment]
        res_fb = mod_fallback(inp)
        np.testing.assert_allclose(np.asarray(res_fb), np.exp(inp), rtol=1e-4)

    # Multi-output and no-output GraphModule execution
    multi_graph = _create_multi_output_graph()
    gen_multi = PyTorchCodeGenerator(multi_graph)
    with patch.dict("sys.modules", {"torch": None}):
        mod_multi: Callable[..., object] = gen_multi._compile_aot_impl(multi_graph)  # type: ignore[assignment]
        res_multi = mod_multi(inp)
        assert isinstance(res_multi, tuple)
        assert len(res_multi) == 2

    no_out_graph = _create_no_output_graph()
    gen_no_out = PyTorchCodeGenerator(no_out_graph)
    with patch.dict("sys.modules", {"torch": None}):
        mod_no_out: Callable[..., object] = gen_no_out._compile_aot_impl(no_out_graph)  # type: ignore[assignment]
        res_no_out = mod_no_out(inp)
        assert isinstance(res_no_out, dict)

    # Multi-input graph with fewer arguments provided
    multi_in_graph = _create_multi_input_graph()
    gen_in = PyTorchCodeGenerator(multi_in_graph)
    with (
        patch.dict("sys.modules", {"torch": None}),
        patch("ml_switcheroo_compiler.interpreter.evaluator.evaluate_graph", return_value={"exp_out": np.exp(inp)}),
    ):
        mod_in: Callable[..., object] = gen_in._compile_aot_impl(multi_in_graph)  # type: ignore[assignment]
        res_in = mod_in(inp)
        assert res_in is not None


def test_mlx_compile_aot_impl() -> None:
    """Verify MLX AOT compilation via mx.compile with raw and Tensor arguments."""
    graph = _create_sample_graph()
    gen = MLXCodeGenerator(graph)

    inp = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    runner_shapeless: Callable[..., object] = gen._compile_aot_impl(graph, shapeless=True)  # type: ignore[assignment]
    res_shapeless = runner_shapeless(inp)
    assert res_shapeless is not None
    np.testing.assert_allclose(np.asarray(res_shapeless), np.exp(inp), rtol=1e-4)

    runner_static: Callable[..., object] = gen._compile_aot_impl(graph, shapeless=False)  # type: ignore[assignment]
    res_static = runner_static(_to_tensor(inp))
    assert res_static is not None
    np.testing.assert_allclose(np.asarray(res_static), np.exp(inp), rtol=1e-4)

    # Load / save / savez / savez_compressed methods
    with (
        patch("mlx.core.load", return_value=inp),
        patch("mlx.core.save"),
        patch("mlx.core.save_safetensors"),
    ):
        assert MLXCodeGenerator.load("dummy.safetensors") is not None
        MLXCodeGenerator.save("dummy.npy", inp)
        MLXCodeGenerator.savez("dummy.safetensors", inp, k=inp)
        MLXCodeGenerator.savez_compressed("dummy.safetensors", inp, k=inp)


def test_mlx_compile_aot_fallbacks() -> None:
    """Verify MLX AOT compilation missing CompiledModel, multi-output, and fallback runner."""
    graph = _create_sample_graph()
    gen = MLXCodeGenerator(graph)
    inp = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    # Missing CompiledModel raises ValueError, handled by except Exception fallback
    with patch.object(gen, "generate", return_value=""):
        callable_fb: Callable[..., object] = gen._compile_aot_impl(graph)  # type: ignore[assignment]
        res = callable_fb(_to_tensor(inp))
        np.testing.assert_allclose(np.asarray(res), np.exp(inp), rtol=1e-4)

    # Multi-output fallback
    multi_graph = _create_multi_output_graph()
    gen_multi = MLXCodeGenerator(multi_graph)
    with patch.object(gen_multi, "generate", return_value=""):
        callable_multi: Callable[..., object] = gen_multi._compile_aot_impl(multi_graph)  # type: ignore[assignment]
        res_multi = callable_multi(inp)
        assert isinstance(res_multi, tuple)
        assert len(res_multi) == 2

    # No-output fallback
    no_out_graph = _create_no_output_graph()
    gen_no_out = MLXCodeGenerator(no_out_graph)
    with patch.object(gen_no_out, "generate", return_value=""):
        callable_no_out: Callable[..., object] = gen_no_out._compile_aot_impl(no_out_graph)  # type: ignore[assignment]
        res_no_out = callable_no_out(inp)
        assert isinstance(res_no_out, dict)

    # Multi-input graph with fewer arguments provided
    multi_in_graph = _create_multi_input_graph()
    gen_in = MLXCodeGenerator(multi_in_graph)
    with (
        patch.object(gen_in, "generate", return_value=""),
        patch("ml_switcheroo_compiler.interpreter.evaluator.evaluate_graph", return_value={"exp_out": np.exp(inp)}),
    ):
        callable_in: Callable[..., object] = gen_in._compile_aot_impl(multi_in_graph)  # type: ignore[assignment]
        res_in = callable_in(inp)
        assert res_in is not None


def test_cupy_compile_aot_impl() -> None:
    """Verify CuPy AOT compilation with CUDA graph recording, Tensor inputs, and fallback."""
    graph = _create_sample_graph()
    gen = CupyGenerator(graph)

    inp = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

    # Runner without CUDA graph
    mock_cp = MagicMock(spec=["asarray"])
    mock_cp.asarray.side_effect = lambda x: x
    with patch.dict("sys.modules", {"cupy": mock_cp}):
        runner: Callable[..., object] = gen._compile_aot_impl(graph)  # type: ignore[assignment]
        res = runner(inp)
        assert res is not None

        res_tensor = runner(_to_tensor(inp))
        assert res_tensor is not None

    # CUDA graph capture mode
    mock_cp_graph = MagicMock()
    mock_instance = MagicMock()
    mock_cp_graph.cuda.Graph.return_value.instantiate.return_value = mock_instance
    with patch.dict("sys.modules", {"cupy": mock_cp_graph}):
        runner_graph: Callable[..., object] = gen._compile_aot_impl(graph, sample_inputs=[inp])  # type: ignore[assignment]
        res_graph = runner_graph(inp)
        assert res_graph is not None
        assert mock_instance.launch.called


def test_cupy_compile_aot_fallbacks() -> None:
    """Verify CuPy forward_fn multi-output, fewer args, and exception fallback."""
    graph = _create_sample_graph()
    gen = CupyGenerator(graph)
    inp = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

    # Exception in try block falls back to forward_fn
    with patch.dict("sys.modules", {"cupy": None}):
        fallback_fn: Callable[..., object] = gen._compile_aot_impl(graph)  # type: ignore[assignment]
        res = fallback_fn(_to_tensor(inp))
        np.testing.assert_allclose(np.asarray(res), np.exp(inp), rtol=1e-4)

    # Multi-output forward_fn
    multi_graph = _create_multi_output_graph()
    gen_multi = CupyGenerator(multi_graph)
    with patch.dict("sys.modules", {"cupy": None}):
        fallback_multi: Callable[..., object] = gen_multi._compile_aot_impl(multi_graph)  # type: ignore[assignment]
        res_multi = fallback_multi(inp)
        assert isinstance(res_multi, tuple)
        assert len(res_multi) == 2

    # No-output forward_fn
    no_out_graph = _create_no_output_graph()
    gen_no_out = CupyGenerator(no_out_graph)
    with patch.dict("sys.modules", {"cupy": None}):
        fallback_no_out: Callable[..., object] = gen_no_out._compile_aot_impl(no_out_graph)  # type: ignore[assignment]
        res_no_out = fallback_no_out(inp)
        assert isinstance(res_no_out, dict)

    # Multi-input graph with fewer arguments provided
    multi_in_graph = _create_multi_input_graph()
    gen_in = CupyGenerator(multi_in_graph)
    with (
        patch.dict("sys.modules", {"cupy": None}),
        patch("ml_switcheroo_compiler.interpreter.evaluator.evaluate_graph", return_value={"exp_out": np.exp(inp)}),
    ):
        fallback_in: Callable[..., object] = gen_in._compile_aot_impl(multi_in_graph)  # type: ignore[assignment]
        res_in = fallback_in(inp)
        assert res_in is not None


def test_tensorflow_compile_aot_impl() -> None:
    """Verify TensorFlow AOT compilation via tf.function(jit_compile=True)."""
    graph = _create_sample_graph()
    gen = TensorFlowCodeGenerator(graph)

    inp = np.array([[0.5, 1.5], [2.5, 3.5]], dtype=np.float32)

    mock_tf = MagicMock()
    mock_compiled_fn = MagicMock(return_value=np.exp(inp))
    mock_tf.function.return_value = mock_compiled_fn
    mock_tf.convert_to_tensor.side_effect = lambda x: x

    with patch.dict("sys.modules", {"tensorflow": mock_tf}):
        runner: Callable[..., object] = gen._compile_aot_impl(graph, jit_compile=True)  # type: ignore[assignment]
        res = runner(inp)
        assert res is not None
        assert mock_tf.function.called


def test_onnx_compile_aot_impl() -> None:
    """Verify ONNX AOT compilation constructing ModelProto and InferenceSession."""
    graph = _create_sample_graph()
    gen = ONNXCodeGenerator(graph)

    inp = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

    # Mock onnxruntime InferenceSession
    mock_ort = MagicMock()
    mock_session = MagicMock()
    mock_session.run.return_value = [np.exp(inp)]
    mock_ort.InferenceSession.return_value = mock_session

    with patch.dict("sys.modules", {"onnxruntime": mock_ort}):
        runner: Callable[..., object] = gen._compile_aot_impl(graph, opset_version=18)  # type: ignore[assignment]
        res = runner(inp)
        assert res is not None
        assert mock_session.run.called

    # Fallback runner when onnxruntime is absent
    runner_fallback: Callable[..., object] = gen._compile_aot_impl(graph)  # type: ignore[assignment]
    res_fallback = runner_fallback(inp)
    assert res_fallback is not None


def test_stablehlo_compile_aot_impl() -> None:
    """Verify StableHLO AOT compilation producing .mlirbc module artifact."""
    graph = _create_sample_graph()
    gen = StableHLOCodeGenerator(graph)

    artifact: Callable[..., object] | object = gen._compile_aot_impl(graph)
    assert hasattr(artifact, "bytecode")
    assert hasattr(artifact, "mlir")
    assert len(artifact.bytecode) > 0
    assert "module" in artifact.mlir

    # Execute artifact
    inp = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    assert callable(artifact)
    res = artifact(inp)
    assert res is not None


def test_dask_compile_aot_impl() -> None:
    """Verify Dask AOT compilation constructing delayed task graph and compute with raw and Tensor args."""
    graph = _create_sample_graph()
    gen = DaskGenerator(graph)

    inp = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

    mock_dask = MagicMock()
    mock_da = MagicMock()
    mock_da.from_array.side_effect = lambda x: x
    mock_dask.delayed.return_value = lambda *args: "dask_task"
    mock_dask.compute.return_value = [np.exp(inp)]

    with patch.dict("sys.modules", {"dask": mock_dask, "dask.array": mock_da}):
        runner: Callable[..., object] = gen._compile_aot_impl(graph)  # type: ignore[assignment]
        res = runner(inp)
        assert res is not None
        assert mock_dask.compute.called

        res_tensor = runner(_to_tensor(inp))
        assert res_tensor is not None


def test_dask_compile_aot_fallbacks() -> None:
    """Verify Dask forward_fn multi-output, fewer args, exception fallback, and import handling."""
    graph = _create_sample_graph()
    gen = DaskGenerator(graph)
    inp = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

    # Exception in try block falls back to forward_fn
    with patch.dict("sys.modules", {"dask": None}):
        fallback_fn: Callable[..., object] = gen._compile_aot_impl(graph)  # type: ignore[assignment]
        res = fallback_fn(_to_tensor(inp))
        np.testing.assert_allclose(np.asarray(res), np.exp(inp), rtol=1e-4)

    # Multi-output forward_fn
    multi_graph = _create_multi_output_graph()
    gen_multi = DaskGenerator(multi_graph)
    with patch.dict("sys.modules", {"dask": None}):
        fallback_multi: Callable[..., object] = gen_multi._compile_aot_impl(multi_graph)  # type: ignore[assignment]
        res_multi = fallback_multi(inp)
        assert isinstance(res_multi, tuple)
        assert len(res_multi) == 2

    # No-output forward_fn
    no_out_graph = _create_no_output_graph()
    gen_no_out = DaskGenerator(no_out_graph)
    with patch.dict("sys.modules", {"dask": None}):
        fallback_no_out: Callable[..., object] = gen_no_out._compile_aot_impl(no_out_graph)  # type: ignore[assignment]
        res_no_out = fallback_no_out(inp)
        assert isinstance(res_no_out, dict)

    # Multi-input graph with fewer arguments provided
    multi_in_graph = _create_multi_input_graph()
    gen_in = DaskGenerator(multi_in_graph)
    with (
        patch.dict("sys.modules", {"dask": None}),
        patch("ml_switcheroo_compiler.interpreter.evaluator.evaluate_graph", return_value={"exp_out": np.exp(inp)}),
    ):
        fallback_in: Callable[..., object] = gen_in._compile_aot_impl(multi_in_graph)  # type: ignore[assignment]
        res_in = fallback_in(inp)
        assert res_in is not None

    # Module top-level import error branch
    with patch.dict(sys.modules, {"dask.array": None}):
        import ml_switcheroo_compiler.backends.dask.generator as dask_gen_mod

        importlib.reload(dask_gen_mod)
        assert dask_gen_mod.da is None

    # Restore dask_gen_mod
    importlib.reload(dask_gen_mod)
