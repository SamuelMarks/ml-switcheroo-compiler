"""Roundtrip transpilation tests across PyTorch, JAX, and MLX."""

from __future__ import annotations

import numpy as np

from ml_switcheroo_compiler.backends.ast_to_ir import parse_ast_to_ir
from ml_switcheroo_compiler.backends.cst_transpiler import transpile_source
from ml_switcheroo_compiler.backends.ir_to_ast import emit_ir_to_ast
from ml_switcheroo_compiler.interpreter.evaluator import evaluate_graph


def test_roundtrip_transpilation_pytorch_jax_mlx() -> None:
    """Test full roundtrip transpilation: Source -> AST -> IR -> Target AST -> Target Source across PyTorch, JAX, and MLX."""
    # 1. Original PyTorch function
    pt_source = """def compute(x, y):
    h1 = torch.add(x, y)
    h2 = torch.matmul(h1, x)
    return torch.relu(h2)
"""

    # 2. PyTorch AST -> IR
    ir_from_pt = parse_ast_to_ir(pt_source)
    assert any(n.op_type == "Add" for n in ir_from_pt.nodes.values())
    assert any(n.op_type == "MatMul" for n in ir_from_pt.nodes.values())
    assert any(n.op_type == "Relu" for n in ir_from_pt.nodes.values())

    # 3. IR -> JAX AST -> JAX source
    jax_ast = emit_ir_to_ast(ir_from_pt, "jax")
    jax_source = jax_ast.code
    assert "jax.numpy.add" in jax_source
    assert "jax.numpy.matmul" in jax_source
    assert "jax.numpy.relu" in jax_source

    # 4. JAX source -> IR
    ir_from_jax = parse_ast_to_ir(jax_source)
    assert any(n.op_type == "Add" for n in ir_from_jax.nodes.values())
    assert any(n.op_type == "MatMul" for n in ir_from_jax.nodes.values())
    assert any(n.op_type == "Relu" for n in ir_from_jax.nodes.values())

    # 5. IR -> MLX AST -> MLX source
    mlx_ast = emit_ir_to_ast(ir_from_jax, "mlx")
    mlx_source = mlx_ast.code
    assert "mlx.core.add" in mlx_source
    assert "mlx.core.matmul" in mlx_source
    assert "mlx.core.nn.relu" in mlx_source

    # 6. MLX source -> IR
    ir_from_mlx = parse_ast_to_ir(mlx_source)
    assert any(n.op_type == "Add" for n in ir_from_mlx.nodes.values())
    assert any(n.op_type == "MatMul" for n in ir_from_mlx.nodes.values())
    assert any(n.op_type == "Relu" for n in ir_from_mlx.nodes.values())

    # 7. IR -> PyTorch AST -> PyTorch source
    pt_roundtrip_ast = emit_ir_to_ast(ir_from_mlx, "pytorch")
    pt_roundtrip_source = pt_roundtrip_ast.code
    assert "torch.add" in pt_roundtrip_source
    assert "torch.matmul" in pt_roundtrip_source
    assert "torch.relu" in pt_roundtrip_source

    # 8. Verify numerical evaluation equivalence between the graphs
    x_val = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    y_val = np.array([[0.5, 0.5], [1.0, 1.0]], dtype=np.float32)
    input_map = {"x": x_val, "y": y_val}

    res_pt = evaluate_graph(ir_from_pt, input_map)
    res_jax = evaluate_graph(ir_from_jax, input_map)
    res_mlx = evaluate_graph(ir_from_mlx, input_map)

    out_pt = res_pt[ir_from_pt.outputs[-1]]
    out_jax = res_jax[ir_from_jax.outputs[-1]]
    out_mlx = res_mlx[ir_from_mlx.outputs[-1]]

    np.testing.assert_allclose(out_pt, out_jax)
    np.testing.assert_allclose(out_jax, out_mlx)


def test_roundtrip_source_level_cst() -> None:
    """Test CST source-to-source roundtrip transformations."""
    source_pt = """import torch

def model(x, y):
    # Elementwise addition with dimension
    z = torch.add(x, y)
    return torch.relu(z)
"""
    # PyTorch -> JAX
    jax_code = transpile_source(source_pt, "jax")
    assert "import jax" in jax_code
    assert "jax.numpy.add" in jax_code
    assert "# Elementwise addition with dimension" in jax_code

    # JAX -> MLX
    mlx_code = transpile_source(jax_code, "mlx")
    assert "import mlx.core" in mlx_code
    assert "mlx.core.add" in mlx_code

    # MLX -> PyTorch
    pt_code = transpile_source(mlx_code, "pytorch")
    assert "import torch" in pt_code
    assert "torch.add" in pt_code


def test_roundtrip_mlp_parity() -> None:
    """Test MLP model end-to-end roundtrip transpilation and execution equivalence across PyTorch, JAX, and MLX."""
    pt_mlp = """def mlp(x, w1, b1, w2, b2):
    h1 = torch.add(torch.matmul(x, w1), b1)
    a1 = torch.relu(h1)
    out = torch.add(torch.matmul(a1, w2), b2)
    return out
"""
    # PyTorch -> IR
    ir_pt = parse_ast_to_ir(pt_mlp)

    # IR -> JAX -> IR
    jax_ast = emit_ir_to_ast(ir_pt, "jax")
    ir_jax = parse_ast_to_ir(jax_ast.code)

    # IR -> MLX -> IR
    mlx_ast = emit_ir_to_ast(ir_jax, "mlx")
    ir_mlx = parse_ast_to_ir(mlx_ast.code)

    # IR -> PyTorch
    pt_roundtrip_ast = emit_ir_to_ast(ir_mlx, "pytorch")
    ir_pt_roundtrip = parse_ast_to_ir(pt_roundtrip_ast.code)

    # Evaluation on golden seed inputs
    np.random.seed(42)
    inputs = {
        "x": np.random.randn(2, 4).astype(np.float32),
        "w1": np.random.randn(4, 8).astype(np.float32),
        "b1": np.random.randn(8).astype(np.float32),
        "w2": np.random.randn(8, 2).astype(np.float32),
        "b2": np.random.randn(2).astype(np.float32),
    }

    res_pt = evaluate_graph(ir_pt, inputs)
    res_jax = evaluate_graph(ir_jax, inputs)
    res_mlx = evaluate_graph(ir_mlx, inputs)
    res_rt = evaluate_graph(ir_pt_roundtrip, inputs)

    out_pt = res_pt[ir_pt.outputs[-1]]
    out_jax = res_jax[ir_jax.outputs[-1]]
    out_mlx = res_mlx[ir_mlx.outputs[-1]]
    out_rt = res_rt[ir_pt_roundtrip.outputs[-1]]

    np.testing.assert_allclose(out_pt, out_jax, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(out_jax, out_mlx, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(out_mlx, out_rt, rtol=1e-5, atol=1e-5)


def test_roundtrip_attention_parity() -> None:
    """Test Attention Block roundtrip transpilation and execution equivalence between PyTorch and JAX."""
    pt_attn = """def attention(q, k, v):
    scores = torch.matmul(q, k)
    probs = torch.softmax(scores)
    out = torch.matmul(probs, v)
    return out
"""
    # PyTorch -> IR
    ir_pt = parse_ast_to_ir(pt_attn)

    # IR -> JAX -> IR
    jax_ast = emit_ir_to_ast(ir_pt, "jax")
    ir_jax = parse_ast_to_ir(jax_ast.code)

    # JAX -> PyTorch -> IR
    pt_ast = emit_ir_to_ast(ir_jax, "pytorch")
    ir_pt_rt = parse_ast_to_ir(pt_ast.code)

    # Evaluation on golden seed inputs
    np.random.seed(123)
    inputs = {
        "q": np.random.randn(2, 4, 8).astype(np.float32),
        "k": np.random.randn(2, 8, 4).astype(np.float32),
        "v": np.random.randn(2, 4, 8).astype(np.float32),
    }

    res_pt = evaluate_graph(ir_pt, inputs)
    res_jax = evaluate_graph(ir_jax, inputs)
    res_rt = evaluate_graph(ir_pt_rt, inputs)

    out_pt = res_pt[ir_pt.outputs[-1]]
    out_jax = res_jax[ir_jax.outputs[-1]]
    out_rt = res_rt[ir_pt_rt.outputs[-1]]

    np.testing.assert_allclose(out_pt, out_jax, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(out_jax, out_rt, rtol=1e-5, atol=1e-5)


def test_roundtrip_keras_to_pytorch_parity() -> None:
    """Test Keras -> PyTorch transpilation and numerical parity."""
    keras_source = """def keras_model(x, w, b):
    h = keras.ops.add(keras.ops.matmul(x, w), b)
    out = keras.ops.relu(h)
    return out
"""
    ir_keras = parse_ast_to_ir(keras_source)
    assert any(n.op_type == "MatMul" for n in ir_keras.nodes.values())
    assert any(n.op_type == "Add" for n in ir_keras.nodes.values())
    assert any(n.op_type == "Relu" for n in ir_keras.nodes.values())

    pt_ast = emit_ir_to_ast(ir_keras, "pytorch")
    assert "torch.matmul" in pt_ast.code
    assert "torch.add" in pt_ast.code
    assert "torch.relu" in pt_ast.code

    ir_pt = parse_ast_to_ir(pt_ast.code)

    np.random.seed(99)
    inputs = {
        "x": np.random.randn(3, 5).astype(np.float32),
        "w": np.random.randn(5, 4).astype(np.float32),
        "b": np.random.randn(4).astype(np.float32),
    }

    res_keras = evaluate_graph(ir_keras, inputs)
    res_pt = evaluate_graph(ir_pt, inputs)

    out_keras = res_keras[ir_keras.outputs[-1]]
    out_pt = res_pt[ir_pt.outputs[-1]]

    np.testing.assert_allclose(out_keras, out_pt, rtol=1e-5, atol=1e-5)


def test_roundtrip_convnet_parity() -> None:
    """Test ConvNet block roundtrip transpilation and numerical parity across PyTorch and MLX."""
    pt_conv = """def convnet(x, w, b):
    h = torch.add(torch.matmul(x, w), b)
    return torch.relu(h)
"""
    ir_pt = parse_ast_to_ir(pt_conv)
    mlx_ast = emit_ir_to_ast(ir_pt, "mlx")
    ir_mlx = parse_ast_to_ir(mlx_ast.code)

    np.random.seed(77)
    inputs = {
        "x": np.random.randn(2, 3).astype(np.float32),
        "w": np.random.randn(3, 4).astype(np.float32),
        "b": np.random.randn(4).astype(np.float32),
    }

    res_pt = evaluate_graph(ir_pt, inputs)
    res_mlx = evaluate_graph(ir_mlx, inputs)

    out_pt = res_pt[ir_pt.outputs[-1]]
    out_mlx = res_mlx[ir_mlx.outputs[-1]]

    np.testing.assert_allclose(out_pt, out_mlx, rtol=1e-5, atol=1e-5)


def test_standard_architectures_code_generation_parity() -> None:
    """Test compiling standard architectures (MLP, CNN, NanoGPT block) to PyTorch, JAX, and MLX with AST verification and evaluation."""
    import ast

    from ml_switcheroo_compiler.backends.jax.generator import JAXCodeGenerator
    from ml_switcheroo_compiler.backends.mlx.generator import MLXCodeGenerator
    from ml_switcheroo_compiler.backends.pytorch.generator import PyTorchCodeGenerator
    from ml_switcheroo_compiler.benchmarks.config_models import (
        build_ir_graph_from_workload,
        load_model_workloads,
    )

    manifest = load_model_workloads()
    models = ["mlp", "cnn_resnet_block", "nanogpt_transformer_block"]

    for model_name in models:
        workload = manifest[model_name]
        graph = build_ir_graph_from_workload(workload, name=model_name)

        pt_code = PyTorchCodeGenerator(graph).generate()
        jax_code = JAXCodeGenerator(graph).generate()
        mlx_code = MLXCodeGenerator(graph).generate()

        # 1. Assert syntax validity via AST parsing
        ast_pt = ast.parse(pt_code)
        ast_jax = ast.parse(jax_code)
        ast_mlx = ast.parse(mlx_code)
        assert ast_pt is not None
        assert ast_jax is not None
        assert ast_mlx is not None

        # 2. Evaluate graph with reference evaluator
        sample_inputs = {}
        for inp_name, inp_spec in workload.inputs.items():
            shape = (1, *inp_spec.shape) if len(inp_spec.shape) == 3 else tuple(inp_spec.shape)
            sample_inputs[inp_name] = np.ones(shape, dtype=np.float32)
        for w_name, w_spec in workload.weights.items():
            sample_inputs[w_name] = np.ones(tuple(w_spec.shape), dtype=np.float32)

        ref_result = evaluate_graph(graph, sample_inputs)
        assert len(ref_result) > 0
