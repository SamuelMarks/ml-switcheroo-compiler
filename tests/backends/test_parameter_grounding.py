"""Tests verifying zero hallucinated parameters and rigorous contract validation."""

from __future__ import annotations

from ml_switcheroo_compiler.backends.snapshot_grounding import (
    SnapshotGroundingEngine,
)


def test_parameter_contract_all_primary_frameworks() -> None:
    """Verify parameter contracts across all primary supported ML backends."""
    engine: SnapshotGroundingEngine = SnapshotGroundingEngine()

    # PyTorch: torch.add(input, other, alpha=1, out=None)
    pt_errs: list[str] = engine.validate_parameter_contract("pytorch", "torch.add", 2, [])
    assert not pt_errs

    # NumPy: numpy.add(x1, x2, out=None, where=True)
    np_errs: list[str] = engine.validate_parameter_contract("numpy", "numpy.add", 2, [])
    assert not np_errs

    # JAX: jax.numpy.add(x1, x2)
    jax_errs: list[str] = engine.validate_parameter_contract("jax", "jax.numpy.add", 2, [])
    assert not jax_errs

    # TensorFlow: tf.math.add(x, y, name=None)
    tf_errs: list[str] = engine.validate_parameter_contract("tensorflow", "tf.math.add", 2, [])
    assert not tf_errs

    # Keras: keras.ops.add(x1, x2)
    keras_errs: list[str] = engine.validate_parameter_contract("keras", "keras.ops.add", 2, [])
    assert not keras_errs

    # MLX: mlx.core.add(a, b)
    mlx_errs: list[str] = engine.validate_parameter_contract("mlx", "mlx.core.add", 2, [])
    assert not mlx_errs


def test_discrepancy_pairs_comprehensive() -> None:
    """Verify that common cross-framework parameter confusion pairs are detected."""
    engine: SnapshotGroundingEngine = SnapshotGroundingEngine()

    # 1. dim vs axis in PyTorch sum
    err1: list[str] = engine.validate_parameter_contract("pytorch", "torch.sum", 1, ["axis"])
    assert any("does not accept keyword 'axis'. Did you mean 'dim'?" in e for e in err1)

    # 2. axis vs dim in NumPy sum
    err2: list[str] = engine.validate_parameter_contract("numpy", "numpy.sum", 1, ["dim"])
    assert any("does not accept keyword 'dim'. Did you mean 'axis'?" in e for e in err2)

    # 3. keepdim vs keepdims in PyTorch mean
    err3: list[str] = engine.validate_parameter_contract("pytorch", "torch.mean", 1, ["keepdims"])
    assert any("does not accept keyword 'keepdims'. Did you mean 'keepdim'?" in e for e in err3)

    # 4. keepdims vs keepdim in JAX mean
    err4: list[str] = engine.validate_parameter_contract("jax", "jax.numpy.mean", 1, ["keepdim"])
    assert any("does not accept keyword 'keepdim'. Did you mean 'keepdims'?" in e for e in err4)


def test_positional_overflow_and_missing_required() -> None:
    """Verify detection of missing required arguments and positional overflows."""
    engine: SnapshotGroundingEngine = SnapshotGroundingEngine()

    # Matmul requires 2 arguments, 0 provided -> missing
    missing: list[str] = engine.validate_parameter_contract("pytorch", "torch.matmul", 0, [])
    assert len(missing) == 1
    assert "Missing required arguments" in missing[0]

    # Matmul takes at most 2 positional args + out keyword -> 5 positional args is overflow
    overflow: list[str] = engine.validate_parameter_contract("pytorch", "torch.matmul", 5, [])
    assert len(overflow) == 1
    assert "Too many positional arguments" in overflow[0]


def test_var_keyword_and_var_positional_endpoints() -> None:
    """Verify endpoints with VAR_KEYWORD and VAR_POSITIONAL accept arbitrary inputs."""
    engine: SnapshotGroundingEngine = SnapshotGroundingEngine()

    # MLX export_to_dot accepts varargs / kwargs
    errs: list[str] = engine.validate_parameter_contract("mlx", "mlx.core.export_to_dot", 2, ["arbitrary_unlisted_kwarg"])
    assert not errs


def test_op_definitions_zero_parameter_hallucinations() -> None:
    """Validate that all standard op definition variants have zero parameter hallucinations."""
    from scripts.validate_parameters import validate_op_definitions_parameters

    engine: SnapshotGroundingEngine = SnapshotGroundingEngine()
    errors: list[str] = validate_op_definitions_parameters(engine)
    all_err_msg: str = "\n".join(errors)
    assert not errors, f"Detected parameter hallucinations in op definitions:\n{all_err_msg}"
