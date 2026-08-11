"""Tests for the operations registry."""

from ml_switcheroo_compiler.ops.registry import backend_mapping_registry as registry


def test_registry_get_op():
    """Test getting an operation."""
    # Transpose should definitely exist.
    op = registry.get_op("transpose")
    assert op is not None
    assert op["description"] == "The Transpose operation."

    # Missing op
    assert registry.get_op("NonExistentOp123") is None


def test_registry_get_eager_mapping():
    """Test getting eager mapping."""
    # Cupy has eager for Transpose
    mapping = registry.get_eager_mapping("cupy", "transpose")
    assert mapping == "cp.transpose"

    # Missing backend
    assert registry.get_eager_mapping("nonexistent_backend", "transpose") is None

    # Missing op
    assert registry.get_eager_mapping("cupy", "NonExistentOp123") is None


def test_registry_get_generator_mapping():
    """Test getting generator mapping."""
    mapping = registry.get_generator_mapping("numpy", "transpose")
    assert mapping == "np.transpose"

    # Missing backend
    assert registry.get_generator_mapping("nonexistent_backend", "transpose") is None

    # Missing op
    assert registry.get_generator_mapping("numpy", "NonExistentOp123") is None
